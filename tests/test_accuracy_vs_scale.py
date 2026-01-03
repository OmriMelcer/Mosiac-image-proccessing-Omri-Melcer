import numpy as np
import cv2
import os
import sys
import time

# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.homography_evaluation import find_rigid_movement_pyramid, ransac_rigid_movement

def warp_image(img, theta_deg, tx, ty):
    """Warps image using CV2 to create ground truth."""
    h, w = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, theta_deg, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    H_gt = np.eye(3)
    H_gt[:2] = M
    warped = cv2.warpAffine(img, M, (w, h))
    return warped, H_gt

def generate_random_sequence(img_base: np.ndarray, steps: int = 50, motion_scale: float = 1.0):
    """
    Generates a sequence with random translations (no rotation to avoid ground truth issues).
    Returns: List of (image, ground_truth_H) tuples.
    """
    sequence = []
    cumulative_H = np.eye(3)
    sequence.append((img_base, cumulative_H.copy()))
    
    for _ in range(steps):
        # Random translation only
        d_x = np.random.uniform(0.0, 0.4 * motion_scale)
        d_y = np.random.uniform(-0.2 * motion_scale, 0.2 * motion_scale)
        
        H_increment = np.array([
            [1.0, 0.0, d_x],
            [0.0, 1.0, d_y],
            [0, 0, 1]
        ])
        
        cumulative_H = H_increment @ cumulative_H
        abs_tx = cumulative_H[0, 2]
        abs_ty = cumulative_H[1, 2]
        
        img_curr, _ = warp_image(img_base, 0.0, abs_tx, abs_ty)
        sequence.append((img_curr, cumulative_H.copy()))
        
    return sequence

def process_sequence_with_scale(sequence, scale=1.0):
    """
    Runs tracking on sequence at a specific scale.
    Returns: (avg_fps, final_error_tx, final_error_ty, final_error_theta)
    """
    errors_tx = []
    errors_ty = []
    errors_theta = []
    
    cumulative_H_est = np.eye(3)
    start_time = time.time()
    
    # Scale the first image
    img_prev_full = sequence[0][0]
    if scale != 1.0:
        new_size = (int(img_prev_full.shape[1] * scale), int(img_prev_full.shape[0] * scale))
        img_prev = cv2.resize(img_prev_full, new_size, interpolation=cv2.INTER_AREA)
    else:
        img_prev = img_prev_full
    
    img_prev = cv2.GaussianBlur(img_prev, (5, 5), 1.0)
    
    for i in range(1, len(sequence)):
        img_curr_full = sequence[i][0]
        H_gt_curr = sequence[i][1]
        
        # Scale current image
        if scale != 1.0:
            new_size = (int(img_curr_full.shape[1] * scale), int(img_curr_full.shape[0] * scale))
            img_curr = cv2.resize(img_curr_full, new_size, interpolation=cv2.INTER_AREA)
        else:
            img_curr = img_curr_full
        
        img_curr = cv2.GaussianBlur(img_curr, (5, 5), 1.0)
        
        # Track at scaled resolution
        H_est_increment, matched = find_rigid_movement_pyramid(img_prev, img_curr)
        
        if H_est_increment is None:
            H_est_increment = np.eye(3)
        
        # Scale the translation back to original image coordinates
        if scale != 1.0:
            H_est_increment[0, 2] /= scale
            H_est_increment[1, 2] /= scale
        
        # Compose cumulative transformation
        cumulative_H_est = H_est_increment @ cumulative_H_est
        
        # Extract parameters
        est_tx = cumulative_H_est[0, 2]
        est_ty = cumulative_H_est[1, 2]
        est_theta = np.degrees(np.arctan2(cumulative_H_est[1, 0], cumulative_H_est[0, 0]))
        
        gt_tx = H_gt_curr[0, 2]
        gt_ty = H_gt_curr[1, 2]
        gt_theta = np.degrees(np.arctan2(H_gt_curr[1, 0], H_gt_curr[0, 0]))
        
        errors_tx.append(est_tx - gt_tx)
        errors_ty.append(est_ty - gt_ty)
        errors_theta.append(est_theta - gt_theta)
        
        img_prev = img_curr
        
    total_time = time.time() - start_time
    avg_fps = (len(sequence)-1) / total_time
    
    return avg_fps, errors_tx[-1], errors_ty[-1], errors_theta[-1]

def test_accuracy_vs_scale_synthetic():
    """
    Test accuracy vs scale tradeoff on synthetic images.
    """
    print("\n" + "="*80)
    print("ACCURACY vs SCALE: Synthetic Image (200x200)")
    print("="*80)
    
    # Create simple synthetic test image
    img = np.zeros((200, 200), dtype=np.float32)
    img[75:125, 75:125] = 255.0
    img = cv2.GaussianBlur(img, (21, 21), 3.0)
    
    print("\nGenerating test sequences...")
    seq_small = generate_random_sequence(img, steps=50, motion_scale=1.0)
    seq_medium = generate_random_sequence(img, steps=50, motion_scale=5.0)
    seq_large = generate_random_sequence(img, steps=50, motion_scale=10.0)
    
    scales = [1.0, 0.75, 0.5, 0.4, 0.3]
    
    for motion_name, seq in [("Small (0-0.4px/frame)", seq_small), 
                              ("Medium (0-2px/frame)", seq_medium),
                              ("Large (0-4px/frame)", seq_large)]:
        print(f"\n--- {motion_name} ---")
        print(f"{'Scale':<8} {'Size':<12} {'FPS':<8} {'Err_X':<10} {'Err_Y':<10} {'Err_Total':<10}")
        print("-" * 80)
        
        for scale in scales:
            fps, err_x, err_y, err_theta = process_sequence_with_scale(seq, scale)
            err_total = np.sqrt(err_x**2 + err_y**2)
            
            h, w = seq[0][0].shape
            new_h, new_w = int(h * scale), int(w * scale)
            size_str = f"{new_w}x{new_h}"
            
            print(f"{scale:<8.2f} {size_str:<12} {fps:<8.1f} {err_x:<10.2f} {err_y:<10.2f} {err_total:<10.2f}")

def test_accuracy_vs_scale_real():
    """
    Test accuracy vs scale tradeoff on real image.
    """
    print("\n" + "="*80)
    print("ACCURACY vs SCALE: Real Image (test1.jpg)")
    print("="*80)
    
    if not os.path.exists("test1.jpg"):
        print("Skipping: test1.jpg not found")
        return
    
    img = cv2.imread("test1.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    print(f"\nOriginal Image Size: {img.shape}")
    print("Generating test sequences...")
    
    seq_small = generate_random_sequence(img, steps=50, motion_scale=1.0)
    seq_medium = generate_random_sequence(img, steps=50, motion_scale=5.0)
    
    scales = [1.0, 0.75, 0.5, 0.4, 0.3]
    
    for motion_name, seq in [("Small (0-0.4px/frame)", seq_small), 
                              ("Medium (0-2px/frame)", seq_medium)]:
        print(f"\n--- {motion_name} ---")
        print(f"{'Scale':<8} {'Size':<12} {'FPS':<8} {'Err_X':<10} {'Err_Y':<10} {'Err_Total':<10}")
        print("-" * 80)
        
        for scale in scales:
            fps, err_x, err_y, err_theta = process_sequence_with_scale(seq, scale)
            err_total = np.sqrt(err_x**2 + err_y**2)
            
            h, w = seq[0][0].shape
            new_h, new_w = int(h * scale), int(w * scale)
            size_str = f"{new_w}x{new_h}"
            
            print(f"{scale:<8.2f} {size_str:<12} {fps:<8.1f} {err_x:<10.2f} {err_y:<10.2f} {err_total:<10.2f}")

def test_single_frame_accuracy():
    """
    Test single-frame accuracy at different scales (no cumulative error).
    """
    print("\n" + "="*80)
    print("SINGLE-FRAME ACCURACY vs SCALE")
    print("="*80)
    
    if not os.path.exists("test1.jpg"):
        print("Skipping: test1.jpg not found")
        return
    
    img1_orig = cv2.imread("test1.jpg")
    img1_orig = cv2.cvtColor(img1_orig, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Create ground truth with 5px motion
    gt_tx, gt_ty = 5.0, 5.0
    img2_orig, _ = warp_image(img1_orig, 0.0, gt_tx, gt_ty)
    
    print(f"\nOriginal Size: {img1_orig.shape}")
    print(f"Ground Truth: tx={gt_tx}, ty={gt_ty}")
    print(f"\n{'Scale':<8} {'Size':<12} {'Est_X':<10} {'Est_Y':<10} {'Err_X':<10} {'Err_Y':<10} {'Err_Total':<10}")
    print("-" * 80)
    
    scales = [1.0, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2]
    
    for scale in scales:
        if scale != 1.0:
            new_size = (int(img1_orig.shape[1] * scale), int(img1_orig.shape[0] * scale))
            img1 = cv2.resize(img1_orig, new_size, interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2_orig, new_size, interpolation=cv2.INTER_AREA)
        else:
            img1 = img1_orig
            img2 = img2_orig
        
        img1 = cv2.GaussianBlur(img1, (5, 5), 1.0)
        img2 = cv2.GaussianBlur(img2, (5, 5), 1.0)
        
        # Estimate motion
        H_est, _ = find_rigid_movement_pyramid(img1, img2)
        
        if H_est is None:
            print(f"{scale:<8.2f} FAILED")
            continue
        
        est_tx = H_est[0, 2] / scale  # Scale back to original coordinates
        est_ty = H_est[1, 2] / scale
        
        err_x = est_tx - gt_tx
        err_y = est_ty - gt_ty
        err_total = np.sqrt(err_x**2 + err_y**2)
        
        size_str = f"{img1.shape[1]}x{img1.shape[0]}"
        print(f"{scale:<8.2f} {size_str:<12} {est_tx:<10.2f} {est_ty:<10.2f} {err_x:<10.2f} {err_y:<10.2f} {err_total:<10.2f}")

if __name__ == "__main__":
    test_single_frame_accuracy()
    test_accuracy_vs_scale_synthetic()
    test_accuracy_vs_scale_real()
