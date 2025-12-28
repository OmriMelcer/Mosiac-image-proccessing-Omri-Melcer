
import numpy as np
import cv2
import os
import sys

# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.homography_evaluation import find_rigid_movement, apply_homography

def create_synthetic_cube(img_size=(200, 200), cube_size=50, cube_pos=(75, 75)):
    """Creates a black image with a white square, blurred for derivatives."""
    img = np.zeros(img_size, dtype=np.float32)
    y, x = cube_pos
    img[y:y+cube_size, x:x+cube_size] = 255.0
    # LK requires smooth gradients. A sharp step edge breaks the linearity assumption Ix*u = It.
    img = cv2.GaussianBlur(img, (21, 21), 3.0)
    return img

def warp_image(img, theta_deg, tx, ty):
    """Warps image using CV2 to create ground truth."""
    h, w = img.shape
    # Rotation center
    center = (w // 2, h // 2)
    
    # Get Rotation Matrix (2x3)
    M = cv2.getRotationMatrix2D(center, theta_deg, 1.0)
    
    # Add Translation
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Create full 3x3 Homography for verification/comparison
    H_gt = np.eye(3)
    H_gt[:2] = M
    
    # Warping
    warped = cv2.warpAffine(img, M, (w, h))
    return warped, H_gt

def test_synthetic_cube():
    print("\n--- Test 1: Synthetic Cube ---")
    img1 = create_synthetic_cube()
    
    # Apply small transform
    gt_theta = 0.5
    gt_tx = 0.5
    gt_ty = 0.5
    
    img2, H_gt = warp_image(img1, gt_theta, gt_tx, gt_ty)
    
    # Run Estimation
    H_est, matched_points = find_rigid_movement(img1, img2)
    
    # Extract params from H_est
    est_theta_rad = np.arctan2(H_est[1, 0], H_est[0, 0])
    est_theta_deg = np.degrees(est_theta_rad)
    est_tx = H_est[0, 2]
    est_ty = H_est[1, 2]
    
    # Extract Ground Truth from Matrix
    real_gt_tx = H_gt[0, 2]
    real_gt_ty = H_gt[1, 2]
    
    print(f"Ground Truth: theta={gt_theta:.2f}, tx={real_gt_tx:.2f}, ty={real_gt_ty:.2f}")
    print(f"Estimated:    theta={est_theta_deg:.2f}, tx={est_tx:.2f}, ty={est_ty:.2f}")
    
    # Create visual debug with points
    if len(matched_points) > 0:
        vis = np.hstack([img1, img2]).astype(np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        w = img1.shape[1]
        
        # Draw all matched points
        for i in range(len(matched_points)):
            pt1 = (int(matched_points[i, 1]), int(matched_points[i, 0]))
            pt2 = (int(matched_points[i, 3]) + w, int(matched_points[i, 2]))
            cv2.circle(vis, pt1, 2, (0, 0, 255), -1)
            cv2.circle(vis, pt2, 2, (0, 0, 255), -1)
            
        # Draw lines for a random subset
        num_lines = min(4, len(matched_points))
        indices = np.random.choice(len(matched_points), num_lines, replace=False)
        for i in indices:
            pt1 = (int(matched_points[i, 1]), int(matched_points[i, 0]))
            pt2 = (int(matched_points[i, 3]) + w, int(matched_points[i, 2]))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1) 
            
        cv2.imwrite("output_synthetic_test.jpg", vis)
        print("Saved output_synthetic_test.jpg with points and lines")
    else:
        print("No matches found for synthetic test visualization!")

def test_real_image():
    print("\n--- Test 2: Real Image (test1.jpg) ---")
    if not os.path.exists("test1.jpg"):
        print("Skipping: test1.jpg not found.")
        return

    img1_color = cv2.imread("test1.jpg")
    if img1_color is None:
        print("Failed to load test1.jpg")
        return
        
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Blur real image to help LK derivative calculation
    img1 = cv2.GaussianBlur(img1, (5, 5), 1.0)
    
    # Warp with VERY small motion for single-step LK
    # 0.1 degree rotation induces ~0.5px motion at 300px radius
    gt_theta = 0.1
    gt_tx = 0.5
    gt_ty = 0.5
    img2, H_gt = warp_image(img1, gt_theta, gt_tx, gt_ty)
    
    H_est, matched_points = find_rigid_movement(img1, img2)
    
    est_theta_rad = np.arctan2(H_est[1, 0], H_est[0, 0])
    est_theta_deg = np.degrees(est_theta_rad)
    est_tx = H_est[0, 2]
    est_ty = H_est[1, 2]
    
    real_gt_tx = H_gt[0, 2]
    real_gt_ty = H_gt[1, 2]

    print(f"Ground Truth: theta={gt_theta:.2f}, tx={real_gt_tx:.2f}, ty={real_gt_ty:.2f}")
    print(f"Estimated:    theta={est_theta_deg:.2f}, tx={est_tx:.2f}, ty={est_ty:.2f}")

    # Visualization of matches
    if len(matched_points) > 0:
        vis = np.hstack([img1, img2]).astype(np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        w = img1.shape[1]
        
        for i in range(len(matched_points)):
            pt1 = (int(matched_points[i, 1]), int(matched_points[i, 0]))
            pt2 = (int(matched_points[i, 3]) + w, int(matched_points[i, 2]))
            cv2.circle(vis, pt1, 2, (0, 0, 255), -1)
            cv2.circle(vis, pt2, 2, (0, 0, 255), -1)

        num_lines = min(4, len(matched_points))
        indices = np.random.choice(len(matched_points), num_lines, replace=False)
        for i in indices:
            pt1 = (int(matched_points[i, 1]), int(matched_points[i, 0]))
            pt2 = (int(matched_points[i, 3]) + w, int(matched_points[i, 2]))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
             
        cv2.imwrite("output_real_matches.jpg", vis)
        print("Saved match visualization to output_real_matches.jpg")

def test_gaussian_noise():
    print("\n--- Test 3: Gaussian Noise ---")
    if not os.path.exists("test1.jpg"):
        print("Skipping: test1.jpg not found.")
        return

    img1_color = cv2.imread("test1.jpg")
    if img1_color is None: return

    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Add Noise
    noise = np.random.normal(0, 10, img1.shape).astype(np.float32)
    img1_noisy = np.clip(img1 + noise, 0, 255)
    
    # Warp
    gt_theta = 1.0
    gt_tx = 5.0
    gt_ty = 5.0
    img2, H_gt = warp_image(img1_noisy, gt_theta, gt_tx, gt_ty)
    
    H_est, _ = find_rigid_movement(img1_noisy, img2)
    
    est_theta_rad = np.arctan2(H_est[1, 0], H_est[0, 0])
    est_theta_deg = np.degrees(est_theta_rad)
    est_tx = H_est[0, 2]
    est_ty = H_est[1, 2]
    
    real_gt_tx = H_gt[0, 2]
    real_gt_ty = H_gt[1, 2]
    
    print(f"Ground Truth: theta={gt_theta:.2f}, tx={real_gt_tx:.2f}, ty={real_gt_ty:.2f}")
    print(f"Estimated:    theta={est_theta_deg:.2f}, tx={est_tx:.2f}, ty={est_ty:.2f}")

def test_pure_translation():
    print("\n--- Test 4: Pure Translation (2.5px) ---")
    img1 = create_synthetic_cube()
    
    # Pure Translation, No Rotation
    gt_theta = 0.0
    gt_tx = 2.5
    gt_ty = 2.5
    
    img2, H_gt = warp_image(img1, gt_theta, gt_tx, gt_ty)
    
    H_est, matched_points = find_rigid_movement(img1, img2)
    
    est_theta_rad = np.arctan2(H_est[1, 0], H_est[0, 0])
    est_theta_deg = np.degrees(est_theta_rad)
    est_tx = H_est[0, 2]
    est_ty = H_est[1, 2]
    
    real_gt_tx = H_gt[0, 2]
    real_gt_ty = H_gt[1, 2]
    
    print(f"Ground Truth: theta={gt_theta:.2f}, tx={real_gt_tx:.2f}, ty={real_gt_ty:.2f}")
    print(f"Estimated:    theta={est_theta_deg:.2f}, tx={est_tx:.2f}, ty={est_ty:.2f}")
    
    if len(matched_points) > 0:
        vis = np.hstack([img1, img2]).astype(np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        w = img1.shape[1]
        for i in range(len(matched_points)):
            pt1 = (int(matched_points[i, 1]), int(matched_points[i, 0]))
            pt2 = (int(matched_points[i, 3]) + w, int(matched_points[i, 2]))
            cv2.circle(vis, pt1, 2, (0, 0, 255), -1)
            cv2.circle(vis, pt2, 2, (0, 0, 255), -1)
        cv2.imwrite("output_pure_translation.jpg", vis)
        print("Saved output_pure_translation.jpg")

def test_cumulative_small_motion():
    print("\n--- Test 5: Cumulative Small Motion (10 x 0.4px) ---")
    img_base = create_synthetic_cube()
    
    total_est_tx = 0.0
    total_est_ty = 0.0
    total_est_theta = 0.0
    
    gt_step_x = 0.4
    gt_step_y = 0.4
    steps = 10
    
    print(f"Running {steps} steps of ({gt_step_x}, {gt_step_y}) pixels...")
    
    # We generate a sequence of images
    # Image 0 is at (0,0)
    current_tx = 0.0
    current_ty = 0.0
    
    img_prev, _ = warp_image(img_base, 0, current_tx, current_ty)
    
    for i in range(steps):
        # Move to next position
        next_tx = current_tx + gt_step_x
        next_ty = current_ty + gt_step_y
        
        img_curr, _ = warp_image(img_base, 0, next_tx, next_ty)
        
        # Estimate motion between prev and curr
        H_est, matched_pts = find_rigid_movement(img_prev, img_curr)
        
        # Accumulate
        est_tx = H_est[0, 2]
        est_ty = H_est[1, 2]
        est_theta = np.degrees(np.arctan2(H_est[1, 0], H_est[0, 0]))
        
        total_est_tx += est_tx
        total_est_ty += est_ty
        total_est_theta += est_theta
        
        # print(f"  Step {i+1}: est=({est_tx:.2f}, {est_ty:.2f}), matches={len(matched_pts)}")
        
        # Update for next itertain
        img_prev = img_curr
        current_tx = next_tx
        current_ty = next_ty
        
    print(f"Total Ground Truth: tx={steps*gt_step_x:.2f}, ty={steps*gt_step_y:.2f}")
    print(f"Total Estimated:    tx={total_est_tx:.2f}, ty={total_est_ty:.2f}")
    print(f"Total Rotation:     theta={total_est_theta:.2f} (Expected 0.0)")



def test_cumulative_rotation_zero_sum():
    print("\n--- Test 6: Zero-Sum Cumulative Rotation ---")
    img_base = create_synthetic_cube()
    
    # Sequence of relative rotations that sum to 0
    relative_thetas = [0.1, -0.2, 0.2, -0.1, 0.3, -0.3]
    print(f"Applying relative rotations: {relative_thetas}")
    print(f"Goal: Total Theta should be near 0.0")
    
    total_est_theta = 0.0
    current_abs_theta = 0.0
    
    # Initial Frame (Theta=0)
    img_prev, _ = warp_image(img_base, current_abs_theta, 0, 0)
    
    for i, d_theta in enumerate(relative_thetas):
        # Calculate next absolute pose
        current_abs_theta += d_theta
        img_curr, _ = warp_image(img_base, current_abs_theta, 0, 0)
        
        # Estimate
        H_est, matched_pts = find_rigid_movement(img_prev, img_curr)
        
        # Extract Theta
        est_theta_rad = np.arctan2(H_est[1, 0], H_est[0, 0])
        est_theta_deg = np.degrees(est_theta_rad)
        
        total_est_theta += est_theta_deg
        
        print(f"  Step {i+1} (GT {d_theta:+.1f}): Est {est_theta_deg:+.3f}")
        
        # Update prev
        img_prev = img_curr
        
    print(f"Total Estimated Rotation: {total_est_theta:.4f} (Expected ~0.0)")



def generate_random_sequence(img_base: np.ndarray, steps: int = 50):
    """
    Generates a sequence of images with random movements.
    Returns: List of (image, ground_truth_pose_abs) tuples.
    Pose is (theta, tx, ty).
    """
    print(f"Generating sequence of {steps} frames...")
    
    sequence = []
    
    # Initial state
    current_tx = 0.0
    current_ty = 0.0
    current_theta = 0.0
    
    # Store initial
    sequence.append((img_base, (current_theta, current_tx, current_ty)))
    
    for _ in range(steps):
        # User Specs:
        # y: random [-0.4, 0.4]
        # theta: random [-0.1, 0.1]
        # x: steady [0, 0.8] -> Uniform(0, 0.8) ensures steady positive drift
        
        d_x = np.random.uniform(0.0, 0.4)
        d_y = np.random.uniform(-0.2, 0.2)
        d_theta = np.random.uniform(-0.1, 0.1)
        
        current_tx += d_x
        current_ty += d_y
        current_theta += d_theta
        
        # Warp
        img_curr, _ = warp_image(img_base, current_theta, current_tx, current_ty)
        sequence.append((img_curr, (current_theta, current_tx, current_ty)))
        
    return sequence

def process_sequence_benchmark(sequence):
    """
    Runs tracking on the generated sequence and benchmarks time.
    """
    import time
    
    print(f"Processing sequence of {len(sequence)} frames...")
    
    errors_tx = []
    errors_ty = []
    errors_theta = []
    
    total_est_tx = 0.0
    total_est_ty = 0.0
    total_est_theta = 0.0
    
    start_time = time.time()
    
    img_prev = sequence[0][0]
    prev_pose = sequence[0][1] # (theta, tx, ty)
    
    # We track incrementally, so we integrate estimated deltas
    # To compare with Abs Pose ground truth
    abs_est_tx = prev_pose[1]
    abs_est_ty = prev_pose[2]
    abs_est_theta = prev_pose[0]
    
    for i in range(1, len(sequence)):
        img_curr = sequence[i][0]
        gt_pose = sequence[i][1] # (theta, tx, ty)
        
        # Track
        step_start = time.time()
        H_est, matched = find_rigid_movement(img_prev, img_curr)
        step_end = time.time()
        
        if H_est is None:
            print(f"  Frame {i}: Tracking Lost! (H_est is None)")
            # Assume 0 motion for this failed step
            d_tx = 0.0
            d_ty = 0.0
            d_theta = 0.0
        else:
            # Extract relative motion
            d_tx = H_est[0, 2]
            d_ty = H_est[1, 2]
            d_theta = np.degrees(np.arctan2(H_est[1, 0], H_est[0, 0]))
        
        # Update Estimates
        abs_est_tx += d_tx
        abs_est_ty += d_ty
        abs_est_theta += d_theta
        
        # Compare with Ground Truth Abs
        gt_theta, gt_tx, gt_ty = gt_pose
        
        err_x = abs_est_tx - gt_tx
        err_y = abs_est_ty - gt_ty
        err_t = abs_est_theta - gt_theta
        
        errors_tx.append(err_x)
        errors_ty.append(err_y)
        errors_theta.append(err_t)
        
        # print(f"  Frame {i}: ms={1000*(step_end-step_start):.1f}, errX={err_x:.2f}, errY={err_y:.2f}")
        
        img_prev = img_curr
        
    total_time = time.time() - start_time
    avg_fps = (len(sequence)-1) / total_time
    
    print(f"--- Benchmark Results ---")
    print(f"Total Time: {total_time:.2f}s for {len(sequence)-1} pairs.")
    print(f"Average FPS: {avg_fps:.2f}")
    
    print(f"Final Drift Errors:")
    print(f"  Tx: {errors_tx[-1]:.2f} px")
    print(f"  Ty: {errors_ty[-1]:.2f} px")
    print(f"  Th: {errors_theta[-1]:.2f} deg")

def test_random_jitter_benchmark():
    print("\n--- Test 8: Random Jitter Benchmark (50 Steps) ---")
    
    # 1. Generate Synthetic Sequence
    print("\n[Synthetic Cube]")
    img_syn = create_synthetic_cube()
    seq_syn = generate_random_sequence(img_syn, steps=50)
    process_sequence_benchmark(seq_syn)
    
    # 2. Generate Real Sequence
    if os.path.exists("test1.jpg"):
        print("\n[Real Image]")
        img_real = cv2.imread("test1.jpg")
        img_real = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY).astype(np.float32)
        img_real = cv2.GaussianBlur(img_real, (5, 5), 1.0) # Always blur
        seq_real = generate_random_sequence(img_real, steps=50)
        process_sequence_benchmark(seq_real)
        
        print("\n--- OpenCV Baseline Benchmark ---")
        process_sequence_benchmark_cv2(seq_real)

def process_sequence_benchmark_cv2(sequence):
    """
    Runs tracking using OpenCV components for comparison.
    1. cv2.goodFeaturesToTrack
    2. cv2.calcOpticalFlowPyrLK
    3. Our RANSAC (to keep pose estimation consistent)
    """
    import time
    
    print(f"Processing sequence (OpenCV Baseline) of {len(sequence)} frames...")
    
    errors_tx = []
    errors_ty = []
    errors_theta = []
    
    abs_est_tx = sequence[0][1][1]
    abs_est_ty = sequence[0][1][2]
    abs_est_theta = sequence[0][1][0]
    
    start_time = time.time()
    
    img_prev = sequence[0][0].astype(np.uint8) # CV2 LK needs uint8 usually, or float32. Let's ensure compatibility.
    # Actually calcOpticalFlowPyrLK accepts floats.
    
    for i in range(1, len(sequence)):
        img_curr = sequence[i][0].astype(np.uint8)
        gt_pose = sequence[i][1]
        
        # 1. Detect
        p0 = cv2.goodFeaturesToTrack(img_prev, maxCorners=50, qualityLevel=0.01, minDistance=3)
        
        if p0 is None or len(p0) < 2:
            d_tx, d_ty, d_theta = 0, 0, 0
        else:
            # 2. Track
            # p0 shape is (N, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(img_prev, img_curr, p0, None, winSize=(15,15), maxLevel=0)
            
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            # 3. RANSAC
            # Convert to our format (N, 2) where cols are (y, x) -> No wait, existing code uses (y, x)?
            # My code:
            #   get_harris_points returns (y, x)
            #   track_features takes (y, x)
            #   compute_rigid_movement takes (y, x)
            # CV2 returns (x, y)
            
            if len(good_new) < 2:
                d_tx, d_ty, d_theta = 0, 0, 0
            else:
                # Flip to (y, x) for our RANSAC
                p1_our = good_old[:, [1, 0]]
                p2_our = good_new[:, [1, 0]]
                
                from src.homography_evaluation import ransac_rigid_movement
                H_est, inliers = ransac_rigid_movement(p1_our, p2_our)
                
                if H_est is None:
                    d_tx, d_ty, d_theta = 0, 0, 0
                else:
                    d_tx = H_est[0, 2]
                    d_ty = H_est[1, 2]
                    d_theta = np.degrees(np.arctan2(H_est[1, 0], H_est[0, 0]))

        # Update Estimates
        abs_est_tx += d_tx
        abs_est_ty += d_ty
        abs_est_theta += d_theta
        
        # Compare
        gt_theta, gt_tx, gt_ty = gt_pose
        
        errors_tx.append(abs_est_tx - gt_tx)
        errors_ty.append(abs_est_ty - gt_ty)
        errors_theta.append(abs_est_theta - gt_theta)
        
        img_prev = img_curr

    total_time = time.time() - start_time
    avg_fps = (len(sequence)-1) / total_time
    
    print(f"--- OpenCV Benchmark Results ---")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Final Drift Errors:")
    print(f"  Tx: {errors_tx[-1]:.2f} px")
    print(f"  Ty: {errors_ty[-1]:.2f} px")
    print(f"  Th: {errors_theta[-1]:.2f} deg")

if __name__ == "__main__":
    test_random_jitter_benchmark()
