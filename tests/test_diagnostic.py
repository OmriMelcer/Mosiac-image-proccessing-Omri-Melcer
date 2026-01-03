"""
Diagnostic test to verify motion estimation is working correctly.
Tests on synthetic data with known ground truth.
"""
import numpy as np
import cv2
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.homography_evaluation import find_rigid_movement_pyramid, ransac_rigid_movement

def create_test_image():
    """Create a test image with features."""
    img = np.zeros((480, 640), dtype=np.float32)
    
    # Add random texture
    np.random.seed(42)
    img = np.random.randint(0, 50, (480, 640)).astype(np.float32)
    
    # Add some strong features (corners)
    for i in range(10):
        y = np.random.randint(50, 430)
        x = np.random.randint(50, 590)
        size = np.random.randint(20, 40)
        img[y:y+size, x:x+size] = np.random.randint(150, 255)
    
    # Blur for gradients
    img = cv2.GaussianBlur(img, (21, 21), 3.0)
    return img

def warp_image(img, tx, ty, theta_deg):
    """Warp image with known transform."""
    h, w = img.shape
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, theta_deg, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    warped = cv2.warpAffine(img, M, (w, h))
    
    # Build full homography
    H_gt = np.eye(3)
    H_gt[:2] = M
    
    return warped, H_gt

def test_synthetic_accuracy():
    """Test both methods on synthetic data with known ground truth."""
    print("\n" + "="*80)
    print("DIAGNOSTIC: Testing on Synthetic Data with Known Ground Truth")
    print("="*80)
    
    img1 = create_test_image()
    
    # Test cases with different motions
    test_cases = [
        (5.0, 0.0, 0.0, "Pure X translation (5px)"),
        (0.0, 3.0, 0.0, "Pure Y translation (3px)"),
        (5.0, 3.0, 0.0, "XY translation (5, 3)"),
        (5.0, 2.0, 0.5, "XY + small rotation"),
        (10.0, 5.0, 0.0, "Large XY translation (10, 5)"),
    ]
    
    print(f"\n{'Test Case':<35} {'GT_TX':<10} {'GT_TY':<10} {'GT_Theta':<10} {'Est_TX':<10} {'Est_TY':<10} {'Est_Theta':<10} {'Err':<10}")
    print("-" * 110)
    
    for tx_gt, ty_gt, theta_gt, desc in test_cases:
        img2, H_gt = warp_image(img1, tx_gt, ty_gt, theta_gt)
        
        # Our method
        H_est, matched = find_rigid_movement_pyramid(img1, img2)
        
        if H_est is not None:
            tx_est = H_est[0, 2]
            ty_est = H_est[1, 2]
            theta_est = np.degrees(np.arctan2(H_est[1, 0], H_est[0, 0]))
            
            err_tx = abs(tx_est - tx_gt)
            err_ty = abs(ty_est - ty_gt)
            err_theta = abs(theta_est - theta_gt)
            err_total = np.sqrt(err_tx**2 + err_ty**2 + err_theta**2)
            
            print(f"{desc:<35} {tx_gt:<10.2f} {ty_gt:<10.2f} {theta_gt:<10.2f} {tx_est:<10.2f} {ty_est:<10.2f} {theta_est:<10.2f} {err_total:<10.2f}")
        else:
            print(f"{desc:<35} {tx_gt:<10.2f} {ty_gt:<10.2f} {theta_gt:<10.2f} {'FAILED':<10} {'FAILED':<10} {'FAILED':<10} {'--':<10}")

def test_opencv_comparison_detailed():
    """Detailed comparison on same synthetic data."""
    print("\n" + "="*80)
    print("DIAGNOSTIC: Detailed OpenCV Comparison")
    print("="*80)
    
    img1 = create_test_image()
    
    # Known motion
    tx_gt, ty_gt, theta_gt = 5.0, 3.0, 0.0
    img2, H_gt = warp_image(img1, tx_gt, ty_gt, theta_gt)
    
    print(f"\nGround Truth: TX={tx_gt}, TY={ty_gt}, Theta={theta_gt}")
    
    # Our method
    print("\n--- Our Method ---")
    H_our, matched_our = find_rigid_movement_pyramid(img1, img2)
    
    if H_our is not None:
        print(f"Transform:\n{H_our}")
        print(f"TX: {H_our[0, 2]:.3f}")
        print(f"TY: {H_our[1, 2]:.3f}")
        print(f"Theta: {np.degrees(np.arctan2(H_our[1, 0], H_our[0, 0])):.3f}°")
        print(f"Inliers: {len(matched_our) if len(matched_our.shape) > 1 else 0}")
    else:
        print("FAILED!")
    
    # OpenCV method
    print("\n--- OpenCV Method ---")
    img1_uint = img1.astype(np.uint8)
    img2_uint = img2.astype(np.uint8)
    
    # Detect
    p0 = cv2.goodFeaturesToTrack(img1_uint, maxCorners=50, qualityLevel=0.01, minDistance=3)
    print(f"Points detected: {len(p0) if p0 is not None else 0}")
    
    if p0 is not None:
        # Track
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1_uint, img2_uint, p0, None, winSize=(15,15), maxLevel=2)
        
        good_new = p1[st==1]
        good_old = p0[st==1]
        print(f"Points tracked: {len(good_new)}")
        
        # Convert to our format
        p1_our = good_old.squeeze()[:, [1, 0]]
        p2_our = good_new.squeeze()[:, [1, 0]]
        
        # Our RANSAC
        H_cv, inliers_cv = ransac_rigid_movement(p1_our, p2_our)
        
        if H_cv is not None:
            print(f"Transform:\n{H_cv}")
            print(f"TX: {H_cv[0, 2]:.3f}")
            print(f"TY: {H_cv[1, 2]:.3f}")
            print(f"Theta: {np.degrees(np.arctan2(H_cv[1, 0], H_cv[0, 0])):.3f}°")
            print(f"Inliers: {len(inliers_cv)}")
        else:
            print("RANSAC FAILED!")
    
    # Compare errors
    if H_our is not None and H_cv is not None:
        print("\n--- Error Analysis ---")
        our_err = np.sqrt((H_our[0,2] - tx_gt)**2 + (H_our[1,2] - ty_gt)**2)
        cv_err = np.sqrt((H_cv[0,2] - tx_gt)**2 + (H_cv[1,2] - ty_gt)**2)
        
        print(f"Our error: {our_err:.3f} px")
        print(f"OpenCV error: {cv_err:.3f} px")

def test_real_video_sample():
    """Test on a few frames from real video to see what's happening."""
    print("\n" + "="*80)
    print("DIAGNOSTIC: Real Video Sample Analysis")
    print("="*80)
    
    from src.video_io import load_video
    
    video_path = "Exercise Inputs-20251225/Garden.mp4"
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    frames_rgb = load_video(video_path)
    frames_gray = np.zeros((len(frames_rgb), frames_rgb[0].shape[0], frames_rgb[0].shape[1]), dtype=np.float32)
    for i in range(len(frames_rgb)):
        frames_gray[i] = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Test first pair
    print(f"\nVideo: {frames_rgb.shape}")
    print("Testing frame 0->1:")
    
    img1 = frames_gray[0]
    img2 = frames_gray[1]
    
    # Our method
    print("\n--- Our Method ---")
    H_our, matched = find_rigid_movement_pyramid(img1, img2)
    if H_our is not None:
        print(f"TX: {H_our[0, 2]:.2f}, TY: {H_our[1, 2]:.2f}, Theta: {np.degrees(np.arctan2(H_our[1, 0], H_our[0, 0])):.3f}°")
        print(f"Matched points: {len(matched) if len(matched.shape) > 1 else 0}")
    else:
        print("FAILED")
    
    # OpenCV
    print("\n--- OpenCV Method ---")
    img1_uint = img1.astype(np.uint8)
    img2_uint = img2.astype(np.uint8)
    
    p0 = cv2.goodFeaturesToTrack(img1_uint, maxCorners=50, qualityLevel=0.01, minDistance=3)
    print(f"Detected points: {len(p0) if p0 is not None else 0}")
    
    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1_uint, img2_uint, p0, None, winSize=(15,15), maxLevel=2)
        good = p1[st==1]
        print(f"Tracked points: {len(good)}")
        
        # Show some flow vectors
        print("\nSample flow vectors (first 5):")
        for i in range(min(5, len(good))):
            dx = good[i,0] - p0[st==1][i,0]
            dy = good[i,1] - p0[st==1][i,1]
            print(f"  Point {i}: dx={dx:.2f}, dy={dy:.2f}")

if __name__ == "__main__":
    test_synthetic_accuracy()
    test_opencv_comparison_detailed()
    test_real_video_sample()
