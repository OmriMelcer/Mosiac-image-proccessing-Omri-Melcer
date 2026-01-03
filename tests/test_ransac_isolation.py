"""
RANSAC Isolation Test
Tests our RANSAC implementation on:
1. Synthetic data with known ground truth
2. First 10 frames of real video with OpenCV points/tracking + our RANSAC
"""

import numpy as np
import cv2
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.video_io import load_video
from src.homography_evaluation import (
    find_rigid_movement_opencv_with_our_ransac,
    ransac_rigid_movement,
    apply_homography
)


def create_synthetic_pattern(img_size=(300, 400)):
    """Creates a textured pattern for better feature detection."""
    img = np.zeros(img_size, dtype=np.float32)
    
    # Add some squares
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
    cv2.rectangle(img, (200, 100), (300, 200), 200, -1)
    cv2.rectangle(img, (100, 180), (180, 260), 180, -1)
    
    # Blur for smooth gradients
    img = cv2.GaussianBlur(img, (21, 21), 3.0)
    return img


def apply_rigid_transform_cv(img, tx, ty, theta_deg):
    """
    Applies rigid transform using OpenCV and returns the warped image + ground truth H matrix.
    
    Args:
        img: Input image
        tx, ty: Translation in pixels (POSITIVE = move right/down)
        theta_deg: Rotation in degrees (POSITIVE = counter-clockwise)
    
    Returns:
        warped_img: Transformed image
        H_gt: 3x3 ground truth homography matrix
    """
    h, w = img.shape
    center = (w // 2, h // 2)
    
    # Get rotation matrix (2x3)
    M = cv2.getRotationMatrix2D(center, theta_deg, 1.0)
    
    # Add translation
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Convert to 3x3 homography
    H_gt = np.eye(3)
    H_gt[:2, :] = M
    
    # Warp image
    warped = cv2.warpAffine(img, M, (w, h))
    
    return warped, H_gt


def extract_transform_params(H):
    """Extract tx, ty, theta from homography matrix."""
    tx = H[0, 2]
    ty = H[1, 2]
    theta_rad = np.arctan2(H[1, 0], H[0, 0])
    theta_deg = np.degrees(theta_rad)
    return tx, ty, theta_deg


def test_synthetic_rightward_motion():
    """Test 1: Real video frame moving RIGHT (positive TX)"""
    print("\n" + "="*80)
    print("TEST 1: REAL VIDEO FRAME - RIGHTWARD MOTION (Positive TX)")
    print("="*80)
    
    # Load a frame from Garden.mp4
    video_path = os.path.join("Exercise Inputs-20251225", "Garden.mp4")
    if not os.path.exists(video_path):
        print("❌ Garden.mp4 not found")
        return
    
    from src.video_io import load_video
    frames_rgb = load_video(video_path)
    img_color = frames_rgb[0]  # Use first frame
    img1 = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Ground truth: Move RIGHT 10px, DOWN 2px, Rotate 1 degree
    gt_tx = 10.0
    gt_ty = 2.0
    gt_theta = 1.0
    
    img2, H_gt = apply_rigid_transform_cv(img1, gt_tx, gt_ty, gt_theta)
    
    print(f"\nGround Truth Transform:")
    print(f"  TX = {gt_tx:.2f} px (RIGHT)")
    print(f"  TY = {gt_ty:.2f} px (DOWN)")
    print(f"  Theta = {gt_theta:.2f}°")
    
    # Test with OpenCV points/tracking + our RANSAC
    H_est, matched = find_rigid_movement_opencv_with_our_ransac(img1, img2)
    
    if H_est is None:
        print("\n❌ FAILED: H_est is None!")
        return
    
    est_tx, est_ty, est_theta = extract_transform_params(H_est)
    
    print(f"\nEstimated Transform:")
    print(f"  TX = {est_tx:.2f} px")
    print(f"  TY = {est_ty:.2f} px")
    print(f"  Theta = {est_theta:.2f}°")
    
    print(f"\nErrors:")
    print(f"  TX Error = {est_tx - gt_tx:.2f} px")
    print(f"  TY Error = {est_ty - gt_ty:.2f} px")
    print(f"  Theta Error = {est_theta - gt_theta:.2f}°")
    print(f"  Matched Inliers = {len(matched)}")
    
    # Check accuracy
    tx_ok = abs(est_tx - gt_tx) < 0.5
    ty_ok = abs(est_ty - gt_ty) < 0.5
    theta_ok = abs(est_theta - gt_theta) < 0.1
    
    if tx_ok and ty_ok and theta_ok:
        print("\n✅ All parameters estimated accurately!")
    else:
        print(f"\n❌ Significant errors detected:")
        if not tx_ok:
            print(f"   TX error too large: {est_tx - gt_tx:.2f}px")
        if not ty_ok:
            print(f"   TY error too large: {est_ty - gt_ty:.2f}px")
        if not theta_ok:
            print(f"   Theta error too large: {est_theta - gt_theta:.2f}°")


def test_synthetic_leftward_motion():
    """Test 2: Real video frame moving LEFT (negative TX)"""
    print("\n" + "="*80)
    print("TEST 2: REAL VIDEO FRAME - LEFTWARD MOTION (Negative TX)")
    print("="*80)
    
    # Load a frame from Garden.mp4
    video_path = os.path.join("Exercise Inputs-20251225", "Garden.mp4")
    if not os.path.exists(video_path):
        print("❌ Garden.mp4 not found")
        return
    
    from src.video_io import load_video
    frames_rgb = load_video(video_path)
    img_color = frames_rgb[0]  # Use first frame
    img1 = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Ground truth: Move LEFT 10px, UP 2px, Rotate -1 degree
    gt_tx = -10.0
    gt_ty = -2.0
    gt_theta = -1.0
    
    img2, H_gt = apply_rigid_transform_cv(img1, gt_tx, gt_ty, gt_theta)
    
    print(f"\nGround Truth Transform:")
    print(f"  TX = {gt_tx:.2f} px (LEFT)")
    print(f"  TY = {gt_ty:.2f} px (UP)")
    print(f"  Theta = {gt_theta:.2f}°")
    
    # Test with OpenCV points/tracking + our RANSAC
    H_est, matched = find_rigid_movement_opencv_with_our_ransac(img1, img2)
    
    if H_est is None:
        print("\n❌ FAILED: H_est is None!")
        return
    
    est_tx, est_ty, est_theta = extract_transform_params(H_est)
    
    print(f"\nEstimated Transform:")
    print(f"  TX = {est_tx:.2f} px")
    print(f"  TY = {est_ty:.2f} px")
    print(f"  Theta = {est_theta:.2f}°")
    
    print(f"\nErrors:")
    print(f"  TX Error = {est_tx - gt_tx:.2f} px")
    print(f"  TY Error = {est_ty - gt_ty:.2f} px")
    print(f"  Theta Error = {est_theta - gt_theta:.2f}°")
    print(f"  Matched Inliers = {len(matched)}")
    
    # Check accuracy
    tx_ok = abs(est_tx - gt_tx) < 0.5
    ty_ok = abs(est_ty - gt_ty) < 0.5
    theta_ok = abs(est_theta - gt_theta) < 0.1
    
    if tx_ok and ty_ok and theta_ok:
        print("\n✅ All parameters estimated accurately!")
    else:
        print(f"\n❌ Significant errors detected:")
        if not tx_ok:
            print(f"   TX error too large: {est_tx - gt_tx:.2f}px")
        if not ty_ok:
            print(f"   TY error too large: {est_ty - gt_ty:.2f}px")
        if not theta_ok:
            print(f"   Theta error too large: {est_theta - gt_theta:.2f}°")


def test_real_video_10_frames(video_name="Garden.mp4"):
    """Test 3: First 10 frames of real video"""
    print("\n" + "="*80)
    print(f"TEST 3: REAL VIDEO - {video_name} (First 10 frames)")
    print("="*80)
    
    video_path = os.path.join("Exercise Inputs-20251225", video_name)
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    # Load video
    frames_rgb = load_video(video_path)
    frames_gray = np.zeros((len(frames_rgb), frames_rgb[0].shape[0], frames_rgb[0].shape[1]), dtype=np.float32)
    for i in range(len(frames_rgb)):
        frames_gray[i] = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    print(f"\nVideo loaded: {len(frames_gray)} frames, resolution: {frames_gray[0].shape}")
    print(f"\n{'Frame':>8} {'TX':>10} {'TY':>10} {'Theta':>10} {'Inliers':>10}")
    print("-" * 60)
    
    # Process first 10 frame pairs
    cumulative_tx = 0.0
    cumulative_ty = 0.0
    cumulative_theta = 0.0
    
    H_cumulative = np.eye(3)
    
    for i in range(min(10, len(frames_gray) - 1)):
        img1 = frames_gray[i]
        img2 = frames_gray[i + 1]
        
        H, matched = find_rigid_movement_opencv_with_our_ransac(img1, img2)
        
        if H is None:
            print(f"{i:>8} {'FAILED':>10} {'FAILED':>10} {'FAILED':>10} {0:>10}")
            continue
        
        # Accumulate
        H_cumulative = H @ H_cumulative
        
        tx, ty, theta = extract_transform_params(H_cumulative)
        num_inliers = len(matched) // 2 if len(matched) > 0 else 0
        
        print(f"{i:>8} {tx:>10.2f} {ty:>10.2f} {theta:>10.3f} {num_inliers:>10}")
    
    print("-" * 60)
    print(f"{'FINAL':>8} {tx:>10.2f} {ty:>10.2f} {theta:>10.3f}")
    
    print("\n📝 EXPECTATION: Camera pans RIGHT → TX should be POSITIVE and large")
    if tx > 0:
        print(f"✅ TX sign looks CORRECT (positive = rightward camera pan)")
    else:
        print(f"❌ TX sign is WRONG (got {tx:.2f}, expected large positive value)")
    
    if abs(ty) < abs(tx) * 0.2:
        print(f"✅ TY is small relative to TX (good for horizontal pan)")
    else:
        print(f"⚠️  TY seems large relative to TX ({ty:.2f} vs {tx:.2f})")


def test_direct_ransac_synthetic():
    """Test 4: Direct RANSAC test with perfect point correspondences"""
    print("\n" + "="*80)
    print("TEST 4: DIRECT RANSAC - Perfect Point Correspondences")
    print("="*80)
    
    # Create synthetic point correspondences with known transform
    # GT: Move RIGHT 15px, DOWN 3px, Rotate 2 degrees
    gt_tx = 15.0
    gt_ty = 3.0
    gt_theta_deg = 2.0
    gt_theta_rad = np.radians(gt_theta_deg)
    
    # Generate points in image 1 (random positions)
    np.random.seed(42)
    N = 50
    p1 = np.random.rand(N, 2) * 200 + 50  # Points in range [50, 250] for both y and x
    
    # Build ground truth transform matrix
    cos_t = np.cos(gt_theta_rad)
    sin_t = np.sin(gt_theta_rad)
    H_gt = np.array([
        [cos_t, -sin_t, gt_tx],
        [sin_t, cos_t, gt_ty],
        [0, 0, 1]
    ])
    
    # Apply transform to get p2
    # Our points are (y, x) format
    # But homography operates on (x, y)
    p1_xy = p1[:, [1, 0]]  # Convert to (x, y)
    p1_h = np.column_stack([p1_xy, np.ones(N)])
    p2_h = (H_gt @ p1_h.T).T
    p2_xy = p2_h[:, :2] / p2_h[:, 2:]
    p2 = p2_xy[:, [1, 0]]  # Convert back to (y, x)
    
    print(f"\nGround Truth Transform:")
    print(f"  TX = {gt_tx:.2f} px")
    print(f"  TY = {gt_ty:.2f} px")
    print(f"  Theta = {gt_theta_deg:.2f}°")
    print(f"  Generated {N} perfect point correspondences")
    
    # Test our RANSAC directly
    H_est, inliers = ransac_rigid_movement(p1, p2, num_iters=1000, threshold=1.0)
    
    if H_est is None:
        print("\n❌ RANSAC returned None!")
        return
    
    est_tx, est_ty, est_theta = extract_transform_params(H_est)
    
    print(f"\nEstimated Transform:")
    print(f"  TX = {est_tx:.2f} px")
    print(f"  TY = {est_ty:.2f} px")
    print(f"  Theta = {est_theta:.2f}°")
    print(f"  Inliers = {len(inliers)}/{N}")
    
    print(f"\nErrors:")
    print(f"  TX Error = {est_tx - gt_tx:.3f} px")
    print(f"  TY Error = {est_ty - gt_ty:.3f} px")
    print(f"  Theta Error = {est_theta - gt_theta_deg:.3f}°")
    
    # This should be nearly perfect
    if abs(est_tx - gt_tx) < 0.1 and abs(est_ty - gt_ty) < 0.1:
        print("\n✅ RANSAC works correctly with perfect correspondences!")
    else:
        print("\n❌ RANSAC has errors even with perfect correspondences!")
        print("   This suggests a bug in compute_rigid_movement or apply_homography")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("RANSAC ISOLATION TESTS")
    print("Testing: OpenCV Points + OpenCV Tracking + OUR RANSAC")
    print("="*80)
    
    # Run all tests
    test_direct_ransac_synthetic()
    test_synthetic_rightward_motion()
    test_synthetic_leftward_motion()
    test_real_video_10_frames("Garden.mp4")
    
    print("\n" + "="*80)
    print("Tests complete! Check results above.")
    print("="*80)
