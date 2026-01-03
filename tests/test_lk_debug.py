"""
Debug LK: Compare our LK vs OpenCV LK on the SAME input points.
This isolates whether the issue is in LK tracking itself.
"""
import numpy as np
import cv2
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.video_io import load_video
from src.homography_evaluation import (
    build_pyramid, compute_gradients, optical_flow_pyramid, 
    ransac_rigid_movement, track_features_pyramid, harris_response, get_harris_points
)

def test_lk_comparison_same_points(video_name: str, frame_idx: int = 0):
    """
    Step 1: Use OpenCV's detected points
    Step 2: Track with BOTH our LK and OpenCV LK
    Step 3: Compare results
    """
    print("\n" + "="*80)
    print(f"LK DEBUG: Same Points, Different Trackers - {video_name}")
    print(f"Frames: {frame_idx} -> {frame_idx+1}")
    print("="*80)
    
    video_path = os.path.join("Exercise Inputs-20251225", video_name)
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    # Load frames
    frames_rgb = load_video(video_path)
    frames_gray = np.zeros((len(frames_rgb), frames_rgb[0].shape[0], frames_rgb[0].shape[1]), dtype=np.float32)
    for i in range(len(frames_rgb)):
        frames_gray[i] = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    img1 = frames_gray[frame_idx]
    img2 = frames_gray[frame_idx + 1]
    
    # ==================== USE OPENCV TO DETECT POINTS ====================
    print("\n--- Detecting Points (OpenCV) ---")
    img1_uint = img1.astype(np.uint8)
    img2_uint = img2.astype(np.uint8)
    
    p0_cv = cv2.goodFeaturesToTrack(img1_uint, maxCorners=50, qualityLevel=0.01, minDistance=3)
    
    if p0_cv is None:
        print("No points detected!")
        return
    
    print(f"Detected: {len(p0_cv)} points")
    
    # Convert OpenCV points to our format (y, x)
    points_our_format = p0_cv.squeeze()[:, [1, 0]]  # (N, 2) - y, x
    
    # ==================== TRACK WITH OPENCV LK ====================
    print("\n--- Tracking with OpenCV LK ---")
    p1_cv, st_cv, err_cv = cv2.calcOpticalFlowPyrLK(
        img1_uint, img2_uint, p0_cv, None, 
        winSize=(15,15), maxLevel=2
    )
    
    # Get successful tracks
    good_new_cv = p1_cv[st_cv==1]
    good_old_cv = p0_cv[st_cv==1]
    
    print(f"Successfully tracked: {len(good_new_cv)}/{len(p0_cv)} points")
    
    # Convert to our format
    p1_cv_our = good_old_cv.squeeze()[:, [1, 0]]
    p2_cv_our = good_new_cv.squeeze()[:, [1, 0]]
    
    # ==================== TRACK WITH OUR PYRAMID LK ====================
    print("\n--- Tracking with Our Pyramid LK ---")
    
    # Build pyramids
    pyr1 = build_pyramid(img1, num_levels=3)
    pyr2 = build_pyramid(img2, num_levels=3)
    
    # Compute gradients
    IX_pyr = []
    IY_pyr = []
    for level_img in pyr1:
        Ix, Iy = compute_gradients(level_img)
        IX_pyr.append(Ix)
        IY_pyr.append(Iy)
    
    # Track using our method
    p1_ours, p2_ours = track_features_pyramid(
        pyr1, pyr2, IX_pyr, IY_pyr, 
        points_our_format, window_size=15
    )
    
    print(f"Successfully tracked: {len(p1_ours)}/{len(points_our_format)} points")
    
    # ==================== COMPARE RESULTS ====================
    print("\n" + "="*80)
    print("COMPARISON: Point-by-Point Differences")
    print("="*80)
    
    # Find common points (that both methods tracked successfully)
    # Match by original position
    common_indices_our = []
    common_indices_cv = []
    
    for i, p_our in enumerate(p1_ours):
        for j, p_cv in enumerate(p1_cv_our):
            # Check if same original point (within 0.5px)
            if np.linalg.norm(p_our - p_cv) < 0.5:
                common_indices_our.append(i)
                common_indices_cv.append(j)
                break
    
    print(f"\nCommon successfully tracked points: {len(common_indices_our)}")
    print(f"Our method only: {len(p1_ours) - len(common_indices_our)}")
    print(f"OpenCV only: {len(p1_cv_our) - len(common_indices_cv)}")
    
    # Compare tracking for common points
    if len(common_indices_our) > 0:
        print(f"\n{'Idx':<6} {'Our DX':<10} {'CV DX':<10} {'Diff DX':<10} {'Our DY':<10} {'CV DY':<10} {'Diff DY':<10} {'Dist':<10}")
        print("-" * 90)
        
        diffs = []
        for k, (i, j) in enumerate(zip(common_indices_our, common_indices_cv)):
            our_dx = p2_ours[i, 1] - p1_ours[i, 1]
            our_dy = p2_ours[i, 0] - p1_ours[i, 0]
            
            cv_dx = p2_cv_our[j, 1] - p1_cv_our[j, 1]
            cv_dy = p2_cv_our[j, 0] - p1_cv_our[j, 0]
            
            diff_dx = our_dx - cv_dx
            diff_dy = our_dy - cv_dy
            dist = np.sqrt(diff_dx**2 + diff_dy**2)
            
            diffs.append(dist)
            
            if k < 15:  # Show first 15
                print(f"{k:<6} {our_dx:<10.3f} {cv_dx:<10.3f} {diff_dx:<10.3f} {our_dy:<10.3f} {cv_dy:<10.3f} {diff_dy:<10.3f} {dist:<10.3f}")
        
        print(f"\n--- Statistics ---")
        print(f"Mean difference: {np.mean(diffs):.3f} px")
        print(f"Median difference: {np.median(diffs):.3f} px")
        print(f"Max difference: {np.max(diffs):.3f} px")
        print(f"Std deviation: {np.std(diffs):.3f} px")
        
        # Histogram of differences
        bins = [0, 0.1, 0.5, 1.0, 2.0, 5.0, 100]
        hist, _ = np.histogram(diffs, bins=bins)
        print(f"\nDifference distribution:")
        for i in range(len(bins)-1):
            print(f"  {bins[i]:.1f}-{bins[i+1]:.1f} px: {hist[i]} points ({100*hist[i]/len(diffs):.1f}%)")
    
    # ==================== RANSAC COMPARISON ====================
    print("\n" + "="*80)
    print("RANSAC: Transform Estimation")
    print("="*80)
    
    # Our LK + Our RANSAC
    if len(p1_ours) >= 4:
        H_our, inliers_our = ransac_rigid_movement(p1_ours, p2_ours)
        if H_our is not None:
            print(f"\nOur LK -> Our RANSAC:")
            print(f"  TX: {H_our[0, 2]:.3f}, TY: {H_our[1, 2]:.3f}, Theta: {np.degrees(np.arctan2(H_our[1, 0], H_our[0, 0])):.3f}°")
            print(f"  Inliers: {len(inliers_our)}/{len(p1_ours)}")
    
    # OpenCV LK + Our RANSAC
    if len(p1_cv_our) >= 4:
        H_cv, inliers_cv = ransac_rigid_movement(p1_cv_our, p2_cv_our)
        if H_cv is not None:
            print(f"\nOpenCV LK -> Our RANSAC:")
            print(f"  TX: {H_cv[0, 2]:.3f}, TY: {H_cv[1, 2]:.3f}, Theta: {np.degrees(np.arctan2(H_cv[1, 0], H_cv[0, 0])):.3f}°")
            print(f"  Inliers: {len(inliers_cv)}/{len(p1_cv_our)}")
    
    # ==================== ANALYZE FAILED TRACKS ====================
    print("\n" + "="*80)
    print("FAILED TRACKS ANALYSIS")
    print("="*80)
    
    # Points that OpenCV tracked but we didn't
    cv_success_set = set()
    for p in p1_cv_our:
        cv_success_set.add((round(p[0], 1), round(p[1], 1)))
    
    our_success_set = set()
    for p in p1_ours:
        our_success_set.add((round(p[0], 1), round(p[1], 1)))
    
    cv_only = cv_success_set - our_success_set
    our_only = our_success_set - cv_success_set
    
    print(f"\nPoints OpenCV tracked but we failed: {len(cv_only)}")
    if len(cv_only) > 0:
        print("Sample locations (y, x):")
        for i, pt in enumerate(list(cv_only)[:5]):
            print(f"  {pt}")
    
    print(f"\nPoints we tracked but OpenCV failed: {len(our_only)}")
    if len(our_only) > 0:
        print("Sample locations (y, x):")
        for i, pt in enumerate(list(our_only)[:5]):
            print(f"  {pt}")

def test_harris_vs_opencv_points(video_name: str, frame_idx: int = 0):
    """
    Direct comparison: Are OUR Harris points the problem?
    Test 1: OpenCV points -> Our LK -> Our RANSAC
    Test 2: Our Harris points -> Our LK -> Our RANSAC
    """
    print("\n" + "="*80)
    print(f"HARRIS POINT QUALITY TEST - {video_name}")
    print(f"Frames: {frame_idx} -> {frame_idx+1}")
    print("="*80)
    
    video_path = os.path.join("Exercise Inputs-20251225", video_name)
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    # Load frames
    frames_rgb = load_video(video_path)
    frames_gray = np.zeros((len(frames_rgb), frames_rgb[0].shape[0], frames_rgb[0].shape[1]), dtype=np.float32)
    for i in range(len(frames_rgb)):
        frames_gray[i] = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    img1 = frames_gray[frame_idx]
    img2 = frames_gray[frame_idx + 1]
    img1_uint = img1.astype(np.uint8)
    
    # ==================== DETECT WITH BOTH METHODS ====================
    print("\n--- Point Detection ---")
    
    # OpenCV points
    p0_cv = cv2.goodFeaturesToTrack(img1_uint, maxCorners=100, qualityLevel=0.01, minDistance=3)
    points_cv = p0_cv.squeeze()[:, [1, 0]]  # Convert to (y, x)
    print(f"OpenCV detected: {len(points_cv)} points")
    
    # Our Harris points (now with relative threshold and 100 max corners)
    harris_resp = harris_response(img1, window_size=3)
    points_harris = get_harris_points(harris_resp, threshold=0.01, max_corners=100)
    print(f"Our Harris detected: {len(points_harris)} points")
    
    # ==================== BUILD PYRAMIDS (once) ====================
    pyr1 = build_pyramid(img1, num_levels=3)
    pyr2 = build_pyramid(img2, num_levels=3)
    
    IX_pyr = []
    IY_pyr = []
    for level_img in pyr1:
        Ix, Iy = compute_gradients(level_img)
        IX_pyr.append(Ix)
        IY_pyr.append(Iy)
    
    # ==================== TEST 1: OpenCV Points -> Our LK ====================
    print("\n--- Test 1: OpenCV Points -> Our LK ---")
    p1_cv, p2_cv = track_features_pyramid(
        pyr1, pyr2, IX_pyr, IY_pyr, 
        points_cv, window_size=15
    )
    print(f"Tracked: {len(p1_cv)}/{len(points_cv)} points ({100*len(p1_cv)/len(points_cv):.1f}% success)")
    
    if len(p1_cv) >= 4:
        H_cv, inliers_cv = ransac_rigid_movement(p1_cv, p2_cv)
        if H_cv is not None:
            print(f"Transform: TX={H_cv[0, 2]:.3f}, TY={H_cv[1, 2]:.3f}, Theta={np.degrees(np.arctan2(H_cv[1, 0], H_cv[0, 0])):.3f}°")
            print(f"RANSAC Inliers: {len(inliers_cv)}/{len(p1_cv)} ({100*len(inliers_cv)/len(p1_cv):.1f}%)")
    
    # ==================== TEST 2: Our Harris Points -> Our LK ====================
    print("\n--- Test 2: Our Harris Points -> Our LK ---")
    p1_harris, p2_harris = track_features_pyramid(
        pyr1, pyr2, IX_pyr, IY_pyr, 
        points_harris, window_size=15
    )
    print(f"Tracked: {len(p1_harris)}/{len(points_harris)} points ({100*len(p1_harris)/len(points_harris):.1f}% success)")
    
    if len(p1_harris) >= 4:
        H_harris, inliers_harris = ransac_rigid_movement(p1_harris, p2_harris)
        if H_harris is not None:
            print(f"Transform: TX={H_harris[0, 2]:.3f}, TY={H_harris[1, 2]:.3f}, Theta={np.degrees(np.arctan2(H_harris[1, 0], H_harris[0, 0])):.3f}°")
            print(f"RANSAC Inliers: {len(inliers_harris)}/{len(p1_harris)} ({100*len(inliers_harris)/len(p1_harris):.1f}%)")
    
    # ==================== SUMMARY COMPARISON ====================
    print("\n" + "="*80)
    print("SUMMARY: Point Selection Impact")
    print("="*80)
    
    tracking_success_cv = 100 * len(p1_cv) / len(points_cv)
    tracking_success_harris = 100 * len(p1_harris) / len(points_harris)
    
    print(f"\nTracking Success Rate:")
    print(f"  OpenCV points: {tracking_success_cv:.1f}%")
    print(f"  Harris points: {tracking_success_harris:.1f}%")
    print(f"  Difference: {tracking_success_cv - tracking_success_harris:+.1f}%")
    
    if len(p1_cv) >= 4 and len(p1_harris) >= 4:
        inlier_rate_cv = 100 * len(inliers_cv) / len(p1_cv)
        inlier_rate_harris = 100 * len(inliers_harris) / len(p1_harris)
        
        print(f"\nRANSAC Inlier Rate:")
        print(f"  OpenCV points: {inlier_rate_cv:.1f}%")
        print(f"  Harris points: {inlier_rate_harris:.1f}%")
        print(f"  Difference: {inlier_rate_cv - inlier_rate_harris:+.1f}%")
        
        print(f"\nTotal Useful Tracks (inliers):")
        print(f"  OpenCV points: {len(inliers_cv)}")
        print(f"  Harris points: {len(inliers_harris)}")
        print(f"  Difference: {len(inliers_cv) - len(inliers_harris):+d}")

if __name__ == "__main__":
    # Test on multiple frame pairs to confirm consistency
    frame_pairs = [0, 5, 10, 15, 20]
    
    print("\n" + "="*80)
    print("PART 1: HARRIS POINT QUALITY TEST - MULTIPLE FRAME PAIRS")
    print("Testing if our Harris points consistently underperform OpenCV points")
    print("="*80)
    
    results = []
    for frame_idx in frame_pairs:
        print(f"\n{'*'*80}")
        print(f"FRAME PAIR: {frame_idx} -> {frame_idx+1}")
        print(f"{'*'*80}")
        test_harris_vs_opencv_points("Garden.mp4", frame_idx=frame_idx)
    
    print("\n\n" + "="*80)
    print("PART 2: DETAILED LK TRACKING COMPARISON")
    print("Comparing our LK vs OpenCV LK on same (OpenCV) points")
    print("="*80)
    
    # Only run detailed comparison on a few frames
    for frame_idx in [0, 10, 20]:
        test_lk_comparison_same_points("Garden.mp4", frame_idx=frame_idx)
        print("\n")
