"""
Full Video Comparison Test
Compares two algorithms on complete videos:
1. Our method: find_rigid_movement_pyramid (uses current NMS setting in code)
2. OpenCV method: OpenCV points + OpenCV LK + Our RANSAC

Usage: Run twice, manually changing num_nms parameter between runs
"""

import numpy as np
import cv2
import os
import sys
import time
from typing import Tuple, List

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.video_io import load_video
from src.homography_evaluation import find_rigid_movement_pyramid, track_features_pyramid, build_pyramid, compute_gradients, ransac_rigid_movement


def process_video_hybrid_method(frames_gray: np.ndarray) -> Tuple[List[np.ndarray], List[int], float]:
    """
    Process video using Hybrid Method:
    OpenCV Points -> Our LK -> Our RANSAC
    """
    transforms = []
    inlier_counts = []
    
    start_time = time.time()
    
    # Limit to 100 frames for faster testing
    limit = min(100, len(frames_gray) - 1)
    
    for i in range(limit):
        img1 = frames_gray[i]
        img2 = frames_gray[i + 1]
        
        # 1. Detect OpenCV points
        img1_uint = img1.astype(np.uint8)
        p0_cv = cv2.goodFeaturesToTrack(img1_uint, maxCorners=200, qualityLevel=0.01, minDistance=10)
        
        if p0_cv is not None and len(p0_cv) > 0:
            # Convert to our format (y, x)
            # Use reshape to avoid squeeze issues with single point
            points = p0_cv.reshape(-1, 2)[:, [1, 0]]
            
            # 2. Track with Our LK
            # We need to build pyramids manually
            # No extra blur, to match find_rigid_movement_pyramid
            
            im_1_pyr = build_pyramid(img1, num_levels=5)
            im_2_pyr = build_pyramid(img2, num_levels=5)
            
            Im1_Ix_pyr = []
            Im1_Iy_pyr = []
            for level_img in im_1_pyr:
                Ix, Iy = compute_gradients(level_img)
                Im1_Ix_pyr.append(Ix)
                Im1_Iy_pyr.append(Iy)
                
            p1, p2 = track_features_pyramid(im_1_pyr, im_2_pyr, Im1_Ix_pyr, Im1_Iy_pyr, points, window_size=15)
            
            # 3. RANSAC
            if len(p1) >= 4:
                H, inliers_indices = ransac_rigid_movement(p1, p2)
                
                # Check if H is None (RANSAC failed)
                if H is None:
                    transforms.append(np.eye(3))
                    inlier_counts.append(0)
                    continue

                # Count inliers (re-verify)
                # ransac_rigid_movement returns inliers indices
                inliers = len(inliers_indices)
                
                transforms.append(H)
                inlier_counts.append(inliers)
            else:
                transforms.append(np.eye(3))
                inlier_counts.append(0)
        else:
            transforms.append(np.eye(3))
            inlier_counts.append(0)
            
    elapsed_time = time.time() - start_time
    fps = limit / elapsed_time
    
    return transforms, inlier_counts, fps


def process_video_our_method(frames_gray: np.ndarray) -> Tuple[List[np.ndarray], List[int], float]:
    """
    Process video using our find_rigid_movement_pyramid function
    (This uses whatever NMS size is currently set in get_harris_points default)
    
    Returns:
        transforms: List of 3x3 homography matrices
        inlier_counts: List of inlier counts per frame
        fps: Frames per second
    """
    transforms = []
    inlier_counts = []
    
    start_time = time.time()
    
    # Limit to 100 frames for faster testing
    limit = min(100, len(frames_gray) - 1)
    
    # Removed downsampling as per user request
    # h, w = frames_gray[0].shape
    # target_pixels = (h * w) // 4
    # print(f"Downsampling to {target_pixels} pixels (1/4 resolution)")

    for i in range(limit):
        img1 = frames_gray[i]
        img2 = frames_gray[i + 1]
        
        # Use the existing pipeline function (default target_pixels=-1)
        H, matched = find_rigid_movement_pyramid(img1, img2)
        
        if H is not None:
            transforms.append(H)
            inlier_counts.append(len(matched) // 2 if len(matched) > 0 else 0)
        else:
            transforms.append(np.eye(3))
            inlier_counts.append(0)
    
    elapsed_time = time.time() - start_time
    fps = limit / elapsed_time
    
    return transforms, inlier_counts, fps



def process_video_opencv_method(frames_gray: np.ndarray) -> Tuple[List[np.ndarray], List[int], float]:
    """
    Process video using OpenCV points + OpenCV LK + Our RANSAC
    
    Returns:
        transforms: List of 3x3 homography matrices
        inlier_counts: List of inlier counts per frame
        fps: Frames per second
    """
    from src.homography_evaluation import ransac_rigid_movement
    
    transforms = []
    inlier_counts = []
    
    start_time = time.time()
    
    # Limit to 100 frames for faster testing
    limit = min(100, len(frames_gray) - 1)
    for i in range(limit):
        img1 = frames_gray[i]
        img2 = frames_gray[i + 1]
        
        img1_uint = img1.astype(np.uint8)
        img2_uint = img2.astype(np.uint8)
        
        # Detect OpenCV points
        p0_cv = cv2.goodFeaturesToTrack(img1_uint, maxCorners=100, qualityLevel=0.01, minDistance=3)
        
        if p0_cv is not None and len(p0_cv) > 0:
            # Track with OpenCV LK
            p1_cv, st_cv, err_cv = cv2.calcOpticalFlowPyrLK(
                img1_uint, img2_uint, p0_cv, None, 
                winSize=(15, 15), maxLevel=2
            )
            
            # Get successful tracks
            good_old = p0_cv[st_cv == 1]
            good_new = p1_cv[st_cv == 1]
            
            if len(good_old) >= 4:
                # Convert to our format (y, x)
                p1_our = good_old.squeeze()[:, [1, 0]]
                p2_our = good_new.squeeze()[:, [1, 0]]
                
                # RANSAC
                H, inliers = ransac_rigid_movement(p1_our, p2_our)
                if H is not None:
                    transforms.append(H)
                    inlier_counts.append(len(inliers))
                else:
                    transforms.append(np.eye(3))
                    inlier_counts.append(0)
            else:
                transforms.append(np.eye(3))
                inlier_counts.append(0)
        else:
            transforms.append(np.eye(3))
            inlier_counts.append(0)
    
    elapsed_time = time.time() - start_time
    fps = limit / elapsed_time
    
    return transforms, inlier_counts, fps


def accumulate_transforms(transforms: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Accumulate transforms and extract TX, TY, Theta at each frame
    
    Returns:
        tx_cumulative: Array of cumulative TX values
        ty_cumulative: Array of cumulative TY values
        theta_cumulative: Array of cumulative theta values (in degrees)
    """
    H_cumulative = np.eye(3)
    tx_values = []
    ty_values = []
    theta_values = []
    
    for H in transforms:
        H_cumulative = H @ H_cumulative
        
        tx_values.append(H_cumulative[0, 2])
        ty_values.append(H_cumulative[1, 2])
        theta_values.append(np.degrees(np.arctan2(H_cumulative[1, 0], H_cumulative[0, 0])))
    
    return np.array(tx_values), np.array(ty_values), np.array(theta_values)


def compute_comparison_stats(tx1, ty1, theta1, tx2, ty2, theta2, label1: str, label2: str):
    """
    Compute comparison statistics between two methods
    """
    tx_diff = tx1 - tx2
    ty_diff = ty1 - ty2
    theta_diff = theta1 - theta2
    
    print(f"\n{'='*80}")
    print(f"COMPARISON: {label1} vs {label2}")
    print(f"{'='*80}")
    
    print(f"\nTX Differences:")
    print(f"  Mean: {np.mean(np.abs(tx_diff)):.3f} px")
    print(f"  Std Dev: {np.std(tx_diff):.3f} px")
    print(f"  Max: {np.max(np.abs(tx_diff)):.3f} px")
    print(f"  Final Cumulative Diff: {tx_diff[-1]:.3f} px")
    
    print(f"\nTY Differences:")
    print(f"  Mean: {np.mean(np.abs(ty_diff)):.3f} px")
    print(f"  Std Dev: {np.std(ty_diff):.3f} px")
    print(f"  Max: {np.max(np.abs(ty_diff)):.3f} px")
    print(f"  Final Cumulative Diff: {ty_diff[-1]:.3f} px")
    
    print(f"\nTheta Differences:")
    print(f"  Mean: {np.mean(np.abs(theta_diff)):.3f}°")
    print(f"  Std Dev: {np.std(theta_diff):.3f}°")
    print(f"  Max: {np.max(np.abs(theta_diff)):.3f}°")
    print(f"  Final Cumulative Diff: {theta_diff[-1]:.3f}°")


def test_video_full_comparison(video_name: str):
    """
    Full comparison test on a complete video - Our method vs OpenCV
    """
    print("\n" + "="*80)
    print(f"FULL VIDEO COMPARISON TEST: {video_name}")
    print("="*80)
    
    video_path = os.path.join("Exercise Inputs-20251225", video_name)
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    # Load video
    print("\nLoading video...")
    frames_rgb = load_video(video_path)
    frames_gray = np.zeros((len(frames_rgb), frames_rgb[0].shape[0], frames_rgb[0].shape[1]), dtype=np.float32)
    for i in range(len(frames_rgb)):
        frames_gray[i] = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    print(f"Video loaded: {len(frames_gray)} frames, resolution: {frames_gray[0].shape}")
    
    # Method 1: Our method (uses current NMS setting)
    print("\n" + "-"*80)
    print("METHOD 1: Our Harris + Our LK + Our RANSAC")
    print("(Using current num_nms setting in get_harris_points)")
    print("-"*80)
    transforms_1, inliers_1, fps_1 = process_video_our_method(frames_gray)
    tx_1, ty_1, theta_1 = accumulate_transforms(transforms_1)
    print(f"Processing complete: {fps_1:.2f} FPS")
    print(f"Average inliers per frame: {np.mean(inliers_1):.1f}")
    print(f"Final cumulative: TX={tx_1[-1]:.2f}, TY={ty_1[-1]:.2f}, Theta={theta_1[-1]:.3f}°")
    
    # Method 2: OpenCV
    print("\n" + "-"*80)
    print("METHOD 2: OpenCV Points + OpenCV LK + Our RANSAC")
    print("-"*80)
    transforms_2, inliers_2, fps_2 = process_video_opencv_method(frames_gray)
    tx_2, ty_2, theta_2 = accumulate_transforms(transforms_2)
    print(f"Processing complete: {fps_2:.2f} FPS")
    print(f"Average inliers per frame: {np.mean(inliers_2):.1f}")
    print(f"Final cumulative: TX={tx_2[-1]:.2f}, TY={ty_2[-1]:.2f}, Theta={theta_2[-1]:.3f}°")

    # 3. Hybrid Method
    print("\n" + "-"*80)
    print("METHOD 3: OpenCV Points + Our LK + Our RANSAC")
    print("-"*80)
    transforms_3, inliers_3, fps_3 = process_video_hybrid_method(frames_gray)
    print(f"Processing complete: {fps_3:.2f} FPS")
    print(f"Average inliers per frame: {np.mean(inliers_3):.1f}")
    
    tx_3, ty_3, theta_3 = accumulate_transforms(transforms_3)
    print(f"Final cumulative: TX={tx_3[-1]:.2f}, TY={ty_3[-1]:.2f}, Theta={theta_3[-1]:.3f}°")
    
    # Performance Comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"\nOur Method: {fps_1:.2f} FPS")
    print(f"OpenCV Method: {fps_2:.2f} FPS")
    print(f"Hybrid Method: {fps_3:.2f} FPS")
    print(f"OpenCV Speedup: {fps_2/fps_1:.2f}x")
    
    # Accuracy Comparison
    compute_comparison_stats(tx_1, ty_1, theta_1, tx_2, ty_2, theta_2, 
                            "Our Method", "OpenCV")
    compute_comparison_stats(tx_3, ty_3, theta_3, tx_2, ty_2, theta_2, 
                            "Hybrid Method", "OpenCV")
    
    # Save results to file
    output_file = f"test_outputs/{video_name.replace('.mp4', '')}_comparison.txt"
    with open(output_file, 'w') as f:
        f.write(f"Full Video Comparison: {video_name}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"Video: {len(frames_gray)} frames, {frames_gray[0].shape}\n\n")
        
        f.write(f"Our Method: {fps_1:.2f} FPS, Avg Inliers: {np.mean(inliers_1):.1f}\n")
        f.write(f"  Final Cumulative: TX={tx_1[-1]:.2f}, TY={ty_1[-1]:.2f}, Theta={theta_1[-1]:.3f}°\n\n")
        
        f.write(f"OpenCV Method: {fps_2:.2f} FPS, Avg Inliers: {np.mean(inliers_2):.1f}\n")
        f.write(f"  Final Cumulative: TX={tx_2[-1]:.2f}, TY={ty_2[-1]:.2f}, Theta={theta_2[-1]:.3f}°\n\n")

        f.write(f"Hybrid Method: {fps_3:.2f} FPS, Avg Inliers: {np.mean(inliers_3):.1f}\n")
        f.write(f"  Final Cumulative: TX={tx_3[-1]:.2f}, TY={ty_3[-1]:.2f}, Theta={theta_3[-1]:.3f}°\n\n")
        
        f.write(f"Performance: OpenCV is {fps_2/fps_1:.2f}x faster\n\n")
        
        f.write(f"Accuracy Differences (Our vs OpenCV):\n")
        f.write(f"  TX: Mean={np.mean(np.abs(tx_1-tx_2)):.3f}px, Std={np.std(tx_1-tx_2):.3f}px, Max={np.max(np.abs(tx_1-tx_2)):.3f}px\n")
        f.write(f"  TY: Mean={np.mean(np.abs(ty_1-ty_2)):.3f}px, Std={np.std(ty_1-ty_2):.3f}px, Max={np.max(np.abs(ty_1-ty_2)):.3f}px\n")
        f.write(f"  Theta: Mean={np.mean(np.abs(theta_1-theta_2)):.3f}°, Std={np.std(theta_1-theta_2):.3f}°, Max={np.max(np.abs(theta_1-theta_2)):.3f}°\n")
        f.write(f"  Final Cumulative Difference: TX={tx_1[-1]-tx_2[-1]:.3f}px, TY={ty_1[-1]-ty_2[-1]:.3f}px, Theta={theta_1[-1]-theta_2[-1]:.3f}°\n")

        f.write(f"Accuracy Differences (Hybrid vs OpenCV):\n")
        f.write(f"  TX: Mean={np.mean(np.abs(tx_3-tx_2)):.3f}px, Std={np.std(tx_3-tx_2):.3f}px, Max={np.max(np.abs(tx_3-tx_2)):.3f}px\n")
        f.write(f"  TY: Mean={np.mean(np.abs(ty_3-ty_2)):.3f}px, Std={np.std(ty_3-ty_2):.3f}px, Max={np.max(np.abs(ty_3-ty_2)):.3f}px\n")
        f.write(f"  Theta: Mean={np.mean(np.abs(theta_3-theta_2)):.3f}°, Std={np.std(theta_3-theta_2):.3f}°, Max={np.max(np.abs(theta_3-theta_2)):.3f}°\n")
        f.write(f"  Final Cumulative Difference: TX={tx_3[-1]-tx_2[-1]:.3f}px, TY={ty_3[-1]-ty_2[-1]:.3f}px, Theta={theta_3[-1]-theta_2[-1]:.3f}°\n")
    
    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("FULL VIDEO COMPARISON: Our Method vs OpenCV")
    print("="*80)
    print("\nNOTE: This uses the current num_nms setting in get_harris_points()")
    print("To test different NMS values, change num_nms default in src/homography_evaluation.py")
    print("="*80)
    
    test_video_full_comparison("Garden.mp4")
    print("\n\n")
    test_video_full_comparison("House.mp4")
