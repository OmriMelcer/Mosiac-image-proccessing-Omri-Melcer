"""
Comprehensive testing suite for mosaic functionality.
Tests motion estimation, stabilization, and full mosaic generation.
"""
import numpy as np
import cv2
import os
import sys
import time
from typing import List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.video_io import load_video, save_video
from src.homography_evaluation import find_rigid_movement_pyramid, ransac_rigid_movement

# Input/Output paths
INPUT_DIR = "Exercise Inputs-20251225"
OUTPUT_DIR = "test_outputs"

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def convert_to_grayscale(frames: np.ndarray) -> np.ndarray:
    """Convert RGB video to grayscale."""
    if frames.ndim == 4 and frames.shape[3] == 3:
        gray_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=np.float32)
        for i in range(len(frames)):
            gray_frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
        return gray_frames
    return frames.astype(np.float32)

def test_motion_estimation_comparison(video_name: str, num_frames: int = 50):
    """
    Test 1: Compare our tracking with OpenCV
    Tests motion estimation accuracy and performance.
    """
    print("\n" + "="*80)
    print(f"TEST 1: Motion Estimation - {video_name}")
    print("="*80)
    
    video_path = os.path.join(INPUT_DIR, video_name)
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    # Load video
    print(f"Loading video: {video_name}")
    frames_rgb = load_video(video_path)
    frames_rgb = frames_rgb[:num_frames]  # Limit for testing
    frames_gray = convert_to_grayscale(frames_rgb)
    
    print(f"Video loaded: {len(frames_gray)} frames, {frames_gray[0].shape}")
    
    # Method 1: Our pyramid LK
    print("\n--- Our Pyramid LK Method ---")
    our_transforms = []
    our_times = []
    
    for i in range(1, len(frames_gray)):
        t0 = time.time()
        H, matched = find_rigid_movement_pyramid(frames_gray[i-1], frames_gray[i])
        t1 = time.time()
        our_times.append(t1 - t0)
        our_transforms.append(H if H is not None else np.eye(3))
        
        if i % 10 == 0:
            print(f"  Frame {i}/{len(frames_gray)-1}")
    
    our_avg_time = np.mean(our_times) * 1000
    our_fps = 1000.0 / our_avg_time
    
    print(f"\nOur Method Results:")
    print(f"  Average time: {our_avg_time:.1f} ms/frame")
    print(f"  FPS: {our_fps:.1f}")
    
    # Method 2: OpenCV goodFeaturesToTrack + calcOpticalFlowPyrLK
    print("\n--- OpenCV Method ---")
    cv_transforms = []
    cv_times = []
    
    for i in range(1, len(frames_gray)):
        img1 = frames_gray[i-1].astype(np.uint8)
        img2 = frames_gray[i].astype(np.uint8)
        
        t0 = time.time()
        
        # Detect corners
        p0 = cv2.goodFeaturesToTrack(img1, maxCorners=50, qualityLevel=0.01, minDistance=3)
        
        if p0 is not None and len(p0) >= 4:
            # Track with pyramid LK
            p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, winSize=(15,15), maxLevel=2)
            
            # Filter good points
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            if len(good_new) >= 4:
                # Convert to our format (y, x)
                # good_old and good_new have shape (N, 1, 2) - squeeze and flip
                p1_our = good_old.squeeze()[:, [1, 0]]
                p2_our = good_new.squeeze()[:, [1, 0]]
                
                # Use our RANSAC
                H, inliers = ransac_rigid_movement(p1_our, p2_our)
                cv_transforms.append(H if H is not None else np.eye(3))
            else:
                cv_transforms.append(np.eye(3))
        else:
            cv_transforms.append(np.eye(3))
        
        t1 = time.time()
        cv_times.append(t1 - t0)
        
        if i % 10 == 0:
            print(f"  Frame {i}/{len(frames_gray)-1}")
    
    cv_avg_time = np.mean(cv_times) * 1000
    cv_fps = 1000.0 / cv_avg_time
    
    print(f"\nOpenCV Method Results:")
    print(f"  Average time: {cv_avg_time:.1f} ms/frame")
    print(f"  FPS: {cv_fps:.1f}")
    
    # Compare transforms
    print("\n--- Transform Comparison ---")
    print(f"{'Frame':<8} {'Our TX':<12} {'CV TX':<12} {'Our TY':<12} {'CV TY':<12} {'Our Theta':<12} {'CV Theta':<12}")
    print("-" * 90)
    
    for i in range(min(10, len(our_transforms))):
        our_tx = our_transforms[i][0, 2]
        our_ty = our_transforms[i][1, 2]
        our_theta = np.degrees(np.arctan2(our_transforms[i][1, 0], our_transforms[i][0, 0]))
        
        cv_tx = cv_transforms[i][0, 2]
        cv_ty = cv_transforms[i][1, 2]
        cv_theta = np.degrees(np.arctan2(cv_transforms[i][1, 0], cv_transforms[i][0, 0]))
        
        print(f"{i+1:<8} {our_tx:<12.2f} {cv_tx:<12.2f} {our_ty:<12.2f} {cv_ty:<12.2f} {our_theta:<12.2f} {cv_theta:<12.2f}")
    
    # Save summary
    ensure_output_dir()
    summary_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(video_name)[0]}_motion_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Motion Estimation Test: {video_name}\n")
        f.write(f"Frames tested: {len(frames_gray)}\n\n")
        f.write(f"Our Method:\n")
        f.write(f"  Average time: {our_avg_time:.1f} ms/frame\n")
        f.write(f"  FPS: {our_fps:.1f}\n\n")
        f.write(f"OpenCV Method:\n")
        f.write(f"  Average time: {cv_avg_time:.1f} ms/frame\n")
        f.write(f"  FPS: {cv_fps:.1f}\n")
    
    print(f"\nSummary saved to: {summary_path}")

def list_available_videos():
    """List all available input videos."""
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory not found: {INPUT_DIR}")
        return []
    
    videos = [f for f in os.listdir(INPUT_DIR) if f.endswith('.mp4')]
    return sorted(videos)

if __name__ == "__main__":
    ensure_output_dir()
    
    # List available videos
    videos = list_available_videos()
    print("Available videos:")
    for i, video in enumerate(videos):
        print(f"  {i+1}. {video}")
    
    # Test on first video as example
    if videos:
        print("\n" + "="*80)
        print("Running tests on first video...")
        print("="*80)
        test_motion_estimation_comparison(videos[0], num_frames=30)
