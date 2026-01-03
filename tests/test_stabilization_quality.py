"""
Test to measure stabilization quality by analyzing motion between consecutive stabilized frames.
Expected: After stabilization, consecutive frames should have ~0 rotation, ~0 TY, only TX.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.video_io import load_video
from src.mosaic import build_mosaic, to_grey_scale
from src import homography_evaluation as he
from scipy.ndimage import gaussian_filter

def test_stabilization_quality(video_path: str, max_frames: int = 50):
    """
    Measure stabilization quality by computing motion between consecutive stabilized frames.
    
    Args:
        video_path: Path to input video
        max_frames: Number of frames to test (for speed)
    """
    print("="*60)
    print("STABILIZATION QUALITY TEST")
    print("="*60)
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Testing first {max_frames} frames")
    print()
    
    # Load video
    frames = load_video(video_path)[:max_frames]
    print(f"Loaded {len(frames)} frames, shape: {frames[0].shape}")
    
    # Build mosaic (includes stabilization)
    print("\nBuilding mosaic and stabilizing frames...")
    mosaic, stabilized_frames = build_mosaic(frames)
    print(f"Stabilized frames shape: {stabilized_frames.shape}")
    
    # Convert stabilized frames to grayscale for motion analysis
    print("\nConverting stabilized frames to grayscale...")
    stabilized_gray = np.zeros((len(stabilized_frames), stabilized_frames.shape[1], stabilized_frames.shape[2]), dtype=np.uint8)
    for i in range(len(stabilized_frames)):
        gray = (to_grey_scale(stabilized_frames[i]) * 255).astype(np.uint8)
        stabilized_gray[i] = gaussian_filter(gray, sigma=1.0).astype(np.uint8)
    
    # Measure motion between consecutive stabilized frames
    print("\nAnalyzing motion between consecutive stabilized frames...")
    print("(Expected: TX only, ~0 TY, ~0 rotation)\n")
    
    tx_values = []
    ty_values = []
    theta_values = []
    
    for i in range(1, len(stabilized_gray)):
        H, _ = he.find_rigid_movement_opencv_with_our_ransac(stabilized_gray[i-1], stabilized_gray[i])
        
        if H is not None:
            tx = H[0, 2]
            ty = H[1, 2]
            theta = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi  # Convert to degrees
            
            tx_values.append(tx)
            ty_values.append(ty)
            theta_values.append(theta)
            
            if i <= 10 or i % 10 == 0:  # Print first 10 and every 10th
                print(f"Frame {i-1} → {i}: TX={tx:7.2f}px, TY={ty:6.2f}px, Theta={theta:6.3f}°")
    
    # Statistics
    print("\n" + "="*60)
    print("STATISTICS:")
    print("="*60)
    
    tx_values = np.array(tx_values)
    ty_values = np.array(ty_values)
    theta_values = np.array(theta_values)
    
    print(f"\nTX (Horizontal Motion):")
    print(f"  Mean: {np.mean(tx_values):.2f} px")
    print(f"  Std:  {np.std(tx_values):.2f} px")
    print(f"  Min:  {np.min(tx_values):.2f} px")
    print(f"  Max:  {np.max(tx_values):.2f} px")
    
    print(f"\nTY (Vertical Drift - should be ~0):")
    print(f"  Mean: {np.mean(ty_values):.2f} px")
    print(f"  Std:  {np.std(ty_values):.2f} px")
    print(f"  Min:  {np.min(ty_values):.2f} px")
    print(f"  Max:  {np.max(ty_values):.2f} px")
    print(f"  Mean Absolute: {np.mean(np.abs(ty_values)):.2f} px")
    
    print(f"\nTheta (Rotation - should be ~0):")
    print(f"  Mean: {np.mean(theta_values):.3f}°")
    print(f"  Std:  {np.std(theta_values):.3f}°")
    print(f"  Min:  {np.min(theta_values):.3f}°")
    print(f"  Max:  {np.max(theta_values):.3f}°")
    print(f"  Mean Absolute: {np.mean(np.abs(theta_values)):.3f}°")
    
    # Assessment
    print("\n" + "="*60)
    print("ASSESSMENT:")
    print("="*60)
    
    ty_threshold = 2.0  # pixels
    theta_threshold = 0.5  # degrees
    
    mean_abs_ty = np.mean(np.abs(ty_values))
    mean_abs_theta = np.mean(np.abs(theta_values))
    
    ty_pass = mean_abs_ty < ty_threshold
    theta_pass = mean_abs_theta < theta_threshold
    
    print(f"TY Stabilization: {'✓ PASS' if ty_pass else '✗ FAIL'} (mean |TY| = {mean_abs_ty:.2f}px, threshold = {ty_threshold}px)")
    print(f"Rotation Stabilization: {'✓ PASS' if theta_pass else '✗ FAIL'} (mean |θ| = {mean_abs_theta:.3f}°, threshold = {theta_threshold}°)")
    
    if ty_pass and theta_pass:
        print("\n✓ Stabilization is working well!")
    else:
        print("\n✗ Stabilization has residual errors that may cause drift/curvature.")
    
    print("="*60)

if __name__ == "__main__":
    # Test with Shinkansen
    video_path = "Exercise Inputs-20251225/Shinkansen.mp4"
    test_stabilization_quality(video_path, max_frames=50)
