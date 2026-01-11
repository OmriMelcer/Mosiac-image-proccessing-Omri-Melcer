"""
Diagnostic test for House video: analyze motion before and after stabilization.
Tests the first 100 frames to understand motion characteristics.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.video_io import load_video
from src.mosaic import (
    build_mosaic, 
    to_grey_scale,
    get_first_to_last_transform,
    get_anchor_frame_by_median_dy,
    apply_stabilization
)
from src import homography_evaluation as he
from scipy.ndimage import gaussian_filter


def analyze_motion(frames_gray, label="Motion"):
    """
    Analyze motion between consecutive frames.
    Returns statistics for TX, TY, and Theta.
    """
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}\n")
    
    tx_values = []
    ty_values = []
    theta_values = []
    
    for i in range(1, len(frames_gray)):
        H, _ = he.find_rigid_movement_opencv_with_our_ransac(frames_gray[i-1], frames_gray[i])
        
        if H is not None:
            tx = H[0, 2]
            ty = H[1, 2]
            theta = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi  # Convert to degrees
            
            tx_values.append(tx)
            ty_values.append(ty)
            theta_values.append(theta)
            
            if i <= 10 or i % 20 == 0:  # Print first 10 and every 20th
                print(f"Frame {i-1} → {i}: TX={tx:7.2f}px, TY={ty:6.2f}px, Theta={theta:6.3f}°")
    
    # Convert to arrays for statistics
    tx_values = np.array(tx_values)
    ty_values = np.array(ty_values)
    theta_values = np.array(theta_values)
    
    # Print statistics
    print(f"\n{'-'*60}")
    print("STATISTICS:")
    print(f"{'-'*60}\n")
    
    print(f"TX (Horizontal Motion):")
    print(f"  Mean:   {np.mean(tx_values):8.2f} px")
    print(f"  Median: {np.median(tx_values):8.2f} px")
    print(f"  Std:    {np.std(tx_values):8.2f} px")
    print(f"  Min:    {np.min(tx_values):8.2f} px")
    print(f"  Max:    {np.max(tx_values):8.2f} px")
    
    print(f"\nTY (Vertical Motion):")
    print(f"  Mean:   {np.mean(ty_values):8.2f} px")
    print(f"  Median: {np.median(ty_values):8.2f} px")
    print(f"  Std:    {np.std(ty_values):8.2f} px")
    print(f"  Min:    {np.min(ty_values):8.2f} px")
    print(f"  Max:    {np.max(ty_values):8.2f} px")
    print(f"  Mean Abs: {np.mean(np.abs(ty_values)):8.2f} px")
    
    print(f"\nTheta (Rotation):")
    print(f"  Mean:   {np.mean(theta_values):8.3f}°")
    print(f"  Median: {np.median(theta_values):8.3f}°")
    print(f"  Std:    {np.std(theta_values):8.3f}°")
    print(f"  Min:    {np.min(theta_values):8.3f}°")
    print(f"  Max:    {np.max(theta_values):8.3f}°")
    print(f"  Mean Abs: {np.mean(np.abs(theta_values)):8.3f}°")
    
    return tx_values, ty_values, theta_values


def test_house_diagnostics(max_frames=100):
    """
    Comprehensive diagnostic for House video:
    1. Analyze original frame motion
    2. Apply stabilization
    3. Analyze stabilized frame motion
    """
    video_path = "Exercise Inputs-20251225/House.mp4"
    
    print("\n" + "="*60)
    print("HOUSE VIDEO DIAGNOSTIC TEST")
    print("="*60)
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Testing first {max_frames} frames")
    
    # Load video
    frames = load_video(video_path)[:max_frames]
    print(f"\nLoaded {len(frames)} frames, shape: {frames[0].shape}")
    
    # Convert to grayscale
    print("\nConverting to grayscale...")
    frames_gray = np.zeros((len(frames), frames.shape[1], frames.shape[2]), dtype=np.uint8)
    blurred_frames = np.zeros_like(frames_gray)
    
    for i in range(len(frames)):
        gray = (to_grey_scale(frames[i]) * 255).astype(np.uint8)
        frames_gray[i] = gray
        blurred_frames[i] = gaussian_filter(gray, sigma=1.0).astype(np.uint8)
    
    # ==========================================
    # STEP 1: Analyze ORIGINAL motion
    # ==========================================
    tx_orig, ty_orig, theta_orig = analyze_motion(
        blurred_frames, 
        label="STEP 1: ORIGINAL FRAME MOTION (Before Stabilization)"
    )
    
    # ==========================================
    # STEP 2: Apply stabilization
    # ==========================================
    print(f"\n{'='*60}")
    print("STEP 2: APPLYING STABILIZATION")
    print(f"{'='*60}\n")
    
    # Get transforms and apply stabilization
    true_transform, transforms, anchor, global_x_chain = get_first_to_last_transform(
        blurred_frames, 
        anchor_index_locator_func=get_anchor_frame_by_median_dy
    )
    
    print(f"Anchor frame selected: {anchor}")
    print(f"Total frames: {len(frames)}")
    
    # Apply stabilization
    stabilized_frames = apply_stabilization(frames, true_transform)
    print(f"Stabilized frames shape: {stabilized_frames.shape}")
    
    # Convert stabilized frames to grayscale
    print("\nConverting stabilized frames to grayscale...")
    stabilized_gray = np.zeros((len(stabilized_frames), stabilized_frames.shape[1], 
                                stabilized_frames.shape[2]), dtype=np.uint8)
    stabilized_blurred = np.zeros_like(stabilized_gray)
    
    for i in range(len(stabilized_frames)):
        gray = (to_grey_scale(stabilized_frames[i]) * 255).astype(np.uint8)
        stabilized_gray[i] = gray
        stabilized_blurred[i] = gaussian_filter(gray, sigma=1.0).astype(np.uint8)
    
    # ==========================================
    # STEP 3: Analyze STABILIZED motion
    # ==========================================
    tx_stab, ty_stab, theta_stab = analyze_motion(
        stabilized_blurred, 
        label="STEP 3: STABILIZED FRAME MOTION (After Stabilization)"
    )
    
    # ==========================================
    # COMPARISON SUMMARY
    # ==========================================
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}\n")
    
    print("TY (Vertical Motion) Reduction:")
    print(f"  Original Mean |TY|: {np.mean(np.abs(ty_orig)):8.2f} px")
    print(f"  Stabilized Mean |TY|: {np.mean(np.abs(ty_stab)):8.2f} px")
    ty_reduction = (1 - np.mean(np.abs(ty_stab)) / np.mean(np.abs(ty_orig))) * 100
    print(f"  Reduction: {ty_reduction:6.1f}%")
    
    print("\nTheta (Rotation) Reduction:")
    print(f"  Original Mean |Theta|: {np.mean(np.abs(theta_orig)):8.3f}°")
    print(f"  Stabilized Mean |Theta|: {np.mean(np.abs(theta_stab)):8.3f}°")
    theta_reduction = (1 - np.mean(np.abs(theta_stab)) / np.mean(np.abs(theta_orig))) * 100
    print(f"  Reduction: {theta_reduction:6.1f}%")
    
    # Assessment
    print(f"\n{'-'*60}")
    print("ASSESSMENT:")
    print(f"{'-'*60}\n")
    
    ty_threshold = 2.0  # pixels
    theta_threshold = 0.5  # degrees
    
    mean_abs_ty_stab = np.mean(np.abs(ty_stab))
    mean_abs_theta_stab = np.mean(np.abs(theta_stab))
    
    ty_pass = mean_abs_ty_stab < ty_threshold
    theta_pass = mean_abs_theta_stab < theta_threshold
    
    print(f"TY Stabilization: {'✓ PASS' if ty_pass else '✗ FAIL'}")
    print(f"  (mean |TY| = {mean_abs_ty_stab:.2f}px, threshold = {ty_threshold}px)")
    
    print(f"\nRotation Stabilization: {'✓ PASS' if theta_pass else '✗ FAIL'}")
    print(f"  (mean |θ| = {mean_abs_theta_stab:.3f}°, threshold = {theta_threshold}°)")
    
    if ty_pass and theta_pass:
        print("\n✓ Stabilization is working well for House video!")
    else:
        print("\n✗ Stabilization has residual errors.")
        if not ty_pass:
            print(f"  - Vertical drift (TY) still present: {mean_abs_ty_stab:.2f}px")
        if not theta_pass:
            print(f"  - Rotation still present: {mean_abs_theta_stab:.3f}°")
    
    print("="*60)
    
    return {
        'original': {'tx': tx_orig, 'ty': ty_orig, 'theta': theta_orig},
        'stabilized': {'tx': tx_stab, 'ty': ty_stab, 'theta': theta_stab},
        'anchor': anchor
    }


if __name__ == "__main__":
    results = test_house_diagnostics(max_frames=100)
