#!/usr/bin/env python3
"""
Test if get_stabilization_transform_one_pixel_shift positions frames correctly
Expected: Frame i should be at position (i - anchor) * direction from anchor
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.video_io import load_video
from src.mosaic import to_grey_scale, get_first_to_last_transform, get_stabilization_transform_one_pixel_shift
from scipy.ndimage import gaussian_filter

def test_transform_positions():
    print("="*60)
    print("TEST: Transform Positioning")
    print("="*60)
    
    # Load first 100 frames
    frames = load_video("Exercise Inputs-20251225/Shinkansen.mp4")[:100]
    print(f"\nLoaded {len(frames)} frames")
    
    # Convert to grayscale and blur (same as build_mosaic_one_column)
    converted_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=frames.dtype)
    blurred_frames = np.zeros_like(converted_frames)
    for i in range(len(frames)):
        converted_frames[i] = (to_grey_scale(frames[i])*255).astype(np.uint8)
        blurred_frames[i] = gaussian_filter(converted_frames[i], sigma=1.0).astype(np.uint8)
    
    # Get transforms with one_pixel_shift
    true_transform, transforms, anchor, global_x_chain = get_first_to_last_transform(
        blurred_frames, get_stabilization_transform_one_pixel_shift
    )
    
    print(f"\nAnchor frame: {anchor}")
    print(f"Direction: {np.sign(true_transform[-1, 0, 2])}")
    direction = np.sign(true_transform[-1, 0, 2])
    
    print("\n" + "="*60)
    print("CHECKING TRANSFORM POSITIONS:")
    print("="*60)
    print(f"{'Frame':<8} {'Expected TX':<15} {'Actual TX':<15} {'Difference':<15}")
    print("-"*60)
    
    errors = []
    for i in range(len(true_transform)):
        expected_tx = direction * (i - anchor)
        actual_tx = true_transform[i, 0, 2]
        diff = actual_tx - expected_tx
        errors.append(abs(diff))
        
        # Print every 10th frame
        if i % 10 == 0 or i == anchor:
            print(f"{i:<8} {expected_tx:<15.2f} {actual_tx:<15.2f} {diff:<15.2f}")
    
    print("-"*60)
    print(f"\nError Statistics:")
    print(f"  Mean absolute error: {np.mean(errors):.3f} pixels")
    print(f"  Max absolute error: {np.max(errors):.3f} pixels")
    print(f"  Std error: {np.std(errors):.3f} pixels")
    
    if np.max(errors) > 2.0:
        print(f"\n❌ FAIL: Transforms are NOT at expected positions!")
        print(f"   Max error {np.max(errors):.2f}px exceeds 2.0px threshold")
    else:
        print(f"\n✓ PASS: Transforms are at expected positions")

if __name__ == "__main__":
    test_transform_positions()
