#!/usr/bin/env python3
"""
Test if canvas dimensions are calculated correctly for build_mosaic_one_column
Expected: width = frame_width + number_of_frames_with_|tx| > 0.5
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.video_io import load_video
from src.mosaic import to_grey_scale, get_first_to_last_transform, get_stabilization_transform_one_pixel_shift
from scipy.ndimage import gaussian_filter

def test_canvas_dimensions():
    print("="*60)
    print("TEST: Canvas Dimension Calculation")
    print("="*60)
    
    # Load first 100 frames
    frames = load_video("Exercise Inputs-20251225/Shinkansen.mp4")[:100]
    print(f"\nLoaded {len(frames)} frames")
    print(f"Frame shape: {frames[0].shape}")
    frame_width = frames.shape[2]
    
    # Convert to grayscale and blur
    converted_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=frames.dtype)
    blurred_frames = np.zeros_like(converted_frames)
    for i in range(len(frames)):
        converted_frames[i] = (to_grey_scale(frames[i])*255).astype(np.uint8)
        blurred_frames[i] = gaussian_filter(converted_frames[i], sigma=1.0).astype(np.uint8)
    
    # Get transforms
    true_transform, transforms, anchor, global_x_chain = get_first_to_last_transform(
        blurred_frames, get_stabilization_transform_one_pixel_shift
    )
    
    print(f"\nAnchor frame: {anchor}")
    
    # Calculate changes_x_chain as done in build_mosaic_one_column
    changes_x_chain = np.zeros(len(global_x_chain))
    sum_rounded_tx = np.zeros(len(global_x_chain))
    changes_x_chain[0] = 0
    sum_rounded_tx[0] = 0
    for i in range(1, len(global_x_chain)):
        changes_x_chain[i] = true_transform[i,0,2] - true_transform[i-1,0,2]
        sum_rounded_tx[i] = sum_rounded_tx[i-1] + round(changes_x_chain[i])
    
    # Count frames with significant motion
    significant_motion_count = 0
    for i in range(len(changes_x_chain)):
        tx = round(changes_x_chain[i])
        if abs(tx) > 0:  # Will be ±1 or 0
            significant_motion_count += 1
    
    print(f"\n" + "="*60)
    print("MOTION ANALYSIS:")
    print("="*60)
    print(f"Frames with |rounded tx| > 0: {significant_motion_count}")
    print(f"Frames with tx = 0: {len(changes_x_chain) - significant_motion_count}")
    
    # Calculate expected canvas width
    expected_canvas_w = frame_width + significant_motion_count
    
    # Calculate actual canvas width (as done in build_mosaic_one_column)
    min_x = np.min(sum_rounded_tx)
    max_x = np.max(sum_rounded_tx)
    actual_canvas_w = int(np.ceil(max_x - min_x + frame_width))
    
    print(f"\n" + "="*60)
    print("CANVAS WIDTH CALCULATION:")
    print("="*60)
    print(f"Frame width: {frame_width}")
    print(f"Frames with motion: {significant_motion_count}")
    print(f"Expected canvas width: {frame_width} + {significant_motion_count} = {expected_canvas_w}")
    print(f"\nsum_rounded_tx range: [{min_x:.0f}, {max_x:.0f}]")
    print(f"Actual formula: ceil({max_x:.0f} - {min_x:.0f} + {frame_width}) = {actual_canvas_w}")
    
    print(f"\n" + "="*60)
    if expected_canvas_w == actual_canvas_w:
        print(f"✓ PASS: Canvas width is correct!")
    else:
        print(f"❌ FAIL: Canvas width mismatch!")
        print(f"   Expected: {expected_canvas_w}")
        print(f"   Actual: {actual_canvas_w}")
        print(f"   Difference: {actual_canvas_w - expected_canvas_w}")
    print("="*60)
    
    # Show some details about sum_rounded_tx
    print(f"\nsum_rounded_tx details:")
    print(f"  First 10 values: {sum_rounded_tx[:10]}")
    print(f"  Last 10 values: {sum_rounded_tx[-10:]}")
    print(f"  Total span: {max_x - min_x:.0f}")

if __name__ == "__main__":
    test_canvas_dimensions()
