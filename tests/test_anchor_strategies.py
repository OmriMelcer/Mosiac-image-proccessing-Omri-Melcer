#!/usr/bin/env python3
"""
Test different anchor selection strategies and compare results
"""
import numpy as np
import sys
import os
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.video_io import load_video
from src.mosaic import (
    to_grey_scale, 
    get_first_to_last_transform,
    apply_stabilization,
    get_anchor_frame_by_mean_dy,
    get_anchor_frame_by_median_dy,
    get_anchor_frame_by_median_rotation
)
from scipy.ndimage import gaussian_filter

def build_mosaic_with_anchor_func(frames: np.ndarray, anchor_func, column_to_build: int = -1):
    """
    Build mosaic using a specific anchor selection function
    """
    if column_to_build == -1:
        column_to_build = frames.shape[2] // 2
    
    # Convert to grayscale and blur
    converted_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=frames.dtype)
    blurred_frames = np.zeros_like(converted_frames)
    for i in range(len(frames)):
        converted_frames[i] = (to_grey_scale(frames[i])*255).astype(np.uint8)
        blurred_frames[i] = gaussian_filter(converted_frames[i], sigma=1.0).astype(np.uint8)
    
    # Get transforms with specified anchor function
    true_transform, transforms, anchor, global_x_chain = get_first_to_last_transform(
        blurred_frames, anchor_func
    )
    
    # Calculate changes
    changes_x_chain = np.zeros(len(global_x_chain))
    sum_rounded_tx = np.zeros(len(global_x_chain))
    changes_x_chain[0] = 0
    sum_rounded_tx[0] = 0
    for i in range(1, len(global_x_chain)):
        changes_x_chain[i] = global_x_chain[i] - global_x_chain[i-1]
        sum_rounded_tx[i] = sum_rounded_tx[i-1] + round(changes_x_chain[i])
    
    cum_tx = global_x_chain[-1]
    stabilized_frames = apply_stabilization(frames, true_transform)
    
    # Calculate canvas
    min_x = np.min(sum_rounded_tx)
    max_x = np.max(sum_rounded_tx)
    canvas_w = int(np.ceil(max_x - min_x + frames.shape[2]))
    canvas = np.zeros((frames.shape[1], canvas_w, frames.shape[3]), dtype=frames.dtype)
    
    # Stitch
    if cum_tx <= 0:  # Left-to-right
        canvas[:, :column_to_build, :] = stabilized_frames[0, :, :column_to_build, :]
        cur_column = column_to_build
        for i in range(len(stabilized_frames)):
            tx = round(changes_x_chain[i])
            if tx >= 0:
                continue
            canvas[:, cur_column:cur_column-tx, :] = stabilized_frames[i, :, column_to_build:column_to_build-tx, :]
            cur_column = cur_column - tx
        canvas[:, cur_column:, :] = stabilized_frames[-1, :, column_to_build:, :]
    else:  # Right-to-left
        canvas[:, canvas_w-column_to_build:, :] = stabilized_frames[0, :, column_to_build:, :]
        cur_column = canvas_w - column_to_build
        for i in range(len(stabilized_frames)):
            tx = round(changes_x_chain[i])
            if tx <= 0:
                continue
            canvas[:, cur_column-tx:cur_column, :] = stabilized_frames[i, :, column_to_build-tx:column_to_build, :]
            cur_column = cur_column - tx
        canvas[:, :cur_column, :] = stabilized_frames[-1, :, :column_to_build, :]
    
    return canvas, anchor

def test_anchor_strategies():
    print("="*60)
    print("ANCHOR STRATEGY COMPARISON TEST")
    print("="*60)
    
    # Load video
    video_path = "Exercise Inputs-20251225/Shinkansen.mp4"
    frames = load_video(video_path)
    print(f"\nLoaded {len(frames)} frames from {video_path}")
    print(f"Frame shape: {frames[0].shape}")
    
    # Test strategies
    strategies = [
        ("median_rotation", get_anchor_frame_by_median_rotation),
        ("mean_dy", get_anchor_frame_by_mean_dy),
        ("median_dy", get_anchor_frame_by_median_dy),
    ]
    
    os.makedirs("anchor_comparison", exist_ok=True)
    
    results = []
    for name, func in strategies:
        print(f"\n{'-'*60}")
        print(f"Testing: {name}")
        print(f"{'-'*60}")
        
        mosaic, anchor = build_mosaic_with_anchor_func(frames, func)
        
        print(f"  Anchor frame: {anchor}")
        print(f"  Mosaic shape: {mosaic.shape}")
        
        # Calculate black pixel percentage
        black_pixels = np.sum(mosaic == 0)
        total_pixels = mosaic.size
        black_percent = 100 * black_pixels / total_pixels
        print(f"  Black pixels: {black_percent:.2f}%")
        
        # Save mosaic
        output_path = f"anchor_comparison/Shinkansen_{name}.png"
        mosaic_bgr = cv2.cvtColor(mosaic.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, mosaic_bgr)
        print(f"  Saved to: {output_path}")
        
        results.append({
            'name': name,
            'anchor': anchor,
            'shape': mosaic.shape,
            'black_percent': black_percent
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Strategy':<20} {'Anchor':<10} {'Width':<10} {'Black %':<10}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['name']:<20} {r['anchor']:<10} {r['shape'][1]:<10} {r['black_percent']:<10.2f}")
    print(f"{'='*60}")
    print(f"\nAll mosaics saved to: anchor_comparison/")

if __name__ == "__main__":
    test_anchor_strategies()
