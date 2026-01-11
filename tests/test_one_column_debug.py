#!/usr/bin/env python3
"""
Debug test for build_mosaic_one_column
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.video_io import load_video
from src.mosaic import build_mosaic_one_column, build_mosaic

def test_one_column_debug():
    print("="*60)
    print("DEBUG TEST: build_mosaic_one_column")
    print("="*60)
    
    # Load ALL frames
    frames = load_video("Exercise Inputs-20251225/Shinkansen.mp4")
    print(f"\nLoaded {len(frames)} frames, shape: {frames[0].shape}")
    
    # Build with one_column method
    print("\n" + "="*60)
    print("Testing build_mosaic_one_column:")
    print("="*60)
    mosaic_1col, stab_1col = build_mosaic_one_column(frames, column_to_build=-1)
    print(f"Mosaic shape: {mosaic_1col.shape}")
    print(f"Stabilized frames shape: {stab_1col.shape}")
    
    # Check for black regions
    black_pixels = np.sum(mosaic_1col == 0)
    total_pixels = mosaic_1col.size
    print(f"Black pixels: {black_pixels}/{total_pixels} ({100*black_pixels/total_pixels:.1f}%)")
    
    # Build with regular method for comparison
    print("\n" + "="*60)
    print("Testing build_mosaic (regular):")
    print("="*60)
    mosaic_reg, stab_reg = build_mosaic(frames, column_to_build=-1)
    print(f"Mosaic shape: {mosaic_reg.shape}")
    print(f"Stabilized frames shape: {stab_reg.shape}")
    
    black_pixels_reg = np.sum(mosaic_reg == 0)
    total_pixels_reg = mosaic_reg.size
    print(f"Black pixels: {black_pixels_reg}/{total_pixels_reg} ({100*black_pixels_reg/total_pixels_reg:.1f}%)")
    
    print("\n" + "="*60)
    print("COMPARISON:")
    print("="*60)
    print(f"Canvas width ratio: {mosaic_1col.shape[1]}/{mosaic_reg.shape[1]} = {mosaic_1col.shape[1]/mosaic_reg.shape[1]:.2f}")
    print(f"Black pixel ratio: {black_pixels/total_pixels:.3f} vs {black_pixels_reg/total_pixels_reg:.3f}")

if __name__ == "__main__":
    test_one_column_debug()
