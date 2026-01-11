"""
Test script to pre-crop video by removing black edges,
then run mosaic generation on the cropped video to verify
edge removal isn't causing mosaic drift issues.
"""
import numpy as np
import cv2
from src.video_io import load_video, save_video
import os

def remove_black_boundaries(frames: np.ndarray, threshold: float = 1.0):
    """
    Remove black boundary columns from raw input frames.
    """
    # Calculate mean intensity per column across all frames
    column_means = np.mean(frames, axis=(0, 1, 3))
    
    # Find first and last columns with mean intensity above threshold
    visible_cols = np.where(column_means > threshold)[0]
    
    if len(visible_cols) == 0:
        print("WARNING: All columns are below threshold! Returning original frames.")
        return frames
    
    left_boundary = visible_cols[0]
    right_boundary = visible_cols[-1] + 1
    
    print(f"Detected black boundaries: left={left_boundary}px, right={frames.shape[2] - right_boundary}px")
    print(f"Original size: {frames.shape[1]}x{frames.shape[2]} → Cropped size: {frames.shape[1]}x{right_boundary - left_boundary}")
    
    # Crop all frames uniformly
    cropped_frames = frames[:, :, left_boundary:right_boundary, :]
    
    # Verify all frames have same dimensions
    assert cropped_frames.shape[0] == frames.shape[0], "Frame count mismatch"
    assert all(cropped_frames[i].shape == cropped_frames[0].shape for i in range(len(cropped_frames))), \
        "Not all frames have same dimensions after cropping"
    
    return cropped_frames


if __name__ == "__main__":
    input_video = "Exercise Inputs-20251225/House.mp4"
    output_video = "output_mosaics/House_precropped.mp4"
    
    print("="*60)
    print("PRE-CROPPING VIDEO TEST")
    print("="*60)
    
    # Load original video
    print(f"\n1. Loading original video: {input_video}")
    frames = load_video(input_video)
    print(f"   Loaded {len(frames)} frames, shape: {frames[0].shape}")
    
    # Remove black edges
    print(f"\n2. Removing black boundaries...")
    cropped_frames = remove_black_boundaries(frames)
    print(f"   Cropped frames shape: {cropped_frames[0].shape}")
    
    # Verify all frames have identical dimensions
    print(f"\n3. Verifying frame consistency...")
    shapes = [f.shape for f in cropped_frames]
    unique_shapes = set(shapes)
    print(f"   Total frames: {len(cropped_frames)}")
    print(f"   Unique shapes: {unique_shapes}")
    if len(unique_shapes) == 1:
        print(f"   ✓ All frames have identical dimensions: {cropped_frames[0].shape}")
    else:
        print(f"   ✗ ERROR: Frames have different dimensions!")
        for i, shape in enumerate(shapes[:10]):
            print(f"     Frame {i}: {shape}")
    
    # Save cropped video
    print(f"\n4. Saving pre-cropped video: {output_video}")
    os.makedirs("output_mosaics", exist_ok=True)
    save_video(cropped_frames, output_video, fps=30)
    print(f"   ✓ Saved {len(cropped_frames)} frames")
    
    print(f"\n{'='*60}")
    print("PRE-CROPPING COMPLETE")
    print("="*60)
    print(f"\nNow run mosaic generation on the pre-cropped video:")
    print(f"  uv run python main.py --mode batch --video House_precropped.mp4")
    print("="*60)
