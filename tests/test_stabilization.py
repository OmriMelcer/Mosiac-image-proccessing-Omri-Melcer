"""
Test 2: Stabilization Visualization
Shows before/after frames for unstable frames (large theta/ty).
"""
import numpy as np
import cv2
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.video_io import load_video
from src import mosaic

OUTPUT_DIR = "test_outputs"
INPUT_DIR = "Exercise Inputs-20251225"

def ensure_output_dir():
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

def test_stabilization_visualization(video_name: str, num_frames: int = 50):
    """
    Visualize stabilization by showing before/after for unstable frames.
    """
    print("\n" + "="*80)
    print(f"TEST 2: Stabilization - {video_name}")
    print("="*80)
    
    video_path = os.path.join(INPUT_DIR, video_name)
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    # Load video
    print(f"Loading video: {video_name}")
    frames_rgb = load_video(video_path)
    frames_rgb = frames_rgb[:num_frames]
    
    print(f"Video loaded: {len(frames_rgb)} frames, {frames_rgb[0].shape}")
    
    # Convert for motion estimation
    frames_gray = convert_to_grayscale(frames_rgb)
    
    # Get transforms
    print("Computing transforms...")
    true_transform, pairwise_transforms, anchor, max_x_change = mosaic.get_first_to_last_transform(frames_gray)
    
    print(f"Anchor frame: {anchor}")
    print(f"Max X change: {max_x_change:.1f}")
    
    # Apply stabilization
    print("Applying stabilization...")
    stabilized_frames = mosaic.apply_stabilization(frames_rgb, true_transform)
    
    # Analyze transforms to find unstable frames
    print("\n--- Transform Analysis ---")
    print(f"{'Frame':<8} {'TX':<10} {'TY':<10} {'Theta(deg)':<12} {'Dist from Anchor':<15}")
    print("-" * 65)
    
    frame_metrics = []
    for i in range(len(true_transform)):
        tx = true_transform[i][0, 2]
        ty = true_transform[i][1, 2]
        theta = np.degrees(np.arctan2(true_transform[i][1, 0], true_transform[i][0, 0]))
        
        # Distance from anchor (y and theta only, x is motion)
        dist = np.sqrt(ty**2 + (theta**2))
        
        frame_metrics.append((i, tx, ty, theta, dist))
        
        if i % 5 == 0 or i == anchor:
            marker = " (ANCHOR)" if i == anchor else ""
            print(f"{i:<8} {tx:<10.2f} {ty:<10.2f} {theta:<12.3f} {dist:<15.2f}{marker}")
    
    # Sort by distance to find most unstable frames
    sorted_metrics = sorted(frame_metrics, key=lambda x: x[4], reverse=True)
    
    # Select 3 most unstable frames (excluding anchor)
    unstable_frames = [m for m in sorted_metrics if m[0] != anchor][:3]
    
    print(f"\n--- Most Unstable Frames ---")
    for frame_idx, tx, ty, theta, dist in unstable_frames:
        print(f"Frame {frame_idx}: TY={ty:.2f}, Theta={theta:.3f}°, Distance={dist:.2f}")
    
    # Create visualizations
    ensure_output_dir()
    base_name = os.path.splitext(video_name)[0]
    
    for frame_idx, tx, ty, theta, dist in unstable_frames:
        # Get before and after
        before = frames_rgb[frame_idx]
        after = stabilized_frames[frame_idx]
        
        # Create side-by-side comparison
        h, w = before.shape[:2]
        comparison = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
        comparison[:, :w] = before
        comparison[:, w+20:] = after
        comparison[:, w:w+20] = 255  # white separator
        
        # Add text labels
        cv2.putText(comparison, "Before", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)
        cv2.putText(comparison, "After", (w+30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2)
        cv2.putText(comparison, f"Frame {frame_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        cv2.putText(comparison, f"TY={ty:.1f} Theta={theta:.2f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Save
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_stab_frame_{frame_idx}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path}")
    
    # Also save anchor frame comparison
    before_anchor = frames_rgb[anchor]
    after_anchor = stabilized_frames[anchor]
    
    h, w = before_anchor.shape[:2]
    comparison = np.zeros((h, w*2 + 20, 3), dtype=np.uint8)
    comparison[:, :w] = before_anchor
    comparison[:, w+20:] = after_anchor
    comparison[:, w:w+20] = 255
    
    cv2.putText(comparison, "Before (Anchor)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 0), 2)
    cv2.putText(comparison, "After (Anchor)", (w+30, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 255, 0), 2)
    cv2.putText(comparison, f"Frame {anchor}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_stab_anchor_{anchor}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"Saved anchor: {output_path}")
    
    print("\nStabilization visualization complete!")

if __name__ == "__main__":
    ensure_output_dir()
    
    # Test on Trees.mp4 (known to have motion)
    test_stabilization_visualization("Trees.mp4", num_frames=50)
