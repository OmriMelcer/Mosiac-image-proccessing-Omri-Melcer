
import os
import cv2
import numpy as np
from src.video_io import load_video
from src.mosaic import get_first_to_last_transform, get_anchor_frame_by_median_dy
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def remove_black_boundaries(frames: np.ndarray, threshold: float = 1.0):
    """
    Remove black boundary columns from raw input frames.
    A column is considered a boundary if its mean pixel value is below the threshold.
    """
    # Calculate mean intensity per column across all frames
    column_means = np.mean(frames, axis=(0, 1, 3))
    visible_cols = np.where(column_means > threshold)[0]
    
    if len(visible_cols) == 0:
        return frames
    
    left_boundary = visible_cols[0]
    right_boundary = visible_cols[-1] + 1
    
    return frames[:, :, left_boundary:right_boundary, :]

def analyze_motion():
    video_path = "Exercise Inputs-20251225/Kessaria.mp4"
    print(f"Loading {video_path}...")
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} does not exist.")
        return

    frames = load_video(video_path)
    print(f"Loaded {len(frames)} frames.")
    
    # Apply the same pre-processing as main.py
    frames = remove_black_boundaries(frames)
    print(f"Frames shape after cropping: {frames.shape}")
    
    # Convert to grayscale for motion estimation
    gray_frames = []
    for f in frames:
        gray_frames.append(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY))
    gray_frames = np.array(gray_frames)
    
    print("Computing transforms...")
    # get_first_to_last_transform returns: true_transform, pairwise_transforms, anchor, global_x_change
    # We want the second return value: pairwise_transforms
    stabilized_transforms, pairwise_transforms, anchor, global_x_change = get_first_to_last_transform(
        gray_frames, 
        get_anchor_frame_by_median_dy
    )
    
    print("\nAnalyzing Pairwise Transforms (Frame i-1 -> Frame i):")
    print("idx | dx      | dy      | dtheta   | Type")
    print("-" * 50)
    
    suspicious_indices = []
    
    dxs = []
    dys = []
    thetas = []

    for i in range(1, len(pairwise_transforms)):
        H = pairwise_transforms[i]
        
        dx = H[0, 2]
        dy = H[1, 2]
        theta = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi
        
        dxs.append(dx)
        dys.append(dy)
        thetas.append(theta)
        
        is_identity = np.allclose(H, np.eye(3))
        
        type_str = "OK"
        if is_identity:
            type_str = "IDENTITY (FAIL?)"
            suspicious_indices.append(i)
        elif abs(dx) > 50 or abs(dy) > 20: # Arbitrary high thresholds
            type_str = "LARGE JUMP"
            suspicious_indices.append(i)
            
        if type_str != "OK" or i < 5 or i > len(pairwise_transforms) - 5:
            print(f"{i:3d} | {dx:7.2f} | {dy:7.2f} | {theta:8.4f} | {type_str}")

    print("-" * 50)
    print(f"Found {len(suspicious_indices)} suspicious frames: {suspicious_indices}")
    
    # Save a plot
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(dxs)
    plt.title('dX per frame')
    plt.ylabel('pixels')
    
    plt.subplot(3, 1, 2)
    plt.plot(dys)
    plt.title('dY per frame')
    plt.ylabel('pixels')

    plt.subplot(3, 1, 3)
    plt.plot(thetas)
    plt.title('dTheta per frame')
    plt.ylabel('degrees')
    
    plt.tight_layout()
    plt.savefig('kessaria_motion_analysis.png')
    print("Saved plot to kessaria_motion_analysis.png")

if __name__ == "__main__":
    analyze_motion()
