"""
Analyze TX (horizontal motion) distribution for House video.
Print statistics and create histogram visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.video_io import load_video
from src.mosaic import to_grey_scale
from src import homography_evaluation as he
from scipy.ndimage import gaussian_filter


def analyze_tx_distribution(video_path: str):
    """
    Analyze TX distribution across all consecutive frame pairs.
    """
    print("="*60)
    print("TX DISTRIBUTION ANALYSIS - HOUSE VIDEO")
    print("="*60)
    print(f"Video: {os.path.basename(video_path)}")
    
    # Load entire video
    frames = load_video(video_path)
    print(f"\nLoaded {len(frames)} frames, shape: {frames[0].shape}")
    
    # Convert to grayscale and blur
    print("Converting to grayscale and applying blur...")
    frames_gray = np.zeros((len(frames), frames.shape[1], frames.shape[2]), dtype=np.uint8)
    blurred_frames = np.zeros_like(frames_gray)
    
    for i in range(len(frames)):
        gray = (to_grey_scale(frames[i]) * 255).astype(np.uint8)
        frames_gray[i] = gray
        blurred_frames[i] = gaussian_filter(gray, sigma=1.0).astype(np.uint8)
    
    # Compute TX for all consecutive pairs
    print(f"\nComputing TX for {len(frames)-1} frame pairs...")
    tx_values = []
    failed_pairs = []
    
    for i in range(1, len(frames)):
        H, _ = he.find_rigid_movement_opencv_with_our_ransac(blurred_frames[i-1], blurred_frames[i])
        
        if H is not None:
            tx = H[0, 2]
            tx_values.append(tx)
        else:
            failed_pairs.append(i)
            tx_values.append(0.0)  # Default to 0 for failed estimation
        
        if i % 50 == 0:
            print(f"  Processed {i}/{len(frames)-1} pairs...")
    
    tx_values = np.array(tx_values)
    
    # Print statistics
    print("\n" + "="*60)
    print("TX STATISTICS:")
    print("="*60)
    print(f"\nTotal frame pairs: {len(tx_values)}")
    print(f"Failed estimations: {len(failed_pairs)}")
    
    print(f"\nTX (Horizontal Motion):")
    print(f"  Mean:    {np.mean(tx_values):8.2f} px")
    print(f"  Median:  {np.median(tx_values):8.2f} px")
    print(f"  Std:     {np.std(tx_values):8.2f} px")
    print(f"  Min:     {np.min(tx_values):8.2f} px")
    print(f"  Max:     {np.max(tx_values):8.2f} px")
    
    # Count near-zero TX values
    near_zero = np.sum(np.abs(tx_values) < 0.1)
    small = np.sum(np.abs(tx_values) < 1.0)
    moderate = np.sum((np.abs(tx_values) >= 1.0) & (np.abs(tx_values) < 10.0))
    large = np.sum(np.abs(tx_values) >= 10.0)
    
    print(f"\nTX Distribution by magnitude:")
    print(f"  |TX| < 0.1 px:     {near_zero:4d} ({near_zero/len(tx_values)*100:5.1f}%)")
    print(f"  |TX| < 1.0 px:     {small:4d} ({small/len(tx_values)*100:5.1f}%)")
    print(f"  1.0 ≤ |TX| < 10:   {moderate:4d} ({moderate/len(tx_values)*100:5.1f}%)")
    print(f"  |TX| ≥ 10 px:      {large:4d} ({large/len(tx_values)*100:5.1f}%)")
    
    # Find largest jumps
    print(f"\nLargest TX jumps (top 10):")
    sorted_indices = np.argsort(np.abs(tx_values))[::-1]
    for idx in sorted_indices[:10]:
        frame_pair = idx + 1
        print(f"  Frame {frame_pair-1} → {frame_pair}: TX = {tx_values[idx]:7.2f} px")
    
    if failed_pairs:
        print(f"\nFailed frame pairs: {failed_pairs}")
    
    # Create visualization
    print("\n" + "="*60)
    print("Creating histogram visualization...")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'TX Distribution Analysis - House Video\n({len(frames)} frames, {len(tx_values)} pairs)', 
                 fontsize=14, fontweight='bold')
    
    # 1. Full histogram
    ax1 = axes[0, 0]
    ax1.hist(tx_values, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(tx_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(tx_values):.2f}')
    ax1.axvline(np.median(tx_values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(tx_values):.2f}')
    ax1.set_xlabel('TX (pixels)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Full TX Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Zoomed histogram (excluding extreme outliers)
    ax2 = axes[0, 1]
    q1, q99 = np.percentile(tx_values, [1, 99])
    filtered_tx = tx_values[(tx_values >= q1) & (tx_values <= q99)]
    ax2.hist(filtered_tx, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax2.axvline(np.mean(filtered_tx), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(filtered_tx):.2f}')
    ax2.axvline(np.median(filtered_tx), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(filtered_tx):.2f}')
    ax2.set_xlabel('TX (pixels)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Zoomed TX Distribution (1st-99th percentile)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. TX over time
    ax3 = axes[1, 0]
    ax3.plot(range(len(tx_values)), tx_values, linewidth=0.8, alpha=0.7)
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax3.axhline(np.mean(tx_values), color='red', linestyle='--', linewidth=1.5, label=f'Mean: {np.mean(tx_values):.2f}')
    ax3.set_xlabel('Frame Pair Index', fontsize=11)
    ax3.set_ylabel('TX (pixels)', fontsize=11)
    ax3.set_title('TX Evolution Over Time', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative TX
    ax4 = axes[1, 1]
    cumulative_tx = np.cumsum(tx_values)
    ax4.plot(range(len(cumulative_tx)), cumulative_tx, linewidth=1.5, color='purple')
    ax4.set_xlabel('Frame Index', fontsize=11)
    ax4.set_ylabel('Cumulative TX (pixels)', fontsize=11)
    ax4.set_title(f'Cumulative TX (Total: {cumulative_tx[-1]:.1f} px)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'test_outputs/house_tx_distribution.png'
    os.makedirs('test_outputs', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    print("="*60)
    
    return tx_values, failed_pairs


if __name__ == "__main__":
    video_path = "Exercise Inputs-20251225/House.mp4"
    tx_values, failed_pairs = analyze_tx_distribution(video_path)
