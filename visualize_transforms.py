"""
Visualization script to debug transform issues.
Shows:
1. First frames until tx > 1 pixel
2. Last frames (backwards) until tx > 1 pixel
3. TX values for each frame
4. Optical flow arrows for 4 inlier points
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.video_io import load_video
from src.mosaic import (
    to_grey_scale, get_first_to_last_transform, 
    get_anchor_frame_by_median_dy, apply_stabilization
)
from src import homography_evaluation as he
from scipy.ndimage import gaussian_filter
import os

def visualize_frame_with_flow(frame1, frame2, H, tx, frame_idx, title_prefix="Frame"):
    """
    Visualize a frame pair with optical flow arrows for inlier points.
    
    Args:
        frame1: First frame (grayscale)
        frame2: Second frame (grayscale)
        H: Homography transform from frame1 to frame2
        tx: Calculated X translation
        frame_idx: Frame index for labeling
        title_prefix: Prefix for plot title
    """
    # Detect features in frame1 using goodFeaturesToTrack
    corners = cv2.goodFeaturesToTrack(
        frame1.astype(np.uint8),
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=20
    )
    
    if corners is None or len(corners) < 4:
        print(f"  Warning: Only {len(corners) if corners is not None else 0} features found")
        return None
    
    # Convert frame to RGB for visualization
    frame1_rgb = cv2.cvtColor(frame1.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # Calculate expected positions using homography
    pts1 = corners.reshape(-1, 2)
    pts1_homogeneous = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_expected = (H @ pts1_homogeneous.T).T
    pts2_expected = pts2_expected[:, :2] / pts2_expected[:, 2:]
    
    # Track actual positions using optical flow
    pts2_actual, status, err = cv2.calcOpticalFlowPyrLK(
        frame1.astype(np.uint8),
        frame2.astype(np.uint8),
        pts1.astype(np.float32),
        None
    )
    
    # Find inliers (where actual matches expected)
    inliers = []
    for i in range(len(pts1)):
        if status[i]:
            dist = np.linalg.norm(pts2_actual[i] - pts2_expected[i])
            if dist < 5:  # Inlier threshold
                inliers.append(i)
    
    # Select up to 4 best inliers (with largest displacements)
    if len(inliers) > 0:
        displacements = [np.linalg.norm(pts2_actual[i] - pts1[i]) for i in inliers]
        sorted_inliers = [inliers[i] for i in np.argsort(displacements)[::-1]]
        selected_inliers = sorted_inliers[:min(4, len(sorted_inliers))]
    else:
        selected_inliers = []
    
    # Draw arrows for selected inliers
    for idx in selected_inliers:
        pt1 = tuple(pts1[idx].astype(int))
        pt2 = tuple(pts2_actual[idx].astype(int))
        
        # Draw point
        cv2.circle(frame1_rgb, pt1, 5, (0, 255, 0), -1)
        # Draw arrow
        cv2.arrowedLine(frame1_rgb, pt1, pt2, (255, 0, 0), 2, tipLength=0.3)
    
    # Add text overlay
    cv2.putText(frame1_rgb, f"{title_prefix} {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame1_rgb, f"tx = {tx:.2f} px", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame1_rgb, f"Inliers: {len(inliers)}/{len(pts1)}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return frame1_rgb, len(inliers), len(pts1)


def main():
    input_video = "Exercise Inputs-20251225/House.mp4"
    output_dir = "output_mosaics/transform_viz"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("TRANSFORM VISUALIZATION")
    print("="*60)
    
    # Load and preprocess video
    print(f"\n1. Loading video: {input_video}")
    frames = load_video(input_video)
    print(f"   Loaded {len(frames)} frames, shape: {frames[0].shape}")
    
    # Remove black boundaries
    from main import remove_black_boundaries
    frames = remove_black_boundaries(frames)
    print(f"   Cropped frames shape: {frames[0].shape}")
    
    # Convert to grayscale and blur
    print(f"\n2. Converting to grayscale and applying blur...")
    converted_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=frames.dtype)
    blurred_frames = np.zeros_like(converted_frames)
    for i in range(len(frames)):
        converted_frames[i] = (to_grey_scale(frames[i])*255).astype(np.uint8)
        blurred_frames[i] = gaussian_filter(converted_frames[i], sigma=1.0).astype(np.uint8)
    
    # Get transforms
    print(f"\n3. Computing transforms...")
    true_transform, transforms, anchor, global_x_chain = get_first_to_last_transform(
        blurred_frames, 
        anchor_index_locator_func=get_anchor_frame_by_median_dy
    )
    
    # Calculate frame-to-frame tx changes
    changes_x_chain = np.zeros(len(global_x_chain))
    changes_x_chain[0] = 0
    for i in range(1, len(global_x_chain)):
        changes_x_chain[i] = global_x_chain[i] - global_x_chain[i-1]
    
    print(f"   Anchor frame: {anchor}")
    print(f"   Cumulative TX: {global_x_chain[-1]:.2f}")
    
    # Apply stabilization
    print(f"\n4. Applying stabilization...")
    stabilized_frames = apply_stabilization(frames, true_transform)
    
    # Find first frames with tx > 1
    print(f"\n5. Finding first frames with |tx| > 1...")
    first_frames = []
    for i in range(1, len(changes_x_chain)):
        if abs(changes_x_chain[i]) > 1:
            first_frames.append(i)
        if len(first_frames) >= 5:
            break
    
    print(f"   First frames with |tx| > 1: {first_frames}")
    
    # Find last frames with tx > 1 (backwards)
    print(f"\n6. Finding last frames with |tx| > 1...")
    last_frames = []
    for i in range(len(changes_x_chain)-1, 0, -1):
        if abs(changes_x_chain[i]) > 1:
            last_frames.append(i)
        if len(last_frames) >= 5:
            break
    last_frames.reverse()
    
    print(f"   Last frames with |tx| > 1: {last_frames}")
    
    # Visualize first frames (side by side)
    print(f"\n7. Visualizing FIRST frames (side-by-side)...")
    fig, axes = plt.subplots(len(first_frames), 2, figsize=(20, 4*len(first_frames)))
    if len(first_frames) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, frame_idx in enumerate(first_frames):
        print(f"   Processing frame {frame_idx}: tx={changes_x_chain[frame_idx]:.2f}")
        
        viz_frame, n_inliers, n_total = visualize_frame_with_flow(
            blurred_frames[frame_idx-1],
            blurred_frames[frame_idx],
            transforms[frame_idx],
            changes_x_chain[frame_idx],
            frame_idx,
            title_prefix="First"
        )
        
        if viz_frame is not None:
            # Show previous frame
            prev_frame_rgb = cv2.cvtColor(blurred_frames[frame_idx-1].astype(np.uint8), cv2.COLOR_GRAY2RGB)
            cv2.putText(prev_frame_rgb, f"Frame {frame_idx-1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            axes[idx, 0].imshow(prev_frame_rgb)
            axes[idx, 0].set_title(f"Frame {frame_idx-1} (Previous)")
            axes[idx, 0].axis('off')
            
            # Show current frame with flow
            axes[idx, 1].imshow(cv2.cvtColor(viz_frame, cv2.COLOR_BGR2RGB))
            axes[idx, 1].set_title(f"Frame {frame_idx}: tx={changes_x_chain[frame_idx]:.2f} px, Inliers: {n_inliers}/{n_total}")
            axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/first_frames_visualization.png", dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir}/first_frames_visualization.png")
    plt.close()
    
    # Visualize last frames (side by side)
    print(f"\n8. Visualizing LAST frames (side-by-side)...")
    fig, axes = plt.subplots(len(last_frames), 2, figsize=(20, 4*len(last_frames)))
    if len(last_frames) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, frame_idx in enumerate(last_frames):
        print(f"   Processing frame {frame_idx}: tx={changes_x_chain[frame_idx]:.2f}")
        
        viz_frame, n_inliers, n_total = visualize_frame_with_flow(
            blurred_frames[frame_idx-1],
            blurred_frames[frame_idx],
            transforms[frame_idx],
            changes_x_chain[frame_idx],
            frame_idx,
            title_prefix="Last"
        )
        
        if viz_frame is not None:
            # Show previous frame
            prev_frame_rgb = cv2.cvtColor(blurred_frames[frame_idx-1].astype(np.uint8), cv2.COLOR_GRAY2RGB)
            cv2.putText(prev_frame_rgb, f"Frame {frame_idx-1}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            axes[idx, 0].imshow(prev_frame_rgb)
            axes[idx, 0].set_title(f"Frame {frame_idx-1} (Previous)")
            axes[idx, 0].axis('off')
            
            # Show current frame with flow
            axes[idx, 1].imshow(cv2.cvtColor(viz_frame, cv2.COLOR_BGR2RGB))
            axes[idx, 1].set_title(f"Frame {frame_idx}: tx={changes_x_chain[frame_idx]:.2f} px, Inliers: {n_inliers}/{n_total}")
            axes[idx, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/last_frames_visualization.png", dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir}/last_frames_visualization.png")
    plt.close()
    
    # Save stabilized frames for visual inspection
    print(f"\n9. Saving sample stabilized frames...")
    sample_indices = [0, anchor, len(stabilized_frames)//2, -1]
    for i in sample_indices:
        frame_rgb = cv2.cvtColor(stabilized_frames[i].astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{output_dir}/stabilized_frame_{i:04d}.png", frame_rgb)
    print(f"   Saved stabilized frames: {sample_indices}")
    
    # Plot tx over time
    print(f"\n10. Plotting TX over time...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot frame-to-frame changes
    ax1.plot(changes_x_chain, 'b-', linewidth=1)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.axhline(y=1, color='g', linestyle='--', alpha=0.3, label='|tx| = 1')
    ax1.axhline(y=-1, color='g', linestyle='--', alpha=0.3)
    ax1.axvline(x=anchor, color='orange', linestyle='--', alpha=0.7, label=f'Anchor (frame {anchor})')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Frame-to-Frame TX (pixels)')
    ax1.set_title('Frame-to-Frame X Translation')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot cumulative displacement
    ax2.plot(global_x_chain, 'r-', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.axvline(x=anchor, color='orange', linestyle='--', alpha=0.7, label=f'Anchor (frame {anchor})')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Cumulative TX (pixels)')
    ax2.set_title('Cumulative X Translation from Frame 0')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/tx_over_time.png", dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_dir}/tx_over_time.png")
    plt.close()
    
    # Statistics
    print(f"\n{'='*60}")
    print("STATISTICS")
    print("="*60)
    print(f"Total frames: {len(frames)}")
    print(f"Anchor frame: {anchor}")
    print(f"Frames with |tx| > 1: {np.sum(np.abs(changes_x_chain) > 1)}")
    print(f"Frames with |tx| > 5: {np.sum(np.abs(changes_x_chain) > 5)}")
    print(f"Frames with |tx| > 10: {np.sum(np.abs(changes_x_chain) > 10)}")
    print(f"Max |tx|: {np.max(np.abs(changes_x_chain)):.2f} px")
    print(f"Mean |tx|: {np.mean(np.abs(changes_x_chain)):.2f} px")
    print(f"Median |tx|: {np.median(np.abs(changes_x_chain)):.2f} px")
    print(f"Total cumulative TX: {global_x_chain[-1]:.2f} px")
    print("="*60)


if __name__ == "__main__":
    main()
