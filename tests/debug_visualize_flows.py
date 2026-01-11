
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from src.video_io import load_video
from src.homography_evaluation import ransac_rigid_movement

def remove_black_boundaries(frames: np.ndarray, threshold: float = 1.0):
    column_means = np.mean(frames, axis=(0, 1, 3))
    visible_cols = np.where(column_means > threshold)[0]
    if len(visible_cols) == 0: return frames
    l, r = visible_cols[0], visible_cols[-1] + 1
    return frames[:, :, l:r, :]

def track_and_analyze(idx, frame1, frame2, output_prefix="debug"):
    print(f"\nAnalyzing Pair {idx-1} -> {idx} ...")
    
    # grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
    
    # 1. Feature Detection (OpenCV)
    # Using same params as homography_evaluation.py
    p0 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.01, minDistance=3)
    
    if p0 is None:
        print("  FAIL: No features detected.")
        return

    # 2. Tracking (OpenCV LK)
    # Using same logic (try blur levels)
    best_len = -1
    best_p0 = None
    best_p1 = None
    
    curr_g1 = gray1.copy()
    curr_g2 = gray2.copy()
    
    # Perform tracking with retries (blurring) to find best tracking set
    for attempt in range(5):
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            curr_g1, curr_g2, p0, None, 
            winSize=(15,15), maxLevel=2
        )
        
        if p1 is not None:
             good_new = p1[st==1]
             good_old = p0[st==1]
             
             if len(good_new) > best_len:
                 best_len = len(good_new)
                 best_p0 = good_old
                 best_p1 = good_new
                 
             if len(p0) > 0 and len(good_new)/len(p0) > 0.6:
                 break # Good enough
        
        # Blur for next iteration
        curr_g1 = cv2.GaussianBlur(curr_g1, (5,5), 1)
        curr_g2 = cv2.GaussianBlur(curr_g2, (5,5), 1)
        
    if best_p0 is None or len(best_p0) < 2:
        print("  FAIL: Tracking lost most points.")
        return

    # 3. Prepare for RANSAC
    # Convert to (y,x) format as used in our codebase for RANSAC
    # best_p0 shape is (N, 1, 2) -> (x, y)
    p0_flat = best_p0.reshape(-1, 2)
    p1_flat = best_p1.reshape(-1, 2)
    
    p0_yx = p0_flat[:, [1, 0]] # (y, x)
    p1_yx = p1_flat[:, [1, 0]] # (y, x)
    
    # 4. RANSAC
    H, inliers_idx = ransac_rigid_movement(p0_yx, p1_yx)
    
    if H is None:
        print("  RANSAC returned None (Identity assumed)")
        H_mat = np.eye(3)
        inliers_idx = []
    else:
        H_mat = H
        print(f"  RANSAC H found: dx={H[0,2]:.2f}, dy={H[1,2]:.2f}")

    # Separate points
    num_points = len(p0_yx)
    all_indices = np.arange(num_points)
    
    if inliers_idx is None or len(inliers_idx) == 0:
        inliers_idx = []
        outliers_idx = all_indices
    else:
        outliers_idx = np.setdiff1d(all_indices, inliers_idx)
        
    p0_inliers = p0_flat[inliers_idx]
    p1_inliers = p1_flat[inliers_idx]
    
    p0_outliers = p0_flat[outliers_idx]
    p1_outliers = p1_flat[outliers_idx]

    # --- VISUALIZATION ---
    
    # A. Side-by-Side Frames
    h, w, c = frame1.shape
    vis_side = np.hstack([frame1, frame2])
    # Draw Inliers lines on side-by-side
    # Point in Frame 1 -> Point in Frame 2 (shifted by width)
    for i in range(len(p0_inliers)):
        pt1 = (int(p0_inliers[i][0]), int(p0_inliers[i][1]))
        pt2 = (int(p1_inliers[i][0]) + w, int(p1_inliers[i][1]))
        cv2.line(vis_side, pt1, pt2, (0, 255, 0), 1)
        
    cv2.imwrite(f"{output_prefix}_{idx}_A_side_by_side.png", cv2.cvtColor(vis_side, cv2.COLOR_RGB2BGR))
    
    # B. Flow Arrows on Frame 1
    vis_arrows = frame1.copy()
    
    # Draw Outliers (Red)
    for i in range(len(p0_outliers)):
        x1, y1 = int(p0_outliers[i][0]), int(p0_outliers[i][1])
        x2, y2 = int(p1_outliers[i][0]), int(p1_outliers[i][1])
        cv2.arrowedLine(vis_arrows, (x1, y1), (x2, y2), (0, 0, 255), 1, tipLength=0.3)
        
    # Draw Inliers (Green)
    for i in range(len(p0_inliers)):
        x1, y1 = int(p0_inliers[i][0]), int(p0_inliers[i][1])
        x2, y2 = int(p1_inliers[i][0]), int(p1_inliers[i][1])
        # Make arrows visible even for small motion
        dx = x2-x1
        dy = y2-y1
        mag = np.sqrt(dx*dx + dy*dy)
        if mag < 2:
            # Draw circle for static points
            cv2.circle(vis_arrows, (x1, y1), 2, (0, 255, 0), -1)
        else:
            cv2.arrowedLine(vis_arrows, (x1, y1), (x2, y2), (0, 255, 0), 2, tipLength=0.3)

    cv2.imwrite(f"{output_prefix}_{idx}_B_arrows.png", cv2.cvtColor(vis_arrows, cv2.COLOR_RGB2BGR))
    
    # C. Statistical Plots
    fig = plt.figure(figsize=(15, 6))
    
    # 1. Spatial Distribution of Points
    ax1 = plt.subplot(1, 2, 1)
    if len(p0_outliers) > 0:
        ax1.scatter(p0_outliers[:, 0], p0_outliers[:, 1], c='red', marker='x', alpha=0.5, label=f'Outliers ({len(p0_outliers)})')
    if len(p0_inliers) > 0:
        ax1.scatter(p0_inliers[:, 0], p0_inliers[:, 1], c='green', marker='o', alpha=0.6, label=f'Inliers ({len(p0_inliers)})')
    
    ax1.set_xlim(0, w)
    ax1.set_ylim(h, 0) # Flip Y
    ax1.set_title(f"Target {idx}: Spatial Distribution of Points")
    ax1.set_xlabel("X (Width)")
    ax1.set_ylabel("Y (Height)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Flow Distribution (dx vs dy)
    ax2 = plt.subplot(1, 2, 2)
    
    # Calculate flows
    if len(p0_outliers) > 0:
        out_dx = p1_outliers[:, 0] - p0_outliers[:, 0]
        out_dy = p1_outliers[:, 1] - p0_outliers[:, 1]
        ax2.scatter(out_dx, out_dy, c='red', marker='x', alpha=0.3, label='Outliers')
        
    if len(p0_inliers) > 0:
        in_dx = p1_inliers[:, 0] - p0_inliers[:, 0]
        in_dy = p1_inliers[:, 1] - p0_inliers[:, 1]
        ax2.scatter(in_dx, in_dy, c='green', marker='o', alpha=0.6, label='Inliers')
        
        # Compute mean/median of inliers
        mean_dx = np.mean(in_dx)
        mean_dy = np.mean(in_dy)
        ax2.scatter([mean_dx], [mean_dy], c='black', marker='*', s=200, label=f'Mean Inlier ({mean_dx:.1f}, {mean_dy:.1f})')
        
    # Plot the RANSAC Result H translation (roughly)
    # Note: H includes rotation, so dx/dy varies by pixel, but for translation-dominant usually close
    h_dx = H_mat[0, 2]
    h_dy = H_mat[1, 2]
    ax2.scatter([h_dx], [h_dy], c='blue', marker='D', s=100, label=f'RANSAC Model ({h_dx:.1f}, {h_dy:.1f})')
    
    ax2.set_title(f"Target {idx}: Motion Vectors Distribution")
    ax2.set_xlabel("dx (X-motion)")
    ax2.set_ylabel("dy (Y-motion)")
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_{idx}_C_plots.png")
    plt.close()

def main():
    video_path = "Exercise Inputs-20251225/Kessaria.mp4"
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return
        
    frames = load_video(video_path)
    frames = remove_black_boundaries(frames)
    
    # Check these specific problematic transitions
    indices = [3, 5, 8, 267]
    
    for idx in indices:
        if idx < len(frames):
            track_and_analyze(idx, frames[idx-1], frames[idx], output_prefix="vis_kessaria")

if __name__ == "__main__":
    main()
