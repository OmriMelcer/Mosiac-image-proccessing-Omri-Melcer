"""
Visualize point correspondences (inliers) for our method vs OpenCV.
Shows which points are being tracked and how they move.
"""
import numpy as np
import cv2
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.video_io import load_video
from src.homography_evaluation import find_rigid_movement_pyramid, ransac_rigid_movement

OUTPUT_DIR = "test_outputs"

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def visualize_correspondences(video_name: str, frame_idx: int = 0):
    """
    Create visual comparison of point correspondences between our method and OpenCV.
    """
    print("\n" + "="*80)
    print(f"VISUALIZING: Point Correspondences - {video_name}")
    print(f"Frames: {frame_idx} -> {frame_idx+1}")
    print("="*80)
    
    video_path = os.path.join("Exercise Inputs-20251225", video_name)
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    # Load frames
    frames_rgb = load_video(video_path)
    frames_gray = np.zeros((len(frames_rgb), frames_rgb[0].shape[0], frames_rgb[0].shape[1]), dtype=np.float32)
    for i in range(len(frames_rgb)):
        frames_gray[i] = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    img1_gray = frames_gray[frame_idx]
    img2_gray = frames_gray[frame_idx + 1]
    img1_rgb = frames_rgb[frame_idx].copy()
    img2_rgb = frames_rgb[frame_idx + 1].copy()
    
    print(f"\nImage shape: {img1_gray.shape}")
    
    # ==================== OUR METHOD ====================
    print("\n--- Our Method ---")
    H_our, matched_our = find_rigid_movement_pyramid(img1_gray, img2_gray)
    
    if H_our is not None and len(matched_our.shape) > 1:
        # matched_our has shape (N, 4) where columns are [y1, x1, y2, x2]
        p1_our = matched_our[:, :2]  # (N, 2) - y, x
        p2_our = matched_our[:, 2:]  # (N, 2) - y, x
        
        print(f"Transform: TX={H_our[0,2]:.2f}, TY={H_our[1,2]:.2f}, Theta={np.degrees(np.arctan2(H_our[1,0], H_our[0,0])):.3f}°")
        print(f"Inliers: {len(p1_our)}")
        
        # Analyze motion by vertical position
        heights = ['Top (0-33%)', 'Middle (33-67%)', 'Bottom (67-100%)']
        h = img1_gray.shape[0]
        for i, (start, end) in enumerate([(0, h//3), (h//3, 2*h//3), (2*h//3, h)]):
            mask = (p1_our[:, 0] >= start) & (p1_our[:, 0] < end)
            if mask.sum() > 0:
                avg_dx = np.mean(p2_our[mask, 1] - p1_our[mask, 1])
                avg_dy = np.mean(p2_our[mask, 0] - p1_our[mask, 0])
                print(f"  {heights[i]}: {mask.sum()} points, avg motion: dx={avg_dx:.2f}, dy={avg_dy:.2f}")
    else:
        print("FAILED!")
        return
    
    # ==================== OPENCV METHOD ====================
    print("\n--- OpenCV Method ---")
    img1_uint = img1_gray.astype(np.uint8)
    img2_uint = img2_gray.astype(np.uint8)
    
    # Detect
    p0 = cv2.goodFeaturesToTrack(img1_uint, maxCorners=50, qualityLevel=0.01, minDistance=3)
    
    if p0 is not None:
        # Track
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1_uint, img2_uint, p0, None, winSize=(15,15), maxLevel=2)
        
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        # Convert to our format for RANSAC
        p1_ransac = good_old.squeeze()[:, [1, 0]]  # (N, 2) - y, x
        p2_ransac = good_new.squeeze()[:, [1, 0]]  # (N, 2) - y, x
        
        H_cv, inliers_cv = ransac_rigid_movement(p1_ransac, p2_ransac)
        
        if H_cv is not None:
            # Get inlier points
            p1_cv = p1_ransac[inliers_cv]
            p2_cv = p2_ransac[inliers_cv]
            
            print(f"Transform: TX={H_cv[0,2]:.2f}, TY={H_cv[1,2]:.2f}, Theta={np.degrees(np.arctan2(H_cv[1,0], H_cv[0,0])):.3f}°")
            print(f"Detected: {len(p0)}, Tracked: {len(good_old)}, Inliers: {len(p1_cv)}")
            
            # Analyze motion by vertical position
            h = img1_gray.shape[0]
            for i, (start, end) in enumerate([(0, h//3), (h//3, 2*h//3), (2*h//3, h)]):
                mask = (p1_cv[:, 0] >= start) & (p1_cv[:, 0] < end)
                if mask.sum() > 0:
                    avg_dx = np.mean(p2_cv[mask, 1] - p1_cv[mask, 1])
                    avg_dy = np.mean(p2_cv[mask, 0] - p1_cv[mask, 0])
                    print(f"  {heights[i]}: {mask.sum()} points, avg motion: dx={avg_dx:.2f}, dy={avg_dy:.2f}")
        else:
            print("RANSAC FAILED!")
            return
    else:
        print("No points detected!")
        return
    
    # ==================== CREATE VISUALIZATIONS ====================
    ensure_output_dir()
    base_name = os.path.splitext(video_name)[0]
    
    # Visualization 1: Our method - side by side
    h, w = img1_rgb.shape[:2]
    vis_our = np.zeros((h, w*2, 3), dtype=np.uint8)
    vis_our[:, :w] = img1_rgb
    vis_our[:, w:] = img2_rgb
    
    # Draw correspondences color-coded by height
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue for Top, Mid, Bottom
    for i in range(len(p1_our)):
        y1, x1 = int(p1_our[i, 0]), int(p1_our[i, 1])
        y2, x2 = int(p2_our[i, 0]), int(p2_our[i, 1])
        
        # Determine color by vertical position
        color_idx = min(2, int(y1 / (h/3)))
        color = colors[color_idx]
        
        # Draw point in frame 1
        cv2.circle(vis_our, (x1, y1), 3, color, -1)
        # Draw point in frame 2
        cv2.circle(vis_our, (w + x2, y2), 3, color, -1)
        # Draw line
        cv2.line(vis_our, (x1, y1), (w + x2, y2), color, 1)
    
    # Add text
    cv2.putText(vis_our, f"Our Method ({len(p1_our)} inliers)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis_our, f"TX={H_our[0,2]:.2f} TY={H_our[1,2]:.2f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_our, f"Theta={np.degrees(np.arctan2(H_our[1,0], H_our[0,0])):.3f}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Legend
    cv2.putText(vis_our, "Red=Top, Green=Mid, Blue=Bottom", (10, h-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_correspondences_our_frame{frame_idx}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(vis_our, cv2.COLOR_RGB2BGR))
    print(f"\nSaved: {output_path}")
    
    # Visualization 2: OpenCV method - side by side
    vis_cv = np.zeros((h, w*2, 3), dtype=np.uint8)
    vis_cv[:, :w] = img1_rgb
    vis_cv[:, w:] = img2_rgb
    
    for i in range(len(p1_cv)):
        y1, x1 = int(p1_cv[i, 0]), int(p1_cv[i, 1])
        y2, x2 = int(p2_cv[i, 0]), int(p2_cv[i, 1])
        
        # Determine color by vertical position
        color_idx = min(2, int(y1 / (h/3)))
        color = colors[color_idx]
        
        cv2.circle(vis_cv, (x1, y1), 3, color, -1)
        cv2.circle(vis_cv, (w + x2, y2), 3, color, -1)
        cv2.line(vis_cv, (x1, y1), (w + x2, y2), color, 1)
    
    cv2.putText(vis_cv, f"OpenCV Method ({len(p1_cv)} inliers)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(vis_cv, f"TX={H_cv[0,2]:.2f} TY={H_cv[1,2]:.2f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_cv, f"Theta={np.degrees(np.arctan2(H_cv[1,0], H_cv[0,0])):.3f}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis_cv, "Red=Top, Green=Mid, Blue=Bottom", (10, h-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_correspondences_cv_frame{frame_idx}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(vis_cv, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")
    
    # Visualization 3: Overlay comparison on single frame
    vis_overlay = img1_rgb.copy()
    
    # Draw our points in red, OpenCV in green
    for i in range(len(p1_our)):
        y, x = int(p1_our[i, 0]), int(p1_our[i, 1])
        cv2.circle(vis_overlay, (x, y), 5, (255, 0, 0), 2)  # Red
    
    for i in range(len(p1_cv)):
        y, x = int(p1_cv[i, 0]), int(p1_cv[i, 1])
        cv2.circle(vis_overlay, (x, y), 3, (0, 255, 0), -1)  # Green
    
    cv2.putText(vis_overlay, f"Red=Our ({len(p1_our)}), Green=OpenCV ({len(p1_cv)})", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_points_overlay_frame{frame_idx}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(vis_overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    ensure_output_dir()
    
    # Test on Garden.mp4
    visualize_correspondences("Garden.mp4", frame_idx=0)
    visualize_correspondences("Garden.mp4", frame_idx=10)
    visualize_correspondences("Garden.mp4", frame_idx=20)
