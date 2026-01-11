"""
Detailed visualization of the last two frames (431→432) to understand
why the homography estimation produces 80 pixels tx when frames are nearly identical.

Shows:
- All detected feature points
- Optical flow arrows
- Inliers (blue) vs Outliers (red)
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.video_io import load_video
from src.mosaic import to_grey_scale
from src import homography_evaluation as he
from main import remove_black_boundaries

def visualize_correspondence_quality(img1, img2, H, all_points_before, all_points_after, inlier_mask):
    """
    Visualize all feature matches with inliers in blue and outliers in red.
    
    Args:
        img1: First frame (grayscale)
        img2: Second frame (grayscale)
        H: Homography matrix
        all_points_before: All matched points in frame1 (N, 2)
        all_points_after: All matched points in frame2 (N, 2)
        inlier_mask: Boolean mask indicating inliers
    """
    # Convert to RGB for visualization
    img1_rgb = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    img2_rgb = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    # Create side-by-side canvas
    h, w = img1.shape
    canvas = np.zeros((h, w*2, 3), dtype=np.uint8)
    canvas[:, :w] = img1_rgb
    canvas[:, w:] = img2_rgb
    
    # Draw all correspondences
    n_inliers = np.sum(inlier_mask)
    n_outliers = len(inlier_mask) - n_inliers
    
    for i in range(len(all_points_before)):
        pt1 = tuple(all_points_before[i].astype(int))
        pt2 = tuple((all_points_after[i] + np.array([w, 0])).astype(int))
        
        if inlier_mask[i]:
            # Inlier: blue
            color = (0, 0, 255)
            thickness = 2
        else:
            # Outlier: red
            color = (255, 0, 0)
            thickness = 1
        
        # Draw circle on both frames
        cv2.circle(canvas, pt1, 5, color, -1)
        cv2.circle(canvas, pt2, 5, color, -1)
        
        # Draw arrow
        cv2.arrowedLine(canvas, pt1, pt2, color, thickness, tipLength=0.2)
    
    # Add text overlay
    cv2.putText(canvas, f"Frame 431 -> 432", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(canvas, f"Inliers: {n_inliers} (blue)", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(canvas, f"Outliers: {n_outliers} (red)", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    if H is not None:
        tx = H[0, 2]
        ty = H[1, 2]
        theta = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi
        cv2.putText(canvas, f"tx={tx:.2f}, ty={ty:.2f}, theta={theta:.2f}deg", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    return canvas


def get_all_correspondences_with_ransac_result(img1, img2):
    """
    Run the full homography estimation pipeline and extract all correspondences
    along with the RANSAC inlier mask.
    
    Returns:
        H: Homography matrix
        pts1: All points in frame1
        pts2: All points in frame2
        inlier_mask: Boolean array indicating inliers
    """
    # Detect features in img1
    corners = cv2.goodFeaturesToTrack(
        img1.astype(np.uint8),
        maxCorners=500,
        qualityLevel=0.01,
        minDistance=10
    )
    
    if corners is None or len(corners) < 4:
        print(f"ERROR: Only {len(corners) if corners is not None else 0} features detected!")
        return None, None, None, None
    
    pts1 = corners.reshape(-1, 2).astype(np.float32)
    
    # Track with optical flow
    pts2, status, err = cv2.calcOpticalFlowPyrLK(
        img1.astype(np.uint8),
        img2.astype(np.uint8),
        pts1,
        None,
        winSize=(21, 21),
        maxLevel=3
    )
    
    # Keep only successfully tracked points
    good_mask = status.ravel() == 1
    pts1_tracked = pts1[good_mask]
    pts2_tracked = pts2[good_mask]
    
    print(f"Features detected: {len(pts1)}")
    print(f"Successfully tracked: {len(pts1_tracked)}")
    
    if len(pts1_tracked) < 4:
        print("ERROR: Too few tracked points!")
        return None, pts1_tracked, pts2_tracked, np.zeros(len(pts1_tracked), dtype=bool)
    
    # Run RANSAC to find homography
    H, inlier_mask = cv2.findHomography(
        pts1_tracked,
        pts2_tracked,
        cv2.RANSAC,
        ransacReprojThreshold=5.0
    )
    
    if inlier_mask is None:
        inlier_mask = np.zeros(len(pts1_tracked), dtype=bool)
    else:
        inlier_mask = inlier_mask.ravel().astype(bool)
    
    print(f"RANSAC inliers: {np.sum(inlier_mask)}/{len(inlier_mask)}")
    
    return H, pts1_tracked, pts2_tracked, inlier_mask


def main():
    print("="*70)
    print("FRAMES 431 → 432 ANALYSIS (the 80-pixel error)")
    print("="*70)
    
    # Load video
    print("\n1. Loading House.mp4...")
    frames = load_video("Exercise Inputs-20251225/House.mp4")
    frames = remove_black_boundaries(frames)
    print(f"   Total frames: {len(frames)} (indices 0-{len(frames)-1})")
    
    # Extract frames 431 and 432 (NOT the last two!)
    frame_431 = frames[431]
    frame_432 = frames[432]
    
    # Convert to grayscale
    gray_431 = (to_grey_scale(frame_431) * 255).astype(np.uint8)
    gray_432 = (to_grey_scale(frame_432) * 255).astype(np.uint8)
    
    print(f"\n2. Frame 431 shape: {gray_431.shape}")
    print(f"   Frame 432 shape: {gray_432.shape}")
    
    # Visual similarity check
    diff = cv2.absdiff(gray_431, gray_432)
    mean_diff = np.mean(diff)
    print(f"\n3. Mean pixel difference: {mean_diff:.2f} (out of 255)")
    
    # Get correspondences and RANSAC results
    print(f"\n4. Running feature detection and optical flow...")
    H, pts1, pts2, inlier_mask = get_all_correspondences_with_ransac_result(gray_431, gray_432)
    
    if H is None:
        print("ERROR: Failed to compute homography!")
        return
    
    # Analyze the matches
    print(f"\n5. Analyzing matches...")
    displacements = np.linalg.norm(pts2 - pts1, axis=1)
    print(f"   Min displacement: {np.min(displacements):.2f} px")
    print(f"   Max displacement: {np.max(displacements):.2f} px")
    print(f"   Mean displacement: {np.mean(displacements):.2f} px")
    print(f"   Median displacement: {np.median(displacements):.2f} px")
    
    # Analyze inliers vs outliers separately
    inlier_displacements = displacements[inlier_mask]
    outlier_displacements = displacements[~inlier_mask]
    
    if len(inlier_displacements) > 0:
        print(f"\n   Inlier displacements:")
        print(f"     Mean: {np.mean(inlier_displacements):.2f} px")
        print(f"     Median: {np.median(inlier_displacements):.2f} px")
        print(f"     Range: [{np.min(inlier_displacements):.2f}, {np.max(inlier_displacements):.2f}]")
    
    if len(outlier_displacements) > 0:
        print(f"\n   Outlier displacements:")
        print(f"     Mean: {np.mean(outlier_displacements):.2f} px")
        print(f"     Median: {np.median(outlier_displacements):.2f} px")
        print(f"     Range: [{np.min(outlier_displacements):.2f}, {np.max(outlier_displacements):.2f}]")
    
    # Check the inlier points specifically
    print(f"\n6. Inlier point locations:")
    inlier_pts1 = pts1[inlier_mask]
    inlier_pts2 = pts2[inlier_mask]
    for i in range(min(10, len(inlier_pts1))):
        dx = inlier_pts2[i, 0] - inlier_pts1[i, 0]
        dy = inlier_pts2[i, 1] - inlier_pts1[i, 1]
        print(f"   Inlier {i}: ({inlier_pts1[i, 0]:.1f}, {inlier_pts1[i, 1]:.1f}) → "
              f"({inlier_pts2[i, 0]:.1f}, {inlier_pts2[i, 1]:.1f}), "
              f"dx={dx:.2f}, dy={dy:.2f}")
    
    # Create visualization
    print(f"\n7. Creating visualization...")
    canvas = visualize_correspondence_quality(gray_431, gray_432, H, pts1, pts2, inlier_mask)
    
    # Save
    output_path = "output_mosaics/last_two_frames_analysis.png"
    cv2.imwrite(output_path, canvas)
    print(f"   Saved: {output_path}")
    
    # Also create a matplotlib version with better layout
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: Original frames side by side
    axes[0].imshow(np.hstack([gray_431, gray_432]), cmap='gray')
    axes[0].set_title("Frame 431 (left) vs Frame 432 (right)")
    axes[0].axis('off')
    
    # Right: Correspondences
    axes[1].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Feature Matches: {np.sum(inlier_mask)} inliers, {len(inlier_mask)-np.sum(inlier_mask)} outliers")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("output_mosaics/last_two_frames_detailed.png", dpi=150, bbox_inches='tight')
    print(f"   Saved: output_mosaics/last_two_frames_detailed.png")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
