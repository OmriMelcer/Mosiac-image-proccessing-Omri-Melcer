
import cv2
import numpy as np
import logging
from src.video_io import load_video
from src.homography_evaluation import find_rigid_movement_opencv_with_our_ransac, ransac_rigid_movement

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_black_boundaries(frames: np.ndarray, threshold: float = 1.0):
    column_means = np.mean(frames, axis=(0, 1, 3))
    visible_cols = np.where(column_means > threshold)[0]
    if len(visible_cols) == 0: return frames
    l, r = visible_cols[0], visible_cols[-1] + 1
    return frames[:, :, l:r, :]

def debug_tracking_failure():
    video_path = "Exercise Inputs-20251225/Kessaria.mp4"
    frames = load_video(video_path)
    frames = remove_black_boundaries(frames)
    
    # Analyze Frame 4 -> 5 (Index 4 is Target, Index 3 is Source... wait)
    # Pairwise loop is range(1, len): i-1 -> i
    # Frame 5 failure means i=5 -> frames[4] -> frames[5]
    
    # Let's check frames[4] and frames[5] (indices are 0-based)
    # The output log said "5 | ... | IDENTITY". This depends on how I printed it.
    # In my script: for i in range(1, len(pairwise_transforms)). i is the index in transforms array.
    # transforms[i] is frames[i-1] -> frames[i]
    # So i=5 is frames[4] -> frames[5].
    
    idx1 = 4
    idx2 = 5
    
    img1 = cv2.cvtColor(frames[idx1], cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(frames[idx2], cv2.COLOR_RGB2GRAY)
    
    print(f"Debug Frames {idx1}->{idx2}")
    
    # 1. Check Initial Points
    p0_cv = cv2.goodFeaturesToTrack(img1, maxCorners=100, qualityLevel=0.01, minDistance=3)
    if p0_cv is None:
        print("FAIL: No features detected in img1")
        return
    print(f"Initial features detected: {len(p0_cv)}")
    
    # 2. Run the function but add print statements (or replicate logic)
    # Replicating logic here to see what happens
    
    current_img1 = img1.copy()
    current_img2 = img2.copy()
    
    best_len = -1
    
    for i in range(5):
        p1_cv, st_cv, err_cv = cv2.calcOpticalFlowPyrLK(
            current_img1, current_img2, p0_cv, None,
            winSize=(15, 15), maxLevel=2
        )
        
        tracked_count = 0
        if p1_cv is not None:
             good_old = p0_cv[st_cv == 1]
             good_new = p1_cv[st_cv == 1]
             tracked_count = len(good_old)
             
        print(f"  Attempt {i} (Blur level {i}): Tracked {tracked_count}/{len(p0_cv)} points ({tracked_count/len(p0_cv)*100:.1f}%)")
        
        # Blur for next
        current_img1 = cv2.GaussianBlur(current_img1, (5, 5), 1)
        current_img2 = cv2.GaussianBlur(current_img2, (5, 5), 1)
        
    # Check what result we actually get calling the real function
    H, matched = find_rigid_movement_opencv_with_our_ransac(img1.astype(np.float32), img2.astype(np.float32))
    
    if H is None:
        print("Function returned None (Identity)")
    else:
        print("Function returned Matrix:")
        print(H)
        dx = H[0, 2]
        dy = H[1, 2]
        print(f"dx={dx}, dy={dy}")

    # Visualize features
    # Draw p0 on img1
    vis1 = frames[idx1].copy()
    for pt in p0_cv:
        x, y = pt.ravel()
        cv2.circle(vis1, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    cv2.imwrite("debug_frame4_features.png", cv2.cvtColor(vis1, cv2.COLOR_RGB2BGR))
    cv2.imwrite("debug_frame4_raw.png", cv2.cvtColor(frames[idx1], cv2.COLOR_RGB2BGR))
    cv2.imwrite("debug_frame5_raw.png", cv2.cvtColor(frames[idx2], cv2.COLOR_RGB2BGR))
    print("Saved debug images.")

if __name__ == "__main__":
    debug_tracking_failure()
