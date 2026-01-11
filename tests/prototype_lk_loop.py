import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from homography_evaluation import ransac_rigid_movement

def proposed_find_rigid_movement_with_loop(img1: np.ndarray, img2: np.ndarray, debug=False) -> tuple:
    """
    Proposed version with retry loop and blur.
    """
    # Convert to uint8 for OpenCV
    img1_uint = img1.astype(np.uint8)
    img2_uint = img2.astype(np.uint8)
    
    # 1. Detect points (ONCE, on original image)
    # We want to track the same features even if we blur the underlying image
    p0_cv = cv2.goodFeaturesToTrack(img1_uint, maxCorners=100, qualityLevel=0.01, minDistance=3)
    
    if p0_cv is None or len(p0_cv) < 2:
        if debug: print("Not enough corners detected initially.")
        return None, np.array([])

    current_img1 = img1_uint.copy()
    current_img2 = img2_uint.copy()
    
    # Variables to store the best result if we fail to hit "perfect" threshold
    best_H = None
    best_matched = np.array([])
    best_inliers_count = -1
    
    # Loop for retries
    max_iters = 5
    for i in range(max_iters):
        if debug: print(f"--- Iteration {i} (Blur applied: {i > 0}) ---")
        
        # 2. Track using OpenCV's calcOpticalFlowPyrLK
        p1_cv, st_cv, err_cv = cv2.calcOpticalFlowPyrLK(
            current_img1, current_img2, p0_cv, None,
            winSize=(15, 15), maxLevel=2
        )
        
        # 3. Filter valid tracks
        if p1_cv is None:
            status_count = 0
            good_old = np.array([])
            good_new = np.array([])
        else:
            good_old = p0_cv[st_cv == 1]
            good_new = p1_cv[st_cv == 1]
            status_count = len(good_old)
            
        if debug: print(f"   Tracked points: {status_count}/{len(p0_cv)}")

        # Success Condition:
        # If we tracked a decent percentage of points (e.g., > 60%)
        # AND we have enough absolute points (e.g., > 10)
        success_threshold = 0.6 * len(p0_cv)
        
        if status_count >= 10 and status_count >= success_threshold:
            if debug: print("   > Success! Threshold met.")
            # Calculate H and return
             # 4. Convert to our format (y, x)
            p1_our = good_old.squeeze()[:, [1, 0]]
            p2_our = good_new.squeeze()[:, [1, 0]]
            
            # 5. RANSAC
            H, inliers = ransac_rigid_movement(p1_our, p2_our)
            if H is not None:
                matched = np.hstack([p1_our[inliers], p2_our[inliers]])
                return H, matched
            else:
                if debug: print("   > RANSAC failed.")

        # If not successful or RANSAC failed, we might want to store this result if it's the "best so far"
        # In case we run out of iterations.
        if status_count >= 4: # Minimum for RANSAC
             p1_our = good_old.squeeze()[:, [1, 0]]
             p2_our = good_new.squeeze()[:, [1, 0]]
             H, inliers = ransac_rigid_movement(p1_our, p2_our)
             
             if H is not None:
                 inlier_count = len(inliers)
                 if inlier_count > best_inliers_count:
                     best_inliers_count = inlier_count
                     best_H = H
                     best_matched = np.hstack([p1_our[inliers], p2_our[inliers]])
        
        # Prepare for next iteration: Blur images
        # NOTE: Verify if we should accumulate blur or apply fresh blur to original.
        # User said "blur and try again". Usually implies increasing blur.
        # Repeating GaussianBlur on already blurred image increases sigma.
        # Or we can just blur the current state.
        if i < max_iters - 1:
            if debug: print("   > Blurring images for next attempt...")
            current_img1 = cv2.GaussianBlur(current_img1, (5, 5), 1)
            current_img2 = cv2.GaussianBlur(current_img2, (5, 5), 1)

    if debug: print("End of loop. Returning best found result.")
    return best_H, best_matched

def test_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return

    # Read two frames
    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read frames")
        return
        
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    print(f"Running prototype on {video_path}...")
    H, matched = proposed_find_rigid_movement_with_loop(gray1, gray2, debug=True)
    
    if H is not None:
        print("\nFinal Result: Homography found.")
        print(f"Matched points: {len(matched)}")
        print("H matrix:\n", H)
    else:
        print("\nFinal Result: FAILED to find homography.")

if __name__ == "__main__":
    video_path = "Exercise Inputs-20251225/House.mp4"
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    test_on_video(video_path)
