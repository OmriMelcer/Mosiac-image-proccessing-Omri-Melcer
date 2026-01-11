
import cv2
import numpy as np
import logging
from src.video_io import load_video
from src.homography_evaluation import ransac_rigid_movement

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_black_boundaries(frames: np.ndarray, threshold: float = 1.0):
    column_means = np.mean(frames, axis=(0, 1, 3))
    visible_cols = np.where(column_means > threshold)[0]
    l, r = visible_cols[0], visible_cols[-1] + 1
    return frames[:, :, l:r, :]

def run_tracking_with_forced_blur(img1, img2, forced_blur_levels=0):
    """
    Simulates the tracking logic but applies blur unconditionally before tracking
    """
    img1_u = img1.astype(np.uint8)
    img2_u = img2.astype(np.uint8)
    
    # Pre-blur if requested
    for _ in range(forced_blur_levels):
        img1_u = cv2.GaussianBlur(img1_u, (5, 5), 1)
        img2_u = cv2.GaussianBlur(img2_u, (5, 5), 1)

    # 1. Detect
    p0_cv = cv2.goodFeaturesToTrack(img1_u, maxCorners=100, qualityLevel=0.01, minDistance=3)
    if p0_cv is None: return None, 0, 0
    
    # 2. Track
    p1_cv, st_cv, err_cv = cv2.calcOpticalFlowPyrLK(
        img1_u, img2_u, p0_cv, None, winSize=(15, 15), maxLevel=2
    )

    if p1_cv is None: return None, 0, 0
    
    good_old = p0_cv[st_cv == 1]
    good_new = p1_cv[st_cv == 1]
    
    if len(good_old) < 2: return None, 0, 0
    
    # 3. RANSAC
    # Convert to (y,x)
    p0_yx = good_old.squeeze().reshape(-1, 2)[:, [1, 0]]
    p1_yx = good_new.squeeze().reshape(-1, 2)[:, [1, 0]]
    
    H, inliers = ransac_rigid_movement(p0_yx, p1_yx)
    
    return H, len(good_old), len(inliers) if inliers is not None else 0

def test_blur_hypothesis():
    video_path = "Exercise Inputs-20251225/Kessaria.mp4"
    frames = load_video(video_path)
    frames = remove_black_boundaries(frames)
    
    # Indices that failed (Identity) or had Large Jumps
    problematic_indices = [3, 5, 8, 267]
    
    print(f"{'Idx':<4} | {'Blur':<4} | {'Found':<5} | {'Inliers':<7} | {'dx':<6} | {'dy':<6} | {'Comment'}")
    print("-" * 60)
    
    for idx in problematic_indices:
        f1 = cv2.cvtColor(frames[idx-1], cv2.COLOR_RGB2GRAY)
        f2 = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2GRAY)
        
        # Test 0 blur (Baseline), 1 blur, 2 blurs
        for blur in [0, 1, 2, 3]:
            H, found, inliers = run_tracking_with_forced_blur(f1, f2, forced_blur_levels=blur)
            
            dx_str = "N/A"
            dy_str = "N/A"
            comment = ""
            
            if H is not None:
                dx = H[0, 2]
                dy = H[1, 2]
                dx_str = f"{dx:.2f}"
                dy_str = f"{dy:.2f}"
                
                if abs(dx) < 0.01 and abs(dy) < 0.01:
                    comment = "IDENTITY (Based?)"
                elif abs(dx) > 20: 
                    comment = "LARGE JUMP"
                else: 
                    comment = "OK"
            else:
                comment = "FAILED"
                
            print(f"{idx:<4} | {blur:<4} | {found:<5} | {inliers:<7} | {dx_str:<6} | {dy_str:<6} | {comment}")
            
        print("-" * 60)

if __name__ == "__main__":
    test_blur_hypothesis()
