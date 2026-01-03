import cv2
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath("src"))

from homography_evaluation import track_features_pyramid, harris_response, get_harris_points, build_pyramid, compute_gradients

def test_lk_isolation():
    # Load House frames
    video_path = "Exercise Inputs-20251225/House.mp4"
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    while len(frames) < 50:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    
    print(f"Loaded {len(frames)} frames")
    
    # Test pair
    idx = 20
    img1 = frames[idx]
    img2 = frames[idx+1]
    print(f"Testing Frames {idx} -> {idx+1}")
    
    # 1. Get points using OUR Harris
    resp = harris_response(img1, k=0.04, window_size=3)
    points = get_harris_points(resp, threshold=0.01, max_corners=200)
    print(f"Found {len(points)} Harris corners")
    
    # 2. Prepare Pyramids
    # Increase levels to handle large motion (24px needs 1/8 or 1/16 scale)
    num_levels = 5
    im_1_pyr = build_pyramid(img1, num_levels=num_levels)
    im_2_pyr = build_pyramid(img2, num_levels=num_levels)
    
    Im1_Ix_pyr = []
    Im1_Iy_pyr = []
    for level_img in im_1_pyr:
        Ix, Iy = compute_gradients(level_img)
        Im1_Ix_pyr.append(Ix)
        Im1_Iy_pyr.append(Iy)

    # 3. Track with OUR Pyramid LK (Default k_iters=3)
    print("Running with k_iters=3...")
    p1_k3, p2_k3 = track_features_pyramid(im_1_pyr, im_2_pyr, Im1_Ix_pyr, Im1_Iy_pyr, points, window_size=15)
    
    # 3b. Track with k_iters=20 (Manual loop)
    print("Running with k_iters=20...")
    from homography_evaluation import optical_flow_pyramid
    valid_p1_k20 = []
    valid_p2_k20 = []
    for point in points:
        dx, dy = optical_flow_pyramid(im_1_pyr, im_2_pyr, Im1_Ix_pyr, Im1_Iy_pyr, point, window_size=15, k_iters=20)
        motion_magnitude = np.sqrt(dx**2 + dy**2)
        if motion_magnitude > 1e-5 and motion_magnitude < 100.0: 
            new_point = (point[0] + dy, point[1] + dx)
            valid_p1_k20.append(point)
            valid_p2_k20.append(new_point)
    
    p1_our = p1_k3 # For compatibility with rest of script
    p2_our = p2_k3
    print(f"Our Pyramid LK tracked: {len(p1_our)} points")
    
    # 4. Track with OpenCV LK (Ground Truth)
    # OpenCV expects (x,y), our points are (y,x) from get_harris_points
    p0_cv = np.fliplr(np.array(points, dtype=np.float32)).reshape(-1, 1, 2)
    
    p1_cv, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0_cv, None, winSize=(15,15), maxLevel=3)
    
    # Count valid status
    valid_cv = np.sum(st)
    print(f"OpenCV LK tracked: {valid_cv} points")
    
    if valid_cv > 0:
        p1_cv_valid = p0_cv[st==1].reshape(-1, 2)
        p2_cv_valid = p1_cv[st==1].reshape(-1, 2)
        diff_cv = p2_cv_valid - p1_cv_valid
        mags_cv = np.sqrt(np.sum(diff_cv**2, axis=1))
        print(f"OpenCV Motion stats: Min={np.min(mags_cv):.4f}, Max={np.max(mags_cv):.4f}, Mean={np.mean(mags_cv):.4f}")

    # Calculate motion magnitude for our points
    if len(p1_k3) > 0:
        diff = p2_k3 - p1_k3
        mags = np.sqrt(np.sum(diff**2, axis=1))
        print(f"Our Motion stats (k=3): Min={np.min(mags):.4f}, Max={np.max(mags):.4f}, Mean={np.mean(mags):.4f}")

    if len(valid_p1_k20) > 0:
        p1_arr = np.array(valid_p1_k20)
        p2_arr = np.array(valid_p2_k20)
        diff = p2_arr - p1_arr
        mags = np.sqrt(np.sum(diff**2, axis=1))
        print(f"Our Motion stats (k=20): Min={np.min(mags):.4f}, Max={np.max(mags):.4f}, Mean={np.mean(mags):.4f}")

    # DEBUG: Trace one point that has large motion
    from homography_evaluation import optical_flow_pyramid
    
    print("\n--- DEBUGGING ONE POINT ---")
    # Check dtype of pyramid levels
    print(f"Pyramid Level 0 dtype: {im_1_pyr[0].dtype}")
    print(f"Pyramid Level 0 range: {np.min(im_1_pyr[0])} - {np.max(im_1_pyr[0])}")
    
    # Find a point that was tracked and has large motion
    # We need to re-run tracking to find indices
    p1_our, p2_our = track_features_pyramid(im_1_pyr, im_2_pyr, Im1_Ix_pyr, Im1_Iy_pyr, points, window_size=15)
    
    if len(p1_our) > 0:
        # Pick a point near the center (360, 640)
        center = np.array([360, 640])
        dists = np.linalg.norm(p1_our - center, axis=1)
        idx_center = np.argmin(dists)
        
        pt = p1_our[idx_center]
        print(f"Tracking point (near center): {pt}")
        
        # Manually call optical_flow_pyramid and print intermediate
        U, V = 0.0, 0.0
        for level in range(len(im_1_pyr)):
            scale = 2 ** (len(im_1_pyr) - level - 1)
            scaled_point = (pt[0] / scale, pt[1] / scale)
            scaled_U = U / scale
            scaled_V = V / scale
            
            # Call iterative for this level
            # We need to import it or just trust the final result
            # Let's just print the inputs
            print(f"Level {level}: scale={scale}, pt={scaled_point}, guess=({scaled_U:.4f}, {scaled_V:.4f})")
            
            # To really debug, we need to see what optical_flow_iterative returns at each level.
            # I will use a trick: I will call optical_flow_iterative directly here.
            from homography_evaluation import optical_flow_iterative
            
            # Calculate detA manually to see what it is
            w = 15 // 2
            iy, ix = round(scaled_point[0]), round(scaled_point[1])
            Ix = Im1_Ix_pyr[level]
            Iy = Im1_Iy_pyr[level]
            if iy-w >= 0 and iy+w+1 <= Ix.shape[0] and ix-w >= 0 and ix+w+1 <= Ix.shape[1]:
                Ix_window = Ix[iy-w:iy+w+1, ix-w:ix+w+1]
                Iy_window = Iy[iy-w:iy+w+1, ix-w:ix+w+1]
                Sxx = np.sum(Ix_window ** 2)
                Sxy = np.sum(Ix_window * Iy_window)
                Syy = np.sum(Iy_window ** 2)
                det_A = Sxx * Syy - Sxy * Sxy
                print(f"  -> Det(A) = {det_A:.6f}")
            
            du, dv = optical_flow_iterative(im_1_pyr[level], im_2_pyr[level], scaled_point, window_size=15, k_iters=3, Ix=Im1_Ix_pyr[level], Iy=Im1_Iy_pyr[level], initial_guess=(scaled_U, scaled_V))
            print(f"  -> Result: du={du:.4f}, dv={dv:.4f}")
            
            U = du * scale
            V = dv * scale
            print(f"  -> Accumulated: U={U:.4f}, V={V:.4f}")

    print(f"Our Pyramid LK tracked: {len(p1_our)} points")

if __name__ == "__main__":
    test_lk_isolation()
