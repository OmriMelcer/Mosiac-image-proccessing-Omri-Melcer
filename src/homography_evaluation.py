import numpy as np
import cv2
from typing import List, Tuple
from scipy.signal import convolve2d
from scipy import ndimage

def compute_gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Ix and Iy gradients using Sobel or central difference.
    """
    kernel_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = convolve2d(img, kernel_X, mode='same') / 8.0
    Iy = convolve2d(img, kernel_Y, mode='same') / 8.0
    return Ix, Iy

def harris_response(img: np.ndarray, k: float = 0.04, window_size: int = 3, Ix: np.ndarray = None, Iy: np.ndarray = None) -> np.ndarray:
    """
    Computes the Harris Corner Response map for the image.
    """
    if Ix is None or Iy is None:
        Ix, Iy = compute_gradients(img)
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    kernel = np.ones((window_size, window_size))
    Sxx = convolve2d(Ixx, kernel, mode='same', boundary='symm')
    Syy = convolve2d(Iyy, kernel, mode='same', boundary='symm')
    Sxy = convolve2d(Ixy, kernel, mode='same', boundary='symm')
    detM = (Sxx * Syy) - (Sxy ** 2)
    traceM = Sxx + Syy
    responses = detM - k * (traceM ** 2)
    return responses

def get_harris_points(harris_response: np.ndarray, threshold: float = 0.01, max_corners: int = 200, num_nms: int = 9) -> np.ndarray:
    """
    Extracts corner points from the response map using thresholding and Non-Maximum Suppression (NMS).
    
    Args:
        harris_response: Harris corner response map
        threshold: Quality level (relative to max response, like OpenCV's qualityLevel)
        max_corners: Maximum number of corners to return
    
    Returns:
        Array of (y, x) coordinates of detected corners
    """
    # 1. Non-Maximum Suppression (NMS) - find local peaks in the spatial map
    local_max_mask = ndimage.maximum_filter(harris_response, size=num_nms) == harris_response
    
    # 2. Relative Thresholding (like OpenCV's qualityLevel)
    # Only consider points with response > threshold * max(response)
    max_response = np.max(harris_response)
    absolute_threshold = threshold * max_response
    thresholded_mask = harris_response > absolute_threshold
    
    # 3. Combine masks
    total_mask = local_max_mask & thresholded_mask
    
    # 4. Get coordinates
    coords = np.argwhere(total_mask)
    
    # 5. Extract values to sort
    values = harris_response[total_mask]
    
    # 6. Sort indices by value (descending)
    sorted_idx = np.argsort(values)[::-1]
    
    # 7. Take top N corners
    if max_corners > 0:
        top_n_idx = sorted_idx[:max_corners]
    else:
        top_n_idx = sorted_idx
    
    return coords[top_n_idx]
    

def optical_flow_iterative(I1: np.ndarray, I2: np.ndarray, point: Tuple[float, float], window_size: int = 15, k_iters: int = 3, Ix: np.ndarray = None, Iy: np.ndarray = None, initial_guess: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    """
    Computes the optical flow (u, v) using Iterative Lucas-Kanade (Newton-Raphson).
    
    Args:
        I1: Template image (at time t)
        I2: Target image (at time t+1)
        point: (y, x) coordinate in I1 to track
        window_size: Size of the window (must be odd)
        k_iters: Number of iterations for refinement
        Ix, Iy: Image gradients of I1 (optional, computed if None)

    Returns:
        (u, v): The estimated flow vector such that I1(y, x) ~= I2(y+v, x+u)
                where u = x_motion and v = y_motion
    """
    cur_y, cur_x = point
    # Cast to int for slicing
    iy, ix = round(cur_y), round(cur_x)
    vel_y,vel_x = 0.0, 0.0
    U,V = initial_guess
    cur_x += U
    cur_y += V
    w = window_size // 2
    if iy-w < 0 or iy+w+1 > I1.shape[0] or ix-w < 0 or ix+w+1 > I1.shape[1]:
        return 0.0, 0.0
    if Ix is None or Iy is None:
        Ix, Iy = compute_gradients(I1)
    w = window_size // 2
    I1_window = I1[iy-w:iy+w+1, ix-w:ix+w+1].astype(np.float32)
    Ix_window = Ix[iy-w:iy+w+1, ix-w:ix+w+1]
    Iy_window = Iy[iy-w:iy+w+1, ix-w:ix+w+1]
    I2_window = I2[iy-w:iy+w+1, ix-w:ix+w+1].astype(np.float32)
    Sxx = np.sum(Ix_window ** 2)
    Sxy = np.sum(Ix_window * Iy_window)
    Syy = np.sum(Iy_window ** 2)
    for _ in range(k_iters):
        It_window = I2_window - I1_window
        Sxt = np.sum(Ix_window * It_window)
        Syt = np.sum(Iy_window * It_window)
        A = np.array([[Sxx, Sxy], [Sxy, Syy]])
        b = np.array([-Sxt, -Syt])
        
        # Check for ill-conditioning (determinant close to zero)
        det_A = Sxx * Syy - Sxy * Sxy
        if det_A < 1e-6: 
            return U, V

        # Solve A*v = b  => v = [u, v] = [x_motion, y_motion]
        try:
            v_sol = np.linalg.solve(A, b)
            vel_x, vel_y = v_sol[0], v_sol[1]
        except np.linalg.LinAlgError:
            return U, V # Return current guess if solve fails
            
        if abs(vel_x) < 1e-3 and abs(vel_y) < 1e-3:
            break
        cur_x, cur_y = cur_x + vel_x, cur_y + vel_y
        mesh_x, mesh_y = np.meshgrid(np.arange(window_size), np.arange(window_size))
        mesh_x = mesh_x - w + cur_x
        mesh_y = mesh_y - w + cur_y
        x_coords = mesh_x.ravel()
        y_coords = mesh_y.ravel()
        coords = np.vstack((y_coords, x_coords))        
        I2_window = ndimage.map_coordinates(I2, coords, order=1, prefilter=False).reshape(window_size, window_size)
        U, V = U + vel_x, V + vel_y
    return U, V  # Return (x_motion, y_motion)
        



def optical_flow_lk_point(I1: np.ndarray, I2: np.ndarray, point: Tuple[float, float], window_size: int = 15, Ix: np.ndarray = None, Iy: np.ndarray = None) -> Tuple[float, float]:
    """
    Calculates the optical flow (u, v) for a single point using Lucas-Kanade.
    Solving: A^T * A * v = A^T * b
    """
    
    y, x = point
    # Cast to int for slicing
    iy, ix = round(y), round(x)
    
    if Ix is None or Iy is None:
        Ix, Iy = compute_gradients(I1)
    w = window_size // 2
    
    # Ensure window is within bounds (logic simplified here)
    if iy-w < 0 or iy+w+1 > I1.shape[0] or ix-w < 0 or ix+w+1 > I1.shape[1]:
        return 0.0, 0.0

    I1_window = I1[iy-w:iy+w+1, ix-w:ix+w+1]
    I2_window = I2[iy-w:iy+w+1, ix-w:ix+w+1]
    Ix_window = Ix[iy-w:iy+w+1, ix-w:ix+w+1]
    Iy_window = Iy[iy-w:iy+w+1, ix-w:ix+w+1]
    
    
    Sxx = np.sum(Ix_window ** 2)
    Sxy = np.sum(Ix_window * Iy_window)
    Syy = np.sum(Iy_window ** 2)
    
    It_window = I2_window - I1_window
    Sxt = np.sum(Ix_window * It_window)
    Syt = np.sum(Iy_window * It_window)
    
    A = np.array([[Sxx, Sxy], [Sxy, Syy]])
    b = np.array([-Sxt, -Syt])
    
    # Solve A*d = b  => d = [du, dv] = [x_motion, y_motion]
    try:
        d_sol = np.linalg.solve(A, b)
        # Return (x_motion, y_motion)
        return d_sol[0], d_sol[1]
    except np.linalg.LinAlgError:
        return 0.0, 0.0
       

def track_features(I1: np.ndarray, I2: np.ndarray, points: np.ndarray, window_size: int = 15, Ix: np.ndarray = None, Iy: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tracks multiple features from I1 to I2.
    Returns:
        valid_p1: Points in I1 that were successfully tracked.
        valid_p2: New coordinates of those points in I2.
    """
    if Ix is None or Iy is None:
        Ix, Iy = compute_gradients(I1)
    valid_p1 = []
    valid_p2 = []
    for point in points:
        dx, dy = optical_flow_iterative(I1, I2, point, window_size, k_iters=5, Ix=Ix, Iy=Iy) 
        motion_magnitude = np.sqrt(dx**2 + dy**2)
        # Lowered threshold to 1e-5 to allow very slow motion (like in House.mp4)
        # but still reject absolute 0.0 which usually indicates convergence failure
        if motion_magnitude > 1e-5 and motion_magnitude < 20.0:  
            new_point = (point[0] + dy, point[1] + dx)
            valid_p1.append(point)
            valid_p2.append(new_point)
    
    return np.array(valid_p1), np.array(valid_p2)

def track_features_pyramid(im_1_pyr: List[np.ndarray], im_2_pyr: List[np.ndarray], IX_pyr: List[np.ndarray], IY_pyr: List[np.ndarray], points: np.ndarray, window_size: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tracks multiple features using pyramid optical flow.
    Args:
        im_1_pyr: Image 1 pyramid (coarse to fine)
        im_2_pyr: Image 2 pyramid (coarse to fine)
        IX_pyr: X-gradients pyramid for Image 1
        IY_pyr: Y-gradients pyramid for Image 1
        points: Harris points detected on original image
        window_size: LK window size
    Returns:
        valid_p1: Points in I1 that were successfully tracked.
        valid_p2: New coordinates of those points in I2.
    """
    valid_p1 = []
    valid_p2 = []
    for point in points:
        dx, dy = optical_flow_pyramid(im_1_pyr, im_2_pyr, IX_pyr, IY_pyr, point, window_size, k_iters=20)
        motion_magnitude = np.sqrt(dx**2 + dy**2)
        # Lowered threshold to 1e-5 to allow very slow motion (like in House.mp4)
        # but still reject absolute 0.0 which usually indicates convergence failure
        if motion_magnitude > 1e-5 and motion_magnitude < 100.0: 
            new_point = (point[0] + dy, point[1] + dx)
            valid_p1.append(point)
            valid_p2.append(new_point)
    
    return np.array(valid_p1), np.array(valid_p2)

def compute_rigid_movement(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Computes Homography H such that p2 ~= H @ p1, using geometry.
    p1, p2 are 2x2 arrays of (y, x) coordinates.
    This function only calculates rigid movement (translation, rotation)
    """
    # Work directly with original points (no normalization)
    y_1_1, x_1_1 = p1[0]
    y_2_1, x_2_1 = p1[1]
    y_1_2, x_1_2 = p2[0]
    y_2_2, x_2_2 = p2[1]
    dy_1 = y_2_1 - y_1_1
    dx_1 = x_2_1 - x_1_1
    dy_2 = y_2_2 - y_1_2
    dx_2 = x_2_2 - x_1_2
    theta_1 = np.arctan2(dy_1, dx_1)
    theta_2 = np.arctan2(dy_2, dx_2)
    # Correct rotation: theta_2 = theta_1 + theta => theta = theta_2 - theta_1
    theta = theta_2 - theta_1
    
    # Wrap theta to (-pi, pi) range to avoid accumulation drift
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    tx = x_1_2 - (cos_t * x_1_1 - sin_t * y_1_1)
    ty = y_1_2 - (sin_t * x_1_1 + cos_t * y_1_1)
    H = np.array([[cos_t, -sin_t, tx], [sin_t, cos_t, ty], [0, 0, 1]])
    return H

def normalize_points(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizes points to have zero mean and unit variance.
    """
    centroid = np.mean(p, axis=0)
    shifted_points = p - centroid
    mean_distance = np.mean (np.linalg.norm(shifted_points, axis=1))
    scale = np.sqrt(2) / mean_distance
    normalized_points = shifted_points * scale
    # T constuction for (x, y) homography: T * [x, y, 1]^T
    # normalized_x = scale * (x - centroid_x)
    # normalized_y = scale * (y - centroid_y)
    # centroid is (y, x) -> centroid[0]=y, centroid[1]=x
    T = np.array([
        [scale, 0, -scale * centroid[1]], 
        [0, scale, -scale * centroid[0]], 
        [0, 0, 1]
    ])
    return normalized_points, T

def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Applies H to points.
    points: Nx2 (y, x)
    Returns: Nx2 (y, x)
    """
    # Convert to homogeneous (x, y, 1). Input points are (y, x).
    # Swap columns to get (x, y)
    points_xy = points[:, [1, 0]]
    hom_ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points_xy, hom_ones])
    
    # H @ P.T -> (3, N)
    projected_h = (H @ points_h.T).T
    
    # Normalize by w
    projected_xy = projected_h[:, :2] / projected_h[:, 2:]
    
    # Convert back to (y, x)
    return projected_xy[:, [1, 0]]

def ransac_rigid_movement(p1: np.ndarray, p2: np.ndarray, num_iters: int = 1000, threshold: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    RANSAC for rigid movement estimation.
    """
    np.random.seed(0)
    best_inliers_args = None
    best_H = None
    best_inliers_count = 0;
    for i in range(num_iters):
        # Robust sampling: Try to find points that are far enough apart
        sample = None
        for _ in range(10): # Try 10 times to find good points
            candidate = np.random.choice(p1.shape[0], 2, replace=False)
            dist = np.linalg.norm(p1[candidate[0]] - p1[candidate[1]])
            if dist >= 20.0: # Minimum 20 pixels distance for stable rotation
                sample = candidate
                break
        
        # Fallback if points are too clustered
        if sample is None:
            sample = np.random.choice(p1.shape[0], 2, replace=False)
            
        H = compute_rigid_movement(p1[sample], p2[sample])
        projected_points = apply_homography(H, p1)
        distances = np.linalg.norm (projected_points - p2, axis=1)
        inliers_mask = distances < threshold
        current_inliers = np.sum(inliers_mask)
        if current_inliers > best_inliers_count:
            best_inliers_count = current_inliers
            best_inliers_args = np.where(inliers_mask)[0]
            best_H = H
    return best_H, best_inliers_args

def find_rigid_movement(img1: np.ndarray, img2: np.ndarray, target_pixels: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orchestrates the process:
    1. Gradients & Harris
    2. Tracking
    3. RANSAC -> Homography
    
    Args:
        img1: First image
        img2: Second image
        target_pixels: Target number of pixels for downsampling. If -1, no downsampling.
                       Example: 200000 means downsample to ~200k pixels
    """
    # Calculate downsampling scale if needed
    scale = 1.0
    if target_pixels > 0:
        current_pixels = img1.shape[0] * img1.shape[1]
        if current_pixels > target_pixels:
            scale = np.sqrt(target_pixels / current_pixels)
            new_h = int(img1.shape[0] * scale)
            new_w = int(img1.shape[1] * scale)
            img1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    Im1_Ix, Im1_Iy = compute_gradients(img1)
    # Im2_Ix, Im2_Iy = compute_gradients(img2)
    Im_1_Harris_responses = harris_response(img1, Ix=Im1_Ix, Iy=Im1_Iy)
    # Im_2_Harris_responses = harris_response(img2, Im2_Ix, Im2_Iy)
    Im_1_Harris_points = get_harris_points(Im_1_Harris_responses)
    # Im_2_Harris_points = get_harris_points(Im_2_Harris_responses)
    p_1, p_2 = track_features(img1, img2, Im_1_Harris_points, Ix=Im1_Ix, Iy=Im1_Iy)
    if len(p_1) < 2:
        return None, np.array([])
    H, inliers = ransac_rigid_movement(p_1, p_2)
    
    # Check if H is None (RANSAC failed)
    if H is None:
        return None, np.array([])
    
    # Scale translations back to original image coordinates
    if scale != 1.0:
        H[0, 2] /= scale
        H[1, 2] /= scale
        # Scale point coordinates back
        matched = np.hstack([p_1[inliers], p_2[inliers]])
        matched /= scale
    else:
        matched = np.hstack([p_1[inliers], p_2[inliers]])
    
    return H, matched

def find_rigid_movement_pyramid(img1: np.ndarray, img2: np.ndarray, target_pixels: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orchestrates the process using pyramid LK:
    1. Build pyramids & compute gradients at each level
    2. Harris detection on original image
    3. Pyramid tracking
    4. RANSAC -> Homography
    
    Args:
        img1: First image
        img2: Second image
        target_pixels: Target number of pixels for downsampling. If -1, no downsampling.
                       Example: 200000 means downsample to ~200k pixels
    """
    # Calculate downsampling scale if needed
    scale = 1.0
    if target_pixels > 0:
        current_pixels = img1.shape[0] * img1.shape[1]
        if current_pixels > target_pixels:
            scale = np.sqrt(target_pixels / current_pixels)
            new_h = int(img1.shape[0] * scale)
            new_w = int(img1.shape[1] * scale)
            img1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Increased pyramid levels to 5 to handle larger motion (e.g. 25px)
    im_1_pyr = build_pyramid(img1, num_levels=5)
    im_2_pyr = build_pyramid(img2, num_levels=5)
    Im1_Ix_pyr = []
    Im1_Iy_pyr = []
    for level_img in im_1_pyr:
        Ix, Iy = compute_gradients(level_img)
        Im1_Ix_pyr.append(Ix)
        Im1_Iy_pyr.append(Iy)
    Im_1_Harris_responses = harris_response(img1, Ix=Im1_Ix_pyr[-1], Iy=Im1_Iy_pyr[-1])
    Im_1_Harris_points = get_harris_points(Im_1_Harris_responses)
    # Track features using pyramid method
    p_1, p_2 = track_features_pyramid(im_1_pyr, im_2_pyr, Im1_Ix_pyr, Im1_Iy_pyr, Im_1_Harris_points, window_size=15)
    if len(p_1) < 2:
        return None, np.array([])
    H, inliers = ransac_rigid_movement(p_1, p_2)
    # Check if H is None (RANSAC failed)
    if H is None:
        return None, np.array([])
    
    # Scale translations back to original image coordinates
    if scale != 1.0:
        H[0, 2] /= scale
        H[1, 2] /= scale
        # Scale point coordinates back
        matched = np.hstack([p_1[inliers], p_2[inliers]])
        matched /= scale
    else:
        matched = np.hstack([p_1[inliers], p_2[inliers]])
    
    return calculate_median_motion_from_inliers_SVD(p_1[inliers], p_2[inliers]), matched

def build_pyramid(img: np.ndarray, num_levels: int) -> List[np.ndarray]:
    """
    Builds Gaussian pyramid for the image.
    """
    pyramid = [img]
    for _ in range(1, num_levels):
        img = ndimage.zoom(img, 0.5, order=1, prefilter=True)
        pyramid.append(img)
    return pyramid[::-1]  # Return from smallest to largest

def optical_flow_pyramid(Im1_pyr, Im2_pyr, IX_pyr, IY_pyr, point: Tuple[float, float], window_size: int = 15, k_iters: int = 3) -> Tuple[float, float]:
    """
    Computes optical flow using pyramids.
    """
    U, V = 0.0, 0.0
    for level in range(len(Im1_pyr)):
        scale = 2 ** (len(Im1_pyr) - level - 1)
        scaled_point = (point[0] / scale, point[1] / scale)
        # Scale accumulated motion down to current level for initial guess
        scaled_U = U / scale
        scaled_V = V / scale
        # Run LK at this level with scaled initial guess
        du, dv = optical_flow_iterative(Im1_pyr[level], Im2_pyr[level], scaled_point, window_size, k_iters, IX_pyr[level], IY_pyr[level], initial_guess=(scaled_U, scaled_V))
        # Add refinement (du, dv are in current level's pixel units)
        U = du * scale  # Scale refinement back to full resolution
        V = dv * scale
    return U, V

def calculate_median_motion_from_inliers_SVD(p1: np.ndarray, p2: np.ndarray)->np.ndarray:
    '''
    Docstring for calculate_median_motion_from_inliers_SVD    :param p1: Description
    :type p1: np.ndarray an array of shape (N, 2), of points (y, x), in imgage 1.
    :param p2: Description an array of shape (N, 2), of points (y, x), in imgage 2.
    :type p2: np.ndarray
    :return: returns the homography matrix H that best fits the rigid movment between p1 and p2. 
    :rtype: ndarray
    '''
    # convert points to (x, y)
    p1_xy = p1[:, [1, 0]]
    p2_xy = p2[:, [1, 0]]
    centroid_p1 = np.mean(p1_xy, axis=0)
    centroid_p2 = np.mean(p2_xy, axis=0)
    centered_p1 = p1_xy - centroid_p1
    centered_p2 = p2_xy - centroid_p2
    H_matrix = centered_p2.T @ centered_p1
    U, S, Vt = np.linalg.svd(H_matrix)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    t = centroid_p2 - R @ centroid_p1
    H = np.eye(3)
    H[0:2, 0:2] = R
    H[:2, 2] = t
    return H

def find_rigid_movement_opencv_with_our_ransac(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uses OpenCV for point detection and tracking, but OUR RANSAC for transform estimation.
    This isolates whether the bug is in our RANSAC or in our point detection/tracking.
    
    Args:
        img1: First image (grayscale float32)
        img2: Second image (grayscale float32)
    
    Returns:
        H: 3x3 homography matrix
        matched: Array of matched points [p1_x, p1_y, p2_x, p2_y] for inliers
    """
    # Convert to uint8 for OpenCV
    img1_uint = img1.astype(np.uint8)
    img2_uint = img2.astype(np.uint8)
    
    # 1. Detect points using OpenCV's goodFeaturesToTrack
    p0_cv = cv2.goodFeaturesToTrack(img1_uint, maxCorners=100, qualityLevel=0.01, minDistance=3)
    
    if p0_cv is None or len(p0_cv) < 2:
        return None, np.array([])
    
    current_img1 = img1_uint.copy()
    current_img2 = img2_uint.copy()
    
    best_len = -1
    best_p1_our = None
    best_p2_our = None
    
    # Try up to 5 times with increasing blur just for tracking
    for i in range(5):
        # 2. Track using OpenCV's calcOpticalFlowPyrLK
        p1_cv, st_cv, err_cv = cv2.calcOpticalFlowPyrLK(
            current_img1, current_img2, p0_cv, None,
            winSize=(15, 15), maxLevel=2
        )
        
        if p1_cv is not None:
             good_old = p0_cv[st_cv == 1]
             good_new = p1_cv[st_cv == 1]
             
             curr_len = len(good_old)
             if curr_len > best_len:
                 best_len = curr_len
                 # 4. Convert to our format (y, x) immediately to save state
                 if curr_len >= 2:
                     best_p1_our = good_old.squeeze()[:, [1, 0]]
                     best_p2_our = good_new.squeeze()[:, [1, 0]]
                 
             # Success condition: tracked > 60% of points
             if len(p0_cv) > 0 and curr_len / len(p0_cv) > 0.6:
                 break
                 
        # Blur for next iteration
        if i < 4:
            current_img1 = cv2.GaussianBlur(current_img1, (5, 5), 1)
            current_img2 = cv2.GaussianBlur(current_img2, (5, 5), 1)
    
    if best_p1_our is None or len(best_p1_our) < 2:
        return None, np.array([])
        
    p1_our = best_p1_our
    p2_our = best_p2_our
    
    # 5. Use OUR RANSAC
    H, inliers = ransac_rigid_movement(p1_our, p2_our)
    
    if H is None:
        return None, np.array([])
    
    matched = np.hstack([p1_our[inliers], p2_our[inliers]])
    
    return H, matched    