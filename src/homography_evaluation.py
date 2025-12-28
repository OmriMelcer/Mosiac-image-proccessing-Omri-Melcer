
import numpy as np
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

def get_harris_points(harris_response: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Extracts corner points from the response map using thresholding and Non-Maximum Suppression (NMS).
    """
    # 1. Non-Maximum Suppression (NMS) - find local peaks in the spatial map
    local_max_mask = ndimage.maximum_filter(harris_response, size=3) == harris_response
    
    # 2. Thresholding
    thresholded_mask = harris_response > threshold
    
    # 3. Combine masks
    total_mask = local_max_mask & thresholded_mask
    
    # 4. Get coordinates
    coords = np.argwhere(total_mask)
    
    # 5. Extract values to sort
    values = harris_response[total_mask]
    
    # 6. Sort indices by value (descending)
    sorted_idx = np.argsort(values)[::-1]
    
    # 7. Take top 50
    top_50_idx = sorted_idx[:50]
    
    return coords[top_50_idx]
    

def optical_flow_iterative(I1: np.ndarray, I2: np.ndarray, point: Tuple[float, float], window_size: int = 15, k_iters: int = 5, Ix: np.ndarray = None, Iy: np.ndarray = None) -> Tuple[float, float]:
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
    U,V = 0.0, 0.0
    w = window_size // 2
    if iy-w < 0 or iy+w+1 > I1.shape[0] or ix-w < 0 or ix+w+1 > I1.shape[1]:
        return 0.0, 0.0
    if Ix is None or Iy is None:
        Ix, Iy = compute_gradients(I1)
    w = window_size // 2
    I1_window = I1[iy-w:iy+w+1, ix-w:ix+w+1]
    Ix_window = Ix[iy-w:iy+w+1, ix-w:ix+w+1]
    Iy_window = Iy[iy-w:iy+w+1, ix-w:ix+w+1]
    I2_window = I2[iy-w:iy+w+1, ix-w:ix+w+1]
    Sxx = np.sum(Ix_window ** 2)
    Sxy = np.sum(Ix_window * Iy_window)
    Syy = np.sum(Iy_window ** 2)
    for _ in range(k_iters):
        It_window = I2_window - I1_window
        Sxt = np.sum(Ix_window * It_window)
        Syt = np.sum(Iy_window * It_window)
        A = np.array([[Sxx, Sxy], [Sxy, Syy]])
        b = np.array([-Sxt, -Syt])
        # Solve A*v = b  => v = [u, v] = [x_motion, y_motion]
        try:
            v_sol = np.linalg.solve(A, b)
            vel_x, vel_y = v_sol[0], v_sol[1]
        except np.linalg.LinAlgError:
            return 0.0, 0.0
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
        if motion_magnitude > 0.01 and motion_magnitude < 10.0:  # Reject (0,0) and huge jumps
            new_point = (point[0] + dy, point[1] + dx)
            valid_p1.append(point)
            valid_p2.append(new_point)
    
    return np.array(valid_p1), np.array(valid_p2)

def compute_rigid_movement(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Computes Homography H such that p2 ~= H @ p1, using geometry.
    p1, p2 are 2x2 arrays of (y, x) coordinates.
    This function on1`ly calculates rigid movment (translation, rotation)
    """
    p_1_normalized,T1 = normalize_points (p1)
    p_2_normalized,T2 = normalize_points (p2)
    y_1_1, x_1_1 = p_1_normalized[0]
    y_2_1, x_2_1 = p_1_normalized[1]
    y_1_2, x_1_2 = p_2_normalized[0]
    y_2_2, x_2_2 = p_2_normalized[1]
    dy_1 = y_2_1 - y_1_1
    dx_1 = x_2_1 - x_1_1
    dy_2 = y_2_2 - y_1_2
    dx_2 = x_2_2 - x_1_2
    theta_1 = np.arctan2(dy_1,dx_1)
    theta_2 = np.arctan2(dy_2,dx_2)
    # Correct rotation: theta_2 = theta_1 + theta => theta = theta_2 - theta_1
    theta = theta_2 - theta_1
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    tx= x_1_2 -(cos_t * x_1_1 - sin_t * y_1_1)
    ty = y_1_2 -(sin_t * x_1_1 + cos_t * y_1_1)
    H = np.array([[cos_t, -sin_t, tx], [sin_t, cos_t, ty], [0, 0, 1]])
    H = np.linalg.inv(T2) @ H @ T1
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
        
        
        


def find_rigid_movement(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orchestrates the process:
    1. Gradients & Harris
    2. Tracking
    3. RANSAC -> Homography
    """
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
        
    return H, np.hstack([p_1[inliers], p_2[inliers]])
    
    
    

