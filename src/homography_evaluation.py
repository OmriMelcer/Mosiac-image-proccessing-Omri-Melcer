
import numpy as np
from typing import List, Tuple
from scipy.signal import convolve2d
from scipy import ndimage

def compute_gradients(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Ix and Iy gradients using Sobel or central difference.
    """
    kernel_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    Ix = convolve2d(img, kernel_X, mode='same')
    Iy = convolve2d(img, kernel_Y, mode='same')
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
    local_max_mask = ndimage.maximum_filter(harris_response, size=3) == harris_response
    thresholded_mask = harris_response > threshold
    total_mask = local_max_mask & thresholded_mask
    return np.argwhere(total_mask)
    

def optical_flow_lk_point(I1: np.ndarray, I2: np.ndarray, point: Tuple[float, float], window_size: int = 15, Ix: np.ndarray = None, Iy: np.ndarray = None) -> Tuple[float, float]:
    """
    Calculates the optical flow (u, v) for a single point using Lucas-Kanade.
    Solving: A^T * A * v = A^T * b
    """
    y, x = point
    # Cast to int for slicing
    iy, ix = int(y), int(x)
    
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
    try:
        v = np.linalg.solve(A, b)
        return v[1], v[0]
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
        u, v = optical_flow_lk_point(I1, I2, point, window_size, Ix, Iy) 
        # Filter out zero motion if it implies failure, or huge motion
        # For now, we trust the output (or we could add a check if u,v == 0.0 and cond check)
        new_point = (point[0] + u, point[1] + v)
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
    T= np.array ([[scale,0,-scale*centroid[0]], [0,scale,-scale*centroid[1]], [0,0,1]])
    return normalized_points, T

    

def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Applies H to points.
    points: Nx2 (y, x)
    Returns: Nx2 (y, x)
    """
    # Convert to homogeneous (x, y, 1)
    3D_points = np.hstack([points, np.ones((points.shape[0], 1))])
    3D_projected_points = H @ 3D_points.T
    projected_points = 3D_projected_points.T[:, :2] / 3D_projected_points.T[:, 2:]
    return projected_points
    



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
    H, inliers = ransac_rigid_movement(p_1, p_2)
    return H, np.hstack([p_1[inliers], p_2[inliers]])
    
    
    

