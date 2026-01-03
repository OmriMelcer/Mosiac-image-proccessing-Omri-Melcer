import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.homography_evaluation import find_rigid_movement, find_rigid_movement_pyramid

def create_synthetic_cube(img_size=(200, 200), cube_size=50, cube_pos=(75, 75)):
    """Creates a black image with a white square, blurred for derivatives."""
    img = np.zeros(img_size, dtype=np.float32)
    y, x = cube_pos
    img[y:y+cube_size, x:x+cube_size] = 255.0
    img = cv2.GaussianBlur(img, (21, 21), 3.0)
    return img

def warp_image(img, theta_deg, tx, ty):
    """Warps image using CV2 to create ground truth."""
    h, w = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, theta_deg, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    H_gt = np.eye(3)
    H_gt[:2] = M
    warped = cv2.warpAffine(img, M, (w, h))
    return warped, H_gt

def test_motion(gt_tx, gt_ty, use_pyramid=False):
    """Test single motion estimation."""
    img1 = create_synthetic_cube()
    img2, H_gt = warp_image(img1, 0, gt_tx, gt_ty)
    
    method = "Pyramid LK" if use_pyramid else "Iterative LK"
    func = find_rigid_movement_pyramid if use_pyramid else find_rigid_movement
    
    H_est, matched = func(img1, img2)
    
    if H_est is None:
        print(f"{method} FAILED for motion ({gt_tx}, {gt_ty})")
        return None, None
    
    est_tx = H_est[0, 2]
    est_ty = H_est[1, 2]
    
    err_x = abs(est_tx - gt_tx)
    err_y = abs(est_ty - gt_ty)
    
    print(f"{method:15s} | GT: ({gt_tx:5.1f}, {gt_ty:5.1f}) | Est: ({est_tx:6.2f}, {est_ty:6.2f}) | Err: ({err_x:5.2f}, {err_y:5.2f}) | Matches: {len(matched) if matched is not None and len(matched) > 0 else 0}")
    
    return err_x, err_y

print("=== Small Motion Tests (0-2px) ===")
for tx in [0.5, 1.0, 1.5, 2.0]:
    test_motion(tx, 0.0, use_pyramid=False)
    test_motion(tx, 0.0, use_pyramid=True)
    print()

print("\n=== Medium Motion Tests (5-15px) ===")
for tx in [5.0, 10.0, 15.0]:
    test_motion(tx, 0.0, use_pyramid=False)
    test_motion(tx, 0.0, use_pyramid=True)
    print()

print("\n=== Large Motion Tests (20-40px) ===")
for tx in [20.0, 30.0, 40.0]:
    test_motion(tx, 0.0, use_pyramid=False)
    test_motion(tx, 0.0, use_pyramid=True)
    print()
