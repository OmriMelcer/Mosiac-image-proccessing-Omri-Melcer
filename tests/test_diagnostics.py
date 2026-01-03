import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.homography_evaluation import find_rigid_movement, compute_rigid_movement, ransac_rigid_movement

def create_synthetic_cube(img_size=(200, 200), cube_size=50, cube_pos=(75, 75)):
    img = np.zeros(img_size, dtype=np.float32)
    y, x = cube_pos
    img[y:y+cube_size, x:x+cube_size] = 255.0
    img = cv2.GaussianBlur(img, (21, 21), 3.0)
    return img

def warp_image(img, theta_deg, tx, ty):
    h, w = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, theta_deg, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    H_gt = np.eye(3)
    H_gt[:2] = M
    warped = cv2.warpAffine(img, M, (w, h))
    return warped, H_gt

print("="*60)
print("DIAGNOSTIC TEST: Checking Ground Truth vs Estimation")
print("="*60)

img1 = create_synthetic_cube()

# Test 1: Pure translation
print("\n--- Test 1: Pure Translation (5px X) ---")
img2, H_gt = warp_image(img1, theta_deg=0, tx=5.0, ty=0.0)
H_est, matches = find_rigid_movement(img1, img2)

print(f"Ground Truth Matrix:\n{H_gt}")
print(f"\nEstimated Matrix:\n{H_est}")
print(f"\nGT tx={H_gt[0,2]:.4f}, ty={H_gt[1,2]:.4f}")
print(f"EST tx={H_est[0,2]:.4f}, ty={H_est[1,2]:.4f}")
print(f"Error: tx={abs(H_est[0,2]-H_gt[0,2]):.4f}, ty={abs(H_est[1,2]-H_gt[1,2]):.4f}")
print(f"Matches: {len(matches) if matches is not None and len(matches) > 0 else 0}")

# Test 2: Small rotation
print("\n--- Test 2: Small Rotation (0.5 deg) ---")
img2, H_gt = warp_image(img1, theta_deg=0.5, tx=0.0, ty=0.0)
H_est, matches = find_rigid_movement(img1, img2)

gt_theta = np.degrees(np.arctan2(H_gt[1,0], H_gt[0,0]))
est_theta = np.degrees(np.arctan2(H_est[1,0], H_est[0,0]))

print(f"GT theta={gt_theta:.4f}, tx={H_gt[0,2]:.4f}, ty={H_gt[1,2]:.4f}")
print(f"EST theta={est_theta:.4f}, tx={H_est[0,2]:.4f}, ty={H_est[1,2]:.4f}")
print(f"Error: theta={abs(est_theta-gt_theta):.4f}, tx={abs(H_est[0,2]-H_gt[0,2]):.4f}, ty={abs(H_est[1,2]-H_gt[1,2]):.4f}")

# Test 3: Rotation + Translation
print("\n--- Test 3: Rotation (0.2 deg) + Translation (2px X) ---")
img2, H_gt = warp_image(img1, theta_deg=0.2, tx=2.0, ty=0.0)
H_est, matches = find_rigid_movement(img1, img2)

gt_theta = np.degrees(np.arctan2(H_gt[1,0], H_gt[0,0]))
est_theta = np.degrees(np.arctan2(H_est[1,0], H_est[0,0]))

print(f"GT theta={gt_theta:.4f}, tx={H_gt[0,2]:.4f}, ty={H_gt[1,2]:.4f}")
print(f"EST theta={est_theta:.4f}, tx={H_est[0,2]:.4f}, ty={H_est[1,2]:.4f}")
print(f"Error: theta={abs(est_theta-gt_theta):.4f}, tx={abs(H_est[0,2]-H_gt[0,2]):.4f}, ty={abs(H_est[1,2]-H_gt[1,2]):.4f}")

# Test 4: Sequential application
print("\n--- Test 4: Sequential Motion (5 steps of 1px X) ---")
print("Testing if errors accumulate...")

img_base = create_synthetic_cube()
cumulative_gt_tx = 0.0
cumulative_est_tx = 0.0

img_prev = img_base
for step in range(1, 6):
    # Ground truth: Move 1px to the right
    gt_tx = step * 1.0
    img_curr, H_gt = warp_image(img_base, theta_deg=0, tx=gt_tx, ty=0.0)
    
    # Estimate motion from prev to curr
    H_est, matches = find_rigid_movement(img_prev, img_curr)
    
    # Extract incremental motion
    delta_tx = H_est[0, 2]
    cumulative_est_tx += delta_tx
    cumulative_gt_tx = gt_tx
    
    error = cumulative_est_tx - cumulative_gt_tx
    
    print(f"  Step {step}: GT_cumulative={cumulative_gt_tx:.2f}, EST_cumulative={cumulative_est_tx:.2f}, Error={error:.4f}")
    
    img_prev = img_curr

print("\n--- Test 5: Checking RANSAC directly ---")
print("Creating perfect synthetic correspondences...")

# Create 4 perfect point pairs for a known 5px X translation
p1 = np.array([
    [50, 50],  # (y, x) format
    [50, 150],
    [150, 50],
    [150, 150]
], dtype=np.float64)

p2 = p1.copy()
p2[:, 1] += 5.0  # Add 5px in X direction

print(f"Point 1 positions:\n{p1}")
print(f"Point 2 positions (GT: 5px shift in X):\n{p2}")

# Direct computation (no RANSAC)
H_direct = compute_rigid_movement(p1, p2)
print(f"\nDirect compute_rigid_movement result:")
print(f"  tx={H_direct[0,2]:.4f}, ty={H_direct[1,2]:.4f}")
print(f"  Expected: tx=5.0, ty=0.0")
print(f"  Error: tx={abs(H_direct[0,2]-5.0):.4f}, ty={abs(H_direct[1,2]):.4f}")

# RANSAC
H_ransac, inliers = ransac_rigid_movement(p1, p2)
print(f"\nRANSAC result:")
print(f"  tx={H_ransac[0,2]:.4f}, ty={H_ransac[1,2]:.4f}")
print(f"  Inliers: {len(inliers)}/{len(p1)}")
print(f"  Error: tx={abs(H_ransac[0,2]-5.0):.4f}, ty={abs(H_ransac[1,2]):.4f}")
