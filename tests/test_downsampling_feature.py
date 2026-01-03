import numpy as np
import cv2
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.homography_evaluation import find_rigid_movement_pyramid

def test_downsampling():
    """
    Test that downsampling parameter works and provides expected speedup.
    """
    print("\n" + "="*80)
    print("TESTING: Downsampling Feature")
    print("="*80)
    
    if not os.path.exists("test1.jpg"):
        print("Skipping: test1.jpg not found")
        return
    
    img1 = cv2.imread("test1.jpg")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img1 = cv2.GaussianBlur(img1, (5, 5), 1.0)
    
    # Create transformed image
    h, w = img1.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 0.0, 1.0)
    M[0, 2] += 5.0
    M[1, 2] += 5.0
    img2 = cv2.warpAffine(img1, M, (w, h))
    
    current_pixels = img1.shape[0] * img1.shape[1]
    
    print(f"\nOriginal Image Size: {img1.shape}")
    print(f"Original Pixels: {current_pixels:,}")
    print(f"Ground Truth: tx=5.0, ty=5.0\n")
    
    # Test different target pixel counts
    import time
    
    targets = [
        (-1, "No downsampling"),
        (400000, "400k pixels (~0.75 scale)"),
        (200000, "200k pixels (~0.5 scale)"),
        (100000, "100k pixels (~0.35 scale)"),
    ]
    
    print(f"{'Target':<25} {'Scale':<10} {'Size':<15} {'Time(ms)':<12} {'FPS':<8} {'Est_X':<10} {'Est_Y':<10} {'Error':<10}")
    print("-" * 105)
    
    for target_pix, desc in targets:
        # Run 5 times for timing
        times = []
        for _ in range(5):
            t0 = time.time()
            H, matched = find_rigid_movement_pyramid(img1, img2, target_pixels=target_pix)
            t1 = time.time()
            times.append(t1 - t0)
        
        avg_time = np.mean(times) * 1000
        fps = 1000.0 / avg_time
        
        if H is not None:
            est_tx = H[0, 2]
            est_ty = H[1, 2]
            err = np.sqrt((est_tx - 5.0)**2 + (est_ty - 5.0)**2)
        else:
            est_tx, est_ty, err = 0, 0, 999
        
        # Calculate actual scale
        if target_pix > 0 and current_pixels > target_pix:
            actual_scale = np.sqrt(target_pix / current_pixels)
            new_h = int(img1.shape[0] * actual_scale)
            new_w = int(img1.shape[1] * actual_scale)
            size_str = f"{new_w}x{new_h}"
            scale_str = f"{actual_scale:.2f}"
        else:
            size_str = f"{w}x{h}"
            scale_str = "1.00"
        
        print(f"{desc:<25} {scale_str:<10} {size_str:<15} {avg_time:<12.1f} {fps:<8.1f} {est_tx:<10.2f} {est_ty:<10.2f} {err:<10.2f}")

if __name__ == "__main__":
    test_downsampling()
