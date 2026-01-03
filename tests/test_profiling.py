import numpy as np
import cv2
import os
import sys
import time

# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.homography_evaluation import (
    get_harris_points, 
    harris_response,
    track_features, 
    track_features_pyramid,
    ransac_rigid_movement,
    build_pyramid,
    compute_gradients
)

def profile_find_rigid_movement(img1, img2, use_pyramid=False, harris_threshold=0.01):
    """
    Manually runs the pipeline with timing for each step.
    Returns dict with timing breakdown and number of points.
    """
    timings = {}
    
    # Step 1: Compute Gradients
    t0 = time.time()
    Im1_Ix, Im1_Iy = compute_gradients(img1)
    t1 = time.time()
    timings['gradients'] = t1 - t0
    
    # Step 2: Harris Response
    t0 = time.time()
    harris_resp = harris_response(img1, Ix=Im1_Ix, Iy=Im1_Iy)
    t1 = time.time()
    timings['harris_response'] = t1 - t0
    
    # Step 3: Get Harris Points
    t0 = time.time()
    points = get_harris_points(harris_resp, threshold=harris_threshold)
    t1 = time.time()
    timings['harris_extraction'] = t1 - t0
    timings['num_points'] = len(points)
    timings['num_points'] = len(points)
    
    if len(points) < 2:
        return timings
    
    # Step 4: Feature Tracking
    t0 = time.time()
    if use_pyramid:
        # Build pyramids
        pyr1 = build_pyramid(img1, num_levels=3)
        pyr2 = build_pyramid(img2, num_levels=3)
        
        # Compute gradients for each level
        IX_pyr = []
        IY_pyr = []
        for level in range(len(pyr1)):
            IX, IY = compute_gradients(pyr1[level])
            IX_pyr.append(IX)
            IY_pyr.append(IY)
        
        t_pyramid = time.time()
        timings['pyramid_build'] = t_pyramid - t0
        
        # Track features
        p1, p2 = track_features_pyramid(pyr1, pyr2, IX_pyr, IY_pyr, points)
        t1 = time.time()
        timings['feature_tracking'] = t1 - t_pyramid
    else:
        p1, p2 = track_features(img1, img2, points, Ix=Im1_Ix, Iy=Im1_Iy)
        t1 = time.time()
        timings['feature_tracking'] = t1 - t0
    
    # Step 5: Check valid points
    if len(p1) < 3:
        return timings
    
    # Step 6: RANSAC
    t0 = time.time()
    H, inliers = ransac_rigid_movement(p1, p2)
    t1 = time.time()
    timings['ransac'] = t1 - t0
    
    return timings

def benchmark_harris_point_counts():
    """
    Tests different Harris thresholds to control number of points.
    """
    print("\n" + "="*70)
    print("PROFILING: Harris Point Count Impact")
    print("="*70)
    
    # Load real image
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
    M[0, 2] += 2.0
    M[1, 2] += 2.0
    img2 = cv2.warpAffine(img1, M, (w, h))
    
    print(f"\nImage Size: {img1.shape}")
    print(f"Testing both Iterative and Pyramid LK\n")
    
    # Test different thresholds (higher threshold = fewer points)
    thresholds = [0.05, 0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003]
    
    for method_name, use_pyramid in [("Iterative LK", False), ("Pyramid LK", True)]:
        print(f"\n--- {method_name} ---")
        print(f"{'Thresh':<8} {'Points':<8} {'Total(ms)':<12} {'Harris(ms)':<12} {'Track(ms)':<12} {'RANSAC(ms)':<12}")
        print("-" * 75)
        
        for thresh in thresholds:
            # Run 5 times and average
            all_timings = []
            for _ in range(5):
                timings = profile_find_rigid_movement(img1, img2, use_pyramid, thresh)
                all_timings.append(timings)
            
            # Average timings
            avg_num_points = int(np.mean([t.get('num_points', 0) for t in all_timings]))
            avg_harris = (np.mean([t.get('harris_response', 0) + t.get('harris_extraction', 0) + t.get('gradients', 0) for t in all_timings]) * 1000)
            avg_track = np.mean([t.get('feature_tracking', 0) for t in all_timings]) * 1000
            avg_ransac = np.mean([t.get('ransac', 0) for t in all_timings]) * 1000
            avg_pyramid = np.mean([t.get('pyramid_build', 0) for t in all_timings]) * 1000
            
            avg_total = avg_harris + avg_track + avg_ransac + avg_pyramid
            
            print(f"{thresh:<8.3f} {avg_num_points:<8} {avg_total:<12.1f} {avg_harris:<12.1f} {avg_track:<12.1f} {avg_ransac:<12.1f}")

def benchmark_image_sizes():
    """
    Tests different downsampling scales.
    """
    print("\n" + "="*70)
    print("PROFILING: Image Downsampling Impact")
    print("="*70)
    
    # Load real image
    if not os.path.exists("test1.jpg"):
        print("Skipping: test1.jpg not found")
        return
    
    img_orig = cv2.imread("test1.jpg")
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    print(f"\nOriginal Image Size: {img_orig.shape}")
    print(f"Testing with threshold=0.01, Pyramid LK\n")
    
    scales = [1.0, 0.75, 0.5, 0.4, 0.3]
    
    print(f"{'Scale':<8} {'Size':<15} {'Points':<8} {'Total(ms)':<12} {'Harris(ms)':<12} {'Track(ms)':<12} {'FPS':<8}")
    print("-" * 80)
    
    for scale in scales:
        # Downsample
        new_size = (int(img_orig.shape[1] * scale), int(img_orig.shape[0] * scale))
        img1 = cv2.resize(img_orig, new_size, interpolation=cv2.INTER_AREA)
        img1 = cv2.GaussianBlur(img1, (5, 5), 1.0)
        
        # Create transformed image (scale translation proportionally)
        h, w = img1.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 0.0, 1.0)
        M[0, 2] += 2.0 * scale
        M[1, 2] += 2.0 * scale
        img2 = cv2.warpAffine(img1, M, (w, h))
        
        # Run 5 times and average
        all_timings = []
        for _ in range(5):
            timings = profile_find_rigid_movement(img1, img2, use_pyramid=True, harris_threshold=0.01)
            all_timings.append(timings)
        
        avg_num_points = int(np.mean([t.get('num_points', 0) for t in all_timings]))
        avg_harris = (np.mean([t.get('harris_response', 0) + t.get('harris_extraction', 0) + t.get('gradients', 0) for t in all_timings]) * 1000)
        avg_track = np.mean([t.get('feature_tracking', 0) for t in all_timings]) * 1000
        avg_ransac = np.mean([t.get('ransac', 0) for t in all_timings]) * 1000
        avg_pyramid = np.mean([t.get('pyramid_build', 0) for t in all_timings]) * 1000
        
        avg_total = avg_harris + avg_track + avg_ransac + avg_pyramid
        fps = 1000.0 / avg_total if avg_total > 0 else 0
        
        size_str = f"{img1.shape[1]}x{img1.shape[0]}"
        print(f"{scale:<8.2f} {size_str:<15} {avg_num_points:<8} {avg_total:<12.1f} {avg_harris:<12.1f} {avg_track:<12.1f} {fps:<8.1f}")

def detailed_pyramid_profiling():
    """
    Deep dive into pyramid building and tracking costs.
    """
    print("\n" + "="*70)
    print("PROFILING: Detailed Pyramid Breakdown")
    print("="*70)
    
    if not os.path.exists("test1.jpg"):
        print("Skipping: test1.jpg not found")
        return
    
    img1 = cv2.imread("test1.jpg")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img1 = cv2.GaussianBlur(img1, (5, 5), 1.0)
    
    h, w = img1.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 0.0, 1.0)
    M[0, 2] += 2.0
    M[1, 2] += 2.0
    img2 = cv2.warpAffine(img1, M, (w, h))
    
    # Get Harris points
    from src.homography_evaluation import harris_response
    Im1_Ix, Im1_Iy = compute_gradients(img1)
    harris_resp = harris_response(img1, Ix=Im1_Ix, Iy=Im1_Iy)
    points = get_harris_points(harris_resp, threshold=0.01)
    
    print(f"\nImage Size: {img1.shape}")
    print(f"Number of Harris Points: {len(points)}")
    
    # Build pyramids with timing
    t0 = time.time()
    pyr1 = build_pyramid(img1, num_levels=3)
    t1 = time.time()
    pyr2 = build_pyramid(img2, num_levels=3)
    t2 = time.time()
    
    print(f"\nPyramid Building:")
    print(f"  Image 1: {(t1-t0)*1000:.1f} ms")
    print(f"  Image 2: {(t2-t1)*1000:.1f} ms")
    print(f"  Total:   {(t2-t0)*1000:.1f} ms")
    
    # Compute gradients with timing
    t0 = time.time()
    IX_pyr = []
    IY_pyr = []
    for level in range(len(pyr1)):
        IX, IY = compute_gradients(pyr1[level])
        IX_pyr.append(IX)
        IY_pyr.append(IY)
    t1 = time.time()
    
    print(f"\nGradient Computation:")
    print(f"  All levels: {(t1-t0)*1000:.1f} ms")
    
    # Track features with timing
    t0 = time.time()
    p1, p2 = track_features_pyramid(pyr1, pyr2, IX_pyr, IY_pyr, points)
    t1 = time.time()
    
    print(f"\nFeature Tracking:")
    print(f"  {len(points)} points: {(t1-t0)*1000:.1f} ms")
    print(f"  Per point: {(t1-t0)*1000/len(points):.2f} ms")
    
    print(f"\nPyramid Levels:")
    for level, img in enumerate(pyr1):
        print(f"  Level {level}: {img.shape}")

if __name__ == "__main__":
    benchmark_harris_point_counts()
    benchmark_image_sizes()
    detailed_pyramid_profiling()
