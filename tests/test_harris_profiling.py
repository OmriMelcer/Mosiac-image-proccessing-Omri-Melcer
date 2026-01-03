import numpy as np
import cv2
import os
import sys
import time
from scipy.signal import convolve2d
from scipy import ndimage

# Ensure src can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def profile_harris_detailed(img):
    """
    Break down Harris corner detection into individual operations and time each.
    """
    timings = {}
    
    # Step 1: Gradients (Sobel)
    t0 = time.time()
    kernel_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = convolve2d(img, kernel_X, mode='same') / 8.0
    Iy = convolve2d(img, kernel_Y, mode='same') / 8.0
    t1 = time.time()
    timings['gradients'] = t1 - t0
    
    # Step 2: Product matrices (element-wise)
    t0 = time.time()
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy
    t1 = time.time()
    timings['products'] = t1 - t0
    
    # Step 3: Window summation (3 convolutions)
    t0 = time.time()
    kernel = np.ones((3, 3))
    Sxx = convolve2d(Ixx, kernel, mode='same', boundary='symm')
    t1 = time.time()
    timings['conv_Sxx'] = t1 - t0
    
    t0 = time.time()
    Syy = convolve2d(Iyy, kernel, mode='same', boundary='symm')
    t1 = time.time()
    timings['conv_Syy'] = t1 - t0
    
    t0 = time.time()
    Sxy = convolve2d(Ixy, kernel, mode='same', boundary='symm')
    t1 = time.time()
    timings['conv_Sxy'] = t1 - t0
    
    timings['conv_total'] = timings['conv_Sxx'] + timings['conv_Syy'] + timings['conv_Sxy']
    
    # Step 4: Harris response calculation
    t0 = time.time()
    detM = (Sxx * Syy) - (Sxy ** 2)
    traceM = Sxx + Syy
    responses = detM - 0.04 * (traceM ** 2)
    t1 = time.time()
    timings['response_calc'] = t1 - t0
    
    # Step 5: Non-Maximum Suppression
    t0 = time.time()
    local_max_mask = ndimage.maximum_filter(responses, size=3) == responses
    t1 = time.time()
    timings['nms'] = t1 - t0
    
    # Step 6: Thresholding
    t0 = time.time()
    thresholded_mask = responses > 0.01
    total_mask = local_max_mask & thresholded_mask
    t1 = time.time()
    timings['thresholding'] = t1 - t0
    
    # Step 7: Extract and sort
    t0 = time.time()
    coords = np.argwhere(total_mask)
    values = responses[total_mask]
    sorted_idx = np.argsort(values)[::-1]
    top_50_idx = sorted_idx[:50]
    points = coords[top_50_idx]
    t1 = time.time()
    timings['extract_sort'] = t1 - t0
    
    return timings, len(points)

def test_harris_bottlenecks():
    """
    Profile Harris at different image sizes to identify bottlenecks.
    """
    print("\n" + "="*80)
    print("HARRIS CORNER DETECTION: Detailed Profiling")
    print("="*80)
    
    if not os.path.exists("test1.jpg"):
        print("Skipping: test1.jpg not found")
        return
    
    img_orig = cv2.imread("test1.jpg")
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    scales = [1.0, 0.75, 0.5, 0.4, 0.3]
    
    print(f"\nOriginal Size: {img_orig.shape}")
    print(f"\n{'Scale':<8} {'Size':<12} {'Total':<10} {'Grad':<10} {'Conv3x':<10} {'NMS':<10} {'Extract':<10}")
    print("-" * 80)
    
    for scale in scales:
        if scale != 1.0:
            new_size = (int(img_orig.shape[1] * scale), int(img_orig.shape[0] * scale))
            img = cv2.resize(img_orig, new_size, interpolation=cv2.INTER_AREA)
        else:
            img = img_orig
        
        # Run 10 times and average
        all_timings = []
        for _ in range(10):
            timings, num_points = profile_harris_detailed(img)
            all_timings.append(timings)
        
        # Average
        avg_grad = np.mean([t['gradients'] for t in all_timings]) * 1000
        avg_conv = np.mean([t['conv_total'] for t in all_timings]) * 1000
        avg_nms = np.mean([t['nms'] for t in all_timings]) * 1000
        avg_extract = np.mean([t['extract_sort'] for t in all_timings]) * 1000
        avg_other = np.mean([t['products'] + t['response_calc'] + t['thresholding'] for t in all_timings]) * 1000
        
        avg_total = avg_grad + avg_conv + avg_nms + avg_extract + avg_other
        
        size_str = f"{img.shape[1]}x{img.shape[0]}"
        print(f"{scale:<8.2f} {size_str:<12} {avg_total:<10.1f} {avg_grad:<10.1f} {avg_conv:<10.1f} {avg_nms:<10.1f} {avg_extract:<10.1f}")
    
    # Detailed breakdown for full size
    print(f"\n" + "="*80)
    print("DETAILED BREAKDOWN (Full Size)")
    print("="*80)
    
    img = img_orig
    all_timings = []
    for _ in range(10):
        timings, _ = profile_harris_detailed(img)
        all_timings.append(timings)
    
    operations = [
        ('Gradient X+Y (Sobel)', 'gradients'),
        ('Products (Ixx, Ixy, Iyy)', 'products'),
        ('Convolution Sxx', 'conv_Sxx'),
        ('Convolution Syy', 'conv_Syy'),
        ('Convolution Sxy', 'conv_Sxy'),
        ('Response Calculation', 'response_calc'),
        ('Non-Maximum Suppression', 'nms'),
        ('Thresholding', 'thresholding'),
        ('Extract & Sort Top-50', 'extract_sort')
    ]
    
    print(f"\n{'Operation':<35} {'Time (ms)':<12} {'% of Total':<12}")
    print("-" * 80)
    
    total_time = sum([np.mean([t[op[1]] for t in all_timings]) * 1000 for op in operations])
    
    for name, key in operations:
        avg_time = np.mean([t[key] for t in all_timings]) * 1000
        pct = (avg_time / total_time) * 100
        print(f"{name:<35} {avg_time:<12.1f} {pct:<12.1f}%")
    
    print(f"\n{'TOTAL':<35} {total_time:<12.1f} {'100.0':<12}%")

def compare_opencv_sobel():
    """
    Compare scipy convolve2d vs OpenCV Sobel.
    """
    print("\n" + "="*80)
    print("COMPARISON: Scipy vs OpenCV Sobel")
    print("="*80)
    
    if not os.path.exists("test1.jpg"):
        print("Skipping: test1.jpg not found")
        return
    
    img = cv2.imread("test1.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    print(f"\nImage Size: {img.shape}")
    print(f"Running 100 iterations each...\n")
    
    # Scipy method
    t0 = time.time()
    for _ in range(100):
        kernel_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        kernel_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        Ix_scipy = convolve2d(img, kernel_X, mode='same') / 8.0
        Iy_scipy = convolve2d(img, kernel_Y, mode='same') / 8.0
    t1 = time.time()
    scipy_time = (t1 - t0) * 10  # ms per iteration
    
    # OpenCV method
    t0 = time.time()
    for _ in range(100):
        Ix_cv = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) / 8.0
        Iy_cv = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    t1 = time.time()
    cv_time = (t1 - t0) * 10  # ms per iteration
    
    print(f"Scipy convolve2d:  {scipy_time:.2f} ms")
    print(f"OpenCV Sobel:      {cv_time:.2f} ms")
    print(f"Speedup:           {scipy_time/cv_time:.2f}x")
    
    # Check accuracy
    kernel_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix_scipy = convolve2d(img, kernel_X, mode='same') / 8.0
    Iy_scipy = convolve2d(img, kernel_Y, mode='same') / 8.0
    Ix_cv = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    Iy_cv = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    
    diff_x = np.abs(Ix_scipy - Ix_cv).max()
    diff_y = np.abs(Iy_scipy - Iy_cv).max()
    print(f"\nMax difference X:  {diff_x:.6f}")
    print(f"Max difference Y:  {diff_y:.6f}")

def compare_boxfilter_methods():
    """
    Compare different methods for the 3x3 box filter convolutions.
    """
    print("\n" + "="*80)
    print("COMPARISON: 3x3 Box Filter Methods")
    print("="*80)
    
    if not os.path.exists("test1.jpg"):
        print("Skipping: test1.jpg not found")
        return
    
    img = cv2.imread("test1.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # Create test matrix
    Ixx = img * img
    
    print(f"\nImage Size: {img.shape}")
    print(f"Running 50 iterations each...\n")
    
    # Method 1: scipy convolve2d
    t0 = time.time()
    for _ in range(50):
        kernel = np.ones((3, 3))
        result1 = convolve2d(Ixx, kernel, mode='same', boundary='symm')
    t1 = time.time()
    scipy_time = (t1 - t0) * 20  # ms per iteration
    
    # Method 2: OpenCV boxFilter
    t0 = time.time()
    for _ in range(50):
        result2 = cv2.boxFilter(Ixx, -1, (3, 3), normalize=False, borderType=cv2.BORDER_REFLECT)
    t1 = time.time()
    cv_time = (t1 - t0) * 20  # ms per iteration
    
    # Method 3: OpenCV filter2D
    t0 = time.time()
    kernel = np.ones((3, 3))
    for _ in range(50):
        result3 = cv2.filter2D(Ixx, -1, kernel, borderType=cv2.BORDER_REFLECT)
    t1 = time.time()
    filter2d_time = (t1 - t0) * 20  # ms per iteration
    
    print(f"scipy convolve2d:  {scipy_time:.2f} ms")
    print(f"cv2.boxFilter:     {cv_time:.2f} ms")
    print(f"cv2.filter2D:      {filter2d_time:.2f} ms")
    print(f"\nSpeedup (boxFilter):  {scipy_time/cv_time:.2f}x")
    print(f"Speedup (filter2D):   {scipy_time/filter2d_time:.2f}x")

if __name__ == "__main__":
    test_harris_bottlenecks()
    compare_opencv_sobel()
    compare_boxfilter_methods()
