# Conversation Summary: Lucas-Kanade Optimization & Benchmarking
**Date:** January 2, 2026

## 1. The Issue
The custom Lucas-Kanade (LK) optical flow implementation was exhibiting two major failure modes:
1.  **Static Scene Instability (`House.mp4`)**: In textureless or static regions, the tracker "hallucinated" large motions (e.g., 35px jumps), causing the mosaic to drift wildly even when the camera was still.
2.  **Large Motion Lag (`Garden.mp4`)**: Fast camera movements (~25px/frame) were being under-tracked (reporting ~10px), causing the mosaic to lose alignment.

## 2. The Investigation & Fixes

### A. Static Instability (Singular Matrices)
*   **Diagnosis**: In flat regions (sky, walls), the gradient matrix $A^T A$ is near-singular. The solver `np.linalg.solve` was producing garbage large values instead of failing gracefully.
*   **Fix**: Added a determinant check in `optical_flow_iterative`:
    ```python
    if np.linalg.det(A) < 1e-6:
        return 0.0, 0.0
    ```
    This forces the tracker to ignore points it cannot confidently track.

### B. Large Motion (Pyramid Depth)
*   **Diagnosis**: The default pyramid levels (3) were insufficient. At level 2 (coarsest), a 25px motion is still ~6px, which is too large for the linearization assumption of LK (which works best for <1px motion).
*   **Fix**: Increased `num_levels` from 3 to **5**.
    *   At level 4 (1/16 scale), 25px becomes ~1.5px, which is trackable.
*   **Tuning**: Increased `k_iters` (iterations per level) from 3 to **20** to ensure convergence.

### C. Noise Reduction
*   **Fix**: Added a **5x5 Gaussian Blur** to the input images before processing to smooth out sensor noise and improve gradient stability.

## 3. Benchmarking & Validation

We created a rigorous testing script `tests/test_full_video_comparison.py` to compare three approaches:

1.  **Method 1: Our Full Pipeline** (Our Harris + Our LK + Our RANSAC)
2.  **Method 2: OpenCV Full** (OpenCV `goodFeatures` + OpenCV `calcOpticalFlow` + Our RANSAC)
3.  **Method 3: Hybrid** (OpenCV `goodFeatures` + Our LK + Our RANSAC)

### Results (Garden.mp4)

| Method | Total X Drift | Total Rotation | Speed | Inliers |
| :--- | :--- | :--- | :--- | :--- |
| **Our Method** | **-448.28 px** | **3.92°** | 2.64 FPS | ~88 |
| **OpenCV** | **-465.07 px** | **4.15°** | 23.72 FPS | ~100 |
| **Hybrid** | **-485.60 px** | **3.18°** | 3.26 FPS | ~191 |

## 4. Key Insights

1.  **Our Pipeline is Valid**: Our method tracks within **~3%** of OpenCV's trajectory. The logic is sound.
2.  **Hybrid Failed**: Using OpenCV points with our tracker produced *worse* results (-485px drift).
    *   **Reason**: OpenCV selects "perfect corners" based on Sobel/Scharr gradients. Our LK uses simple central differences. Points optimized for one gradient method aren't necessarily optimal for the other.
    *   **Conclusion**: Our Harris detector is better suited for our LK tracker than OpenCV's detector is.
3.  **Performance**: The speed gap (2.6 FPS vs 24 FPS) is due to Python vs C++. The logic cannot be optimized much further in pure Python without moving to C/C++ extensions.

## 5. Next Steps
Proceed to generating the actual mosaics using the now-validated `find_rigid_movement` pipeline.

## 5. Guideline Update (Jan 2, 2026)
**We will use OpenCV for performance.**
- The restriction on using OpenCV for core algorithms is lifted.
- We should prioritize performance, using `cv2` functions where appropriate (e.g., `cv2.calcOpticalFlowPyrLK`, `cv2.warpAffine`).
