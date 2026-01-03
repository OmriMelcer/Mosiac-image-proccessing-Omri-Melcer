# Motion Estimation Debugging Session Summary

## Current Status: Debugging Lucas-Kanade Optical Flow Issues

**Date:** December 30, 2025  
**Repository:** Mosiac-image-proccessing-Omri-Melcer  
**Branch:** main

---

## Problem Statement

We are testing our mosaic/stabilization pipeline on real video inputs (Garden.mp4, Trees.mp4, etc.) and discovered that our motion estimation produces **different results than OpenCV**, despite using similar algorithms.

### Key Symptoms:
- **Performance:** Our method: 4.2 FPS vs OpenCV: 14.7 FPS
- **Transform Discrepancies on Garden.mp4 Frame 0→1:**
  - Our method: TX=-3.59, TY=-1.46, Theta=0.122°
  - OpenCV: TX=-3.67, TY=-0.20, Theta=0.033°
  - **TY differs by >1px**, Theta differs significantly

---

## Diagnostic Process & Findings

### 1. Initial Testing (`test_mosaic_suite.py`)
- Compared our pyramid LK method vs OpenCV's calcOpticalFlowPyrLK
- Both use our RANSAC implementation for consistency
- Results showed consistent differences in estimated transforms

### 2. Synthetic Data Testing (`test_diagnostic.py`)
**Key Finding:** Our method **fails badly on rotation**
- Pure translation (5px): Error 0.13px ✅
- **Translation + 0.5° rotation:** Error 3.48px ❌❌❌
  - GT: TX=5.0, TY=2.0 → Estimated: TX=3.22, TY=4.82
- This suggests rotation estimation is fundamentally broken

### 3. Point Correspondence Visualization (`test_visualize_correspondences.py`)
**Critical Discovery:**
- ALL Harris points (both our method and OpenCV) are concentrated in **top 1/3 of image**
- Garden.mp4 has sky/trees in upper portion with good texture
- **Average optical flow is nearly identical:**
  - Our method: dx=-4.55, dy=0.32
  - OpenCV: dx=-4.51, dy=0.30
- But number of inliers differs: 38 vs 50

### 4. Component Isolation (`test_lk_debug.py`)
**Test Design:**
```
Test 1: OpenCV points → Our LK → Our RANSAC
Test 2: OpenCV points → OpenCV LK → Our RANSAC  
Test 3: Our Harris points → Our LK → Our RANSAC
```

**Initial Results (Frame 0→1):**
- Test 1 (CV points + Our LK): Successfully tracked points
- Test 2 (CV points + CV LK): Reference baseline
- Test 3 (Our points + Our LK): Current production pipeline

---

## Current Hypothesis: Harris Points Are the Root Cause

### Evidence:
1. ✅ **RANSAC is identical** across all tests → not the issue
2. ✅ **Average optical flow matches** → raw LK computation works
3. ❌ **Our Harris points cause tracking failures** → 12 points fail (38 inliers vs 50)

### Suspected Mechanism:
- Our Harris corner detector may be selecting **low-quality points**
- These points are harder to track reliably
- Failed tracks → fewer inliers → worse RANSAC conditioning
- Poor spatial distribution (all in top 1/3) makes rotation estimation unstable

---

## Code Structure Overview

### Main Components:
```
src/
├── homography_evaluation.py
│   ├── compute_gradients()          # Sobel via scipy.convolve2d
│   ├── harris_response()            # Harris corner response map
│   ├── get_harris_points()          # NMS + thresholding + top-50
│   ├── optical_flow_iterative()     # Newton-Raphson LK
│   ├── optical_flow_pyramid()       # Coarse-to-fine pyramid LK
│   ├── track_features_pyramid()     # Batch tracking wrapper
│   ├── ransac_rigid_movement()      # RANSAC for rigid transform
│   ├── find_rigid_movement()        # Full pipeline (iterative LK)
│   └── find_rigid_movement_pyramid() # Full pipeline (pyramid LK)
├── mosaic.py                        # Stabilization & mosaic building
└── video_io.py                      # Video loading/saving

tests/
├── test_mosaic_suite.py             # Main comparison test
├── test_diagnostic.py               # Synthetic ground truth tests
├── test_visualize_correspondences.py # Visual point correspondence analysis
└── test_lk_debug.py                 # Component isolation tests
```

### Recent Features Added:
- **Downsampling support:** Both `find_rigid_movement` functions now accept `target_pixels` parameter
  - Automatically downsamples images for performance
  - Scales transforms back to original coordinates
  - Example: `H, matched = find_rigid_movement_pyramid(img1, img2, target_pixels=200000)`

---

## Performance Profiling Results

### Harris Detection Bottleneck:
- **Full resolution (1024×765):** 112ms total
  - Gradients (Sobel): 37ms (33%)
  - 3× Box filters: 51ms (46%)
  - NMS: 11ms (10%)

### Optimization Options Identified (NOT YET IMPLEMENTED):
1. **OpenCV replacements** (rejected - user wants no-OpenCV solution):
   - cv2.Sobel: 12x faster
   - cv2.boxFilter: 24x faster
   - Potential: 3.7x overall speedup

2. **Downsampling** (tested, works well):
   - Scale 0.5: 9.6 FPS, 0.36px error ✅
   - Scale 0.35: 11.4 FPS, 0.85px error ✅

---

## Next Steps (WHERE WE LEFT OFF)

### Immediate Action Required:
**Run `test_lk_debug.py` on multiple frame pairs** to confirm Harris point quality hypothesis:

```bash
cd /Users/omrimelcer/Documents/university/2026_fall/image\ proccesing/ex_4_imageproc
uv run python tests/test_lk_debug.py
```

### Modify test to check frames: 0→1, 5→6, 10→11, 15→16, 20→21

**Expected outcome:**
- If Harris points consistently cause more tracking failures → Harris is the problem
- If pattern is inconsistent → LK implementation may have bugs

### Potential Fixes (Once Root Cause Confirmed):

#### If Harris Points Are Bad:
1. **Lower threshold:** Currently 0.01, try 0.005 or 0.003 to get more/better points
2. **Spatial tiling:** Divide image into grid, take top N points per cell
3. **Multi-scale Harris:** Run on pyramid level 0 or 1 (coarser resolution)
4. **Better NMS:** Increase NMS window from 3×3 to 5×5 or 7×7

#### If LK Has Bugs:
1. Check coordinate scaling between pyramid levels
2. Verify initial_guess is applied correctly
3. Check boundary conditions and window clipping
4. Validate gradient computation at each level

---

## Test Files Status

### Working Tests:
- ✅ `test_mosaic_suite.py` - Comparison framework functional
- ✅ `test_diagnostic.py` - Synthetic tests reveal rotation issue
- ✅ `test_visualize_correspondences.py` - Creates useful visualizations
- ✅ `test_downsampling_feature.py` - Validates downsampling works
- ✅ `test_harris_profiling.py` - Performance breakdown
- ✅ `test_accuracy_vs_scale.py` - Accuracy/performance tradeoffs

### In Progress:
- 🔄 `test_lk_debug.py` - Component isolation, needs multi-frame testing

### Not Yet Created:
- ❌ `test_stabilization.py` - Visualize before/after stabilization
- ❌ Full mosaic generation test
- ❌ Incremental tracking test (detect once, track for N frames)

---

## Important Configuration & Constraints

### User Preferences:
- **No OpenCV dependencies** for core algorithms (only for I/O and visualization)
- Prefer algorithmic improvements over library optimizations
- Want comprehensive testing before touching production code (mosaic.py)

### Current Parameters:
- Harris: threshold=0.01, window_size=3, top 50 points
- LK: window_size=15, k_iters=3
- Pyramid: 3 levels
- RANSAC: 1000 iterations, 1.0px threshold

---

## Key Files to Review

### For Harris Issues:
- `src/homography_evaluation.py` lines 12-60 (harris_response, get_harris_points)

### For LK Issues:
- `src/homography_evaluation.py` lines 66-230 (optical_flow functions)

### For RANSAC Issues:
- `src/homography_evaluation.py` lines 231-306 (ransac_rigid_movement)

---

## Questions to Answer

1. **Why do our Harris points fail to track reliably?**
   - Are they near edges/boundaries?
   - Are response values actually high quality?
   - Is NMS too aggressive or not aggressive enough?

2. **Why does rotation estimation fail so badly?**
   - Is it due to poor spatial distribution of points?
   - Is RANSAC failing to find good rigid model?
   - Is the rotation parameterization incorrect?

3. **Can we improve spatial distribution without changing algorithms?**
   - Grid-based sampling?
   - Stratified random sampling by image region?

---

## Command Reference

### Run Tests:
```bash
# Full comparison test
uv run python tests/test_mosaic_suite.py

# Diagnostic tests
uv run python tests/test_diagnostic.py

# Visualize correspondences
uv run python tests/test_visualize_correspondences.py

# LK component debugging
uv run python tests/test_lk_debug.py

# Profiling
uv run python tests/test_harris_profiling.py
```

### Output Location:
All test outputs saved to: `test_outputs/`

---

## Summary for Next Agent

**You are debugging why our Lucas-Kanade optical flow produces different results than OpenCV's implementation.**

**Current suspicion:** Our Harris corner detector selects poor-quality points that fail to track reliably, leading to fewer inliers and worse transform estimates.

**Immediate task:** Modify and run `test_lk_debug.py` on multiple frame pairs (0→1, 5→6, 10→11, 15→16, 20→21) to confirm whether Harris point quality is consistently the issue.

**After confirmation:** Implement fixes to Harris point selection (spatial tiling, lower threshold, multi-scale detection, or better NMS).

**Do NOT:** 
- Use OpenCV for core algorithms (only I/O and visualization)
- Modify `src/mosaic.py` until testing is complete
- Implement optimizations until correctness is established
