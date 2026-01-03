∑# Video Stabilization Project - Status Report
**Date:** January 1, 2026  
**Repository:** Mosiac-image-proccessing-Omri-Melcer  
**Branch:** main

---

## Executive Summary

We discovered and partially fixed Harris point quality issues, achieving 88% tracking success (up from 77.6%). However, full video testing revealed **catastrophic errors in transform estimation** that persist even with OpenCV's perfect point tracking. The problem is definitively in our rigid transform computation or accumulation logic, not in point detection/tracking.

---

## Problem Evolution

### Phase 1: Harris Point Quality Issues ✅ Partially Solved
**Initial Problem:** Our method achieved 4.2 FPS vs OpenCV's 14.7 FPS, with significant transform discrepancies.

**Root Cause Identified:** Harris corner detector selected poor-quality points
- Only 77.6% tracking success vs OpenCV's 99.2%
- Lost ~13 points per frame on average
- Points clustered in small regions (poor spatial distribution)

**Fixes Applied:**
1. ✅ Increased from 50 → 100 points detected
2. ✅ Changed to relative threshold (OpenCV-style: threshold × max_response)
3. ✅ Increased NMS window from 3×3 → 7×7 (then 9×9)

**Results:**
- Tracking success: **77.6% → 88.0%** (+10.4 percentage points)
- Gap vs OpenCV: **21.6% → 11.0%** (cut in half!)
- Average inliers: 39 → 84 per frame

**Remaining Gap:** Still 11% worse than OpenCV, but acceptable for pure Python implementation.

---

### Phase 2: Transform Computation Crisis 🔴 CRITICAL BUG FOUND

**Breakthrough Discovery:** Testing on House.mp4 revealed huge errors **even with OpenCV's perfect tracking**:

**Test Setup:**
- Method 1: Our Harris + Our LK + Our RANSAC
- Method 2: OpenCV points + OpenCV LK + Our RANSAC

**House.mp4 Results (435 frames):**
```
Method 1: TX=-1845px, TY=-300px, Theta=-4.7°
Method 2: TX=-2657px, TY=-187px, Theta=+4.0°

Differences:
  TX: 812px difference!
  TY: 113px difference
  Theta: 8.7° difference!
```

**Critical Insight:** Since Method 2 uses OpenCV's proven point detection and tracking, but still produces wrong results, **the bug MUST be in:**
1. `ransac_rigid_movement()` 
2. `compute_rigid_movement()` / `calculate_median_motion_from_inliers_SVD()`
3. Transform accumulation logic in test

---

## Implementation History

### Harris Detection Improvements
| Parameter | Original | Current | Impact |
|-----------|----------|---------|--------|
| max_corners | 50 | 100 | More candidates |
| threshold | Absolute 0.01 | 0.01 × max_response | Image-adaptive |
| NMS window | 3×3 | 9×9 | Better spacing (~5.7px min) |
| Tracking success | 77.6% | 88.0% | +10.4% |

### SVD Refinement Implementation
**Goal:** Use ALL inliers (not just 2 random points) to compute final transform

**Implementation:** `calculate_median_motion_from_inliers_SVD()`
```python
# Kabsch algorithm: optimal rotation via SVD
H_matrix = centered_p2.T @ centered_p1
U, S, Vt = np.linalg.svd(H_matrix)
R = U @ Vt
if np.linalg.det(R) < 0:
    Vt[-1, :] *= -1
    R = U @ Vt
t = centroid_p2 - R @ centroid_p1
```

**Expected:** More robust (using N points instead of 2)  
**Actual Result:** Made Garden.mp4 results WORSE! ❌

**Before SVD:**
- TX diff: Mean=42px, Final=-60px
- TY diff: Mean=13px, Final=-74px

**After SVD:**
- TX diff: Mean=156px, Final=-492px 😱
- TY diff: Mean=54px, Final=-199px 😱

**Conclusion:** SVD implementation is mathematically correct but reveals that the underlying transform computation has fundamental issues.

---

## Current Code Status

### Working Components ✅
- `compute_gradients()` - Sobel gradients
- `harris_response()` - Harris corner response
- `get_harris_points()` - NMS + thresholding with improvements
- `optical_flow_pyramid()` - Pyramid Lucas-Kanade tracking
- `track_features_pyramid()` - Batch feature tracking
- Point detection: 88% success rate (acceptable)
- LK tracking: 0.03px difference from OpenCV (excellent)

### Suspected Buggy Components 🔴
- `compute_rigid_movement()` - 2-point rigid transform (used in RANSAC iterations)
- `calculate_median_motion_from_inliers_SVD()` - SVD-based refinement (new)
- `ransac_rigid_movement()` - RANSAC wrapper
- Transform accumulation in `test_full_video_comparison.py`

### Key Files
- [src/homography_evaluation.py](src/homography_evaluation.py) - All motion estimation code
- [tests/test_full_video_comparison.py](tests/test_full_video_comparison.py) - Full video testing
- [tests/test_lk_debug.py](tests/test_lk_debug.py) - Component isolation tests

---

## Test Results Summary

### Garden.mp4 (488 frames, horizontal pan)
| Method | FPS | Avg Inliers | Final TX | Final TY | Final Theta |
|--------|-----|-------------|----------|----------|-------------|
| Our Method | 4.24 | 33.1 | -3409px | -840px | 19.1° |
| OpenCV Method | 14.58 | 95.0 | -2917px | -641px | 18.3° |
| **Difference** | 3.44x slower | -61.9 | **-492px** | **-199px** | **0.8°** |

### House.mp4 (435 frames, horizontal pan)
| Method | FPS | Avg Inliers | Final TX | Final TY | Final Theta |
|--------|-----|-------------|----------|----------|-------------|
| Our Method | 4.38 | 6.6 | -1845px | -300px | -4.7° |
| OpenCV Method | 14.16 | 87.3 | -2657px | -187px | +4.0° |
| **Difference** | 3.23x slower | -80.7 | **+812px** | **-113px** | **-8.7°** |

**Expected Results:** Horizontal camera pan should produce:
- Large TX (thousands of pixels) ✓
- Small TY (<100px typically) ❓
- Small Theta (<5° typically) ❓

**Actual:** Both methods fail these sanity checks!

---

## Next Steps (Priority Order)

### 1. 🔥 IMMEDIATE: Synthetic Transform Verification
Create ground-truth test to isolate the bug:
```python
# Generate synthetic data with known transform
# GT: TX=100, TY=50, Theta=5°
p1 = synthetic_points()
H_gt = build_rigid_transform(tx=100, ty=50, theta=5)
p2 = apply_homography(H_gt, p1)

# Test our functions
H_estimated = calculate_median_motion_from_inliers_SVD(p1, p2)

# Compare
error = compare_transforms(H_estimated, H_gt)
```

**Goal:** Determine if bug is in:
- Transform computation itself
- Coordinate system handling (y,x vs x,y)
- Normalization/denormalization
- Accumulation logic

### 2. Investigate Specific Hypotheses

**Hypothesis A: Coordinate System Bug**
- Our points are (y, x) format
- Homography expects (x, y) format
- We swap in multiple places - might have inconsistency

**Hypothesis B: Normalization Issue**
- `normalize_points()` scales and centers
- Denormalization: `H = inv(T2) @ H @ T1`
- Might be incorrect for rotation around non-zero centroids

**Hypothesis C: Accumulation Error**
- Test accumulates: `H_cumulative = H @ H_cumulative`
- Should it be: `H_cumulative = H_cumulative @ H`?
- Matrix multiplication order matters!

### 3. If All Else Fails
- Use OpenCV's `estimateAffinePartial2D()` as reference
- Compare our implementation step-by-step against OpenCV source
- Add extensive logging to trace coordinate transformations

---

## Technical Debts & Considerations

### Performance
- Our method: ~4.3 FPS
- OpenCV method: ~14.5 FPS
- **3.4x slower** - acceptable for Python implementation

### Code Quality
- ✅ Good separation of concerns
- ✅ Well-documented functions
- ✅ Comprehensive test suite
- ❌ Need more unit tests with ground truth
- ❌ Need better error handling in edge cases

### Algorithm Choices
- Harris + LK: Classic, well-understood ✅
- RANSAC: Standard approach ✅
- SVD refinement: Theoretically optimal, but revealing bugs ⚠️

---

## Key Learnings

1. **Point quality matters less than expected** - Even with only 88% success, we should get reasonable results if transform estimation is correct

2. **Component isolation is crucial** - Testing with OpenCV points + OpenCV LK revealed the bug isn't in tracking

3. **Synthetic data is essential** - Can't debug complex geometric transforms on real data without ground truth

4. **SVD is not a magic bullet** - Statistically optimal doesn't mean bug-free; revealed deeper issues

5. **Transform accumulation is sensitive** - Small per-frame errors compound exponentially over 400+ frames

---

## Open Questions

1. **Why does SVD make results worse?** 
   - Is there a sign error?
   - Is translation computation wrong?
   - Is denormalization incorrect?

2. **Why do both methods give unrealistic theta values?**
   - House.mp4: Expected <2°, got -4.7° and +4.0°
   - Suggests systematic bias in rotation estimation

3. **Is RANSAC selecting good inliers?**
   - High inlier counts (95 for OpenCV) suggest yes
   - But maybe threshold is too loose?

4. **Are we accumulating transforms correctly?**
   - Order: `H_cumulative = H @ H_cumulative` or reverse?
   - Frame of reference: global vs local transforms?

---

## Files Modified This Session
- [src/homography_evaluation.py](src/homography_evaluation.py#L35-L72) - `get_harris_points()` improvements
- [src/homography_evaluation.py](src/homography_evaluation.py#L454-L480) - Added `calculate_median_motion_from_inliers_SVD()`
- [tests/test_lk_debug.py](tests/test_lk_debug.py) - Enhanced multi-frame testing
- [tests/test_full_video_comparison.py](tests/test_full_video_comparison.py) - Full video comparison framework
- [test_outputs/harris_points_analysis.txt](test_outputs/harris_points_analysis.txt) - Detailed analysis
- [test_outputs/harris_improvements_comparison.txt](test_outputs/harris_improvements_comparison.txt) - Before/after comparison
- [test_outputs/Garden_comparison.txt](test_outputs/Garden_comparison.txt) - Garden.mp4 results
- [test_outputs/House_comparison.txt](test_outputs/House_comparison.txt) - House.mp4 results

---

## Contact Points for Debugging

**When synthetic tests are ready, focus on:**
1. Single transform with known GT (TX=100, TY=50, Theta=5°)
2. Verify `compute_rigid_movement()` with 2 points
3. Verify `calculate_median_motion_from_inliers_SVD()` with N points
4. Test accumulation of 10 small transforms
5. Check if errors compound linearly or exponentially

**Expected outcome:** Synthetic tests will pinpoint the exact line where the bug occurs.

---

## Agent Notes

**User Preferences:**
- Prefers targeted fixes over full rewrites
- Wants algorithmic understanding before code changes
- No OpenCV for core algorithms (only I/O)
- Appreciates thorough explanations

**Code Review Philosophy:**
- Always check existing code before proposing changes
- Explain the math/logic first
- Verify assumptions with tests before implementing

**Current Blocker:** Need synthetic ground truth tests to isolate transform computation bug before proceeding further.
