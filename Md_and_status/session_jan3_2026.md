# Session Summary: January 3, 2026
## Video Mosaic Generation - Implementation & Testing

---

## Overview
Continued development of video mosaic generation pipeline. Focus on integrating components, debugging stabilization, and implementing single-column stitching approach.

---

## Accomplishments Today

### 1. ✅ Pipeline Integration
**Task:** Connect video loading → mosaic building → output saving

**Implementation:**
- Updated `main.py` to process videos from `Exercise Inputs-20251225/` folder
- Added command-line arguments for video selection and output directory
- Integrated `build_mosaic()` to return both canvas and stabilized frames
- Successfully processed Garden.mp4 (10 frames) and Shinkansen.mp4 (231 frames)

### 2. ✅ Fixed Critical Bugs

**Bug 1: Image Format Conversion**
- **Problem:** `to_grey_scale()` returns [0,1] float, but OpenCV needs [0,255] uint8
- **Solution:** Fixed in `homography_evaluation.py` - multiply by 255 before converting to uint8
- **Impact:** Frames no longer completely black after stabilization

**Bug 2: Canvas Sizing Error**
- **Problem:** Rounding individual TX values accumulates error (44px vs 42.79px actual)
- **Root Cause:** Canvas width calculated from float range, but stitching uses rounded integers
- **Solution:** Created `sum_rounded_tx` array that accumulates rounded deltas
- **Result:** Canvas sized correctly (1324px instead of 1323px)

**Bug 3: Stitching Logic**
- **Problem:** Off-by-one errors in strip placement, missing final frame strip
- **Solution:** Fixed slice indices in left-to-right stitching branch
- **Result:** No more black columns or gaps in output

### 3. ✅ Stabilization Quality Testing

**Created:** `tests/test_stabilization_quality.py`

**Purpose:** Measure if stabilization actually removes rotation and Y-drift

**Method:**
1. Build mosaic (includes stabilization)
2. Run motion estimation on consecutive **stabilized** frames
3. Measure TX, TY, and theta between frames

**Results (Shinkansen.mp4, 50 frames):**
```
TX (Horizontal):  Mean=7.51px (expected - camera panning)
TY (Vertical):    Mean=0.25px (threshold: 2.0px) ✓ PASS
Theta (Rotation): Mean=0.11°  (threshold: 0.5°)  ✓ PASS
```

**Conclusion:** Stabilization is working correctly! Curving in output is NOT due to failed stabilization.

---

## Key Discoveries

### Issue: Visible Seams and Curvature in Mosaics
**Observation:** Shinkansen.mp4 shows curved "banana" shape and misalignments between stitches

**Initial Hypothesis:** Residual rotation errors accumulating
- **Test Result:** ❌ Ruled out - stabilization passes quality test

**Current Hypothesis:** Wide strip stitching causes internal misalignment
- When copying 5-11 pixel wide strips, there's scene motion within each strip
- This creates visible seams at stitch boundaries

**Proposed Solution:** Single-column stitching

---

## Implementation: Single-Column Stitching

### New Functions

**1. `get_stabilization_transform_one_pixel_shift()`**
```python
def get_stabilization_transform_one_pixel_shift(transforms, anchor):
    direction = np.sign(transforms[-1][0,2])
    for i in range(len(transforms)):
        tx = transforms[i][0,2]
        target_mat = np.eye(3)
        if abs(tx * direction) > 1:  # Only if moved > 1px
            target_mat[0,2] = -tx + direction * (anchor - i)
        else:
            target_mat[0,2] = -tx  # Keep original if sub-pixel
        transforms[i] = np.dot(transforms[i], target_mat)
    return transforms
```

**Strategy:**
- Position frames at integer pixel intervals from anchor: `(anchor - i) * direction`
- Direction determined by overall camera motion: `np.sign(transforms[-1][0,2])`
- Frames with sub-1px motion keep original position (ignored during stitching)

**2. `build_mosaic_one_column()` (In Progress)**
- Uses `get_stabilization_transform_one_pixel_shift` for positioning
- Extracts only single column from each frame at `column_to_build`
- Simplifies stitching logic - no wide strips

---

## Current Status

### ✅ Working Components
1. Video loading (`video_io.py`)
2. Motion estimation (OpenCV goodFeaturesToTrack + calcOpticalFlowPyrLK + our RANSAC)
3. Transform accumulation and anchor selection
4. Stabilization (TY and rotation removal) - **validated by tests**
5. Wide-strip mosaic building (`build_mosaic()`)
6. Output saving

### 🔧 In Progress
1. Single-column stitching implementation (`build_mosaic_one_column()`)
2. Sub-pixel motion handling strategy

### 📋 Open Questions
1. **Sub-pixel frames:** Currently skips frames with < 0.5px motion. Should we:
   - Option A: Accumulate fractional motion, extract column when ≥1px (most accurate)
   - Option B: Skip sub-pixel frames (simpler, loses temporal resolution)
   - Option C: Round to nearest integer from anchor (acceptable error)
   
2. **Direction calculation:** Using `(anchor - i)` vs `(i - anchor)` for frame positioning

---

## Test Videos Used

| Video | Frames | Resolution | Motion Type | Result |
|-------|--------|------------|-------------|--------|
| Garden.mp4 | 488 | 720×1280 | Right-to-left pan | ✓ Success (gaps fixed) |
| Shinkansen.mp4 | 231 | 428×240 | Left-to-right pan | ✓ Success (shows curvature) |

---

## Next Steps

1. **Complete single-column implementation:**
   - Finalize `build_mosaic_one_column()` stitching logic
   - Decide on sub-pixel motion handling (Option A recommended)
   - Test on Shinkansen.mp4 to see if seams disappear

2. **Test suite expansion:**
   - Add test for one-pixel positioning accuracy
   - Verify column extraction at correct positions

3. **Process all input videos:**
   - Garden.mp4
   - House.mp4
   - Iguazu.mp4
   - Kessaria.mp4
   - Shinkansen.mp4
   - boat.mp4

4. **Compare methods:**
   - Wide-strip vs single-column quality comparison
   - Performance benchmarking

---

## Technical Notes

### Coordinate System
- Transforms use (x, y) convention
- But `scipy.ndimage.affine_transform` expects (row, col) = (y, x)
- Swap implemented in `apply_stabilization()` with row/column swaps

### Transform Chain
1. **Pairwise:** Frame[i-1] → Frame[i]
2. **Accumulate:** Frame[0] → Frame[i]
3. **Re-anchor:** Frame[anchor] → Frame[i]
4. **Stabilize:** Zero out rotation & TY, position for stitching

### Canvas Sizing
- Must use **rounded** cumulative TX values
- Formula: `max(rounded_tx) - min(rounded_tx) + frame_width`
- Critical for avoiding index errors

---

## Files Modified Today

### Core Implementation
- `src/mosaic.py` - Added one-pixel shift function, fixed stitching bugs
- `src/homography_evaluation.py` - Fixed uint8 conversion (multiply by 255)
- `main.py` - Updated to return stabilized_frames from build_mosaic

### Testing
- `tests/test_stabilization_quality.py` - **NEW** - Validates stabilization success

### Status Documentation
- `Md_and_status/session_jan3_2026.md` - **NEW** - This file

---

## Performance Notes

From stabilization test (Shinkansen.mp4, 50 frames):
- Motion estimation: ~0.5 seconds per frame
- Stabilization (warping): ~0.1 seconds per frame
- Total: ~30 seconds for 50 frames

For 231-frame video: ~3-4 minutes processing time

---

## Conclusions

1. **Pipeline is functional** - Successfully processes videos end-to-end
2. **Stabilization works correctly** - Validated by automated testing
3. **Visual artifacts are real** - Not due to bugs, but algorithmic limitations of wide-strip stitching
4. **Single-column approach promising** - Should eliminate seam artifacts
5. **Sub-pixel handling needs decision** - Recommend Option A (accumulate fractional motion)

The project is in good shape with a clear path forward for improving output quality.
