# Harris Corner Detector Code Review

## Suspicious Issues to Investigate

### 1. **CRITICAL: Coordinate System Convention**
```python
coords = np.argwhere(total_mask)  # Returns (row, col) = (y, x) ✅
```
- `np.argwhere` returns coordinates as **(row, col)** which is **(y, x)** ✓
- This matches our convention for LK tracking ✓
- **Status**: Likely correct

### 2. **POTENTIAL ISSUE: NMS Filter Size Mismatch**
```python
local_max_mask = ndimage.maximum_filter(harris_response, size=3) == harris_response
```
**OpenCV's behavior:**
- OpenCV's `goodFeaturesToTrack` uses `minDistance=3` which means points must be at least 3 pixels apart
- This is measured as Euclidean distance, NOT a 3x3 NMS window

**Our behavior:**
- We use a 3x3 NMS window from `maximum_filter`
- This only ensures points are local maxima within 1.5 pixels
- **Two corners 2 pixels apart diagonally would BOTH pass our NMS**
- OpenCV would reject one of them (distance = 2 < 3)

**Impact:** 
- We might select clustered, nearby points
- These nearby points may have similar appearance and confuse LK tracking
- **SUSPICION LEVEL: HIGH** 🔴

### 3. **Threshold Normalization Issue**
```python
thresholded_mask = harris_response > threshold  # threshold=0.01
```

**OpenCV's behavior:**
- Uses `qualityLevel=0.01` 
- This means: keep points with response > 0.01 * max(harris_response)
- **It's a RELATIVE threshold**

**Our behavior:**
- We use absolute threshold of 0.01
- Not normalized by max response
- Depending on image brightness/contrast, this could select very different points

**Impact:**
- Dark images: might not find enough points
- Bright high-contrast images: might find too many weak corners
- **SUSPICION LEVEL: MEDIUM** 🟡

### 4. **Gradient Computation Differences**
```python
kernel_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
kernel_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
Ix = convolve2d(img, kernel_X, mode='same') / 8.0
Iy = convolve2d(img, kernel_Y, mode='same') / 8.0
```

**Issue:** Kernel_Y appears to be TRANSPOSED
- For Y-gradients, we want vertical differences
- Standard Sobel Y kernel should be:
  ```
  [[ 1  2  1]
   [ 0  0  0]
   [-1 -2 -1]]
  ```
  This computes: bottom_row - top_row = positive when image gets brighter downward

**Our kernel:**
  ```
  [[ 1  2  1]
   [ 0  0  0]
   [-1 -2 -1]]
  ```
  This is correct! ✅

**Status**: Correct

### 5. **Box Filter for Structure Tensor**
```python
kernel = np.ones((window_size, window_size))  # window_size=3
Sxx = convolve2d(Ixx, kernel, mode='same', boundary='symm')
```

**Observation:**
- Using 3x3 box filter for structure tensor
- OpenCV typically uses Gaussian weighting or larger windows
- Small window = more noise sensitive

**Impact:**
- Harris response may be noisier
- Selected corners may be less stable
- **SUSPICION LEVEL: LOW-MEDIUM** 🟡

### 6. **No Sub-Pixel Refinement**
```python
coords = np.argwhere(total_mask)  # Integer coordinates only
```

**OpenCV's behavior:**
- Can return sub-pixel corner locations
- Uses corner refinement algorithms

**Our behavior:**
- Only integer pixel coordinates
- No refinement

**Impact:**
- Up to 0.5 pixel offset from true corner location
- Could cause LK to start from sub-optimal position
- **SUSPICION LEVEL: MEDIUM** 🟡

## Summary of Suspicions (Priority Order)

1. **🔴 HIGH: NMS Window Too Small**
   - Using 3x3 NMS vs OpenCV's 3-pixel Euclidean distance
   - May select clustered points that confuse tracking
   - **FIX:** Implement proper minimum distance filtering

2. **🟡 MEDIUM: Absolute vs Relative Threshold**
   - We use absolute 0.01, OpenCV uses 0.01 * max(response)
   - Image-dependent behavior
   - **FIX:** Normalize threshold by max response

3. **🟡 MEDIUM: No Sub-Pixel Refinement**
   - Integer coordinates only
   - Up to 0.5px positioning error
   - **FIX:** Add corner refinement step

4. **🟡 LOW-MEDIUM: Small Box Filter Window**
   - 3x3 window may be noisy
   - **FIX:** Try window_size=5 or use Gaussian weights

## Recommended Tests

1. **Test with 100 points instead of 50**
   - More points = better chance of good ones
   - If success rate improves, confirms point quality issue

2. **Visualize exact point locations**
   - Plot our Harris points vs OpenCV points on same image
   - Check for clustering vs good distribution

3. **Check response values**
   - Compare harris_response values at our points vs OpenCV points
   - Are we selecting weak corners?
