# Project Context: Video Mosaic Generation

## 1. Objective
Create a "rectified strip" video mosaic (panorama) from a video sequence. The output should stabilize the video to remove Y-translation and Rotation, leaving only "Pure X" horizontal progression, then stitch vertical strips from each frame.

## 2. Core Components

### `src/homography_evaluation.py`
- **Purpose**: Low-level motion estimation.
- **Key Methods**:
    - `find_rigid_movement`: Estimates rigid homography (rotation + translation) between pairwise frames.
    - Uses **Iterative Lucas-Kanade** optical flow with `scipy.ndimage.map_coordinates`.
    - Uses **RANSAC** for robust estimation.
    - **Coordinate System**: Returns matrices in standard $(x, y)$ convention.

### `src/mosaic.py`
- **Purpose**: High-level mosaic logic.
- **Key Algorithms**:
    - **Accumulation**: Computes cumulative transforms $H_{0 \to i}$ by chaining pairwise results ($H_{0 \to i} = H_{i-1 \to i} \times H_{0 \to i-1}$).
    - **Anchor Selection**: Selects the frame with the **median cumulative theta** as the reference (Anchor) to minimize global rotation drift.
    - **Stabilization**:
        - Recalculates all transforms relative to the Anchor ($H_{anchor \to i}$).
        - Computes "Stabilization Transform" ($H_{stabilized \to i}$) which effectively zeroes out Y and Theta relative to the anchor, keeping only X.
    - **Mosaic Building (`build_mosaic`)**:
        - **Logic**: Iterates frames, calculating the integer `dx` (horizontal shift) for each step.
        - **Placement**: Pastes a vertical strip of width `dx` onto the canvas.
        - **Canvas Sizing**: Calculated dynamically based on the Global X range (`max_x - min_x`).
        - **Safety**: Calculates a `start_offset` to handle negative X coordinates (dragging the mosaic to the left of the anchor) without crashing indices.
    - **Technical Details**:
        - Uses `scipy.ndimage.affine_transform`.
        - **Must swap coordinates**: Matrices are $(x, y)$, but `affine_transform` expects $(row, col)$ / $(y, x)$. This swap is implemented in `apply_stabilization`.
        - **RGB Support**: Iterates over channels (R, G, B) to apply the 2D warp to 3D image data.

### `src/video_io.py`
- **Purpose**: Testing and Utils.
- **Features**:
    - `generate_synthetic_video`: Creates a moving square for validation.
    - `load/save_video`.

## 3. Current Status
- **Implemented**: Full pipeline (Motion Est -> Stabilization -> Stitching).
- **Verified**:
    - Matrix multiplication order (fixed).
    - Anchor selection (implemented).
    - Coordinate swapping for Scipy (implemented).
    - Canvas sizing and negative index safety (implemented).
- **In Progress**: Synthetic verification test (`uv run ...`) to confirm pixel-perfect output.

## 4. User Preferences & Rules (CRITICAL)
1.  **Work Process**: Follow this cycle: **Issue -> Discussion -> Testing Suggestion -> Test & Discuss Result -> Further Testing -> Code Change Suggestion -> Code Change**.
2.  **Code Preservation**: **DO NOT rewrite entire files.** The user prefers targeted, minimal fixes to specific lines or blocks. Respect the existing structure (e.g., specific loop implementations) unless functionally broken.
2.  **Algorithm First**: Explain the *logic* of a fix before changing code.
3.  **No "Optimization" Rewrites**: Do not refactor for "cleanliness" if it deletes user-written comments or changes their recognized variable names without good reason.
4.  **Tools**: Use `uv run` for executing python scripts.
5.  **Performance**: **Use OpenCV** for core algorithms (optical flow, warping) to ensure high performance.
6.  **3-Channel Handling**: Convert to Grayscale for motion estimation (CV2 or simple mean), but warp the original RGB frames for the output.
