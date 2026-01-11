# Main Function Requirements

## Overview
The main.py file should support two distinct operational modes:

---

## Mode 1: Batch Mosaic Generation with Retry Logic

### Description
Process all video files in the `Exercise Inputs` folder and generate mosaic panoramas from each one.

### Behavior
1. **Input**: Iterate through all video files in the `Exercise Inputs` directory
2. **Processing**: For each video file:
   - Attempt to generate a mosaic using the standard pipeline
   - Save the output mosaic to the `output_mosaics` folder
   
3. **Failure Handling** (Retry with Blur):
   - If mosaic generation fails for any video:
     - Apply Gaussian blur to the video frames
     - Retry mosaic generation
     - Repeat this process up to **3 times** with increasing blur levels
     - Log each retry attempt
   - After 3 failed attempts, skip the video and move to the next one
   
4. **Output**: 
   - Save successful mosaics with meaningful filenames (e.g., `Garden_mosaic.jpg`)
   - Log all successes and failures to a summary file

### Pseudocode
```
for each video_file in Exercise Inputs:
    success = False
    blur_level = 0
    attempts = 0
    
    while not success and attempts < 3:
        try:
            if blur_level > 0:
                frames = apply_blur(video_frames, blur_level)
            mosaic = generate_mosaic(frames)
            save_mosaic(mosaic)
            success = True
        except Exception as e:
            attempts += 1
            blur_level += 1
            log_failure(video_file, attempts, e)
    
    if not success:
        log_complete_failure(video_file)
```

---

## Mode 2: Progressive Mosaic Video Generation (Column Scanning)

### Description
Create a 2-second animated video showing different perspective views of the same panorama by varying which column is extracted from each frame during stitching.

### Key Concept
**All mosaics use ALL frames from the video.** The only difference between mosaics is the `column_to_build` parameter - which column position in each frame is used for stitching. This creates different "viewpoints" or "perspectives" of the same panoramic scene.

### Parameters
- **Division Factor**: Default = 10, or accept from CLI argument
- **Output Duration**: 2 seconds
- **Frame Rate**: 30 fps (60 frames total for 2 seconds)

### Behavior

1. **Input**: Select a video file (from CLI or default)

2. **Mosaic Generation** (Different Column Perspectives):
   - Get the original video frame width `W`
   - Calculate step size: `step = W / division_factor`
   - Generate `division_factor` mosaics, each using **ALL frames** but different column positions:
     - Mosaic 1: Build from column position `1 * step` (e.g., column 128 if W=1280, divisions=10)
     - Mosaic 2: Build from column position `2 * step` (e.g., column 256)
     - Mosaic 3: Build from column position `3 * step` (e.g., column 384)
     - ...
     - Mosaic 9: Build from column position `9 * step` (e.g., column 1152)
     - Mosaic 10: Build from column position `10 * step - 1` (e.g., column 1279, to stay within bounds)
   
   **Important**: Each mosaic processes the same set of frames; only the extraction column varies. This changes the perspective/viewpoint of the resulting panorama.
   
3. **Video Creation**:
   - Create animation sequence: `1 → 2 → 3 → ... → 9 → 10 → 9 → ... → 3 → 2 → 1`
   - This creates a "scanning" effect through different perspectives
   - Total unique mosaics: `2 * division_factor - 1` (e.g., 19 for divisions=10)
   - Distribute frames evenly across 60 frames (2 seconds @ 30 fps)
   - Each mosaic appears approximately 3 times in the sequence
   
4. **Output**:
   - Save as `[video_name]_progressive.avi`
   - Creates a smooth "perspective scanning" effect

### Pseudocode
```
# Parse CLI arguments
division_factor = parse_cli_argument() or 10
video_file = select_video_file()

# Load all frames once
frames = load_video(video_file)
width = frames.shape[2]
step = width / division_factor
mosaics = []

# Generate mosaics from different column perspectives
for i in range(1, division_factor + 1):
    column_position = int(i * step)
    if column_position >= width:
        column_position = width - 1  # Stay within bounds
    
    # Build mosaic using ALL frames, but extract from column_position
    mosaic = build_mosaic(frames, column_to_build=column_position)
    mosaics.append(mosaic)

# Create bidirectional sequence for scanning effect
forward_sequence = mosaics  # 1 to 10
backward_sequence = mosaics[-2:0:-1]  # 9 to 1 (exclude endpoints to avoid duplicates)
full_sequence = forward_sequence + backward_sequence

# Generate 2-second video at 30fps (60 frames)
output_frames = interpolate_or_repeat(full_sequence, target_frames=60)
save_video("output_progressive.avi", output_frames, fps=30)
```

---

## CLI Interface

### Suggested Command Structure

```bash
# Mode 1: Batch processing
python main.py --mode batch

# Mode 2: Progressive video (default division=10)
python main.py --mode progressive --input Garden.mp4

# Mode 2: Progressive video (custom division)
python main.py --mode progressive --input House.mp4 --divisions 15
```

### Arguments
- `--mode`: `batch` or `progressive` (required)
- `--input`: Input video file path (required for progressive mode)
- `--divisions`: Number of divisions for progressive mode (default=10)
- `--output-dir`: Output directory (default=output_mosaics)

---

## Implementation Notes

1. **Error Handling**: Wrap all mosaic generation in try-except blocks
2. **Logging**: Use Python logging module to track progress and errors
3. **Blur Strategy**: Use `cv2.GaussianBlur()` with kernel sizes (5,5), (9,9), (13,13)
4. **Column Position**: For progressive mode, calculate column positions as `int(i * width / division_factor)` for i in 1..division_factor
5. **Video Writing**: Use `cv2.VideoWriter` or `save_video()` function for progressive video output
6. **Progress Display**: Show progress bars using `tqdm` if available
7. **Perspective Effect**: The column scanning creates different viewpoints - left columns show more of the right side of scenes, right columns show more of the left side

---

## Testing Checklist

- [ ] Mode 1 processes all videos in Exercise Inputs
- [ ] Mode 1 retry logic triggers on failure
- [ ] Mode 1 blur increases with each retry
- [ ] Mode 1 logs all attempts and failures
- [ ] Mode 2 generates correct number of intermediate mosaics
- [ ] Mode 2 creates smooth 2-second video
- [ ] Mode 2 accepts custom division factor from CLI
- [ ] CLI arguments parse correctly
- [ ] Both modes handle edge cases (empty folder, corrupted video, etc.)
