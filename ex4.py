import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from src.video_io import load_video, save_video
from src.mosaic import build_mosaic
import logging
from PIL import Image
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def remove_black_boundaries(frames: np.ndarray, threshold: float = 1.0):
    """
    Remove black boundary columns from raw input frames.
    A column is considered a boundary if its mean pixel value is below the threshold.
    
    Args:
        frames: Input frames (N, H, W, C)
        threshold: Mean pixel intensity threshold to consider a column as visible (default: 1.0)
    
    Returns:
        Cropped frames with black boundaries removed
    """
    # Calculate mean intensity per column across all frames
    # Shape: (N, H, W, C) -> mean over axes (0, 1, 3) -> shape (W,)
    column_means = np.mean(frames, axis=(0, 1, 3))
    
    # Find first and last columns with mean intensity above threshold
    visible_cols = np.where(column_means > threshold)[0]
    
    if len(visible_cols) == 0:
        logger.warning("  All columns are below threshold! Returning original frames.")
        return frames
    
    left_boundary = visible_cols[0]
    right_boundary = visible_cols[-1] + 1  # +1 for exclusive slicing
    
    logger.info(f"  Detected black boundaries: left={left_boundary}px, right={frames.shape[2] - right_boundary}px")
    logger.info(f"  Original size: {frames.shape[1]}x{frames.shape[2]} → Cropped size: {frames.shape[1]}x{right_boundary - left_boundary}")
    
    # Crop all frames uniformly
    cropped_frames = frames[:, :, left_boundary:right_boundary, :]
    
    # Verify all frames have same dimensions after cropping
    assert cropped_frames.shape[0] == frames.shape[0], "Frame count mismatch after cropping"
    assert cropped_frames.shape[1] == frames.shape[1], "Height mismatch after cropping"
    assert cropped_frames.shape[3] == frames.shape[3], "Channel count mismatch after cropping"
    assert all(cropped_frames[i].shape == cropped_frames[0].shape for i in range(len(cropped_frames))), \
        "Not all frames have same dimensions after cropping"
    
    return cropped_frames


def process_video_with_retry(input_path: str, output_dir: str, column_to_build: int = -1, max_retries: int = 3, second_half: bool = False, first_half: bool = False):
    """
    Process a single video file to create a mosaic (simplified - no retry logic).
    
    Args:
        input_path: Path to input video file
        output_dir: Directory to save output mosaic image
        column_to_build: Which column of each frame to use for stitching (-1 for center)
        max_retries: Maximum number of retry attempts with blur (UNUSED - kept for compatibility)
    
    Returns:
        mosaic: The generated mosaic image, or None if failed
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {os.path.basename(input_path)}")
    logger.info(f"{'='*60}")
    
    # Load video
    logger.info("Loading video...")
    frames = load_video(input_path)
    logger.info(f"  Loaded {len(frames)} frames, shape: {frames[0].shape}")
    
    # Remove black boundaries from raw input frames
    logger.info("\n  Checking for black boundaries in input frames...")
    # Check if video filename contains "precropped" to skip edge removal
    if "precropped" in os.path.basename(input_path).lower():
        logger.info("  Video is already pre-cropped, skipping edge removal")
        logger.info(f"  Frame shape: {frames[0].shape}")
    else:
        frames = remove_black_boundaries(frames)
        logger.info(f"  Cropped frames shape: {frames[0].shape}")
    
    # Optional: Use only part of frames
    if second_half:
        start_idx = len(frames) // 2
        frames = frames[start_idx:]
        logger.info(f"  Using second half only: frames {start_idx} to {start_idx + len(frames)} ({len(frames)} frames)")
    elif first_half:
        end_idx = len(frames) // 2
        frames = frames[:end_idx]
        logger.info(f"  Using first half only: frames 0 to {end_idx} ({len(frames)} frames)")
    
    video_name = Path(input_path).stem
    
    try:
        logger.info("\n  Building mosaic...")
        logger.info("  - Converting frames to grayscale for motion estimation")
        logger.info("  - Computing rigid transforms (OpenCV + RANSAC)")
        logger.info("  - Finding optimal anchor frame (median dy)")
        logger.info("  - Stabilizing frames (removing Y-drift and rotation)")
        logger.info("  - Stitching into panorama...")
        
        # Build mosaic from cropped frames
        mosaic, mosaic_f, stabilized_frames = build_mosaic(frames)
        logger.info(f"  Final mosaic shape: {mosaic.shape}")
        logger.info(f"  Final fractional mosaic shape: {mosaic_f.shape}")
        
        # Save output
        output_path = os.path.join(output_dir, f"{video_name}_mosaic.png")
        logger.info(f"  Saving mosaic to: {output_path}")
        
        # Convert RGB to BGR for OpenCV saving
        mosaic_bgr = cv2.cvtColor(mosaic.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, mosaic_bgr)
        
        # Save fractional output
        output_path_f = os.path.join(output_dir, f"{video_name}_mosaic_fractional.png")
        logger.info(f"  Saving fractional mosaic to: {output_path_f}")
        mosaic_bgr_f = cv2.cvtColor(mosaic_f.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path_f, mosaic_bgr_f)
        
        logger.info(f"  ✓ Successfully created mosaic for {video_name}")
        return mosaic, mosaic_f
        
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_progressive_mosaic_video(input_path: str, output_dir: str, num_divisions: int = 10):
    """
    Create a 2-second mosaic video by varying the column extraction position.
    Uses the optimized build_mosaic function that generates all mosaics in one pass.
    
    Args:
        input_path: Path to input video file
        output_dir: Directory to save output video
        num_divisions: Number of column positions to scan through (default 10)
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Creating Progressive Mosaic Video: {os.path.basename(input_path)}")
    logger.info(f"{'='*60}")
    
    # Load video
    logger.info("Loading video...")
    frames = load_video(input_path)
    logger.info(f"  Loaded {len(frames)} frames, shape: {frames[0].shape}")
    
    # Remove black boundaries from raw input frames (consistency with batch mode)
    logger.info("\n  Checking for black boundaries in input frames...")
    if "precropped" in os.path.basename(input_path).lower():
        logger.info("  Video is already pre-cropped, skipping edge removal")
    else:
        frames = remove_black_boundaries(frames)
        logger.info(f"  Cropped frames shape: {frames[0].shape}")
    
    video_name = Path(input_path).stem
    
    logger.info(f"\n  Generating {num_divisions} mosaics with different column perspectives...")
    logger.info(f"  This will process all frames once and create multiple mosaics efficiently")
    
    try:
        # Build all mosaics in one pass using the optimized function
        # Returns shape: (num_divisions, height, width, channels)
        # We use the fractional stitching (second return value) as requested
        _, mosaics, _ = build_mosaic(frames, amount_of_frames=num_divisions)
        logger.info(f"  ✓ Generated {len(mosaics)} mosaics (fractional), shape: {mosaics.shape}")
        
        # Create subdirectory for mosaic frames
        mosaic_frames_dir = os.path.join(output_dir, f"{video_name}_progressive_frames")
        os.makedirs(mosaic_frames_dir, exist_ok=True)
        
        # Save individual mosaic frames as images
        logger.info(f"\n  Saving individual mosaic frames to: {mosaic_frames_dir}")
        for i in range(len(mosaics)):
            mosaic_path = os.path.join(mosaic_frames_dir, f"{video_name}_mosaic_{i+1:02d}.png")
            mosaic_bgr = cv2.cvtColor(mosaics[i].astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(mosaic_path, mosaic_bgr)
        logger.info(f"  ✓ Saved {len(mosaics)} mosaic frames")
    except Exception as e:
        logger.error(f"  ✗ Failed to generate mosaics: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create video sequence: 1→2→...→n→(n-1)→...→2→1 (bidirectional scanning)
    logger.info(f"\n  Creating video sequence (forward and backward scan)...")
    # Convert to list for easier manipulation
    mosaics_list = [mosaics[i] for i in range(len(mosaics))]
    video_sequence = mosaics_list + mosaics_list[-2:0:-1]  # Forward + reverse (excluding endpoints)
    logger.info(f"  Sequence length: {len(video_sequence)} unique perspective views")
    
    # Target: 2 seconds at 30 fps = 60 frames
    fps = 30
    total_frames_needed = 60
    frames_per_mosaic = max(1, total_frames_needed // len(video_sequence))
    
    logger.info(f"  Video specs: {fps} FPS, {total_frames_needed} frames, ~{frames_per_mosaic} frames per mosaic")
    
    # Build frame sequence by repeating each mosaic
    video_frames = []
    for mosaic in video_sequence:
        for _ in range(frames_per_mosaic):
            video_frames.append(mosaic)
    
    # Ensure exactly 2 seconds (pad or trim if needed)
    while len(video_frames) < total_frames_needed:
        video_frames.append(video_frames[-1])
    video_frames = video_frames[:total_frames_needed]
    
    # Convert to numpy array
    video_frames = np.array(video_frames, dtype=np.uint8)
    logger.info(f"  Total video frames: {len(video_frames)}")
    logger.info(f"  Video array shape: {video_frames.shape}, dtype: {video_frames.dtype}")
    
    # Save video
    output_path = os.path.join(output_dir, f"{video_name}_progressive.mp4")
    logger.info(f"\n  Saving video to: {output_path}")
    save_video(video_frames, output_path, fps=fps)
    
    logger.info(f"  ✓ Successfully created progressive mosaic video for {video_name}")
    logger.info(f"    Duration: {len(video_frames)/fps:.2f} seconds")
    logger.info(f"    Resolution: {video_frames[0].shape[1]}x{video_frames[0].shape[0]}")
    logger.info(f"    Frames: {len(video_frames)}")
    
    return True
    logger.info(f"    Frames: {len(video_frames)}")
    
    return True

def generate_panorama(input_frames_path, n_out_frames):
    """
    Main entry point for ex4
    :param input_frames_path : path to a dir with input video frames.
        We will test your code with a dir that has K frames, each in the format
        "frame_i:05d.jpg" (e.g., frame_00000.jpg, frame_00001.jpg, frame_00002.jpg, ...).
    :param n_out_frames: number of generated panorama frames
    :return: A list of generated panorma frames (of size n_out_frames),
        each list item should be a PIL image of a generated panorama.
    """
    logger.info(f"Generating panorama from {input_frames_path} with {n_out_frames} output frames")
    
    # load frames
    frame_files = sorted(glob.glob(os.path.join(input_frames_path, "*.jpg")))
    if not frame_files:
        logger.warning(f"No .jpg files found in {input_frames_path}, trying .png")
        frame_files = sorted(glob.glob(os.path.join(input_frames_path, "*.png")))
        
    if not frame_files:
        raise ValueError(f"No image frames found in {input_frames_path}")
        
    frames_list = []
    for f in frame_files:
        img = cv2.imread(f)
        if img is None:
            continue
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames_list.append(img)
        
    if not frames_list:
        raise ValueError("Failed to load any frames")
        
    frames = np.array(frames_list)
    logger.info(f"Loaded {len(frames)} frames of shape {frames[0].shape}")
    
    # Remove black boundaries if necessary (using existing logic)
    # Note: We assume the test inputs might have black boundaries like the exercise inputs
    frames = remove_black_boundaries(frames)
    
    # Generate mosaics
    # build_mosaic returns: canvas_array, canvas_array_f, stabilized_frames
    # We prefer canvas_array_f (fractional) for better quality
    
    try:
        if n_out_frames == 1:
             # build_mosaic returns single array for amount=1
             _, mosaic_f, _ = build_mosaic(frames, amount_of_frames=1)
             numpy_mosaics = [mosaic_f]
        else:
             # build_mosaic returns array of arrays for amount>1 (N, H, W, C)
             _, mosaics_f, _ = build_mosaic(frames, amount_of_frames=n_out_frames)
             numpy_mosaics = [mosaics_f[i] for i in range(len(mosaics_f))]
             
        # Convert to PIL Images
        pil_images = []
        for mos in numpy_mosaics:
            # Ensure uint8
            if mos.dtype != np.uint8:
                mos = mos.astype(np.uint8)
            pil_images.append(Image.fromarray(mos))
            
        return pil_images
        
    except Exception as e:
        logger.error(f"Error generating panorama: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    """
    Video Mosaic Generator with two modes:
    1. Batch Mode: Process all videos to create panoramic mosaics with retry logic
    2. Progressive Mode: Create animated mosaic videos showing progressive building
    """
    parser = argparse.ArgumentParser(
        description='Video Mosaic Generator - Creates panoramic mosaics from video sequences',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: Batch process all videos (with retry on failure)
  python main.py --mode batch
  
  # Mode 1: Process specific video with retry
  python main.py --mode batch --video Garden.mp4
  
  # Mode 2: Create progressive mosaic video (default 10 divisions)
  python main.py --mode progressive --video Garden.mp4
  
  # Mode 2: Create progressive mosaic video with 15 divisions
  python main.py --mode progressive --video House.mp4 --divisions 15
        """
    )
    
    parser.add_argument('--mode', type=str, choices=['batch', 'progressive'], required=True,
                        help='Mode: batch (create mosaics with retry) or progressive (create animated video)')
    parser.add_argument('--input', type=str, default='Exercise Inputs-20251225',
                        help='Input directory containing video files (default: Exercise Inputs-20251225)')
    parser.add_argument('--output', type=str, default='output_mosaics',
                        help='Output directory for mosaic images/videos (default: output_mosaics)')
    parser.add_argument('--video', type=str, default=None,
                        help='Process a specific video file (e.g., Garden.mp4) - if not specified, processes all videos')
    parser.add_argument('--column', type=int, default=-1,
                        help='Column index to extract from each frame (-1 for center, batch mode only)')
    parser.add_argument('--divisions', type=int, default=20,
                        help='Number of progressive steps for progressive mode (default: 20)')
    parser.add_argument('--retries', type=int, default=3,
                        help='Maximum retry attempts with blur on failure for batch mode (default: 3)')
    parser.add_argument('--second-half', action='store_true',
                        help='Process only second half of video frames')
    parser.add_argument('--first-half', action='store_true',
                        help='Process only first half of video frames')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("VIDEO MOSAIC GENERATOR")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Input directory: {args.input}")
    logger.info(f"Output directory: {args.output}")
    if args.mode == 'batch':
        logger.info(f"Max retries: {args.retries}")
    else:
        logger.info(f"Progressive divisions: {args.divisions}")
    logger.info("="*60)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Get list of videos to process
    input_dir = args.input
    if args.video:
        # Process single video
        video_files = [args.video]
    else:
        # Process all videos in input directory
        video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
        video_files = [f for f in os.listdir(input_dir) 
                      if any(f.endswith(ext) for ext in video_extensions)]
        video_files.sort()
    
    if not video_files:
        logger.error(f"No video files found in {input_dir}")
        return
    
    logger.info(f"\nFound {len(video_files)} video(s) to process:")
    for i, vf in enumerate(video_files, 1):
        logger.info(f"  {i}. {vf}")
    
    # Process each video
    successful = 0
    failed = 0
    
    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        
        if args.mode == 'batch':
            result = process_video_with_retry(input_path, args.output, args.column, args.retries, args.second_half, args.first_half)
            if result is not None:
                successful += 1
            else:
                failed += 1
        else:  # progressive mode
            result = create_progressive_mosaic_video(input_path, args.output, args.divisions)
            if result:
                successful += 1
            else:
                failed += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Successful: {successful}/{len(video_files)}")
    logger.info(f"Failed: {failed}/{len(video_files)}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()
