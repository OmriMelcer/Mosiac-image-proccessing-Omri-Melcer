
import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from src.video_io import load_video, save_video
from src.mosaic import build_mosaic

def process_video(input_path: str, output_dir: str, column_to_build: int = -1):
    """
    Process a single video file to create a mosaic.
    
    Args:
        input_path: Path to input video file
        output_dir: Directory to save output mosaic image
        column_to_build: Which column of each frame to use for stitching (-1 for center)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(input_path)}")
    print(f"{'='*60}")
    
    # Load video
    print("Loading video...")
    frames = load_video(input_path)
    print(f"  Loaded {len(frames)} frames, shape: {frames[0].shape}")
    
    # Build mosaic
    print("Building mosaic...")
    print("  - Converting frames to grayscale for motion estimation")
    print("  - Applying Gaussian blur for stability")
    print("  - Computing rigid transforms (OpenCV + RANSAC)")
    print("  - Finding optimal anchor frame (median rotation)")
    print("  - Stabilizing frames (removing Y-drift and rotation)")
    print("  - Stitching into panorama...")
    
    mosaic, stabilized_frames = build_mosaic(frames, column_to_build=column_to_build)
    print(f"  Final mosaic shape: {mosaic.shape}")
    
    # Save output
    video_name = Path(input_path).stem
    output_path = os.path.join(output_dir, f"{video_name}_mosaic.png")
    print(f"Saving mosaic to: {output_path}")
    
    # Convert RGB to BGR for OpenCV saving
    mosaic_bgr = cv2.cvtColor(mosaic.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, mosaic_bgr)
    
    print(f"✓ Successfully created mosaic for {video_name}")
    return mosaic

def main():
    """
    Main pipeline:
    1. Load Video from Exercise Inputs folder
    2. Motion Estimation (OpenCV goodFeaturesToTrack + OpenCV LK + Our RANSAC)
    3. Stabilization (Median Reference Frame, Remove Y-drift and Rotation)
    4. Mosaic Construction (Stitch vertical strips)
    """
    parser = argparse.ArgumentParser(description='Video Mosaic Generator')
    parser.add_argument('--input', type=str, default='Exercise Inputs-20251225',
                        help='Input directory containing video files')
    parser.add_argument('--output', type=str, default='output_mosaics',
                        help='Output directory for mosaic images')
    parser.add_argument('--video', type=str, default=None,
                        help='Process a specific video file (e.g., Garden.mp4)')
    parser.add_argument('--column', type=int, default=-1,
                        help='Column index to extract from each frame (-1 for center)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VIDEO MOSAIC GENERATOR")
    print("="*60)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Using: OpenCV goodFeaturesToTrack + OpenCV calcOpticalFlowPyrLK + RANSAC")
    print("="*60)
    
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
        print(f"No video files found in {input_dir}")
        return
    
    print(f"\nFound {len(video_files)} video(s) to process:")
    for i, vf in enumerate(video_files, 1):
        print(f"  {i}. {vf}")
    
    # Process each video
    successful = 0
    failed = 0
    
    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        try:
            process_video(input_path, args.output, args.column)
            successful += 1
        except Exception as e:
            print(f"✗ Failed to process {video_file}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{len(video_files)}")
    print(f"Failed: {failed}/{len(video_files)}")
    print(f"Output directory: {args.output}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
