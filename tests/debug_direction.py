
import numpy as np
import cv2
from src.video_io import load_video
from src.mosaic import get_first_to_last_transform, get_anchor_frame_by_median_dy, to_grey_scale
from scipy.ndimage import gaussian_filter

def check_direction():
    print("Loading Shinkansen.mp4...")
    frames = load_video('Exercise Inputs-20251225/Shinkansen.mp4')
    # Process first 30 frames to save time, should be enough for direction
    frames = frames[:30] 
    
    print("Preprocessing...")
    converted_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=frames.dtype)
    blurred_frames = np.zeros_like(converted_frames)
    for i in range(len(frames)):
        converted_frames[i] = (to_grey_scale(frames[i])*255).astype(np.uint8)
        blurred_frames[i] = gaussian_filter(converted_frames[i], sigma=1.0).astype(np.uint8)
        
    print("Calculating transforms...")
    true_transform, transforms, anchor, global_x_chain = get_first_to_last_transform(
        blurred_frames, 
        anchor_index_locator_func=get_anchor_frame_by_median_dy
    )
    
    cum_tx = global_x_chain[-1]
    print(f"Cumulative X change (first 30 frames): {cum_tx}")
    if cum_tx > 0:
        print("Direction: Scene moves RIGHT (Camera pans LEFT)")
        print("Mosaic filling order: Right-to-Left")
        print("Right side of mosaic comes from: Frame 0 (First Stitches)")
    else:
        print("Direction: Scene moves LEFT (Camera pans RIGHT)")
        print("Mosaic filling order: Left-to-Right")
        print("Right side of mosaic comes from: Frame N (Last Stitches)")

if __name__ == "__main__":
    check_direction()
