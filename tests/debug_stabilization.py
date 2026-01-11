
import numpy as np
import cv2
import os
from src.video_io import load_video
from src.mosaic import get_first_to_last_transform, get_anchor_frame_by_median_dy, to_grey_scale, apply_stabilization
from scipy.ndimage import gaussian_filter

def save_stabilized_frames():
    print("Loading Shinkansen.mp4...")
    frames = load_video('Exercise Inputs-20251225/Shinkansen.mp4')
    # Process first 50 frames to save time
    frames = frames[:50]
    
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
    
    print(f"Anchor frame index: {anchor}")
    
    print("Applying stabilization...")
    stabilized_frames = apply_stabilization(frames, true_transform)
    
    output_dir = "test_outputs/debug_stabilized"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving first 10 stabilized frames to {output_dir}...")
    for i in range(10):
        output_path = os.path.join(output_dir, f"stabilized_frame_{i:03d}.png")
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(stabilized_frames[i].astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, frame_bgr)
        print(f"Saved {output_path}")

    # Also save the anchor frame for comparison
    anchor_path = os.path.join(output_dir, f"stabilized_frame_anchor_{anchor:03d}.png")
    anchor_bgr = cv2.cvtColor(stabilized_frames[anchor].astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(anchor_path, anchor_bgr)
    print(f"Saved Anchor: {anchor_path}")

if __name__ == "__main__":
    save_stabilized_frames()
