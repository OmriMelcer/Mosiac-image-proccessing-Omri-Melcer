
import numpy as np
import cv2
from typing import List, Tuple, Union

def load_video(path: str) -> np.ndarray:
    """
    Loads a video from the specified path.
    Returns:
        np.ndarray: Video data as (frames, height, width) or (frames, height, width, channels).
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV loads as BGR, convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames)

def save_video(frames: np.ndarray, path: str, fps: int = 30):
    """
    Saves a sequence of frames to a video file.
    """
    if len(frames) == 0:
        return
    height, width = frames[0].shape[:2]
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # For MP4
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Safer for AVI across systems
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Expecting RGB, convert back to BGR for OpenCV
        if frame.ndim == 3 and frame.shape[2] == 3:
            out.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
        else:
            # Grayscale
            out.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2BGR))
            
    out.release()

def generate_synthetic_video(type: str = 'translation', size: Tuple[int, int] = (300, 300), num_frames: int = 30) -> np.ndarray:
    """
    Generates a synthetic video for testing.
    Args:
        type: 'translation' (moves X) or 'rotation' (around center) or 'mixed'
    """
    H, W = size
    frames = np.zeros((num_frames, H, W, 3), dtype=np.uint8)
    
    # White background
    frames[:] = 255
    
    # Object: A red square with some texture (random noise) to help feature tracking
    obj_size = 50
    obj = np.random.randint(0, 150, (obj_size, obj_size, 3), dtype=np.uint8)
    obj[:, :, 0] = 255 # make it reddish
    
    center_y, center_x = H // 2, W // 2
    
    for i in range(num_frames):
        # Calculate position/rotation
        dx, dy, angle = 0, 0, 0
        
        if type == 'translation':
            dx = int(i * 3) # Move 3 px per frame to right
        elif type == 'rotation':
            angle = i * 2 # Rotate 2 degrees per frame
        elif type == 'mixed':
            dx = int(i * 2)
            angle = i * 1
            
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        
        # We warp the "base scene" which has the square in middle?
        # Simpler: Just place the square at new coords.
        # But rotation is easier via warp.
        
        # Create base frame with square in center
        base = np.full((H, W, 3), 255, dtype=np.uint8)
        base[center_y-obj_size//2 : center_y+obj_size//2, 
             center_x-obj_size//2 : center_x+obj_size//2] = obj
             
        # Apply transformation
        frames[i] = cv2.warpAffine(base, M, (W, H), borderValue=(255, 255, 255))
        
    return frames
