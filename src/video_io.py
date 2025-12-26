
import numpy as np
import cv2
from typing import List, Tuple, Union

def load_video(path: str) -> np.ndarray:
    """
    Loads a video from the specified path.
    Returns:
        np.ndarray: Video data as (frames, height, width) or (frames, height, width, channels).
    """
    pass

def save_video(frames: np.ndarray, path: str, fps: int = 30):
    """
    Saves a sequence of frames to a video file.
    """
    pass

def generate_synthetic_video(type: str = 'translation', size: Tuple[int, int] = (300, 300), num_frames: int = 30) -> np.ndarray:
    """
    Generates a synthetic video for testing.
    Args:
        type: 'translation' or 'rotation'
    """
    pass
