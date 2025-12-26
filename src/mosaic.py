
import numpy as np
from typing import List

def find_optimal_reference_frame(transforms: List[np.ndarray]) -> int:
    """
    Finds the reference frame index that minimizes global distortion.
    Based on the median of cumulative rotation and Y-translation.
    """
    pass

def normalize_vertical_motion(frames: np.ndarray, transforms: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray], bool]:
    """
    Detects if the dominant motion is vertical.
    If so, rotates frames and transforms 90 degrees to treat it as a horizontal panorama.
    Returns:
        (frames, transforms, is_transposed)
    """
    pass

def build_mosaic(frames: np.ndarray, transforms: List[np.ndarray]) -> np.ndarray:
    """
    Warps all frames according to the transforms and stitches them into a single mosaic.
    """
    pass
