import numpy as np
from typing import List
import homography_evaluation as he

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
    # TODO: will be implimented later - for now Only horizontal panoramas are supported.
    pa

def get_first_to_last_transform (frames: np.ndarray):
    """
    input: frames as a sequence of images ,shape (frames, height, width) (grey scale)
    output: an array of transforms between two consecutive frames, and an array of accumaltive transforms from 1st to i-th, and the index of the optimal reference frame, the max change in x and y (after theta rotation)
    """
    dx = np.zeros(len(frames)-1)
    dy = np.zeros(len(frames)-1)
    d_theta = np.zeros(len(frames)-1)
    transforms = np.zeros((len(frames)-1, 3, 3))
    accumaltive_transforms = np.zeros((len(frames)-1, 3, 3))
    total_transform = np.eye(3)
    for i in range(len(frames)-1):
        cur_transform = he.get_transform_between_frames(frames[i], frames[i+1])
        transforms[i] = cur_transform
        total_transform = np.dot(total_transform, cur_transform)
        accumaltive_transforms[i] = total_transform
        dx[i] += cur_transform[0,2]
        dy[i] += cur_transform[1,2]
        d_theta[i] += np.arctan2(cur_transform[1,0], cur_transform[0,0])

def build_mosaic(frames: np.ndarray, transforms: List[np.ndarray]) -> np.ndarray:
    """
    Warps all frames according to the transforms and stitches them into a single mosaic.
    """
    pass

def 
