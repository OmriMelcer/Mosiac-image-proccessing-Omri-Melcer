import numpy as np
from typing import List, Tuple
from . import homography_evaluation as he
from scipy.ndimage import affine_transform, gaussian_filter
from skimage import color as sk

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
    pass

def get_first_to_last_transform (frames: np.ndarray, func_transform_to_stabilization = get_stabilization_transform):
    """
    input: frames as a sequence of images ,shape (frames, height, width) (grey scale)
    output: transition transforms to stabalize the frames, anchor index, max change in x. 
    """
    dx = np.zeros(len(frames))
    dy = np.zeros(len(frames))
    min_y = np.inf
    max_y = -np.inf
    d_theta = np.zeros(len(frames))
    transforms = np.zeros((len(frames), 3, 3))
    accumaltive_transforms = np.zeros((len(frames), 3, 3))
    total_transform = np.eye(3)
    dx[0] = 0
    dy[0] = 0
    d_theta[0] = 0
    transforms[0] = np.eye(3)
    accumaltive_transforms[0] = np.eye(3)
    for i in range(1,len(frames)):
        cur_transform, _ = he.find_rigid_movement_opencv_with_our_ransac(frames[i-1], frames[i])
        if cur_transform is None:
            cur_transform = np.eye(3)
        transforms[i] = cur_transform
        total_transform = np.dot(cur_transform, total_transform)
        accumaltive_transforms[i] = total_transform
        dx[i] = accumaltive_transforms[i,0,2]
        dy[i] = accumaltive_transforms[i,1,2]
        d_theta[i] = np.arctan2(accumaltive_transforms[i,1,0], accumaltive_transforms[i,0,0])
        min_y = min(min_y, dy[i])
        max_y = max(max_y, dy[i])
    median_theta = np.median(d_theta)
    anchor = np.argmin(np.abs(d_theta - median_theta))
    new_transforms = recalculate_transforms(accumaltive_transforms, anchor)
    global_x_change = new_transforms[:,0,2].copy()
    true_transform = func_transform_to_stabilization(new_transforms,anchor)
    return true_transform, transforms,anchor, global_x_change

def get_stabilization_transform(transforms: np.ndarray, anchor : int  =  -1):
    """
    input: an array of from anchor to i-th frame. transforms[i] is the transform from anchor to i-th frame
    we will create a transform that will only move frame i to anchor in the y and theta direction
    output: the stabilization transform
    """
    for i in range (len(transforms)):
        tx = transforms[i][0,2]
        target_mat = np.eye(3)
        target_mat[0,2] = -tx
        transforms[i] = np.dot(transforms[i], target_mat)
    return transforms

def get_stabilization_transform_one_pixel_shift(transforms: np.ndarray, anchor: int):
    """
    input: an array of from anchor to i-th frame. transforms[i] is the transform from anchor to i-th frame
    we will create a transform that will only move frame i to anchor in the y and theta direction
    output: the stabilization transform
    """
    direction = np.sign(transforms[-1][0,2])
    for i in range (len(transforms)):
        tx = transforms[i][0,2]
        target_mat = np.eye(3)
        if abs(tx* direction) >1: # only adjust if we have moved more than 1 pixel in the dominant direction
            target_mat[0,2] = -tx+direction*(anchor - i)
        else:
            target_mat[0,2] = -tx
        transforms[i] = np.dot(transforms[i], target_mat)
    return transforms


def get_total_x_change_after_stabilization(transforms: np.ndarray):
    min_x = np.min(transforms[:,0,2])
    max_x = np.max(transforms[:,0,2])
    return max_x - min_x
    
def recalculate_transforms(accumaltive_transforms: np.ndarray, anchor: int):
    """
    Recalculates the accumaltive transforms from the anchor index
    """
    new_accumlative_transforms = np.zeros((len(accumaltive_transforms), 3, 3))
    new_accumlative_transforms[anchor] = np.eye(3)
    inverse_H = np.linalg.inv(accumaltive_transforms[anchor])
    for i in range (len(accumaltive_transforms)):
        new_accumlative_transforms[i] = np.dot( accumaltive_transforms[i], inverse_H)
    return new_accumlative_transforms

    
def apply_stabilization(frames: np.ndarray, true_transforms: np.ndarray):
    '''
    Docstring for apply_stabilization
    
    :param frames: an arrat of frames of 3 channels to be stabalized
    :type frames: np.ndarray
    :param true_transforms: homogrpahy transforms to stabalize the frames in Y and theta direction
    :type true_transforms: np.ndarray
    '''
    stabalized_frames = np.zeros_like(frames)
    for i in range (len(frames)):
        H = true_transforms[i].copy()
        # Swap rows 0 and 1 (y <-> x exchange in output)
        H[[0, 1], :] = H[[1, 0], :]
        # Swap cols 0 and 1 (y <-> x exchange in input)
        H[:, [0, 1]] = H[:, [1, 0]]
        
        matrix_rot = H[0:2, 0:2]
        matrix_trans = H[0:2, 2]
       # supposed to for on all three channels 
        func = lambda channel: affine_transform( frames[i,:,:,channel], matrix_rot, offset=matrix_trans, order=1, output_shape=frames[i,:,:,channel].shape)
        stabalized_frames[i] = np.stack([func(c) for c in range(frames.shape[3])], axis=-1)
    return stabalized_frames

def to_grey_scale(img: np.ndarray):
    """
    Convert an image to grayscale.

    This function handles various image formats and converts them to grayscale representation.
    It supports both 2D (already grayscale) and 3D (RGB/RGBA) images, and automatically
    normalizes uint8 images to float64 in the range [0, 1].

    Note: The .copy() calls are necessary to avoid potential in-place modifications by
    scikit-image functions, ensuring the original image array is not altered.

    Parameters
    ----------
    img : np.ndarray
        Input image as a numpy array. Can be:
        - 2D array (H, W) for grayscale images
        - 3D array (H, W, 3) for RGB images
        - 3D array (H, W, 4) for RGBA images
        Dtype can be uint8 or float64.

    Returns
    -------
    np.ndarray
        Grayscale image as a 2D numpy array with dtype float64 in range [0, 1].

    Raises
    ------
    ValueError
        If the image shape is not supported for grayscale conversion.
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float64) / 255.0
    if img.ndim == 2:
        return img
    elif img.ndim == 3:
        if img.shape[2] == 3:
            return sk.rgb2gray(img.copy()) # RGB
        elif img.shape[2] == 4:
            return sk.rgb2gray(sk.color.rgba2rgb(img.copy())) #RGBA
    raise ValueError("Unsupported image shape for grayscale conversion.")


def build_mosaic(frames: np.ndarray, column_to_build: int =-1 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warps all frames according to the transforms and stitches them into a single mosaic.
    input: asequence of frames that are matrices by 3 channels.
    output: (mosiac by 3 channels, stabilized_frames array)
    """
    # REMOVED FRAME LIMIT - process all frames
    # frames = frames[:100]
    
    if column_to_build == -1:
        column_to_build = frames.shape[2] // 2
    converted_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=frames.dtype)
    blurred_frames = np.zeros_like(converted_frames)
    for i in range(len(frames)):
        converted_frames[i] = (to_grey_scale(frames[i])*255).astype(np.uint8)
        blurred_frames[i] = gaussian_filter(converted_frames[i], sigma=1.0).astype(np.uint8)
    true_transform, transforms, anchor, global_x_chain = get_first_to_last_transform(blurred_frames)
    changes_x_chain = np.zeros(len(global_x_chain))
    sum_rounded_tx = np.zeros(len(global_x_chain))
    changes_x_chain[0] = 0
    sum_rounded_tx[0] = 0
    for i in range (1, len(global_x_chain)):
        changes_x_chain[i] = global_x_chain[i] - global_x_chain[i-1]
        sum_rounded_tx[i] = sum_rounded_tx[i-1] + round(changes_x_chain[i])
    # --- Global X Logic ---
    # Reconstruct absolute positions relative to the start (Frame 0) based on the pairwise chain.
    cum_tx = global_x_chain[-1]
    stabilized_frames = apply_stabilization(frames, true_transform)
    # Calculate bounds based on the cumulative scan
    min_x = np.min(sum_rounded_tx)
    max_x = np.max(sum_rounded_tx)
    # Canvas Width: Span + Frame Width
    canvas_w = int(np.ceil(max_x - min_x + frames.shape[2]))
    # Initialize Canvas
    canvas = np.zeros((frames.shape[1], canvas_w, frames.shape[3]), dtype=frames.dtype)
    # dominent movment left to right then tx is negative (ignoring all 0 movment frames and positive movments))
    if cum_tx <=0:
        canvas[:,:column_to_build,:] = stabilized_frames[0,:,:column_to_build,:]
        cur_column = column_to_build 
        for i in range (0, len(stabilized_frames)):
            tx = round (changes_x_chain[i])
            if (tx >=0):
                continue
            try:
                canvas[:,cur_column : cur_column-tx,:] = stabilized_frames[i,:,column_to_build : column_to_build-tx,:]
            except IndexError as e:
                raise IndexError(f"Error processing frame {i}: cur_column={cur_column}, tx={tx}, column_to_build={column_to_build}, canvas_w={canvas_w}, stabilized_frames shape={stabilized_frames.shape}") from e
            cur_column = cur_column - tx #tx is negative so this is addition
        #now add the rest of the last frame to the right side of the canvas
        canvas[:,cur_column:,:]= stabilized_frames[-1,:,column_to_build:,:]
    #dominent movment right to left (ignoring all 0 movment frames and negative movments))
    else:
        canvas[:,canvas_w-column_to_build:,:]= stabilized_frames[0,:,column_to_build:,:]
        cur_column = canvas_w - column_to_build
        for i in range (0, len(stabilized_frames)):
            tx = round (changes_x_chain[i])
            if (tx <=0):
                continue
            try:
                canvas[:,cur_column - tx : cur_column,:] = stabilized_frames[i,:,column_to_build-tx : column_to_build,:]
            except IndexError as e:
                raise IndexError(f"Error processing frame {i}: cur_column={cur_column}, tx={tx}, column_to_build={column_to_build}, canvas_w={canvas_w}, stabilized_frames shape={stabilized_frames.shape}") from e
            cur_column = cur_column - tx #tx is positive so this is subtraction and moves left
        #now add the rest of the last frame to the left side of the canvas
        canvas[:,:cur_column,:]= stabilized_frames[-1,:,:column_to_build,:]
    return canvas, stabilized_frames 


def build_mosaic_one_column(frames: np.ndarray, column_to_build: int =-1 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warps all frames according to the transforms and stitches them into a single mosaic.
    input: asequence of frames that are matrices by 3 channels.
    output: (mosiac by 3 channels, stabilized_frames array)
    """
    # REMOVED FRAME LIMIT - process all frames
    # frames = frames[:100]
    
    if column_to_build == -1:
        column_to_build = frames.shape[2] // 2
    converted_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=frames.dtype)
    blurred_frames = np.zeros_like(converted_frames)
    for i in range(len(frames)):
        converted_frames[i] = (to_grey_scale(frames[i])*255).astype(np.uint8)
        blurred_frames[i] = gaussian_filter(converted_frames[i], sigma=1.0).astype(np.uint8)
    true_transform, transforms, anchor, global_x_chain = get_first_to_last_transform(blurred_frames, get_stabilization_transform_one_pixel_shift)
    changes_x_chain = np.zeros(len(global_x_chain))
    sum_rounded_tx = np.zeros(len(global_x_chain))
    changes_x_chain[0] = 0
    sum_rounded_tx[0] = 0
    for i in range (1, len(global_x_chain)):
        changes_x_chain[i] = true_transform[i,0,2] - true_transform[i-1,0,2]
        sum_rounded_tx[i] = sum_rounded_tx[i-1] + round(changes_x_chain[-1])
    # --- Global X Logic ---
    # Reconstruct absolute positions relative to the start (Frame 0) based on the pairwise chain.
    cum_tx = np.sum(sum_rounded_tx)
    stabilized_frames = apply_stabilization(frames, true_transform)
    # Calculate bounds based on the cumulative scan
    min_x = np.min(sum_rounded_tx)
    max_x = np.max(sum_rounded_tx)
    # Canvas Width: Span + Frame Width
    canvas_w = int(np.ceil(max_x - min_x + frames.shape[2]))
    # Initialize Canvas
    canvas = np.zeros((frames.shape[1], canvas_w, frames.shape[3]), dtype=frames.dtype)
    # dominent movment left to right then tx is negative (ignoring all 0 movment frames and positive movments))
    if cum_tx <=0:
        canvas[:,:column_to_build,:] = stabilized_frames[0,:,:column_to_build,:]
        cur_column = column_to_build 
        for i in range (0, len(stabilized_frames)):
            tx = round (changes_x_chain[i])
            if (tx >=0):
                continue
            try:
                canvas[:,cur_column : cur_column-tx,:] = stabilized_frames[i,:,column_to_build : column_to_build-tx,:]
            except IndexError as e:
                raise IndexError(f"Error processing frame {i}: cur_column={cur_column}, tx={tx}, column_to_build={column_to_build}, canvas_w={canvas_w}, stabilized_frames shape={stabilized_frames.shape}") from e
            cur_column = cur_column - tx #tx is negative so this is addition
        #now add the rest of the last frame to the right side of the canvas
        canvas[:,cur_column:,:]= stabilized_frames[-1,:,column_to_build:,:]
    #dominent movment right to left (ignoring all 0 movment frames and negative movments))
    else:
        canvas[:,canvas_w-column_to_build:,:]= stabilized_frames[0,:,column_to_build:,:]
        cur_column = canvas_w - column_to_build
        for i in range (0, len(stabilized_frames)):
            tx = round (changes_x_chain[i])
            if (tx <=0):
                continue
            try:
                canvas[:,cur_column - tx : cur_column,:] = stabilized_frames[i,:,column_to_build-tx : column_to_build,:]
            except IndexError as e:
                raise IndexError(f"Error processing frame {i}: cur_column={cur_column}, tx={tx}, column_to_build={column_to_build}, canvas_w={canvas_w}, stabilized_frames shape={stabilized_frames.shape}") from e
            cur_column = cur_column - tx #tx is positive so this is subtraction and moves left
        #now add the rest of the last frame to the left side of the canvas
        canvas[:,:cur_column,:]= stabilized_frames[-1,:,:column_to_build,:]
    return canvas, stabilized_frames 
    
    

