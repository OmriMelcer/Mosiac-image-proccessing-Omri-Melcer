import numpy as np
from typing import List, Tuple
from . import homography_evaluation as he
from scipy.ndimage import affine_transform, gaussian_filter
from skimage import color as sk
import cv2


def normalize_vertical_motion(frames: np.ndarray, transforms: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray], bool]:
    """
    Detects if the dominant motion is vertical.
    If so, rotates frames and transforms 90 degrees to treat it as a horizontal panorama.
    Returns:
        (frames, transforms, is_transposed)
    """
    # TODO: will be implimented later - for now Only horizontal panoramas are supported.
    pass

def calculate_safe_margins(frames_shape: Tuple[int, int], true_transforms: np.ndarray) -> Tuple[int, int]:
    """
    Calculates the safe horizontal margins to avoid black triangular regions caused by rotation/stabilization.
    Returns (safe_min_x, safe_max_x)
    """
    h, w = frames_shape[0], frames_shape[1]
    corners = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [w, h, 1],
        [0, h, 1]
    ]).T  # 3x4 matrix

    min_valid_x_list = []
    max_valid_x_list = []

    for H in true_transforms:
        # We need to map the Input Frame corners to the Output (Stabilized) Frame.
        # apply_stabilization uses affine_transform(..., H, ...).
        # This implies H maps Output -> Input.
        # So to map Input -> Output, we need inv(H).
        try:
            inv_H = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            continue

        # Map corners: X_out = inv_H * X_in
        transformed_corners = np.dot(inv_H, corners)
        # Normalize by z (though for affine it should be 1)
        transformed_corners /= transformed_corners[2, :]
        
        x_coords = transformed_corners[0, :]
        
        # For a rotated rectangle, the "valid" region for a full vertical strip
        # is bounded by the "innermost" corners.
        # Left bound: max(Top-Left X, Bottom-Left X)
        # Right bound: min(Top-Right X, Bottom-Right X)
        
        # Corners order: TL(0), TR(1), BR(2), BL(3)
        x_tl, x_tr, x_br, x_bl = x_coords[0], x_coords[1], x_coords[2], x_coords[3]
        
        current_min_valid = max(x_tl, x_bl)
        current_max_valid = min(x_tr, x_br)
        
        min_valid_x_list.append(current_min_valid)
        max_valid_x_list.append(current_max_valid)

    # We need a margin that is safe for ALL frames
    if not min_valid_x_list:
        return 0, w

    global_safe_min = int(np.ceil(np.max(min_valid_x_list)))
    global_safe_max = int(np.floor(np.min(max_valid_x_list)))
    
    # Clamp to image bounds
    global_safe_min = max(0, global_safe_min)
    global_safe_max = min(w, global_safe_max)
    
    return global_safe_min, global_safe_max

def get_first_to_last_transform (frames: np.ndarray, anchor_index_locator_func):
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
    # median_theta = np.median(d_theta)
    # anchor = np.argmin(np.abs(d_theta - median_theta))
    anchor = anchor_index_locator_func(dy, d_theta)
    new_transforms = recalculate_transforms(accumaltive_transforms, anchor)
    global_x_change = new_transforms[:,0,2].copy()
    true_transform = get_stabilization_transform(new_transforms)
    return true_transform, transforms,anchor, global_x_change

def get_anchor_frame_by_mean_dy(dy: np.ndarray, d_theta: np.ndarray) -> int:
    """
    Finds the anchor frame index based on mean Y-translation.
    """
    mean_dy = np.mean(dy)
    anchor = np.argmin(np.abs(dy - mean_dy))
    return anchor

def get_anchor_frame_by_median_dy(dy: np.ndarray, d_theta: np.ndarray) -> int:
    """
    Finds the anchor frame index based on median Y-translation.
    """
    median_dy = np.median(dy)
    anchor = np.argmin(np.abs(dy - median_dy))
    return anchor

def get_anchor_frame_by_median_rotation(d_y: np.ndarray, d_theta: np.ndarray) -> int:
    """
    Finds the anchor frame index based on median rotation.
    """
    median_theta = np.median(d_theta)
    anchor = np.argmin(np.abs(d_theta - median_theta))
    return anchor

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


def build_mosaic(frames: np.ndarray,  anchor_index_locator_func = None, amount_of_frames: int = None):
    """
    Warps all frames according to the transforms and stitches them into a single mosaic.
    input: asequence of frames that are matrices by 3 channels.
    output: (mosiac by 3 channels, stabilized_frames array)
    """
    if anchor_index_locator_func is None:
        anchor_index_locator_func = get_anchor_frame_by_median_dy
    if amount_of_frames is None or amount_of_frames == 1:
        amount_of_frames = 1
        column_to_build = frames.shape[2] // 2
    converted_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=frames.dtype)
    for i in range(len(frames)):
        converted_frames[i] = (to_grey_scale(frames[i])*255).astype(np.uint8)
    true_transform, transforms, anchor, global_x_chain = get_first_to_last_transform(converted_frames, anchor_index_locator_func=anchor_index_locator_func)
    changes_x_chain = np.zeros(len(global_x_chain))
    sum_rounded_tx = np.zeros(len(global_x_chain))
    changes_x_chain[0] = 0
    sum_rounded_tx[0] = 0
    # Sum full changes ignoring the fractional parts
    for i in range (1, len(global_x_chain)):
        changes_x_chain[i] = global_x_chain[i] - global_x_chain[i-1]
        sum_rounded_tx[i] = sum_rounded_tx[i-1] + round(changes_x_chain[i])
    # Sum full changes including fractional parts
    sum_all_tx = np.zeros(len(global_x_chain))
    sum_all_tx[0] = 0
    for i in range (1, len(global_x_chain)):
        sum_all_tx[i] = sum_all_tx[i-1] + changes_x_chain[i]
    # Sum full changes ignoring the fractional parts

    cum_tx = global_x_chain[-1]
    stabilized_frames = apply_stabilization(frames, true_transform)
    # Calculate bounds based on the cumulative scan
    min_x = np.min(sum_rounded_tx)
    max_x = np.max(sum_rounded_tx)
    # Alternatively, for fractional tx handling:
    min_x_f = np.min(sum_all_tx)
    max_x_f = np.max(sum_all_tx)

    # Canvas Width: Span + Frame Width
    canvas_w = int(np.ceil(max_x - min_x + frames.shape[2]))
    #canvas with for fractional tx handling
    canvas_w_f = int(np.ceil(max_x_f - min_x_f + frames.shape[2]))
    # Initialize Canvas
    canvas_array = np.zeros((amount_of_frames,frames.shape[1], canvas_w, frames.shape[3]), dtype=frames.dtype)
    #intialize canvas for fractional tx handling
    canvas_array_f = np.zeros((amount_of_frames,frames.shape[1], canvas_w_f, frames.shape[3]), dtype=frames.dtype)    
    # Calculate safe margins to avoid black triangular regions
    safe_min_x, safe_max_x = calculate_safe_margins((frames.shape[1], frames.shape[2]), true_transform)
    valid_width = safe_max_x - safe_min_x
    print(f"Valid scanning width: {valid_width} (from {safe_min_x} to {safe_max_x})")
    
    # Use the safe valid width for scanning (safe_min_x to safe_max_x)
    width_partical = valid_width // amount_of_frames
    
    # adjust for 
    if amount_of_frames ==1:
        fill_canvas_from_stabilized_frames(canvas_array[0], stabilized_frames, changes_x_chain, column_to_build, cum_tx)
        fill_canvas_from_stabilized_frames_fractional(canvas_array_f[0], stabilized_frames, changes_x_chain, column_to_build, cum_tx)
        return canvas_array[0], canvas_array_f[0], stabilized_frames
    for i in range(amount_of_frames):
        # Scan strictly within safe margins
        column_to_build = safe_min_x + i * width_partical
        
        # Ensure we don't exceed safe_max_x
        if column_to_build >= safe_max_x:
            column_to_build = safe_max_x - 1
            
        fill_canvas_from_stabilized_frames(canvas_array[i], stabilized_frames, changes_x_chain, column_to_build, cum_tx)
        fill_canvas_from_stabilized_frames_fractional(canvas_array_f[i], stabilized_frames, changes_x_chain, column_to_build, cum_tx)
    return canvas_array, canvas_array_f, stabilized_frames


def fill_canvas_from_stabilized_frames(canvas: np.ndarray, stabilized_frames: np.ndarray, changes_x_chain: np.ndarray, column_to_build: int, cum_tx: int):
    """
    Fills the mosaic canvas from the stabilized frames based on the dominant motion direction.
    input: (canvas by 3 channels, stabilized_frames array, column to build from)
    output: filled canvas
    """
    canvas_w = canvas.shape[1]
    
    # Find the last frame with significant movement (> 0.5 pixels)
    last_significant_idx = 0
    for i in range(len(changes_x_chain) - 1, -1, -1):
        if abs(changes_x_chain[i]) > 0.5:
            last_significant_idx = i
            break
            
    if cum_tx <=0:
        canvas[:,:column_to_build,:] = stabilized_frames[0,:,:column_to_build,:]
        cur_column = column_to_build 
        
        # Iterate only up to the last significant frame (exclusive)
        # We will paste the remainder of the last_significant_frame afterwards
        for i in range (0, last_significant_idx):
            tx = round (changes_x_chain[i])
            if (tx >=0):
                continue
            try:
                canvas[:,cur_column : cur_column-tx,:] = stabilized_frames[i,:,column_to_build : column_to_build-tx,:]
            except IndexError as e:
                raise IndexError(f"Error processing frame {i}: cur_column={cur_column}, tx={tx}, column_to_build={column_to_build}, canvas_w={canvas_w}, stabilized_frames shape={stabilized_frames.shape}") from e
            cur_column = cur_column - tx #tx is negative so this is addition
            
        #now add the rest of the last significant frame to the right side of the canvas
        remaining_canvas = canvas_w - cur_column
        remaining_frame = stabilized_frames.shape[2] - column_to_build
        # Safety check: only copy what fits
        columns_to_copy = min(remaining_canvas, remaining_frame)
        
        # Use last_significant_idx instead of -1
        canvas[:,cur_column:cur_column+columns_to_copy,:] = stabilized_frames[last_significant_idx,:,column_to_build:column_to_build+columns_to_copy,:]
    #dominent movment right to left (ignoring all 0 movment frames and negative movments))
    else:
        right_side_of_first_frame = stabilized_frames.shape[2] - column_to_build
        canvas[:,canvas_w-right_side_of_first_frame:,:]= stabilized_frames[0,:,column_to_build:,:]
        cur_column = canvas_w - right_side_of_first_frame
        
        # Iterate only up to the last significant frame (exclusive)
        for i in range (0, last_significant_idx):
            tx = round (changes_x_chain[i])
            if (tx <=0):
                continue
            try:
                canvas[:,cur_column - tx : cur_column,:] = stabilized_frames[i,:,column_to_build-tx : column_to_build,:]
            except IndexError as e:
                raise IndexError(f"Error processing frame {i}: cur_column={cur_column}, tx={tx}, column_to_build={column_to_build}, canvas_w={canvas_w}, stabilized_frames shape={stabilized_frames.shape}") from e
            cur_column = cur_column - tx #tx is positive so this is subtraction and moves left
            
        #now add the rest of the last significant frame to the left side of the canvas
        # Use last_significant_idx instead of -1
        canvas[:,:cur_column,:] = stabilized_frames[last_significant_idx,:,:cur_column,:]
     
def fill_canvas_from_stabilized_frames_fractional(canvas: np.ndarray, stabilized_frames: np.ndarray, changes_x_chain: np.ndarray, column_to_build: int, cum_tx: int):
    """
    Fills the mosaic canvas from the stabilized frames based on the dominant motion direction.
    input: (canvas by 3 channels, stabilized_frames array, column to build from), input tx is a fractional change bewteen frames. 
    not rounded to integer
    canvas_w is the sum of all the fractional changes + original frame width
    output: filled canvas
    """
    canvas_w = canvas.shape[1]
    resedual_tx = 0.0
    
    # Find the last frame with significant movement (> 0.5 pixels)
    last_significant_idx = 0
    for i in range(len(changes_x_chain) - 1, -1, -1):
        if abs(changes_x_chain[i]) > 0.5:
            last_significant_idx = i
            break
            
    if cum_tx <=0:
        canvas[:,:column_to_build,:] = stabilized_frames[0,:,:column_to_build,:]
        cur_column = column_to_build
        
        # Iterate only up to the last significant frame (exclusive)
        for i in range (0, last_significant_idx):
            tx_with_resedue = changes_x_chain[i] + resedual_tx
            resedual_tx = tx_with_resedue - round(tx_with_resedue)
            tx = round (tx_with_resedue)
            
            if (tx >= 0):
                continue
            try:
                canvas[:,cur_column : cur_column-tx,:] = stabilized_frames[i,:,column_to_build : column_to_build-tx,:]
            except IndexError as e:
                raise IndexError(f"Error processing frame {i}: cur_column={cur_column}, tx={tx}, column_to_build={column_to_build}, canvas_w={canvas_w}, stabilized_frames shape={stabilized_frames.shape}") from e
            cur_column = cur_column - tx #tx is negative so this is addition
            
        #now add the rest of the last significant frame to the right side of the canvas
        remaining_canvas = canvas_w - cur_column
        remaining_frame = stabilized_frames.shape[2] - column_to_build
        columns_to_copy = min(remaining_canvas, remaining_frame)
        canvas[:,cur_column:cur_column+columns_to_copy,:] = stabilized_frames[last_significant_idx,:,column_to_build:column_to_build+columns_to_copy,:]
    #dominent movment right to left (ignoring all 0 movment frames and negative movments))
    else:
        right_side_of_first_frame = stabilized_frames.shape[2] - column_to_build
        canvas[:,canvas_w-right_side_of_first_frame:,:]= stabilized_frames[0,:,column_to_build:,:]
        cur_column = canvas_w - right_side_of_first_frame
        
        # Iterate only up to the last significant frame (exclusive)
        for i in range (0, last_significant_idx):
            tx_with_resedue = changes_x_chain[i] + resedual_tx
            resedual_tx = tx_with_resedue - round(tx_with_resedue)
            tx = round (tx_with_resedue)
            if (tx <=0):
                continue
            try:
                canvas[:,cur_column - tx : cur_column,:] = stabilized_frames[i,:,column_to_build-tx : column_to_build,:]
            except IndexError as e:
                raise IndexError(f"Error processing frame {i}: cur_column={cur_column}, tx={tx}, column_to_build={column_to_build}, canvas_w={canvas_w}, stabilized_frames shape={stabilized_frames.shape}") from e
            cur_column = cur_column - tx #tx is positive so this is subtraction and moves left
            
        #now add the rest of the last significant frame to the left side of the canvas
        # Use last_significant_idx instead of -1
        # Fix: copy from the left side of the frame (0 to column_to_build)
        canvas[:,:cur_column,:] = stabilized_frames[last_significant_idx,:,:cur_column,:]
     