import numpy as np
from typing import List, Tuple
import homography_evaluation as he
from scipy.ndimage import affine_transform

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

def get_first_to_last_transform (frames: np.ndarray):
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
        cur_transform = he.find_rigid_movement(frames[i-1], frames[i])
        transforms[i] = cur_transform
        total_transform = np.dot(cur_transform, total_transform)
        accumaltive_transforms[i] = total_transform
        dx[i] += cur_transform[0,2]
        dy[i] += cur_transform[1,2]
        d_theta[i] += np.arctan2(cur_transform[1,0], cur_transform[0,0])
        min_y = min(min_y, dy[i])
        max_y = max(max_y, dy[i])
    median_theta = np.median(d_theta)
    anchor = np.argmin(np.abs(d_theta - median_theta))
    new_transforms = recalculate_transforms(accumaltive_transforms, anchor)
    true_transform = get_stabilization_transform(new_transforms)
    return true_transform, transforms,anchor, get_total_x_change_after_stabilization(new_transforms)

def get_stabilization_transform(transforms: np.ndarray):
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
    stabalized_frames = np.zeros_like(frames)
    for i in range (len(frames)):
        H = true_transforms[i].copy()
        # Swap rows 0 and 1 (y <-> x exchange in output)
        H[[0, 1], :] = H[[1, 0], :]
        # Swap cols 0 and 1 (y <-> x exchange in input)
        H[:, [0, 1]] = H[:, [1, 0]]
        
        matrix_rot = H[0:2, 0:2]
        matrix_trans = H[0:2, 2]
        
        stabilized_frames[i] = affine_transform(
            frames[i], 
            matrix_rot, 
            offset=matrix_trans, 
            order=1, 
            output_shape=frames[i].shape
        )
    return stabilized_frames
    

def build_mosaic(frames: np.ndarray, column_to_build: int =-1 ) -> np.ndarray:
    """
    Warps all frames according to the transforms and stitches them into a single mosaic.
    """
    if column_to_build == -1:
        column_to_build = frames.shape[2] // 2
        
    true_transform, transforms, anchor, max_x_change = get_first_to_last_transform(frames)
    
    # --- Global X Logic ---
    # Reconstruct absolute positions relative to the start (Frame 0) based on the pairwise chain.
    global_x_chain = np.zeros(len(frames))
    cum_tx = 0
    for i in range(1, len(frames)):
        cum_tx += transforms[i][0,2] 
        global_x_chain[i] = cum_tx
        
    stabilized_frames = apply_stabilization(frames, true_transform)
    
    # Calculate bounds based on the cumulative scan
    min_x = np.min(global_x_chain)
    max_x = np.max(global_x_chain)
    
    # Canvas Width: Span + Frame Width
    canvas_w = int(np.ceil(max_x - min_x + frames.shape[2]))
    
    # Initialize Canvas
    if frames.ndim == 4:
         canvas = np.zeros((frames.shape[1], canvas_w, frames.shape[3]), dtype=frames.dtype)
    else:
         canvas = np.zeros((frames.shape[1], canvas_w), dtype=frames.dtype)

    
    # Start Position:
    # If min_x is negative, we must shift our "zero" to the right.
    start_offset = -min_x if min_x < 0 else 0
    
    # Current cursor on canvas
    canvas_cursor = int(start_offset) 
    
    # Approximate start - paste the first column(s)
    # Note: anchor for frame 0 is at 'column_to_build', but adjusted by start_offset
    # Logic: The loop below handles i=1..N. We need to initialize the "state" at i=0.
    # Just paste the single column for Frame 0?
    if frames.ndim == 4:
        canvas[:, canvas_cursor, :] = stabilized_frames[0][:, column_to_build, :]
    else:
        canvas[:, canvas_cursor] = stabilized_frames[0][:, column_to_build]

    for i in range(1, len(frames)):
        dx = transforms[i][0,2]
        dx_int = int(round(dx))
        
        # If no movement, skip or just move cursor?
        if dx_int == 0:
            continue
            
        if dx_int > 0:
            # Moving Right: Paste strip of width dx
            width = dx_int
            # Strip from frame: Center to Center + Width
            # (Assuming we revealed pixels to the right)
            strip = stabilized_frames[i][:, column_to_build : column_to_build + width]
            
            # Place on canvas
            dest_end = canvas_cursor + width
            # Check bounds (Source and Dest)
            h, w_strip = strip.shape[:2]
            
            if canvas_cursor + w_strip <= canvas.shape[1]:
                if frames.ndim == 4:
                     canvas[:, canvas_cursor : canvas_cursor + w_strip, :] = strip
                else:
                     canvas[:, canvas_cursor : canvas_cursor + w_strip] = strip
            
            canvas_cursor += width
            
        elif dx_int < 0:
            # Moving Left
            width = abs(dx_int)
            # Strip from frame: Center - Width to Center
            # (Assuming we revealed pixels to the left)
            # Ensure valid splice
            start_col = column_to_build - width
            if start_col < 0: start_col = 0 # safety
            
            strip = stabilized_frames[i][:, start_col : column_to_build]
            
            # Place "backwards" relative to current cursor?
            # Actually, if we moved left, the NEW pixels belong to the left of the old ones?
            # But the loop iterates chronologically. Frame i is to the LEFT of Frame i-1.
            # So canvas_cursor should move LEFT.
            
            dest_start = canvas_cursor - width
            if dest_start >= 0:
                 # We paste INTO the gap we just stepped back over
                 if frames.ndim == 4:
                     canvas[:, dest_start : canvas_cursor, :] = strip
                 else:
                     canvas[:, dest_start : canvas_cursor] = strip
            
            canvas_cursor -= width
            
    # Fill tail?
    # canvas[:, canvas_cursor:] = ... 
    
    return canvas 

        
    
    

