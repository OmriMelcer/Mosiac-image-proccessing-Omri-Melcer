
import numpy as np
from src.mosaic import calculate_safe_margins

def test_margins():
    # Simulate a 100x100 image
    h, w = 100, 100
    
    # Case 1: Identity (No rotation)
    H_identity = np.eye(3)
    
    # Case 2: Rotation by 45 degrees (counter-clockwise)
    # Center of rotation (50, 50)
    # But our transforms are usually global. Let's just use a simple rotation around (0,0) for testing logic.
    theta = np.radians(10)
    c, s = np.cos(theta), np.sin(theta)
    # Rotation matrix (Output -> Input)
    # x_in = c*x_out - s*y_out
    # y_in = s*x_out + c*y_out
    H_rot = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    
    # Case 3: Translation (should shift the valid window)
    # x_in = x_out + 10 -> x_out = x_in - 10
    # Valid region shifts left.
    H_trans = np.array([
        [1, 0, 10],
        [0, 1, 0],
        [0, 0, 1]
    ])
    
    transforms = np.array([H_identity, H_rot, H_trans])
    
    print("Testing calculate_safe_margins...")
    safe_min, safe_max = calculate_safe_margins((h, w), transforms)
    
    print(f"Image Size: {w}x{h}")
    print(f"Safe Min X: {safe_min}")
    print(f"Safe Max X: {safe_max}")
    
    # Expected behavior:
    # Rotation 10 deg:
    # TL(0,0) -> (0,0)
    # BL(0,100) -> (100*sin(10), 100*cos(10)) = (17.3, 98.4)
    # So Left Margin should be at least 17.3 (approx 18)
    
    # Translation 10px:
    # x_out = x_in - 10.
    # TL(0,0) -> (-10, 0)
    # TR(100,0) -> (90, 0)
    # So Right Margin should be at most 90.
    
    # Combined: [18, 90] approx.

if __name__ == "__main__":
    test_margins()
