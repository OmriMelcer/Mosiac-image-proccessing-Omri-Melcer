
import numpy as np
import cv2
from src.video_io import load_video
from src.mosaic import build_mosaic

def test_negative_indexing():
    print("Loading Shinkansen.mp4...")
    frames = load_video('Exercise Inputs-20251225/Shinkansen.mp4')
    width = frames.shape[2]
    
    # build_mosaic signature: (frames, anchor_index_locator_func=None, amount_of_frames=None)
    # It handles column_to_build internally if amount_of_frames=1, but we can't pass it directly?
    # Wait, looking at build_mosaic code:
    # if amount_of_frames is None or amount_of_frames == 1:
    #     amount_of_frames = 1
    #     column_to_build = frames.shape[2] // 2
    
    # We need to modify build_mosaic to accept column_to_build or use the progressive mode logic
    # Actually, let's just modify the test to use the internal logic or patch it.
    # But wait, main.py calls it with column_to_build?
    # Let's check main.py again.
    pass

def test_negative_indexing_patched():
    print("Loading Shinkansen.mp4...")
    frames = load_video('Exercise Inputs-20251225/Shinkansen.mp4')
    
    # We can't easily pass column_to_build to build_mosaic in its current form without changing the signature.
    # But we can use the progressive mode which iterates columns.
    # Or we can just call fill_canvas_from_stabilized_frames directly if we had the stabilized frames.
    
    # Let's use the progressive mode with many divisions to force a small column index.
    # If width is ~600, and we want col=5, we need step=5.
    # divisions = 600 / 5 = 120.
    
    divisions = 120
    print(f"Testing with divisions={divisions} to trigger small column indices")
    
    try:
        mosaics, _ = build_mosaic(frames, amount_of_frames=divisions)
        print("Mosaics built successfully")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_negative_indexing_patched()
