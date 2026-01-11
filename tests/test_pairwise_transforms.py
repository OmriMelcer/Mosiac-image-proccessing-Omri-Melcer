"""
Test to isolate whether the problem is in:
1. Pairwise homography estimation (homography_evaluation)
2. Transform accumulation (mosaic.py)

We'll compute direct frame-to-frame transforms and compare to accumulated transforms.
"""
import numpy as np
import cv2
from src.video_io import load_video
from src.mosaic import to_grey_scale
from src import homography_evaluation as he
from scipy.ndimage import gaussian_filter
from main import remove_black_boundaries

def test_frame_pair(frame1, frame2, frame_idx, label=""):
    """Test a single frame pair and extract tx."""
    H, inliers = he.find_rigid_movement_opencv_with_our_ransac(frame1, frame2)
    
    if H is None:
        print(f"  {label} Frame {frame_idx}: FAILED to compute homography")
        return None, 0
    
    tx = H[0, 2]
    ty = H[1, 2]
    theta = np.arctan2(H[1, 0], H[0, 0]) * 180 / np.pi
    
    print(f"  {label} Frame {frame_idx}: tx={tx:7.2f}, ty={ty:7.2f}, theta={theta:6.2f}°, inliers={inliers}")
    return H, inliers


def main():
    print("="*70)
    print("PAIRWISE TRANSFORM TEST")
    print("Testing if problem is in homography_evaluation vs accumulation")
    print("="*70)
    
    # Load and preprocess
    print("\n1. Loading House.mp4...")
    frames = load_video("Exercise Inputs-20251225/House.mp4")
    frames = remove_black_boundaries(frames)
    print(f"   Frames: {len(frames)}, shape: {frames[0].shape}")
    
    # Convert to grayscale and blur
    print("\n2. Converting to grayscale and blurring...")
    blurred_frames = np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=frames.dtype)
    for i in range(len(frames)):
        gray = (to_grey_scale(frames[i])*255).astype(np.uint8)
        blurred_frames[i] = gaussian_filter(gray, sigma=1.0).astype(np.uint8)
    
    # Test problematic frame pairs
    test_pairs = [
        (2, 3, "First with large tx"),
        (431, 432, "Last with HUGE tx"),
        (14, 15, "Another large tx"),
        (0, 1, "Very first pair"),
        (215, 216, "Around anchor"),
    ]
    
    print("\n3. Testing PAIRWISE transforms (direct frame-to-frame):")
    print("-"*70)
    
    pairwise_results = {}
    for frame1_idx, frame2_idx, description in test_pairs:
        print(f"\n{description}:")
        H, inliers = test_frame_pair(
            blurred_frames[frame1_idx],
            blurred_frames[frame2_idx],
            frame2_idx,
            "Pairwise"
        )
        pairwise_results[(frame1_idx, frame2_idx)] = (H, inliers)
    
    # Now compute accumulated transforms and compare
    print("\n" + "="*70)
    print("4. Computing ACCUMULATED transforms (as done in mosaic.py):")
    print("-"*70)
    
    # Compute all pairwise transforms sequentially
    transforms = [np.eye(3)]  # First frame is identity
    for i in range(1, len(blurred_frames)):
        H, _ = he.find_rigid_movement_opencv_with_our_ransac(
            blurred_frames[i-1],
            blurred_frames[i]
        )
        if H is None:
            H = np.eye(3)
        transforms.append(H)
    
    # Accumulate transforms
    accumulated = [np.eye(3)]
    total = np.eye(3)
    for i in range(1, len(transforms)):
        total = np.dot(transforms[i], total)
        accumulated.append(total.copy())
    
    # Extract dx values from accumulated transforms
    dx_accumulated = [accumulated[i][0, 2] for i in range(len(accumulated))]
    
    # Compute frame-to-frame changes from accumulated
    changes_from_accumulated = [0]
    for i in range(1, len(dx_accumulated)):
        changes_from_accumulated.append(dx_accumulated[i] - dx_accumulated[i-1])
    
    print("\nComparison of PAIRWISE vs ACCUMULATED frame-to-frame tx:")
    print("-"*70)
    for frame1_idx, frame2_idx, description in test_pairs:
        print(f"\n{description} (frames {frame1_idx} → {frame2_idx}):")
        
        H_pair, inliers_pair = pairwise_results[(frame1_idx, frame2_idx)]
        if H_pair is not None:
            tx_direct = H_pair[0, 2]
            print(f"  Direct pairwise tx:        {tx_direct:7.2f} px (inliers: {inliers_pair})")
        
        # For accumulated, we need the individual step
        tx_accum_step = transforms[frame2_idx][0, 2]
        print(f"  Pairwise in chain:         {tx_accum_step:7.2f} px")
        print(f"  Change from accumulated:   {changes_from_accumulated[frame2_idx]:7.2f} px")
        
        if H_pair is not None:
            diff = abs(tx_direct - tx_accum_step)
            print(f"  Difference:                {diff:7.2f} px {'✓ MATCH' if diff < 0.1 else '✗ MISMATCH'}")
    
    # Special focus on frame 432
    print("\n" + "="*70)
    print("5. DETAILED ANALYSIS of Frame 432 (tx = -79.21):")
    print("="*70)
    
    print("\nDirect computation frame 431→432:")
    H_direct, inliers = test_frame_pair(blurred_frames[431], blurred_frames[432], 432, "Direct")
    
    print("\nMulti-step accumulation 0→431→432:")
    print(f"  Accumulated tx at frame 431: {dx_accumulated[431]:7.2f}")
    print(f"  Accumulated tx at frame 432: {dx_accumulated[432]:7.2f}")
    print(f"  Difference (frame-to-frame): {changes_from_accumulated[432]:7.2f}")
    
    print("\nPairwise transform 431→432 (from chain):")
    H_chain = transforms[432]
    print(f"  tx from chain: {H_chain[0, 2]:7.2f}")
    print(f"  ty from chain: {H_chain[1, 2]:7.2f}")
    print(f"  theta from chain: {np.arctan2(H_chain[1, 0], H_chain[0, 0]) * 180/np.pi:6.2f}°")
    
    # Visual similarity check
    print("\n" + "="*70)
    print("6. Visual similarity check (frames 431 vs 432):")
    print("="*70)
    
    diff = cv2.absdiff(blurred_frames[431], blurred_frames[432])
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    
    print(f"  Mean pixel difference: {mean_diff:.2f} (out of 255)")
    print(f"  Max pixel difference:  {max_diff:.2f}")
    print(f"  Frames are: {'NEARLY IDENTICAL' if mean_diff < 5 else 'SIGNIFICANTLY DIFFERENT'}")
    
    # Check for possible duplicates or near-duplicates
    print("\n7. Checking for duplicate/similar frame pairs:")
    print("-"*70)
    similar_pairs = []
    for i in range(len(blurred_frames) - 1):
        diff = np.mean(cv2.absdiff(blurred_frames[i], blurred_frames[i+1]))
        if diff < 2.0:  # Very similar
            similar_pairs.append((i, i+1, diff, changes_from_accumulated[i+1]))
    
    print(f"  Found {len(similar_pairs)} nearly identical frame pairs:")
    for i, (idx1, idx2, diff, tx) in enumerate(similar_pairs[:10]):
        print(f"    Frames {idx1}→{idx2}: mean_diff={diff:.2f}, tx={tx:7.2f}")
    if len(similar_pairs) > 10:
        print(f"    ... and {len(similar_pairs)-10} more")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print("If pairwise and accumulated tx values MATCH:")
    print("  → Problem is in HOMOGRAPHY EVALUATION (bad feature matching)")
    print("If pairwise and accumulated tx values DIFFER:")
    print("  → Problem is in ACCUMULATION logic")
    print("="*70)


if __name__ == "__main__":
    main()
