"""
Visual verification of TX estimation correctness for House video.
Creates diagnostic images for large TX jumps and zero-motion frames.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.video_io import load_video
from src.mosaic import to_grey_scale
from src import homography_evaluation as he
from scipy.ndimage import gaussian_filter, shift


def create_verification_image(frame1, frame2, tx, pair_idx, label, output_dir):
    """
    Create a diagnostic image showing:
    - Top: Frame1 and Frame2 side by side
    - Middle: Frame2 shifted by TX overlaid on Frame1
    - Bottom: Difference image
    """
    h, w = frame1.shape[:2]
    
    # Convert to uint8 if needed
    if frame1.dtype == np.float64:
        frame1 = (frame1 * 255).astype(np.uint8)
    if frame2.dtype == np.float64:
        frame2 = (frame2 * 255).astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'{label}\nFrame {pair_idx-1} → {pair_idx}, Estimated TX = {tx:.2f} px', 
                 fontsize=14, fontweight='bold')
    
    # Row 1: Original frames
    axes[0, 0].imshow(frame1, cmap='gray')
    axes[0, 0].set_title(f'Frame {pair_idx-1}', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(frame2, cmap='gray')
    axes[0, 1].set_title(f'Frame {pair_idx}', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Row 2: Alignment check - shift frame2 and overlay
    # Apply the estimated shift to frame2 (opposite direction to compensate for motion)
    # If TX = -79 (moved left), shift frame2 right by +79 to align
    shifted_frame2 = shift(frame2, shift=(0, -tx), order=1, mode='constant', cval=0)
    
    # Create overlay: frame1 in red, shifted_frame2 in green
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 0] = frame1  # Red channel
    overlay[:, :, 1] = shifted_frame2  # Green channel
    # Where they align, it will appear yellow/gray
    
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Overlay: Frame1(Red) + Shifted Frame2(Green)\n(Yellow/Gray = Good Alignment)', 
                        fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Show shifted frame alone
    axes[1, 1].imshow(shifted_frame2, cmap='gray')
    axes[1, 1].set_title(f'Frame {pair_idx} shifted by {-tx:.2f}px (to compensate)', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Row 3: Difference images
    diff_before = np.abs(frame1.astype(np.float32) - frame2.astype(np.float32))
    diff_after = np.abs(frame1.astype(np.float32) - shifted_frame2.astype(np.float32))
    
    im1 = axes[2, 0].imshow(diff_before, cmap='hot', vmin=0, vmax=100)
    axes[2, 0].set_title(f'Difference BEFORE shift\nMean: {np.mean(diff_before):.2f}', 
                        fontsize=11, fontweight='bold')
    axes[2, 0].axis('off')
    plt.colorbar(im1, ax=axes[2, 0], fraction=0.046)
    
    im2 = axes[2, 1].imshow(diff_after, cmap='hot', vmin=0, vmax=100)
    axes[2, 1].set_title(f'Difference AFTER shift\nMean: {np.mean(diff_after):.2f}', 
                        fontsize=11, fontweight='bold')
    axes[2, 1].axis('off')
    plt.colorbar(im2, ax=axes[2, 1], fraction=0.046)
    
    plt.tight_layout()
    
    # Save
    filename = f'{label.lower().replace(" ", "_")}_pair_{pair_idx-1}_{pair_idx}.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    # Calculate improvement metric
    improvement = (np.mean(diff_before) - np.mean(diff_after)) / np.mean(diff_before) * 100
    
    return output_path, improvement


def verify_tx_estimation(video_path: str, num_large_jumps: int = 5, num_zero_motion: int = 5):
    """
    Verify TX estimation by visual inspection of frame pairs.
    """
    print("="*60)
    print("TX ESTIMATION VERIFICATION - HOUSE VIDEO")
    print("="*60)
    
    # Create output directory
    output_dir = 'test_outputs/house_tx_verification'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load video
    frames = load_video(video_path)
    print(f"\nLoaded {len(frames)} frames")
    
    # Convert to grayscale
    print("Converting to grayscale...")
    frames_gray = np.zeros((len(frames), frames.shape[1], frames.shape[2]), dtype=np.uint8)
    blurred_frames = np.zeros_like(frames_gray)
    
    for i in range(len(frames)):
        gray = (to_grey_scale(frames[i]) * 255).astype(np.uint8)
        frames_gray[i] = gray
        blurred_frames[i] = gaussian_filter(gray, sigma=1.0).astype(np.uint8)
    
    # Compute TX for all pairs
    print(f"Computing TX for all frame pairs...")
    tx_values = []
    pair_indices = []
    
    for i in range(1, len(frames)):
        H, _ = he.find_rigid_movement_opencv_with_our_ransac(blurred_frames[i-1], blurred_frames[i])
        
        if H is not None:
            tx = H[0, 2]
            tx_values.append(tx)
            pair_indices.append(i)
    
    tx_values = np.array(tx_values)
    pair_indices = np.array(pair_indices)
    
    # ==========================================
    # STEP 1: Verify large TX jumps
    # ==========================================
    print(f"\n{'='*60}")
    print(f"STEP 1: VERIFYING LARGE TX JUMPS (Top {num_large_jumps})")
    print(f"{'='*60}\n")
    
    # Find largest jumps
    sorted_indices = np.argsort(np.abs(tx_values))[::-1]
    large_jump_results = []
    
    for rank, idx in enumerate(sorted_indices[:num_large_jumps], 1):
        pair_idx = pair_indices[idx]
        tx = tx_values[idx]
        
        print(f"[{rank}/{num_large_jumps}] Verifying Frame {pair_idx-1} → {pair_idx}: TX = {tx:.2f} px")
        
        # Create verification image
        output_path, improvement = create_verification_image(
            frames_gray[pair_idx-1], 
            frames_gray[pair_idx],
            tx, 
            pair_idx,
            f'Large Jump {rank}',
            output_dir
        )
        
        large_jump_results.append({
            'rank': rank,
            'pair': f'{pair_idx-1}→{pair_idx}',
            'tx': tx,
            'improvement': improvement,
            'file': output_path
        })
        
        print(f"   Difference improvement after shift: {improvement:.1f}%")
        print(f"   Saved to: {output_path}")
    
    # ==========================================
    # STEP 2: Verify zero-motion frames
    # ==========================================
    print(f"\n{'='*60}")
    print(f"STEP 2: VERIFYING ZERO-MOTION FRAMES (Sample {num_zero_motion})")
    print(f"{'='*60}\n")
    
    # Find near-zero motion frames
    zero_motion_indices = np.where(np.abs(tx_values) < 0.1)[0]
    print(f"Found {len(zero_motion_indices)} frame pairs with |TX| < 0.1 px")
    
    # Sample evenly across the video
    if len(zero_motion_indices) > num_zero_motion:
        sample_indices = np.linspace(0, len(zero_motion_indices)-1, num_zero_motion, dtype=int)
        zero_motion_sample = zero_motion_indices[sample_indices]
    else:
        zero_motion_sample = zero_motion_indices
    
    zero_motion_results = []
    
    for rank, idx in enumerate(zero_motion_sample, 1):
        pair_idx = pair_indices[idx]
        tx = tx_values[idx]
        
        # Check if frames are literally identical
        frame1 = frames_gray[pair_idx-1]
        frame2 = frames_gray[pair_idx]
        
        pixel_diff = np.sum(np.abs(frame1.astype(np.float32) - frame2.astype(np.float32)))
        mean_diff = pixel_diff / (frame1.shape[0] * frame1.shape[1])
        are_identical = (pixel_diff == 0)
        
        print(f"[{rank}/{num_zero_motion}] Verifying Frame {pair_idx-1} → {pair_idx}: TX = {tx:.4f} px")
        print(f"   Mean pixel difference: {mean_diff:.4f}")
        print(f"   Frames identical: {are_identical}")
        
        # Create verification image
        output_path, improvement = create_verification_image(
            frames_gray[pair_idx-1], 
            frames_gray[pair_idx],
            tx, 
            pair_idx,
            f'Zero Motion {rank}',
            output_dir
        )
        
        zero_motion_results.append({
            'rank': rank,
            'pair': f'{pair_idx-1}→{pair_idx}',
            'tx': tx,
            'mean_diff': mean_diff,
            'identical': are_identical,
            'file': output_path
        })
        
        print(f"   Saved to: {output_path}\n")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print(f"{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}\n")
    
    print("LARGE TX JUMPS:")
    print(f"  {'Rank':<6} {'Pair':<12} {'TX (px)':<10} {'Improvement':<12}")
    print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*12}")
    for r in large_jump_results:
        print(f"  {r['rank']:<6} {r['pair']:<12} {r['tx']:>8.2f}   {r['improvement']:>9.1f}%")
    
    avg_improvement = np.mean([r['improvement'] for r in large_jump_results])
    print(f"\n  Average difference improvement: {avg_improvement:.1f}%")
    
    if avg_improvement > 30:
        print(f"  ✓ Large TX estimates appear CORRECT (good alignment improvement)")
    elif avg_improvement > 10:
        print(f"  ~ Large TX estimates are REASONABLE but could be better")
    else:
        print(f"  ✗ Large TX estimates may be INCORRECT (poor alignment improvement)")
    
    print(f"\nZERO-MOTION FRAMES:")
    print(f"  {'Rank':<6} {'Pair':<12} {'TX (px)':<12} {'Mean Diff':<12} {'Identical'}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*12} {'-'*9}")
    for r in zero_motion_results:
        print(f"  {r['rank']:<6} {r['pair']:<12} {r['tx']:>10.4f}   {r['mean_diff']:>10.4f}   {r['identical']}")
    
    num_identical = sum([r['identical'] for r in zero_motion_results])
    print(f"\n  Identical frame pairs: {num_identical}/{len(zero_motion_results)}")
    
    if num_identical >= len(zero_motion_results) * 0.8:
        print(f"  ✓ Zero-motion frames are mostly DUPLICATE/FREEZE frames (real)")
    elif num_identical >= len(zero_motion_results) * 0.3:
        print(f"  ~ Zero-motion is MIXED (some real, some estimation)")
    else:
        print(f"  ✗ Zero-motion frames likely represent ESTIMATION ERRORS")
    
    print(f"\nAll verification images saved to: {output_dir}/")
    print(f"{'='*60}")
    
    return large_jump_results, zero_motion_results


if __name__ == "__main__":
    video_path = "Exercise Inputs-20251225/House.mp4"
    large_results, zero_results = verify_tx_estimation(video_path, num_large_jumps=5, num_zero_motion=5)
