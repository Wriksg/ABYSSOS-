import numpy as np
import warnings
from skimage.registration import phase_cross_correlation

def estimate_shifts(stack, masks, ref_idx, band=3):
    """
    Estimates sub-pixel shifts between frames.
    shifts[k] = shift that moves frame k onto the reference frame.
    stack: (K, 4, H, W) float array
    masks: (K, H, W) binary/bool array (1 = clear, 0 = cloud)
    ref_idx: index of the anchor frame
    band: index of the band to register on (default 3 = B8 NIR)
    """
    K, C, H, W = stack.shape
    shifts = np.full((K, 2), np.nan, dtype=np.float32)
    
    ref_img = stack[ref_idx, band] * masks[ref_idx]
    
    # Track if all calculated shifts are suspiciously small
    all_tiny = True 

    print(f"\n--- Estimating Shifts (Anchor: Frame {ref_idx}, Band {band}) ---")
    
    for i in range(K):
        if i == ref_idx:
            shifts[i] = [0.0, 0.0]
            print(f"Frame {i}: [0.0000, 0.0000] (Reference)")
            continue
            
        # Calculate overlap (intersection of clear pixels)
        overlap_mask = masks[ref_idx] * masks[i]
        overlap_fraction = overlap_mask.sum() / (H * W)
        
        if overlap_fraction < 0.4:
            print(f"Frame {i}: [NaN, NaN] (Overlap < 40%, dropping)")
            continue
            
        moving_img = stack[i, band] * masks[i]
        
        # Calculate shift (Returns dy, dx)
        shift, error, diffphase = phase_cross_correlation(
            ref_img, moving_img, upsample_factor=100
        )
        
        # Check geographic bounds
        if abs(shift[0]) > 3.0 or abs(shift[1]) > 3.0:
            print(f"Frame {i}: [NaN, NaN] (Shift {shift} exceeded 3.0px, dropping)")
            continue
            
        shifts[i] = shift
        print(f"Frame {i}: [{shift[0]:.4f}, {shift[1]:.4f}]")
        
        if abs(shift[0]) >= 0.05 or abs(shift[1]) >= 0.05:
            all_tiny = False

    if K > 1 and all_tiny:
        warnings.warn(
            "\n🚨 WARNING: All valid shifts are under 0.05 px! "
            "Phase correlation is failing and returning trivial zeros. "
            "Check if images are empty, perfectly identical, or if mask multiplication destroyed features."
        )

    return shifts