import os
import numpy as np
from pathlib import Path
from skimage.registration import phase_cross_correlation
from skimage.transform import resize
import warnings

# Suppress noisy skimage/HF warnings
warnings.filterwarnings("ignore")

try:
    from datasets import load_dataset
except ImportError:
    print("FATAL: Please install datasets package -> pip install datasets")
    exit(1)

def compute_real_shifts(lrs):
    """
    Computes real sub-pixel shifts between LR passes using Phase Cross-Correlation.
    lrs: (K, C, H, W) numpy array
    Returns: shifts (K, 2) in (dy, dx) LR pixel units.
    """
    K = lrs.shape[0]
    shifts = np.zeros((K, 2), dtype=np.float32)
    
    # We use the first pass (i=0) as the anchor/reference frame.
    # We use Band 0 (usually Red or NIR) as the correlation reference because it has high contrast.
    ref_img = lrs[0, 0] 
    
    for i in range(1, K):
        tgt_img = lrs[i, 0]
        # upsample_factor=100 matches your degrade.py test tolerance
        shift, error, diffphase = phase_cross_correlation(ref_img, tgt_img, upsample_factor=100)
        
        # phase_cross_correlation returns (dy, dx). 
        shifts[i] = shift
        
    return shifts

def main():
    out_dir = Path("abyssos_data/train")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("🌍 Connecting to WorldStrat (Streaming Mode)...")
    
    # We use streaming=True so it doesn't try to download 100GB of data.
    # We will just pull the first N items from the stream.
    try:
        # The community version of WorldStrat on HF
        dataset = load_dataset('corrupt_chowder/WorldStrat', split='train', streaming=True)
    except Exception as e:
        print(f"Dataset access error: {e}")
        print("Falling back to standard WorldStrat repository...")
        dataset = load_dataset('worldstrat/worldstrat', split='train', streaming=True)

    iterator = iter(dataset)
    
    # Configuration
    K = 4            # Number of passes to extract
    C = 4            # Optical bands (RGB + NIR)
    LR_SIZE = 64     # 64x64 LR (10m)
    SCALE = 4
    HR_SIZE = LR_SIZE * SCALE # 256x256 HR (2.5m)
    
    num_tiles_needed = 50
    tiles_saved = 0
    
    print(f"🔥 Streaming and processing {num_tiles_needed} real AOIs...")
    
    while tiles_saved < num_tiles_needed:
        try:
            item = next(iterator)
        except StopIteration:
            break
            
        # 1. EXTRACT HR (SPOT 6/7)
        # Standardize shape to (C, H, W). WorldStrat format can vary, so we safely resize.
        hr_image = np.array(item['hr']) # Usually (H, W, C)
        if hr_image.shape[-1] > C:
            hr_image = hr_image.transpose(2, 0, 1)[:C]
        elif hr_image.ndim == 3:
            hr_image = hr_image.transpose(2, 0, 1)
            
        # Ensure exact 256x256 shape
        hr_tensor = np.zeros((C, HR_SIZE, HR_SIZE), dtype=np.float32)
        for c in range(min(C, hr_image.shape[0])):
            hr_tensor[c] = resize(hr_image[c], (HR_SIZE, HR_SIZE), anti_aliasing=True)

        # 2. EXTRACT LR (Sentinel-2 Temporal Stack)
        # WorldStrat has multiple temporal revisits. We slice out K passes.
        lr_stack = np.array(item['lr']) # Typically (Temporal, H, W, C)
        if lr_stack.shape[0] < K:
            continue # Skip AOIs that don't have enough clear passes
            
        lrs_tensor = np.zeros((K, C, LR_SIZE, LR_SIZE), dtype=np.float32)
        for k in range(K):
            frame = lr_stack[k]
            if frame.ndim == 3:
                frame = frame.transpose(2, 0, 1) # to (C, H, W)
            for c in range(min(C, frame.shape[0])):
                lrs_tensor[k, c] = resize(frame[c], (LR_SIZE, LR_SIZE), anti_aliasing=True)

        # 3. COMPUTE SHIFTS
        # This is where your physics model gets its real labels!
        shifts_tensor = compute_real_shifts(lrs_tensor)

        # 4. MASKS & SAR (Mocking for now to unblock training)
        # Masks: Assume 100% clear for these selected HR pairs.
        masks_tensor = np.ones((K, 1, LR_SIZE, LR_SIZE), dtype=np.float32)
        clear_fraction = np.array([1.0], dtype=np.float32)
        
        # SAR: Backend-1 hasn't delivered radar yet. Injecting Sentinel-1 style speckle noise.
        # Mean -10dB, var 2dB (typical C-band land response)
        sar_tensor = np.random.normal(loc=-10.0, scale=2.0, size=(2, LR_SIZE, LR_SIZE)).astype(np.float32)

        # 5. NORMALIZE OPTICAL DATA (0 to 1 range)
        lrs_tensor = np.clip(lrs_tensor / 10000.0, 0, 1) if lrs_tensor.max() > 10 else lrs_tensor
        hr_tensor = np.clip(hr_tensor / 10000.0, 0, 1) if hr_tensor.max() > 10 else hr_tensor

        # 6. SAVE TO DISK
        np.savez(
            out_dir / f"worldstrat_{tiles_saved:03d}.npz",
            lrs=lrs_tensor,
            masks=masks_tensor,
            shifts=shifts_tensor,
            sar=sar_tensor,
            clear_fraction=clear_fraction,
            hr=hr_tensor
        )
        
        tiles_saved += 1
        print(f"  -> Saved Tile {tiles_saved}/{num_tiles_needed} (Computed shifts: max dx/dy = {np.abs(shifts_tensor).max():.2f} px)")

    print(f"\n✅ Unblocked! {tiles_saved} real multi-frame tiles saved to {out_dir}")

if __name__ == "__main__":
    main()