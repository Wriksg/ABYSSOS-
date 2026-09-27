import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model.mfsr import MFSR
from core.audit.auditor import Tile, audit

def run_bug_hunt():
    print("🕵️  INITIATING TRUTH AUDITOR BUG HUNT...\n")

    # Load the trained model
    device = torch.device("cpu") # Keep it simple for diagnostics
    model = MFSR(num_optical_bands=4, num_sar_bands=2, scale=4).to(device)
    
    weights_path = "abyssos_weights/best.pt"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
    else:
        print(f"❌ Weights not found at {weights_path}")
        return

    # Find ONE real tile
    data_dir = Path("abyssos_data/train") # or real_eval
    real_files = list(data_dir.glob("*.npz"))
    if not real_files:
        print("❌ No real tiles found to test.")
        return
        
    f = real_files[0]
    print(f"📂 Loaded Real Tile: {f.name}")
    data = np.load(f)

    # Reconstruct the exact Tile format for the auditor
    tile = Tile(
        lr=torch.from_numpy(data['lrs']).unsqueeze(0),
        masks=torch.from_numpy(data['masks']).unsqueeze(0),
        shifts=torch.from_numpy(data['shifts']).unsqueeze(0),
        sar=torch.from_numpy(data['sar']).unsqueeze(0),
        clear_fraction=torch.from_numpy(data['clear_fraction']).unsqueeze(0),
        meta={"id": "bug_hunt_tile"}
    )
    hr = torch.from_numpy(data['hr']) # Ground Truth

    # Run Auditor
    with torch.no_grad():
        ar = audit(model, tile, scale=4)
    
    err = (ar.sr - hr).abs().mean(dim=0) # Mean across channels

    print("\n" + "="*50)
    print("1. ALIGNMENT & SHAPE VERIFICATION")
    print("="*50)
    print(f"  Ground Truth (HR) : {hr.shape}")
    print(f"  Output (SR)       : {ar.sr.shape}")
    print(f"  Error Map (err)   : {err.shape}")
    print(f"  Consistency (c)   : {ar.c.shape}")
    print(f"  Variance (v)      : {ar.v.shape}")
    
    if hr.shape[-2:] != ar.sr.shape[-2:] or err.shape != ar.c.shape:
        print("  🚨 BUG DETECTED: Shape/Grid mismatch! The arrays are not aligned.")
    else:
        print("  ✅ SHAPES ALIGNED. No coordinate frame mismatch.")

    print("\n" + "="*50)
    print("2. SCALE & UNITS VERIFICATION")
    print("="*50)
    def print_scale(name, t):
        print(f"  {name:10s} | Min: {t.min().item():.4f}, Max: {t.max().item():.4f}, Mean: {t.mean().item():.4f}")
    
    print_scale("HR (Truth)", hr)
    print_scale("SR (Model)", ar.sr)
    print_scale("err", err)
    
    # Unit Mismatch Logic
    if hr.max() > 10 and ar.sr.max() <= 1.0:
        print("  🚨 BUG DETECTED: SCALE MISMATCH! HR is in raw DN (e.g., 0-10000) but SR is 0-1 normalized.")
    elif ar.sr.max() > 10 and hr.max() <= 1.0:
        print("  🚨 BUG DETECTED: SCALE MISMATCH! SR is raw DN but HR is 0-1 normalized.")
    else:
        print("  ✅ UNITS ALIGNED. Both arrays occupy the same mathematical space.")

    print("\n" + "="*50)
    print("3. SANITY CHECK (Bright Feature Localization)")
    print("="*50)
    # Find the top 5% brightest pixels in the real HR image
    bright_threshold = torch.quantile(hr.mean(dim=0), 0.95)
    bright_mask = hr.mean(dim=0) > bright_threshold
    
    err_on_bright = err[bright_mask].mean().item()
    err_on_dark = err[~bright_mask].mean().item()
    
    print(f"  Mean Error on Brightest 5% of features: {err_on_bright:.4f}")
    print(f"  Mean Error on rest of image           : {err_on_dark:.4f}")
    
    if abs(err_on_bright - err_on_dark) < 0.01 and err.mean().item() > 0.3:
        print("  🚨 BUG DETECTED: Error is uniform across distinct physical features.")
        print("     This happens if HR is random noise, or if the images are completely misregistered (shifted massively).")
    else:
        print("  ✅ SANITY CHECK PASSED. Error reacts to physical image structure.")

    print("\n" + "="*50)
    print("4. STATISTICAL VALIDITY WARNING")
    print("="*50)
    print("  ⚠️ SAMPLE SIZE ALERT (n=1 tile)")
    print("  Regardless of the Spearman correlation on this single image, n=1 is")
    print("  statistically meaningless for global threshold calibration. A single")
    print("  cloud shadow or specific landscape type (e.g., all water) will heavily")
    print("  skew `c` and `v`. Do not deploy these thresholds to production.")
    print("  You need at minimum 10-20 diverse tiles to establish a stable AUROC.")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_bug_hunt()