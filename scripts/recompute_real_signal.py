import os
import sys
import torch
import numpy as np
from pathlib import Path
try:
    from scipy.stats import spearmanr
except ImportError:
    print("Please run: pip install scipy")
    exit()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model.mfsr import MFSR
from core.model.degrade import degrade
from core.audit.auditor import Tile, audit

def recompute_and_correlate():
    print("🔧 --- RECOMPUTING SIGNALS (BUG-FIXED) ---")
    
    device = torch.device("cpu")
    model = MFSR(num_optical_bands=4, num_sar_bands=2, scale=4).to(device)
    
    weights_path = "abyssos_weights/best.pt"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
    else:
        print("❌ Weights not found. Run training first.")
        return

    # REQUIRE n >= 5. Use the train directory to ensure we have enough tiles.
    data_dir = Path("abyssos_data/train")
    files = list(data_dir.glob("*.npz"))[:10] # Grab up to 10 tiles
    
    if len(files) < 5:
        print(f"❌ Sample size too small (n={len(files)}). Gather at least 5 tiles before running.")
        return
        
    print(f"📂 Processing {len(files)} tiles for statistical validity...\n")
    
    all_c, all_v, all_err = [], [], []
    
    with torch.no_grad():
        for i, f in enumerate(files):
            data = np.load(f)
            
            # 1. Load Data
            lr = torch.from_numpy(data['lrs']).unsqueeze(0)
            hr = torch.from_numpy(data['hr'])
            tile = Tile(
                lr=lr,
                masks=torch.from_numpy(data['masks']).unsqueeze(0),
                shifts=torch.from_numpy(data['shifts']).unsqueeze(0),
                sar=torch.from_numpy(data['sar']).unsqueeze(0),
                clear_fraction=torch.from_numpy(data['clear_fraction']).unsqueeze(0),
                meta={}
            )
            
            # 2. Run Auditor
            ar = audit(model, tile, scale=4)
            
            # 3. FIX THE SCALE BUG: Radiometric Anchoring
            # Shift SR brightness to match the reference LR frame (channel by channel)
            lr_ref = lr[0, 0] # (4, 64, 64)
            sr_mean = ar.sr.mean(dim=(-2,-1), keepdim=True)
            lr_mean = lr_ref.mean(dim=(-2,-1), keepdim=True)
            sr_corrected = ar.sr - sr_mean + lr_mean
            
            # 4. FIX THE SHAPE BUG: Strict Alignment
            # Ensure everything collapses to exactly (256, 256)
            err = (sr_corrected - hr).abs().mean(dim=0).squeeze()
            
            # Since we shifted SR, we must recompute c (Consistency) based on the corrected SR
            K = tile.lr.shape[1]
            res_list = []
            for k in range(K):
                lr_est = degrade(sr_corrected.unsqueeze(0), tile.shifts[:, k], scale=4)
                diff = (lr_est - tile.lr[:, k]).abs().mean(dim=1, keepdim=True)
                res_list.append(diff)
            
            res_stack = torch.stack(res_list, dim=1)
            c_lr = torch.nanmedian(res_stack, dim=1).values
            c = torch.nn.functional.interpolate(c_lr, scale_factor=4, mode='nearest').squeeze()
            
            # Variance (v) is immune to scale shifts, just fix the shape
            v = ar.v.squeeze()
            
            # Verify strict shape match
            assert err.shape == c.shape == v.shape == (256, 256), f"Shape mismatch: err{err.shape}, c{c.shape}, v{v.shape}"
            
            all_err.append(err.numpy().flatten())
            all_c.append(c.numpy().flatten())
            all_v.append(v.numpy().flatten())
            print(f"  -> Processed Tile {i+1} | SR Mean (Corrected): {sr_corrected.mean().item():.4f} | HR Mean: {hr.mean().item():.4f}")

    err_flat = np.concatenate(all_err)
    c_flat = np.concatenate(all_c)
    v_flat = np.concatenate(all_v)

    print("\n" + "="*50)
    print("📈 BUG-FIXED CORRELATION RESULTS")
    print("="*50)
    print(f"Dataset Size : {len(files)} tiles ({len(err_flat)} pixels)")
    print(f"Err Stats    : Mean {err_flat.mean():.4f}, Std {err_flat.std():.4f}")
    
    rho_c, _ = spearmanr(c_flat, err_flat)
    rho_v, _ = spearmanr(v_flat, err_flat)
    
    print(f"Spearman 'c' (Physics)  vs Err : {rho_c:.4f}")
    print(f"Spearman 'v' (Variance) vs Err : {rho_v:.4f}")
    print("="*50)

if __name__ == "__main__":
    recompute_and_correlate()