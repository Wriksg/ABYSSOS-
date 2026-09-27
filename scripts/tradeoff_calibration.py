import os
import sys
import torch
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model.mfsr import MFSR
from core.model.degrade import degrade
from core.audit.auditor import Tile, audit

def run_tradeoff_calibration():
    print("🚀 --- TRUTH AUDITOR: PRECISION-COVERAGE TRADEOFF ---")
    
    device = torch.device("cpu")
    model = MFSR(num_optical_bands=4, num_sar_bands=2, scale=4).to(device)
    model.load_state_dict(torch.load("abyssos_weights/best.pt", map_location=device))
    model.eval()

    files = list(Path("abyssos_data/train").glob("*.npz"))[:10]
    n_tiles = len(files)
    if n_tiles == 0:
        print("❌ No real tiles found.")
        return
        
    print(f"📂 Processing {n_tiles} bug-fixed real tiles...\n")
    
    all_c, all_v, all_err = [], [], []
    
    with torch.no_grad():
        for f in files:
            data = np.load(f)
            lr = torch.from_numpy(data['lrs']).unsqueeze(0)
            hr = torch.from_numpy(data['hr'])
            tile = Tile(
                lr=lr, masks=torch.from_numpy(data['masks']).unsqueeze(0),
                shifts=torch.from_numpy(data['shifts']).unsqueeze(0),
                sar=torch.from_numpy(data['sar']).unsqueeze(0),
                clear_fraction=torch.from_numpy(data['clear_fraction']).unsqueeze(0), meta={}
            )
            ar = audit(model, tile, scale=4)
            
            # Radiometric Anchoring
            lr_ref = lr[0, 0]
            sr_corrected = ar.sr - ar.sr.mean(dim=(-2,-1), keepdim=True) + lr_ref.mean(dim=(-2,-1), keepdim=True)
            
            # Strict Shape Alignment
            err = (sr_corrected - hr).abs().mean(dim=0).squeeze()
            
            res_list = []
            for k in range(tile.lr.shape[1]):
                lr_est = degrade(sr_corrected.unsqueeze(0), tile.shifts[:, k], scale=4)
                diff = (lr_est - tile.lr[:, k]).abs().mean(dim=1, keepdim=True)
                res_list.append(diff)
            
            res_stack = torch.stack(res_list, dim=1)
            c_lr = torch.nanmedian(res_stack, dim=1).values
            c = torch.nn.functional.interpolate(c_lr, scale_factor=4, mode='nearest').squeeze()
            v = ar.v.squeeze()
            
            all_err.append(err.numpy().flatten())
            all_c.append(c.numpy().flatten())
            all_v.append(v.numpy().flatten())

    err_flat = np.concatenate(all_err)
    c_flat = np.concatenate(all_c)
    v_flat = np.concatenate(all_v)
    total_pixels = len(err_flat)

    # Calculate AUROC & Delta
    delta = np.percentile(err_flat, 25)
    y_true = (err_flat > delta).astype(int)
    c_norm = (c_flat - c_flat.min()) / (c_flat.max() - c_flat.min() + 1e-8)
    v_norm = (v_flat - v_flat.min()) / (v_flat.max() - v_flat.min() + 1e-8)
    auroc = roc_auc_score(y_true, c_norm + v_norm)

    print(f"📊 REAL AUROC: {auroc:.4f}")
    print(f"   (Baseline Random Precision: 25.0% due to 25th percentile delta)\n")

    # PRECOMPUTE ENTIRE GRID
    grid_results = []
    pcts = np.linspace(5, 95, 30)
    for tc in np.percentile(c_flat, pcts):
        for tv in np.percentile(v_flat, pcts):
            mask = (c_flat < tc) & (v_flat < tv)
            cov = mask.sum()
            if cov == 0: continue
            prec = (err_flat[mask] < delta).sum() / cov
            grid_results.append({'tc': tc, 'tv': tv, 'cov': cov, 'cov_pct': (cov/total_pixels)*100, 'prec_pct': prec*100})

    # 1. GENERATE TRADEOFF CURVE
    targets = [99.0, 95.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0]
    print("="*55)
    print(" 📈 PRECISION vs. COVERAGE TRADEOFF CURVE ")
    print("="*55)
    print(f"{'Target Prec':<15} | {'Actual Prec':<15} | {'Coverage %':<15}")
    print("-" * 55)
    
    for t in targets:
        valid = [r for r in grid_results if r['prec_pct'] >= t]
        if valid:
            best = max(valid, key=lambda x: x['cov_pct'])
            print(f" >= {t:>5.1f}%      |    {best['prec_pct']:>6.2f}%       |    {best['cov_pct']:>6.2f}%")
        else:
            print(f" >= {t:>5.1f}%      |    Not Achievable   |    ---")

    # 2. SELECT OPTIMAL OPERATING POINT
    # Find highest precision where coverage is meaningful (>15%)
    min_coverage_pct = 15.0
    viable_points = [r for r in grid_results if r['cov_pct'] >= min_coverage_pct]
    
    if viable_points:
        op = max(viable_points, key=lambda x: x['prec_pct'])
    else:
        # Fallback to max precision available if 15% coverage is strictly impossible
        op = max(grid_results, key=lambda x: x['prec_pct'])

    print("\n" + "="*55)
    print(" 🎯 SELECTED OPERATING POINT ")
    print("="*55)
    print(f" Opted for highest precision maintaining >15% coverage.")
    print(f" -> Precision Achieved : {op['prec_pct']:>6.2f}% (vs 25% baseline)")
    print(f" -> Coverage Achieved  : {op['cov_pct']:>6.2f}% of pixels")
    print(f" -> Selected tau_c     : {op['tc']:.5f}")
    print(f" -> Selected tau_v     : {op['tv']:.5f}")

    # 3. SAVE RIGOROUS JSON
    taus = {
        "tau_c": float(op['tc']),
        "tau_v": float(op['tv']),
        "tau_a": 0.5,
        "metrics": {
            "real_auroc": float(auroc),
            "calibrated_precision_pct": float(op['prec_pct']),
            "calibrated_coverage_pct": float(op['cov_pct']),
            "n_tiles_used": n_tiles,
            "baseline_precision_pct": 25.0
        },
        "warning": "Calibrated at constrained precision due to synthetic domain gap."
    }
    
    os.makedirs("weights", exist_ok=True)
    out_file = "weights/taus.json"
    with open(out_file, "w") as f:
        json.dump(taus, f, indent=4)
        
    print(f"\n💾 Saved rigorously labeled thresholds to: {out_file}")

if __name__ == "__main__":
    run_tradeoff_calibration()