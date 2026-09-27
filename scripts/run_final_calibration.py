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

def run_final_calibration(target_precision=0.95):
    print("🚀 --- FINAL TRUTH AUDITOR CALIBRATION ---")
    
    device = torch.device("cpu")
    model = MFSR(num_optical_bands=4, num_sar_bands=2, scale=4).to(device)
    model.load_state_dict(torch.load("abyssos_weights/best.pt", map_location=device))
    model.eval()

    files = list(Path("abyssos_data/train").glob("*.npz"))[:10]
    print(f"📂 Processing {len(files)} bug-fixed tiles...")
    
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
            
            # Radiometric Anchoring (Fixing the scale bug)
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

    err_flat, c_flat, v_flat = np.concatenate(all_err), np.concatenate(all_c), np.concatenate(all_v)
    total_pixels = len(err_flat)

    print("\n" + "="*55)
    print(" 1. ERROR DISTRIBUTION & DELTA SELECTION ")
    print("="*55)
    print(f"Err Stats -> Mean: {err_flat.mean():.4f}, Std: {err_flat.std():.4f}")
    
    # Select delta dynamically. 0.03 is arbitrary. 
    # We define the "top 25% most accurate pixels" as our ground truth threshold.
    delta = np.percentile(err_flat, 25)
    print(f"Selected Delta (25th percentile of error): {delta:.4f}")
    print(f"(Pixels with error < {delta:.4f} are considered physically accurate)")

    print("\n" + "="*55)
    print(" 2. AUROC CALCULATION ")
    print("="*55)
    y_true = (err_flat > delta).astype(int)
    c_norm = (c_flat - c_flat.min()) / (c_flat.max() - c_flat.min() + 1e-8)
    v_norm = (v_flat - v_flat.min()) / (v_flat.max() - v_flat.min() + 1e-8)
    
    auroc = roc_auc_score(y_true, c_norm + v_norm)
    print(f"REAL AUROC (on actual model output): {auroc:.4f}")

    print("\n" + "="*55)
    print(" 3. GRID SEARCH (Target Precision: 95%) ")
    print("="*55)
    best_tau_c, best_tau_v, max_cov = None, None, -1
    pcts = np.linspace(10, 90, 20)
    
    for tc in np.percentile(c_flat, pcts):
        for tv in np.percentile(v_flat, pcts):
            mask = (c_flat < tc) & (v_flat < tv)
            cov = mask.sum()
            if cov == 0: continue
            
            precision = (err_flat[mask] < delta).sum() / cov
            if precision >= target_precision and cov > max_cov:
                max_cov = cov
                best_tau_c = tc
                best_tau_v = tv

    coverage_pct = (max_cov / total_pixels) * 100 if max_cov > 0 else 0.0

    if best_tau_c is None:
        print("⚠️ FAILED to reach 95% precision. Falling back to median.")
        best_tau_c, best_tau_v = np.median(c_flat), np.median(v_flat)
        mask = (c_flat < best_tau_c) & (v_flat < best_tau_v)
        actual_precision = (err_flat[mask] < delta).sum() / mask.sum()
        coverage_pct = (mask.sum() / total_pixels) * 100
        print(f"Fallback Coverage: {coverage_pct:.2f}% | Precision: {actual_precision*100:.2f}%")
    else:
        print(f"✅ Found Thresholds -> tau_c: {best_tau_c:.5f}, tau_v: {best_tau_v:.5f}")
        print(f"✅ VERIFIED Coverage: {coverage_pct:.2f}% of pixels")
        if coverage_pct < 5.0:
            print("\n🚨 RED FLAG: Coverage is tiny (< 5%).")
            print("The thresholds are technically valid, but they verify almost nothing.")
            print("Report this honestly: 'To maintain 95% certainty, we had to be highly aggressive.'")

    print("\n" + "="*55)
    print(" ⚠️ SYSTEM CAVEAT FOR DOCUMENTATION ⚠️")
    print(" Calibrated on 10 tiles — expand to 20+ for production-grade stability.")
    print("="*55)

    taus = {"tau_c": float(best_tau_c), "tau_v": float(best_tau_v), "tau_a": 0.5, "real_auroc": float(auroc)}
    os.makedirs("weights", exist_ok=True)
    with open("weights/real_taus.json", "w") as f:
        json.dump(taus, f, indent=4)

if __name__ == "__main__":
    run_final_calibration()