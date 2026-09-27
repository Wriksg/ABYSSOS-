import numpy as np
import json
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score

def calibrate(dump_dir="abyssos_calibration_data", delta=0.03, target_precision=0.95):
    """
    Grid-searches thresholds to ensure 95% of pixels labeled 'VERIFIED' 
    are actually physically accurate (err < delta).
    """
    dump_dir = Path(dump_dir)
    files = list(dump_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz dumps found in {dump_dir}.")

    cs, vs, errs = [], [], []
    for f in files:
        d = np.load(f)
        cs.append(d['c'].flatten())
        vs.append(d['v'].flatten())
        errs.append(d['err'].flatten())

    c_all = np.concatenate(cs)
    v_all = np.concatenate(vs)
    err_all = np.concatenate(errs)

    # Calculate REAL AUROC (Hallucination detection capability)
    # y_true = 1 if the model hallucinated/failed (err > delta)
    y_true = (err_all > delta).astype(int)
    
    c_norm = (c_all - c_all.min()) / (c_all.max() - c_all.min() + 1e-8)
    v_norm = (v_all - v_all.min()) / (v_all.max() - v_all.min() + 1e-8)
    score = c_norm + v_norm # High score = likely hallucination

    try:
        auroc = roc_auc_score(y_true, score)
        print(f"📊 REAL AUROC (Hallucination Detection Score): {auroc:.4f}")
    except ValueError:
        print("⚠️ AUROC skipped (Only one class present in this batch).")

    # Grid Search for taus
    best_tau_c, best_tau_v, max_coverage = None, None, -1
    percentiles = np.linspace(10, 90, 20)
    
    for tc in np.percentile(c_all, percentiles):
        for tv in np.percentile(v_all, percentiles):
            # How many pixels does this threshold claim are VERIFIED?
            verified_mask = (c_all < tc) & (v_all < tv)
            coverage = verified_mask.sum()
            if coverage == 0: continue
            
            # Of those, how many actually had an error lower than delta?
            correct = (err_all[verified_mask] < delta).sum()
            precision = correct / coverage
            
            if precision >= target_precision and coverage > max_coverage:
                max_coverage = coverage
                best_tau_c, best_tau_v = tc, tv

    if best_tau_c is None:
        print("⚠️ Warning: Could not achieve 95% precision. Falling back to safe medians.")
        best_tau_c, best_tau_v = np.median(c_all), np.median(v_all)

    taus = {"tau_c": float(best_tau_c), "tau_v": float(best_tau_v), "tau_a": 0.5}
    os.makedirs("weights", exist_ok=True)
    with open("weights/taus.json", "w") as f:
        json.dump(taus, f, indent=4)
        
    print(f"✅ Calibration saved to weights/taus.json: {taus}")
    return taus