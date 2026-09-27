import numpy as np
import json
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score

def calibrate(dump_dir="abyssos_calibration_data", delta=0.03, target_precision=0.95):
    dump_dir = Path(dump_dir)
    files = list(dump_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz dumps found in {dump_dir}")

    cs, vs, errs = [], [], []
    for f in files:
        d = np.load(f)
        cs.append(d['c'].flatten())
        vs.append(d['v'].flatten())
        errs.append(d['err'].flatten())

    c_all, v_all, err_all = np.concatenate(cs), np.concatenate(vs), np.concatenate(errs)

    y_true = (err_all > delta).astype(int)
    
    # ---------------------------------------------------------
    # SAFETY CHECK: DEGENERATE DISTRIBUTION
    # ---------------------------------------------------------
    unique_classes = np.unique(y_true)
    if len(unique_classes) == 1:
        print("\n" + "!"*60)
        print("🚨 [FATAL WARNING] CALIBRATION IS INVALID 🚨")
        print("!"*60)
        print(f"Reason: y_true has only one class (Value: {unique_classes[0]}).")
        print(f"Fraction of pixels with err > {delta} is {np.mean(y_true)*100:.1f}%.")
        print("This means your calibration dump has no realistic variation (likely synthetic).")
        print("DO NOT REPORT ANY AUROC NUMBERS FROM THIS RUN.")
        print("Writing fallback taus purely to prevent downstream UI crashes, but they are FAKE.")
        print("!"*60 + "\n")
        
        # Write dummy fallback to unblock Srijoni's UI, but explicitly flag it
        taus = {"tau_c": float(np.median(c_all)), "tau_v": float(np.median(v_all)), "tau_a": 0.5, "IS_VALID": False}
        os.makedirs("weights", exist_ok=True)
        with open("weights/taus.json", "w") as f:
            json.dump(taus, f, indent=4)
        return taus

    # ---------------------------------------------------------
    # VALID CALIBRATION
    # ---------------------------------------------------------
    c_norm = (c_all - c_all.min()) / (c_all.max() - c_all.min() + 1e-8)
    v_norm = (v_all - v_all.min()) / (v_all.max() - v_all.min() + 1e-8)
    
    auroc = roc_auc_score(y_true, c_norm + v_norm)
    print(f"\n📊 REAL AUROC (Hallucination Detection): {auroc:.4f}")

    best_tau_c, best_tau_v, max_cov = None, None, -1
    pcts = np.linspace(10, 90, 20)
    
    for tc in np.percentile(c_all, pcts):
        for tv in np.percentile(v_all, pcts):
            mask = (c_all < tc) & (v_all < tv)
            cov = mask.sum()
            if cov == 0: continue
            
            precision = (err_all[mask] < delta).sum() / cov
            if precision >= target_precision and cov > max_cov:
                max_cov, best_tau_c, best_tau_v = cov, tc, tv

    if best_tau_c is None:
        print("⚠️ Warning: Could not achieve 95% precision. Falling back to medians.")
        best_tau_c, best_tau_v = np.median(c_all), np.median(v_all)

    taus = {"tau_c": float(best_tau_c), "tau_v": float(best_tau_v), "tau_a": 0.5, "IS_VALID": True}
    os.makedirs("weights", exist_ok=True)
    with open("weights/taus.json", "w") as f:
        json.dump(taus, f, indent=4)
        
    print(f"✅ Calibration saved: {taus}")
    return taus