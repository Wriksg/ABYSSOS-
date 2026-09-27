import numpy as np
import json
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score

def run_real_calibration(dump_dir="abyssos_calibration_data", delta=0.03, target_precision=0.95):
    dump_dir = Path(dump_dir)
    
    # 1. Load ONLY real data (Explicitly block the smoke test)
    files = [f for f in dump_dir.glob("*.npz") if "smoke_test" not in f.name]
    
    if not files:
        print(f"❌ No real .npz dumps found in {dump_dir}.")
        return
        
    print(f"📂 Loaded {len(files)} real dump file(s).")
    
    cs, vs, errs = [], [], []
    for f in files:
        d = np.load(f)
        cs.append(d['c'].flatten())
        vs.append(d['v'].flatten())
        errs.append(d['err'].flatten())

    c_all = np.concatenate(cs)
    v_all = np.concatenate(vs)
    err_all = np.concatenate(errs)
    total_pixels = len(err_all)

    print(f"📈 Real Err Stats - Mean: {err_all.mean():.4f}, Std: {err_all.std():.4f}")
    
    # 2. Check the Distribution
    y_true = (err_all > delta).astype(int)
    fail_rate = y_true.mean() * 100
    print(f"⚠️ Pixels exceeding delta={delta}: {fail_rate:.2f}%")
    
    if fail_rate == 0 or fail_rate == 100:
        print("❌ FATAL: Distribution is degenerate. Cannot compute AUROC.")
        return

    # 3. Compute REAL AUROC
    c_norm = (c_all - c_all.min()) / (c_all.max() - c_all.min() + 1e-8)
    v_norm = (v_all - v_all.min()) / (v_all.max() - v_all.min() + 1e-8)
    
    auroc = roc_auc_score(y_true, c_norm + v_norm)
    
    print("\n" + "="*55)
    print(f" 📊 REAL AUROC (on actual model output): {auroc:.4f} ")
    print("="*55)

    # 4. Grid Search for >= 95% Precision
    best_tau_c, best_tau_v, max_cov = None, None, -1
    pcts = np.linspace(10, 90, 20)
    
    print(f"\n🔍 Searching for thresholds (Target Precision: {target_precision*100}%)...")
    for tc in np.percentile(c_all, pcts):
        for tv in np.percentile(v_all, pcts):
            mask = (c_all < tc) & (v_all < tv)
            cov = mask.sum()
            if cov == 0: continue
            
            # Precision: Of the pixels we claim are VERIFIED, how many actually have err < delta?
            precision = (err_all[mask] < delta).sum() / cov
            if precision >= target_precision and cov > max_cov:
                max_cov = cov
                best_tau_c = tc
                best_tau_v = tv

    coverage_pct = (max_cov / total_pixels) * 100 if max_cov > 0 else 0.0

    if best_tau_c is None:
        print("\n⚠️ FAILED to find thresholds that achieve 95% precision.")
        print("Fallback to median (THIS WILL NOT HIT 95% PRECISION).")
        best_tau_c, best_tau_v = np.median(c_all), np.median(v_all)
        mask = (c_all < best_tau_c) & (v_all < best_tau_v)
        actual_precision = (err_all[mask] < delta).sum() / mask.sum() if mask.sum() > 0 else 0
        coverage_pct = (mask.sum() / total_pixels) * 100
        print(f"Fallback Coverage: {coverage_pct:.2f}%, Fallback Precision: {actual_precision*100:.2f}%")
    else:
        print(f"✅ Found Thresholds: tau_c = {best_tau_c:.5f}, tau_v = {best_tau_v:.5f}")
        print(f"✅ VERIFIED Coverage: {coverage_pct:.2f}% of pixels")
        
        if coverage_pct < 5.0:
            print("\n🚨 RED FLAG: Coverage is tiny (< 5%).")
            print("   The thresholds are technically valid (they achieve 95% precision),")
            print("   but they verify almost nothing. This is practically useless for the UI demo.")
            print("   You may need to lower target_precision (e.g. 0.85) or raise delta.")

    # 5. Save separately to avoid overwriting/mixing with smoke tests
    taus = {
        "tau_c": float(best_tau_c), 
        "tau_v": float(best_tau_v), 
        "tau_a": 0.5, 
        "IS_VALID": True, 
        "real_auroc": float(auroc), 
        "coverage_pct": float(coverage_pct)
    }
    
    os.makedirs("weights", exist_ok=True)
    out_file = "weights/real_taus.json"
    with open(out_file, "w") as f:
        json.dump(taus, f, indent=4)
        
    print(f"\n💾 Saved REAL thresholds safely to: {out_file}")

if __name__ == "__main__":
    run_real_calibration()