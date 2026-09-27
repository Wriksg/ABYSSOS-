import numpy as np
from pathlib import Path
try:
    from scipy.stats import spearmanr
except ImportError:
    print("Please run: pip install scipy")
    exit()

def diagnose_continuous_signal(dump_dir="abyssos_calibration_data"):
    dump_dir = Path(dump_dir)
    files = [f for f in dump_dir.glob("*.npz") if "smoke_test" not in f.name]
    
    if not files:
        print("❌ No real dumps found.")
        return
        
    cs, vs, errs = [], [], []
    for f in files:
        d = np.load(f)
        cs.append(d['c'].flatten())
        vs.append(d['v'].flatten())
        errs.append(d['err'].flatten())

    c = np.concatenate(cs)
    v = np.concatenate(vs)
    err = np.concatenate(errs)

    print(f"📊 Loaded {len(files)} real tile(s) | Total Pixels: {len(err)}")
    print(f"Err Stats -> Mean: {err.mean():.4f}, Std: {err.std():.4f}, Min: {err.min():.4f}, Max: {err.max():.4f}\n")

    # 1. SPEARMAN CORRELATION (Continuous relationship)
    # This proves if the signal exists independently of thresholds
    rho_c, p_c = spearmanr(c, err)
    rho_v, p_v = spearmanr(v, err)
    
    print("🔍 CONTINUOUS CORRELATION (Spearman Rank):")
    print(f"  Consistency (c) vs Error : {rho_c:.4f}")
    print(f"  Variance (v) vs Error    : {rho_v:.4f}")
    
    if rho_c > 0.3 or rho_v > 0.3:
        print("  -> ✅ SIGNAL DETECTED. The Auditor works, the delta was just wrong.\n")
    else:
        print("  -> ❌ NO STRONG SIGNAL. The model is guessing blindly on real data.\n")

    # 2. BINNED STATS (Monotonicity Check)
    def print_binned_stats(name, signal_arr, target_err):
        print(f"📈 Binned {name.upper()} vs Mean Error:")
        # Split into 4 quartiles based on the signal
        quantiles = np.quantile(signal_arr, [0.25, 0.50, 0.75])
        
        bins = [
            ("Low (0-25%)   ", signal_arr <= quantiles[0]),
            ("Mid-Low (25-50%)", (signal_arr > quantiles[0]) & (signal_arr <= quantiles[1])),
            ("Mid-High(50-75%)", (signal_arr > quantiles[1]) & (signal_arr <= quantiles[2])),
            ("High (75-100%)", signal_arr > quantiles[2])
        ]
        
        for label, mask in bins:
            bin_err = target_err[mask].mean() if mask.sum() > 0 else 0
            print(f"   {label}: {bin_err:.4f}")

    print_binned_stats("Consistency (c)", c, err)
    print("\n")
    print_binned_stats("Variance (v)", v, err)

    # 3. DATA-DRIVEN DELTA RECOMMENDATION
    # Use the 25th percentile of the real error. 
    # This guarantees exactly 25% of pixels are "Verified" and 75% are "Hallucinations",
    # ensuring the AUROC math will always execute perfectly on this dataset.
    p25_delta = np.percentile(err, 25)
    print(f"\n💡 DATA-DRIVEN DELTA RECOMMENDATION:")
    print(f"   If you want to run calibrate.py on this exact tile right now,")
    print(f"   change delta=0.03 to delta={p25_delta:.4f}")
    print(f"   This splits the tile into the 'best 25%' vs the 'worst 75%'.")

if __name__ == "__main__":
    diagnose_continuous_signal()