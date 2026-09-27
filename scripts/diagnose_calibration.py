import numpy as np
from pathlib import Path

def diagnose(dump_dir="abyssos_calibration_data", delta=0.03):
    dump_dir = Path(dump_dir)
    files = list(dump_dir.glob("*.npz"))
    if not files:
        print("❌ No dump files found.")
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

    print("\n📊 --- CALIBRATION DUMP DIAGNOSTICS ---")
    
    def print_stats(name, arr):
        print(f"{name:4s} | Min: {arr.min():.4f} | Max: {arr.max():.4f} | Mean: {arr.mean():.4f} | Std: {arr.std():.4f}")
        print(f"       Percentiles -> 10th: {np.percentile(arr, 10):.4f} | 50th: {np.percentile(arr, 50):.4f} | 90th: {np.percentile(arr, 90):.4f} | 99th: {np.percentile(arr, 99):.4f}")

    print_stats("err", err)
    print_stats("c", c)
    print_stats("v", v)

    # Check the Delta threshold
    frac_failed = (err > delta).mean() * 100
    print("\n🚨 --- DEGENERATE LABEL CHECK ---")
    print(f"Target Delta Threshold: {delta}")
    print(f"Pixels exceeding Delta: {frac_failed:.2f}%")
    
    if frac_failed == 0.0 or frac_failed == 100.0:
        print("\n❌ FATAL: You have a degenerate distribution (Only 1 class).")
        print("AUROC cannot be calculated. Your threshold calibration will be meaningless.")
    else:
        print("\n✅ SUCCESS: You have a mix of classes. Calibration is valid.")

if __name__ == "__main__":
    diagnose()