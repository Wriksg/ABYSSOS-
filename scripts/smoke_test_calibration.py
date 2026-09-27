import torch
import numpy as np
from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.audit.calibrate import calibrate

def generate_smoke_test_dump():
    out = Path("abyssos_calibration_data")
    out.mkdir(exist_ok=True)
    
    # 1. Base clean pixels (70% of image)
    err = np.random.uniform(0.0, 0.02, size=(256, 256))
    c = np.random.uniform(0.0, 0.05, size=(256, 256))
    v = np.random.uniform(0.0, 0.05, size=(256, 256))
    
    # 2. Inject "Hallucinations" (30% of image)
    # We purposefully make c and v higher where err is high so AUROC is detectable
    hallucination_mask = np.random.rand(256, 256) > 0.7
    err[hallucination_mask] = np.random.uniform(0.04, 0.1)  # > delta
    c[hallucination_mask] = np.random.uniform(0.08, 0.2)
    v[hallucination_mask] = np.random.uniform(0.08, 0.2)

    np.savez(out / "smoke_test.npz", err=err, c=c, v=v, sr=np.zeros((4,256,256)), sar=np.zeros((2,64,64)))
    print("✅ Created SMOKE_TEST dump with controlled 30% hallucination rate.")

if __name__ == "__main__":
    generate_smoke_test_dump()
    print("\n--- RUNNING CALIBRATE ON SMOKE TEST ---")
    calibrate(dump_dir="abyssos_calibration_data", delta=0.03)
