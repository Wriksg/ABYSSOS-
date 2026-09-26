import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def export_for_ui(npz_path, out_dir="ui_layers"):
    """
    Converts the raw .npz Truth Auditor dumps into ready-to-use PNG layers
    for Srijoni's Streamlit Analyst Console.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Unpacking {npz_path} for UI...")
    data = np.load(npz_path)
    
    sr = data['sr']  # (4, H, W)
    c = data['c']    # Consistency
    v = data['v']    # Variance Disagreement
    
    # 1. EXPORT THE SUPER-RESOLVED SATELLITE IMAGE (RGB)
    # Extract RGB bands (0, 1, 2) and normalize to 0-255 for PNG
    rgb = sr[:3].transpose(1, 2, 0)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    
    sr_path = out / "1_super_resolved_output.png"
    plt.imsave(sr_path, rgb)
    print(f"  -> Saved SR Image: {sr_path}")

    # 2. EXPORT THE EVIDENCE MAP (The "Phantom" UI)
    # Rules based on the calibration dump:
    # - If Variance (v) is high, the model is hallucinating -> PRIOR-ONLY (RED)
    # - If Consistency (c) is good -> VERIFIED (GREEN)
    
    # Normalize c and v for visualization
    c_norm = np.clip(c / np.percentile(c, 95), 0, 1).squeeze()
    v_norm = np.clip(v / np.percentile(v, 95), 0, 1).squeeze()
    
    H, W = c_norm.shape
    evidence_map = np.zeros((H, W, 4), dtype=np.float32) # RGBA
    
    for i in range(H):
        for j in range(W):
            if v_norm[i, j] > 0.6:  # High disagreement = Hallucination
                evidence_map[i, j] = [1.0, 0.0, 0.0, 0.6] # Red overlay (PRIOR-ONLY)
            elif c_norm[i, j] < 0.3: # Low error = Verified Physics
                evidence_map[i, j] = [0.0, 1.0, 0.0, 0.4] # Green overlay (VERIFIED)
            else:
                evidence_map[i, j] = [0.0, 0.0, 0.0, 0.0] # Transparent (Normal)

    ev_path = out / "2_evidence_overlay.png"
    plt.imsave(ev_path, evidence_map)
    print(f"  -> Saved Evidence UI Overlay: {ev_path}")
    print("\n✅ Hand-off ready! Srijoni can now drop these directly into Streamlit.")

if __name__ == "__main__":
    # Test it on the dump you made earlier!
    test_file = "abyssos_calibration_data/test_tile.npz"
    if os.path.exists(test_file):
        export_for_ui(test_file)
    else:
        print("Run `python scripts/dump_for_calibration.py` first to generate the test file!")