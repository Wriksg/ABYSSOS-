import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model.mfsr import MFSR
from core.audit.auditor import Tile, audit

def run_auditor_test():
    print("--- STARTING WEEK 3 AUDITOR TEST ---\n")
    
    # 1. Setup Dummy Single Tile (B=1)
    K, C, H, W = 6, 4, 64, 64
    scale = 4
    
    print("Generating unbatched dummy Tile (K=6 frames)...")
    tile = Tile(
        lr=torch.rand(1, K, C, H, W),
        masks=torch.ones(1, K, 1, H, W), 
        shifts=torch.rand(1, K, 2),
        sar=torch.rand(1, 2, H, W),
        clear_fraction=torch.tensor([0.8]),
        meta={"aoi": "Kolkata", "date": "2026-05-12"}
    )
    
    # 2. Init Model
    print("Initializing MFSR Model...")
    model = MFSR(num_optical_bands=C, num_sar_bands=2, scale=scale)
    
    # 3. Run Auditor
    print("Running Truth Auditor (8 TTA passes + consistency physics)...")
    result = audit(model, tile, scale=scale)
    
    # 4. Verify the Contract (Shapes)
    print("\nVerifying Contract C Shapes:")
    print(f"  -> SR Output: {result.sr.shape} (Expected: [4, 256, 256])")
    print(f"  -> Consistency (c): {result.c.shape} (Expected: [1, 256, 256])")
    print(f"  -> Variance (v): {result.v.shape} (Expected: [1, 256, 256])")
    print(f"  -> Residual Stack (res): {result.res.shape} (Expected: [6, 1, 64, 64])")
    
    assert result.sr.shape == (C, H*scale, W*scale), "SR shape wrong"
    assert result.c.shape == (1, H*scale, W*scale), "c shape wrong"
    assert result.v.shape == (1, H*scale, W*scale), "v shape wrong"
    assert result.res.shape == (K, 1, H, W), "res shape wrong"
    
    print("\n✅ AUDITOR TEST PASSED! Srijoni is fully unblocked.")

if __name__ == "__main__":
    run_auditor_test()