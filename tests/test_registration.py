import os, sys, torch, numpy as np
from scipy.ndimage import shift
from skimage.registration import phase_cross_correlation
import torch.nn.functional as F
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.harvest.register import estimate_shifts
from core.model.degrade import degrade

def create_test_image(shape=(64, 64)):
    y, x = np.mgrid[-shape[0]//2 : shape[0]//2, -shape[1]//2 : shape[1]//2]
    return np.exp(-(x**2 + y**2) / 20.0)

def test_skimage_phase_correlation():
    print("\n--- Test 1: Base Skimage Phase Correlation ---")
    img = create_test_image((64, 64))
    dy, dx = 0.4, -0.7
    shifted_img = shift(img, shift=(dy, dx), order=3)
    
    est, _, _ = phase_cross_correlation(img, shifted_img, upsample_factor=100)
    print(f"Injected: [{dy:.4f}, {dx:.4f}] -> Recovered: [{est[0]:.4f}, {est[1]:.4f}]")
    
    # FIX: Assert est == -injected
    assert abs(est[0] - (-dy)) < 0.05, "Y shift recovery failed!"
    assert abs(est[1] - (-dx)) < 0.05, "X shift recovery failed!"
    print("✅ Base shift recovery passed.")

def test_estimate_shifts_synthetic_stack():
    print("\n--- Test 2: Synthetic Stack estimate_shifts() ---")
    K, C, H, W = 8, 4, 64, 64
    ref_idx = 0
    base_img = create_test_image((H, W))
    
    stack = np.zeros((K, C, H, W))
    masks = np.ones((K, H, W)) 
    injected_shifts = np.zeros((K, 2))
    
    for i in range(K):
        if i == ref_idx:
            stack[i, 3] = base_img 
        else:
            dy, dx = np.random.uniform(-0.5, 0.5, size=2)
            injected_shifts[i] = [dy, dx]
            stack[i, 3] = shift(base_img, shift=(dy, dx), order=3)

    est_shifts = estimate_shifts(stack, masks, ref_idx=ref_idx, band=3)
    
    # FIX: Compare est vs -injected
    errors = np.abs(est_shifts[1:] - (-injected_shifts[1:]))
    max_err = np.max(errors)
    assert max_err < 0.05, f"estimate_shifts() failed! Max error {max_err}"
    print("✅ Stack estimation passed.")

def test_shift_convention_end_to_end():
    print("\n--- Test 3: End-to-End Shift Convention Check ---")
    files = list(Path("abyssos_data/train").glob("*.npz"))
    if not files:
        print("❌ No real tiles found.")
        return
        
    data = np.load(files[0])
    np_lrs = data['lrs']
    lrs = torch.from_numpy(np_lrs).unsqueeze(0)  # (1, K, 4, H, W)
    K = lrs.shape[1]
    
    # 1. Crude SR from reference frame
    ref_idx = 0
    lr_ref = lrs[:, ref_idx]
    sr = F.interpolate(lr_ref, scale_factor=4, mode='bicubic', align_corners=False)
    
    # 2. Get real shifts from register.py
    masks = np.ones((K, np_lrs.shape[2], np_lrs.shape[3]))
    calc_shifts = estimate_shifts(np_lrs, masks, ref_idx=ref_idx, band=3)
    
    # 3. Test Native vs Flipped
    for k in range(1, min(K, 3)): # Just test 2 frames to save output space
        s_tensor = torch.from_numpy(calc_shifts[k:k+1])
        
        err_base = F.l1_loss(lr_ref, lrs[:, k]).item()
        err_native = F.l1_loss(degrade(sr, s_tensor, scale=4), lrs[:, k]).item()
        err_flipped = F.l1_loss(degrade(sr, -s_tensor, scale=4), lrs[:, k]).item()
        
        print(f"Frame {k} | Shift calculated by register.py: {calc_shifts[k]}")
        print(f"  Baseline L1 (No shift) : {err_base:.5f}")
        print(f"  L1 with Native Shift   : {err_native:.5f}")
        print(f"  L1 with Flipped Shift  : {err_flipped:.5f}")
        
        if err_native < err_flipped:
            print("  -> degrade() expects the NATIVE sign.")
        else:
            print("  -> degrade() expects the FLIPPED sign.")

if __name__ == "__main__":
    test_skimage_phase_correlation()
    test_estimate_shifts_synthetic_stack()
    test_shift_convention_end_to_end()