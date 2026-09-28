import sys, os, numpy as np
from scipy.ndimage import gaussian_filter, fourier_shift
from scipy.fft import fft2, ifft2
from skimage.registration import phase_cross_correlation
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.harvest.register import estimate_shifts_v2

def test_skimage_phase_correlation():
    print("--- Test 1: Phase Correlation Diagnostics (Analytical) ---")
    dy, dx = 0.4, -0.7
    y, x = np.mgrid[-32:32, -32:32]
    img = np.exp(-(x**2 + y**2) / 20.0)
    shifted = np.exp(-((x - dx)**2 + (y - dy)**2) / 20.0)
    
    est, _, _ = phase_cross_correlation(img, shifted, upsample_factor=100)
    print(f"Injected: [{dy:.4f}, {dx:.4f}] -> Recovered: [{-est[0]:.4f}, {-est[1]:.4f}]")
    
    assert abs(-est[0] - dy) < 0.05, f"Y shift failed! Got {-est[0]}"
    assert abs(-est[1] - dx) < 0.05, f"X shift failed! Got {-est[1]}"
    print("✅ Test 1 Strict Recovery Passed.\n")

def test_registration_real_tile():
    print("--- Test 2: V2 Joint Solve (Real Tile Synthetics) ---")
    files = list(Path("abyssos_data/tiles").glob("*.npz")) + list(Path("abyssos_data/train").glob("*.npz"))
    if not files: return
        
    data = np.load(files[0])
    # FIX: Safely handle both Harvester (lr) and Fallback (lrs) naming conventions
    lr_key = 'lr' if 'lr' in data else 'lrs'
    
    real_frame = data[lr_key][0]
    K = 4
    real_masks = np.ones((K, 64, 64))
    
    noise_std = np.std(real_frame[3] - gaussian_filter(real_frame[3], 1))
    stack, injected = np.zeros((K, 4, 64, 64)), np.zeros((K, 2))
    
    for k in range(K):
        if k == 0:
            injected[k] = [0.0, 0.0]
            shifted = real_frame
        else:
            dy, dx = np.random.uniform(-0.5, 0.5, size=2)
            injected[k] = [dy, dx]
            shifted = np.zeros_like(real_frame)
            for b in range(4):
                fft_img = fft2(real_frame[b])
                shifted_fft = fourier_shift(fft_img, shift=(dy, dx))
                shifted[b] = np.real(ifft2(shifted_fft))
                
        stack[k] = shifted + np.random.normal(0, noise_std, shifted.shape)
        
    v2_shifts, conf = estimate_shifts_v2(stack, real_masks, ref_idx=0, band=3)
    
    valid = ~np.isnan(v2_shifts).any(axis=1)
    err_v2 = np.abs(v2_shifts[valid][1:] - (-injected[valid][1:]))
    
    if len(err_v2) > 0:
        print(f"V2 Joint -> Median Err: {np.median(err_v2):.4f} px | P95 Err: {np.percentile(err_v2, 95):.4f} px")
        assert np.percentile(err_v2, 95) < 0.05, f"V2 failed to hit <0.05px p95 target! Got {np.percentile(err_v2, 95):.4f}"
        print("✅ V2 Sub-pixel Registration Self-Test Passed.\n")

if __name__ == "__main__":
    test_skimage_phase_correlation()
    test_registration_real_tile()
