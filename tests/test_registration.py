import sys, os, numpy as np
import scipy
from scipy.ndimage import shift, gaussian_filter, sobel
import skimage
from skimage.registration import phase_cross_correlation
from skimage.filters import window
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def sobel_mag(img):
    return np.hypot(sobel(img, axis=1), sobel(img, axis=0))

print("--- DIAGNOSTIC ENVIRONMENT ---")
print(f"Python: {sys.version.split()[0]}")
print(f"NumPy: {np.__version__} | SciPy: {scipy.__version__} | skimage: {skimage.__version__}\n")

def analyze_and_test(name, img, dy, dx):
    grad_y = np.mean(np.abs(np.diff(img, axis=0)))
    grad_x = np.mean(np.abs(np.diff(img, axis=1)))
    
    print(f"[{name}]")
    print(f"  Shape: {img.shape}, Dtype: {img.dtype}, Std: {img.std():.4f}")
    print(f"  Mean Abs Grad Y: {grad_y:.4f} | X: {grad_x:.4f}")
    
    shifted = shift(img, (dy, dx), order=3, mode="reflect")
    
    # Base
    est_base, _, _ = phase_cross_correlation(img, shifted, upsample_factor=100)
    
    # Variant 1: Crop 16px
    c = 16
    img_c, sft_c = img[c:-c, c:-c], shifted[c:-c, c:-c]
    img_c_sub, sft_c_sub = img_c - img_c.mean(), sft_c - sft_c.mean()
    est_crop, _, _ = phase_cross_correlation(img_c_sub, sft_c_sub, upsample_factor=100)
    
    # Variant 2: Hann Window
    hann = window('hann', img.shape)
    img_h, sft_h = (img - img.mean()) * hann, (shifted - shifted.mean()) * hann
    est_hann, _, _ = phase_cross_correlation(img_h, sft_h, upsample_factor=100)
    
    print(f"  Injected   : [{dy:.4f}, {dx:.4f}]")
    print(f"  Base Recov : [{-est_base[0]:.4f}, {-est_base[1]:.4f}]")
    print(f"  Crop Recov : [{-est_crop[0]:.4f}, {-est_crop[1]:.4f}]")
    print(f"  Hann Recov : [{-est_hann[0]:.4f}, {-est_hann[1]:.4f}]\n")
    
    return -est_crop[0], -est_crop[1]

def test_skimage_phase_correlation():
    print("--- Test 1: Phase Correlation Diagnostics ---")
    dy, dx = 0.4, -0.7
    
    # Image A: Textured Synthetic
    img_a = gaussian_filter(np.random.rand(256, 256), 2)
    est_y, est_x = analyze_and_test("A. Textured Synthetic", img_a, dy, dx)
    
    # Strict assertion on Image A using the Crop method
    assert abs(est_y - dy) < 0.05, f"Y shift failed! Got {est_y}"
    assert abs(est_x - dx) < 0.05, f"X shift failed! Got {est_x}"
    print("✅ Test 1 Strict Recovery Passed (Interior Crop).\n")

    # Image B & C: Real Tile
    files = list(Path("abyssos_data/tiles").glob("*.npz")) + list(Path("abyssos_data/train").glob("*.npz"))
    if files:
        data = np.load(files[0])
        real_b08 = (data.get('lr') if 'lr' in data else data.get('lrs'))[0, 3]
        analyze_and_test("B. Real B08 Crop", real_b08, dy, dx)
        analyze_and_test("C. Real B08 Sobel Magnitude", sobel_mag(real_b08), dy, dx)

def test_gradient_energy():
    print("--- Test 2: Gradient Energy Diversity ---")
    files = list(Path("abyssos_data/tiles").glob("*.npz")) + list(Path("abyssos_data/train").glob("*.npz"))
    for f in files[:5]:
        data = np.load(f)
        img = (data.get('lr') if 'lr' in data else data.get('lrs'))[0, 3]
        grad_y = np.sum(np.abs(np.diff(img, axis=0)))
        grad_x = np.sum(np.abs(np.diff(img, axis=1)))
        
        ratio = min(grad_y, grad_x) / max(grad_y, grad_x)
        print(f"Tile {f.name} | Y/X Gradient Ratio: {ratio:.2f}")
        
        if ratio < 0.20:
            print("  🚨 low texture on one axis, registration unreliable on this tile")
        else:
            print("  ✅ Texture healthy.")

if __name__ == "__main__":
    test_skimage_phase_correlation()
    test_gradient_energy()