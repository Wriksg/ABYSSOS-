import torch
from skimage.registration import phase_cross_correlation

import sys
import os
# Adjust path to import core if running as a raw python script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model.degrade import degrade

def test_degrade_recovers_injected_shift():
    # 1. Setup Golden HR (Requires gradients to test differentiability)
    # We use random noise here because phase_cross_correlation needs high-frequency 
    # texture to lock onto a perfect sub-pixel shift.
    torch.manual_seed(42)
    hr = torch.rand(1, 1, 256, 256, requires_grad=True)

    # 2. Degrade with zero shift
    shift_0 = torch.tensor([[0.0, 0.0]])
    a = degrade(hr, shift_0, scale=4, sigma_hr=2.1)

    # 3. Degrade with injected shift (dy=0.37, dx=-0.62)
    shift_1 = torch.tensor([[0.37, -0.62]])
    b = degrade(hr, shift_1, scale=4, sigma_hr=2.1)

    # 4. Test differentiability
    loss = b.sum()
    loss.backward()
    assert hr.grad is not None, "FATAL: degrade() broke the computational graph!"
    assert not torch.all(hr.grad == 0), "FATAL: Gradients are completely zero."

    # 5. Measure Phase Correlation
    # Skimage expects numpy arrays of shape (H, W)
    img_a = a[0, 0].detach().numpy()
    img_b = b[0, 0].detach().numpy()

    # phase_cross_correlation(reference, moving)
    est, _, _ = phase_cross_correlation(img_a, img_b, upsample_factor=100)
    
    print(f"Injected  (dy, dx) = (0.37, -0.62)")
    print(f"Recovered (dy, dx) = ({est[0]:.4f}, {est[1]:.4f})")

    # 6. Verify against expectations
    assert abs(est[0] - 0.37) < 0.1, f"Y-shift failed! Expected ~0.37, got {est[0]}"
    assert abs(est[1] + 0.62) < 0.1, f"X-shift failed! Expected ~-0.62, got {est[1]}"
    print("SUCCESS: Degrade operator round-trip passed!")

if __name__ == "__main__":
    test_degrade_recovers_injected_shift()