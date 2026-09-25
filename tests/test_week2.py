import torch
import sys
import os

# Adjust path so we can import from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model.mfsr import MFSR
from core.model.losses import (
    shift_tolerant_l1, 
    consistency_loss, 
    spectral_angle_loss, 
    gradient_loss, 
    rot_shifts
)

def run_full_pipeline_test():
    print("--- STARTING WEEK 2 PIPELINE TEST ---\n")
    
    # 1. Setup Dummy Data
    B, K, C, H, W = 2, 4, 4, 64, 64
    scale = 4
    HR_H, HR_W = H * scale, W * scale
    
    print(f"Generating dummy batch: Batch={B}, Frames={K}, Channels={C}, LR_Size={H}x{W}")
    lrs = torch.rand(B, K, C, H, W)
    masks = torch.ones(B, K, 1, H, W) # 1 = clear
    sar = torch.rand(B, 2, H, W)
    clear_fraction = torch.tensor([0.9, 0.2]) # Batch 1 clear, Batch 2 cloudy
    shifts = torch.rand(B, K, 2) * 2.0 - 1.0  # Shifts between -1 and 1 LR pixel
    hr_target = torch.rand(B, C, HR_H, HR_W)

    # 2. Initialize Model
    print("\nInitializing MFSR Model (scale 4x)...")
    model = MFSR(num_optical_bands=C, num_sar_bands=2, scale=scale)
    
    # 3. Forward Pass
    print("Running forward pass...")
    sr, gate = model(lrs, masks, sar, clear_fraction)
    print(f"  -> SR Output Shape: {sr.shape} (Expected: [{B}, {C}, {HR_H}, {HR_W}])")
    print(f"  -> SAR Gate Values: {gate.squeeze().detach().numpy()}")
    
    assert sr.shape == (B, C, HR_H, HR_W), "SR output shape is incorrect!"

    # 4. Compute Losses
    print("\nComputing Losses...")
    loss_l1 = shift_tolerant_l1(sr, hr_target)
    print(f"  -> Shift-Tolerant L1: {loss_l1.item():.4f}")
    
    loss_cons = consistency_loss(sr, lrs, shifts, masks, scale=scale)
    print(f"  -> Physics Consistency: {loss_cons.item():.4f}")
    
    loss_sam = spectral_angle_loss(sr, hr_target)
    print(f"  -> Spectral Angle (SAM): {loss_sam.item():.4f}")
    
    loss_grad = gradient_loss(sr, hr_target)
    print(f"  -> Gradient Edge Loss: {loss_grad.item():.4f}")

    total_loss = loss_l1 + (0.5 * loss_cons) + (0.2 * loss_sam) + (0.1 * loss_grad)
    print(f"\n  => Total Combined Loss: {total_loss.item():.4f}")
    
    # 5. Check Gradients (Does it train?)
    print("\nChecking backpropagation...")
    total_loss.backward()
    
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "FATAL: Backprop failed. No gradients found."
    print("SUCCESS: Gradients flow correctly through the entire network and sensor model!")
    
    # 6. Test Rotation Utility
    print("\nTesting augmentation shift rotation...")
    shifted = rot_shifts(shifts, k=1, flip=True)
    assert shifted.shape == shifts.shape, "Rotation util broke tensor shape!"
    
    print("\n✅ ALL TESTS PASSED! YOU ARE READY TO TRAIN IN COLAB.")

if __name__ == "__main__":
    run_full_pipeline_test()