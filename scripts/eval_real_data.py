import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model.mfsr import MFSR
from core.model.losses import spectral_angle_loss

def calc_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0: return 100.0
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()

def evaluate_checkpoint(weights_path, real_data_dir, synthetic_val_psnr=30.0):
    print(f"🔍 Loading weights from: {weights_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MFSR(num_optical_bands=4, num_sar_bands=2, scale=4).to(device)
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"❌ Failed to load weights: {e}")
        return

    data_dir = Path(real_data_dir)
    real_files = list(data_dir.glob("*.npz"))
    
    if not real_files:
        print(f"❌ No .npz files found in {real_data_dir}. Drop real Sentinel-2 tiles here!")
        return
        
    print(f"🌍 Found {len(real_files)} real Sentinel-2 tiles. Running Evaluation...\n")
    
    out_vis_dir = Path("eval_visuals")
    out_vis_dir.mkdir(exist_ok=True)
    
    avg_sr_psnr, avg_bicubic_psnr = 0.0, 0.0
    avg_sr_sam, avg_bicubic_sam = 0.0, 0.0
    
    with torch.no_grad():
        for i, f in enumerate(real_files):
            # Load real data
            data = np.load(f)
            lrs = torch.from_numpy(data['lrs']).unsqueeze(0).to(device)       # (1, K, 4, H, W)
            masks = torch.from_numpy(data['masks']).unsqueeze(0).to(device)   
            shifts = torch.from_numpy(data['shifts']).unsqueeze(0).to(device) 
            sar = torch.from_numpy(data['sar']).unsqueeze(0).to(device)       
            c_frac = torch.from_numpy(data['clear_fraction']).unsqueeze(0).to(device)
            hr = torch.from_numpy(data['hr']).unsqueeze(0).to(device)         # Ground Truth
            
            # 1. Bicubic Baseline (Upsample reference frame)
            bicubic = F.interpolate(lrs[:, 0], scale_factor=4, mode='bicubic', align_corners=False)
            
            # 2. MFSR Prediction
            sr, _ = model(lrs, masks, sar, c_frac)
            
            # 3. Metrics
            b_psnr = calc_psnr(bicubic, hr)
            s_psnr = calc_psnr(sr, hr)
            
            b_sam = spectral_angle_loss(bicubic, hr).item()
            s_sam = spectral_angle_loss(sr, hr).item()
            
            avg_bicubic_psnr += b_psnr; avg_sr_psnr += s_psnr
            avg_bicubic_sam += b_sam; avg_sr_sam += s_sam
            
            print(f"Tile {i+1}: SR PSNR = {s_psnr:.2f}dB | Bicubic = {b_psnr:.2f}dB  ||  SR SAM = {s_sam:.4f} | Bicubic = {b_sam:.4f}")
            
            # 4. Save Visuals (Extract RGB, normalize 0-1)
            plt.figure(figsize=(15, 5))
            def prep_img(t): return np.clip(t[0, :3].cpu().numpy().transpose(1, 2, 0) * 2.5, 0, 1)
            
            plt.subplot(1, 4, 1); plt.imshow(prep_img(lrs[:, 0])); plt.title("LR (10m)")
            plt.subplot(1, 4, 2); plt.imshow(prep_img(bicubic)); plt.title(f"Bicubic ({b_psnr:.1f}dB)")
            plt.subplot(1, 4, 3); plt.imshow(prep_img(sr)); plt.title(f"Ábyssos SR ({s_psnr:.1f}dB)")
            plt.subplot(1, 4, 4); plt.imshow(prep_img(hr)); plt.title("Ground Truth (2.5m)")
            
            plt.savefig(out_vis_dir / f"compare_{i+1}.png", bbox_inches='tight')
            plt.close()

    # Calculate Averages
    N = len(real_files)
    avg_sr_psnr /= N; avg_bicubic_psnr /= N
    avg_sr_sam /= N; avg_bicubic_sam /= N
    
    print("\n" + "="*50)
    print(" 📊 FINAL REAL DATA EVALUATION REPORT")
    print("="*50)
    print(f"Bicubic Baseline : PSNR = {avg_bicubic_psnr:.2f}dB | SAM = {avg_bicubic_sam:.4f}")
    print(f"Synthetic Trained: PSNR = {avg_sr_psnr:.2f}dB | SAM = {avg_sr_sam:.4f}")
    
    psnr_gap = avg_sr_psnr - avg_bicubic_psnr
    synth_gap = synthetic_val_psnr - avg_sr_psnr
    
    print("\n 🚩 SYSTEM DIAGNOSIS:")
    if psnr_gap < 0:
        print("❌ OVERFIT ALERT: Model performs WORSE than bicubic on real data.")
        print("   Conclusion: Only good for proving the pipeline runs. DO NOT use this checkpoint for the UI Demo.")
    elif psnr_gap < 0.5:
        print("⚠️ WEAK GENERALIZATION: Model is barely beating bicubic.")
        print(f"   (It scored {synthetic_val_psnr}dB on synthetic data, but dropped to {avg_sr_psnr:.2f}dB on real).")
        print("   Conclusion: It learned some edges, but you urgently need Backend-1's real dataset for the final demo.")
    else:
        print("✅ SUCCESS: The synthetic physics generalized to the real world!")
        print(f"   Model beats bicubic by +{psnr_gap:.2f}dB on unseen real Sentinel-2 tiles.")
        print("   Conclusion: You can safely use this checkpoint to generate Auditor UI signals right now.")
    print("="*50)
    print(f"🖼️ Side-by-side visuals saved to: {out_vis_dir.absolute()}")

if __name__ == "__main__":
    # Replace synthetic_val_psnr with the final validation PSNR your Colab output printed.
    evaluate_checkpoint(
        weights_path="abyssos_weights/best.pt", 
        real_data_dir="abyssos_data/real_eval",
        synthetic_val_psnr=30.0 
    )