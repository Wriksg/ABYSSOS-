import os
import argparse
import csv
import torch
import numpy as np   # <--- ADDED THIS LINE
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model.mfsr import MFSR
from core.model.losses import (
    shift_tolerant_l1, 
    consistency_loss, 
    spectral_angle_loss, 
    gradient_loss
)

# ---------------------------------------------------------
# 1. REAL DATASET LOADER
# ---------------------------------------------------------
class AbyssosDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.data_dir = os.path.join(data_dir, split)
        # Get all .npz files in the directory
        if os.path.exists(self.data_dir):
            self.files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        else:
            self.files = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the real satellite data array from disk
        data = np.load(self.files[idx])
        
        # Convert numpy arrays to PyTorch tensors
        return {
            "lrs": torch.from_numpy(data['lrs']).float(),                
            "masks": torch.from_numpy(data['masks']).float(),            
            "shifts": torch.from_numpy(data['shifts']).float(),          
            "sar": torch.from_numpy(data['sar']).float(),                
            "clear_fraction": torch.from_numpy(data['clear_fraction']).float(), 
            "hr": torch.from_numpy(data['hr']).float()                   
        }

# ---------------------------------------------------------
# 2. METRICS
# ---------------------------------------------------------
def calc_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0: return torch.tensor(100.0)
    return 20 * torch.log10(max_val / torch.sqrt(mse))

# ---------------------------------------------------------
# 3. MAIN TRAINING LOOP
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load Data
    train_loader = DataLoader(AbyssosDataset(args.data, "train"), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(AbyssosDataset(args.data, "val"), batch_size=args.batch_size, shuffle=False)

    # Initialize Model, Optimizer, Scheduler, AMP Scaler
    model = MFSR(num_optical_bands=4, num_sar_bands=2, scale=4).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler('cuda')

    # Logging setup
    csv_file = open(os.path.join(args.out, "metrics.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_PSNR", "Bicubic_PSNR"])

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            lrs = batch["lrs"].to(device)
            masks = batch["masks"].to(device)
            shifts = batch["shifts"].to(device)
            sar = batch["sar"].to(device)
            clear_frac = batch["clear_fraction"].to(device)
            hr = batch["hr"].to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                sr, _ = model(lrs, masks, sar, clear_frac)
                
                # The combined loss from the contract
                l_l1 = shift_tolerant_l1(sr, hr)
                l_cons = consistency_loss(sr, lrs, shifts, masks, scale=4)
                l_sam = spectral_angle_loss(sr, hr)
                l_grad = gradient_loss(sr, hr)
                
                loss = l_l1 + (0.5 * l_cons) + (0.2 * l_sam) + (0.1 * l_grad)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # ---------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------
        model.eval()
        val_loss, val_psnr, bicubic_psnr = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                lrs = batch["lrs"].to(device)
                masks = batch["masks"].to(device)
                shifts = batch["shifts"].to(device)
                sar = batch["sar"].to(device)
                clear_frac = batch["clear_fraction"].to(device)
                hr = batch["hr"].to(device)

                with autocast('cuda'):
                    sr, _ = model(lrs, masks, sar, clear_frac)
                    
                    l_l1 = shift_tolerant_l1(sr, hr)
                    l_cons = consistency_loss(sr, lrs, shifts, masks, scale=4)
                    l_sam = spectral_angle_loss(sr, hr)
                    l_grad = gradient_loss(sr, hr)
                    loss = l_l1 + (0.5 * l_cons) + (0.2 * l_sam) + (0.1 * l_grad)

                val_loss += loss.item()
                val_psnr += calc_psnr(sr, hr).item()
                
                # Compute Baseline: simple bicubic upsample of the reference frame (i=0)
                baseline_sr = F.interpolate(lrs[:, 0], scale_factor=4, mode='bicubic', align_corners=False)
                bicubic_psnr += calc_psnr(baseline_sr, hr).item()

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        bicubic_psnr /= len(val_loader)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} (Bicubic: {bicubic_psnr:.2f})")
        csv_writer.writerow([epoch, train_loss, val_loss, val_psnr, bicubic_psnr])
        csv_file.flush()

        # Checkpointing
        torch.save(model.state_dict(), os.path.join(args.out, "last.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            print(f"  -> Saved new best.pt (Val Loss: {val_loss:.4f})")

    csv_file.close()

if __name__ == "__main__":
    main()