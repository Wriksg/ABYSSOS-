import os, argparse, torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model.mfsr import MFSR
from core.model.losses import shift_tolerant_l1, consistency_loss, spectral_angle_loss, gradient_loss
from scripts.train import AbyssosDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="abyssos_data")
    parser.add_argument("--init_weights", type=str, default="abyssos_weights/best.pt")
    parser.add_argument("--out", type=str, default="abyssos_weights/best_finetuned.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5) # Lower LR for finetuning
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Fine-tuning on {device}...")

    # Load Model & Weights
    model = MFSR(num_optical_bands=4, num_sar_bands=2, scale=4).to(device)
    if os.path.exists(args.init_weights):
        model.load_state_dict(torch.load(args.init_weights, map_location=device))
        print(f"✅ Loaded synthetic checkpoint: {args.init_weights}")
    else:
        print("❌ Init weights not found. Exiting.")
        return

    # Data
    train_loader = DataLoader(AbyssosDataset(args.data, "train"), batch_size=8, shuffle=True)
    if len(train_loader.dataset) == 0:
        print("❌ No real data found in abyssos_data/train. Add Backend-1 data first.")
        return

    opt = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for b in train_loader:
            lrs, masks, shifts = b["lrs"].to(device), b["masks"].to(device), b["shifts"].to(device)
            sar, c_frac, hr = b["sar"].to(device), b["clear_fraction"].to(device), b["hr"].to(device)
            
            opt.zero_grad()
            with autocast(device.type):
                sr, _ = model(lrs, masks, sar, c_frac)
                loss = shift_tolerant_l1(sr, hr) + 0.5 * consistency_loss(sr, lrs, shifts, masks, 4) + \
                       0.2 * spectral_angle_loss(sr, hr) + 0.1 * gradient_loss(sr, hr)
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch}/{args.epochs} | Loss: {epoch_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"✅ Fine-tuning complete. Saved to {args.out}")

if __name__ == "__main__":
    main()