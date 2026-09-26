import os
import torch
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from datasets import load_dataset
import torchvision.transforms.functional as TF

# Import your own physics model to generate the synthetic satellite passes!
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model.degrade import degrade

def main():
    out_dir = Path("abyssos_data/train")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("🌍 WorldStrat is gated. Pivoting to RESISC45 (Real Earth Textures)...")
    
    # RESISC45 is a stable, public HF dataset containing 31,500 satellite images (256x256).
    dataset = load_dataset('timm/resisc45', split='train', streaming=True)
    iterator = iter(dataset)
    
    K = 4            
    C = 4            # We need 4 bands (RGB + NIR)
    LR_SIZE = 64     
    SCALE = 4
    HR_SIZE = 256    
    
    num_tiles = 50
    tiles_saved = 0
    
    print(f"🔥 Generating {num_tiles} synthetic multi-frame stacks from real satellite data...")
    
    while tiles_saved < num_tiles:
        try:
            item = next(iterator)
        except StopIteration:
            break
            
        # 1. EXTRACT HR (Real Earth Image, 256x256 RGB)
        pil_img = item['image']
        hr_rgb = TF.to_tensor(pil_img) # (3, 256, 256), scaled 0-1
        
        # We need a 4th channel (NIR) to match Sentinel-2 format. 
        # We will mock NIR by averaging the Red and Green channels.
        nir_channel = (hr_rgb[0:1] + hr_rgb[1:2]) / 2.0
        hr_tensor = torch.cat([hr_rgb, nir_channel], dim=0) # Shape: (4, 256, 256)
        
        # 2. GENERATE K SUB-PIXEL SHIFTED PASSES USING YOUR OWN PHYSICS ENGINE
        # We randomly shift the satellite between -1 and +1 LR pixels for each pass
        shifts_tensor = (torch.rand(K, 2) * 2.0) - 1.0 
        
        lrs_tensor = torch.zeros(K, C, LR_SIZE, LR_SIZE)
        
        for k in range(K):
            # We must add a batch dimension for degrade() -> (1, C, H, W)
            hr_batch = hr_tensor.unsqueeze(0) 
            shift_batch = shifts_tensor[k].unsqueeze(0)
            
            # Use YOUR physics model to simulate what Sentinel-2 would see!
            lr_pass = degrade(hr_batch, shift_batch, scale=SCALE, sigma_hr=2.1)
            lrs_tensor[k] = lr_pass.squeeze(0)

        # 3. MASKS & SAR (Mocked to unblock the network)
        masks_tensor = torch.ones(K, 1, LR_SIZE, LR_SIZE)
        clear_fraction = torch.tensor([1.0])
        # Random C-band radar speckle
        sar_tensor = torch.randn(2, LR_SIZE, LR_SIZE) * 2.0 - 10.0 

        # 4. SAVE TO DISK (Matching Backend-1's contract exactly)
        np.savez(
            out_dir / f"fallback_{tiles_saved:03d}.npz",
            lrs=lrs_tensor.numpy().astype(np.float32),
            masks=masks_tensor.numpy().astype(np.float32),
            shifts=shifts_tensor.numpy().astype(np.float32),
            sar=sar_tensor.numpy().astype(np.float32),
            clear_fraction=clear_fraction.numpy().astype(np.float32),
            hr=hr_tensor.numpy().astype(np.float32)
        )
        
        tiles_saved += 1
        print(f"  -> Generated Tile {tiles_saved}/{num_tiles} (Passes: {K}, Bands: {C})")

    print(f"\n✅ UNBLOCKED! Dataset saved to {out_dir}. You are ready for Colab.")

if __name__ == "__main__":
    main()