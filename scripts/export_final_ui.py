import os, sys, torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model.mfsr import MFSR
from core.audit.auditor import Tile, audit

def export_final():
    out = Path("ui_layers"); out.mkdir(exist_ok=True)
    device = torch.device("cpu")
    model = MFSR(num_optical_bands=4, num_sar_bands=2, scale=4).to(device)
    model.load_state_dict(torch.load("abyssos_weights/best_finetuned.pt", map_location=device))
    model.eval()

    data = np.load(list(Path("abyssos_data/train").glob("*.npz"))[0])
    tile = Tile(
        lr=torch.from_numpy(data['lrs']).unsqueeze(0), masks=torch.from_numpy(data['masks']).unsqueeze(0),
        shifts=torch.from_numpy(data['shifts']).unsqueeze(0), sar=torch.from_numpy(data['sar']).unsqueeze(0),
        clear_fraction=torch.from_numpy(data['clear_fraction']).unsqueeze(0), meta={}
    )
    
    with torch.no_grad(): ar = audit(model, tile, scale=4)
    
    # 1. Save SR
    rgb = np.clip(ar.sr[:3].numpy().transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    plt.imsave(out / "1_finetuned_sr.png", rgb)
    
    # 2. Save Evidence Map (using new taus)
    import json
    with open("weights/taus.json") as f: taus = json.load(f)
    tc, tv = taus['tau_c'], taus['tau_v']
    
    ev_map = np.zeros((256, 256, 4), dtype=np.float32)
    c, v = ar.c.numpy().squeeze(), ar.v.numpy().squeeze()
    
    ev_map[v > tv] = [1.0, 0.0, 0.0, 0.6]  # Red (PRIOR-ONLY)
    ev_map[(c < tc) & (v < tv)] = [0.0, 1.0, 0.0, 0.4]  # Green (VERIFIED)
    
    plt.imsave(out / "2_finetuned_evidence.png", ev_map)
    print("✅ Final fine-tuned UI layers exported to ui_layers/")

if __name__ == "__main__": export_final()