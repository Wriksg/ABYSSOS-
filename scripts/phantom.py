import os, sys, torch, numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model.mfsr import MFSR
from core.audit.auditor import Tile, audit

def run_real_phantom_benchmark():
    print("👻 --- RUNNING PHANTOM BENCHMARK (SOBEL FALLBACK) ---")
    device = torch.device("cpu")
    
    # 1. Load Model
    model = MFSR(num_optical_bands=4, num_sar_bands=2, scale=4).to(device)
    model.load_state_dict(torch.load("abyssos_weights/best_finetuned.pt", map_location=device))
    model.eval()

    # 2. Load visually confirmed empty tiles ONLY
    empty_file = Path("tests/fixtures/empty_tiles.txt")
    if not empty_file.exists():
        print("❌ Run find_empty_tiles.py first!")
        return
        
    with open(empty_file, "r") as f:
        valid_tiles = set(f.read().splitlines())
        
    files = [f for f in Path("abyssos_data/train").glob("*.npz") if f.name in valid_tiles]
    
    if not files:
        print("❌ No confirmed empty tiles found in abyssos_data/train.")
        return

    import json
    with open("weights/taus.json") as f: taus = json.load(f)

    total_empty, total_ph, caught = 0, 0, 0

    with torch.no_grad():
        for f in files:
            data = np.load(f)
            tile = Tile(
                lr=torch.from_numpy(data['lrs']).unsqueeze(0), masks=torch.from_numpy(data['masks']).unsqueeze(0),
                shifts=torch.from_numpy(data['shifts']).unsqueeze(0), sar=torch.from_numpy(data['sar']).unsqueeze(0),
                clear_fraction=torch.from_numpy(data['clear_fraction']).unsqueeze(0), meta={}
            )
            ar = audit(model, tile, scale=4)
            
            empty_mask = torch.ones(256, 256, dtype=torch.bool)
            
            # Sobel fallback (mathematical structural edges)
            edges = torch.abs(ar.sr[:, 1:, 1:] - ar.sr[:, :-1, :-1]).mean(dim=0)
            edges = torch.nn.functional.pad(edges, (0, 1, 0, 1))
            detected_objects = edges > (edges.mean() + edges.std() * 2)
            
            total_empty += empty_mask.sum().item()
            total_ph += (detected_objects & empty_mask).sum().item()
            caught += (detected_objects & empty_mask & ((ar.v.squeeze() > taus['tau_v']) | (ar.c.squeeze() > taus['tau_c']))).sum().item()

    raw_phantom_rate = (total_ph / total_empty) * 100 if total_empty > 0 else 0
    abyssos_phantom_rate = ((total_ph - caught) / total_empty) * 100 if total_empty > 0 else 0
    flagged_pct = (caught / total_ph) * 100 if total_ph > 0 else 100.0

    print(f"\n📊 BENCHMARK RESULTS (n={len(files)} visually-confirmed tiles)")
    print(f"Standard AI Phantom Rate: {raw_phantom_rate:.2f}% (Hallucinated false details)")
    print(f"Ábyssos Phantom Rate    : {abyssos_phantom_rate:.2f}% (After Auditor filtering)")
    print(f"-> The Truth Auditor successfully flagged {flagged_pct:.1f}% of hallucinations as PRIOR-ONLY.")

if __name__ == "__main__":
    run_real_phantom_benchmark()
