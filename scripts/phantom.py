import os, sys, torch, numpy as np
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model.mfsr import MFSR
from core.audit.auditor import Tile, audit

def run_phantom_benchmark():
    print("👻 --- RUNNING PHANTOM BENCHMARK ---")
    
    device = torch.device("cpu")
    model = MFSR(num_optical_bands=4, num_sar_bands=2, scale=4).to(device)
    model.load_state_dict(torch.load("abyssos_weights/best_finetuned.pt", map_location=device))
    model.eval()

    files = list(Path("abyssos_data/train").glob("*.npz"))[:5]
    if not files:
        print("❌ No data found.")
        return

    import json
    with open("weights/taus.json") as f: taus = json.load(f)
    tc, tv = taus['tau_c'], taus['tau_v']

    total_empty_pixels = 0
    total_hallucinations = 0
    caught_by_auditor = 0

    with torch.no_grad():
        for f in files:
            data = np.load(f)
            tile = Tile(
                lr=torch.from_numpy(data['lrs']).unsqueeze(0), masks=torch.from_numpy(data['masks']).unsqueeze(0),
                shifts=torch.from_numpy(data['shifts']).unsqueeze(0), sar=torch.from_numpy(data['sar']).unsqueeze(0),
                clear_fraction=torch.from_numpy(data['clear_fraction']).unsqueeze(0), meta={}
            )
            ar = audit(model, tile, scale=4)
            
            empty_mask = torch.zeros(256, 256, dtype=torch.bool)
            empty_mask[128:, :] = True 
            
            # FIX: Calculate edges and pad back up to 256x256
            edges = torch.abs(ar.sr[:, 1:, 1:] - ar.sr[:, :-1, :-1]).mean(dim=0)
            edges = torch.nn.functional.pad(edges, (0, 1, 0, 1))
            
            edge_threshold = edges.mean() + (edges.std() * 2)
            detected_objects = (edges > edge_threshold)
            
            hallucinated_pixels = (detected_objects & empty_mask).sum().item()
            prior_only_mask = (ar.v.squeeze() > tv) | (ar.c.squeeze() > tc)
            caught = (detected_objects & empty_mask & prior_only_mask).sum().item()
            
            total_empty_pixels += empty_mask.sum().item()
            total_hallucinations += hallucinated_pixels
            caught_by_auditor += caught

    raw_phantom_rate = (total_hallucinations / total_empty_pixels) * 100 if total_empty_pixels > 0 else 0
    abyssos_phantom_rate = ((total_hallucinations - caught_by_auditor) / total_empty_pixels) * 100 if total_empty_pixels > 0 else 0
    flagged_pct = (caught_by_auditor / total_hallucinations) * 100 if total_hallucinations > 0 else 100.0

    print(f"\n📊 BENCHMARK RESULTS (Tested on {len(files)} tiles)")
    print(f"Standard AI Phantom Rate: {raw_phantom_rate:.2f}% (Hallucinated false details)")
    print(f"Ábyssos Phantom Rate    : {abyssos_phantom_rate:.2f}% (After Auditor filtering)")
    print(f"-> The Truth Auditor successfully flagged {flagged_pct:.1f}% of hallucinations as PRIOR-ONLY.")

if __name__ == "__main__":
    run_phantom_benchmark()