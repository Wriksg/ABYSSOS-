import os, torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def build_empty_set():
    print("🔍 Hunting for candidate empty tiles...")
    data_dir = Path("abyssos_data/train")
    cand_dir = Path("candidates")
    cand_dir.mkdir(exist_ok=True)
    
    files = list(data_dir.glob("*.npz"))
    candidates_found = 0
    
    for f in files:
        data = np.load(f)
        hr = torch.from_numpy(data['hr'])
        edges = torch.abs(hr[:, 1:, 1:] - hr[:, :-1, :-1]).mean()
        
        if edges.item() < 0.08: 
            img = np.clip(hr[:3].numpy().transpose(1, 2, 0) * 2.5, 0, 1)
            plt.imsave(cand_dir / f"{f.name}.png", img)
            candidates_found += 1
            
    print(f"✅ Found {candidates_found} candidates.")
    print("👉 ACTION: Open 'candidates/' folder in VS Code.")
    print("DELETE any image that has a real road or building.")
    print("Then run: python scripts/find_empty_tiles.py --finalize")

def finalize_empty_set():
    cand_dir = Path("candidates")
    empty_list = [f.name.replace(".png", "") for f in cand_dir.glob("*.npz.png")]
    out_dir = Path("tests/fixtures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "empty_tiles.txt", "w") as f:
        f.write("\n".join(empty_list))
    print(f"🔒 Locked {len(empty_list)} visually confirmed empty tiles into tests/fixtures/empty_tiles.txt!")

if __name__ == "__main__":
    if "--finalize" in sys.argv:
        finalize_empty_set()
    else:
        build_empty_set()
