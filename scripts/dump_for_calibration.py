import torch, numpy as np, os, sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model.mfsr import MFSR
from core.audit.auditor import Tile, audit

def generate_calibration_dump():
    out = Path("abyssos_calibration_data"); out.mkdir(parents=True, exist_ok=True)
    model = MFSR(num_optical_bands=4, num_sar_bands=2, scale=4).eval()
    
    # Mock single tile
    dummy_tile = Tile(
        lr=torch.rand(1, 4, 4, 64, 64), masks=torch.ones(1, 4, 1, 64, 64),
        shifts=torch.rand(1, 4, 2), sar=torch.rand(1, 2, 64, 64),
        clear_fraction=torch.tensor([0.7]), meta={"id": "test_tile"}
    )
    r = audit(model, dummy_tile, scale=4)
    err = (r.sr - torch.rand(4, 256, 256)).abs().mean(dim=0)
    
    np.savez(out / "test_tile.npz", c=r.c.numpy(), v=r.v.numpy(), err=err.numpy(), sar=dummy_tile.sar.squeeze(0).numpy(), sr=r.sr.numpy())
    print("✅ Dump complete! Tell Srijoni the .npz files are ready.")

if __name__ == "__main__": generate_calibration_dump()