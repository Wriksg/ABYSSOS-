import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict

# Import our tools from Week 2
from core.model.degrade import degrade
from core.model.losses import rot_shifts

@dataclass
class Tile:
    """A standard data structure representing one Sentinel-2/1 observation tile."""
    lr: torch.Tensor             # (1, K, C, H, W)
    masks: torch.Tensor          # (1, K, 1, H, W)
    shifts: torch.Tensor         # (1, K, 2)
    sar: torch.Tensor            # (1, C_sar, H, W)
    clear_fraction: torch.Tensor # (1,)
    meta: Dict                   # Metadata dict for Srijoni's Analyst Console

@dataclass
class AuditResult:
    """Contract C: The exact structure Srijoni expects for the Analyst Console."""
    sr: torch.Tensor             # (C, H*scale, W*scale)
    c: torch.Tensor              # (1, H*scale, W*scale)
    v: torch.Tensor              # (1, H*scale, W*scale)
    res: torch.Tensor            # (K, 1, H, W)
    masks: torch.Tensor          # (K, 1, H, W)
    meta: Dict

def audit(model: torch.nn.Module, tile: Tile, scale: int = 4) -> AuditResult:
    """
    The Truth Auditor. Evaluates an unbatched tile to produce evidence signals.
    """
    model.eval()
    B, K, C, H, W = tile.lr.shape
    assert B == 1, "Auditor expects a single unbatched tile (B=1)"

    with torch.no_grad():
        # ---------------------------------------------------------
        # 1. ENSEMBLE DISAGREEMENT (Signal `v`)
        # Probes the model's null-space by applying Test-Time Augmentation (TTA).
        # Real details stay stable; hallucinated details jitter.
        # ---------------------------------------------------------
        outs = []
        for k in range(4):
            for flip in (False, True):
                # Spatial transformations (works for both 4D and 5D tensors)
                def t(z): return (z.flip(-1) if flip else z).rot90(k, dims=(-2, -1))
                # Inverse spatial transformations
                def it(z): return z.rot90(-k, dims=(-2, -1)).flip(-1) if flip else z.rot90(-k, dims=(-2, -1))

                # Augment the inputs
                lr_aug = t(tile.lr)
                masks_aug = t(tile.masks)
                sar_aug = t(tile.sar)
                shifts_aug = rot_shifts(tile.shifts, k, flip)

                # Forward pass
                sr_aug, _ = model(lr_aug, masks_aug, sar_aug, tile.clear_fraction)

                # Inverse augment the output and store
                outs.append(it(sr_aug))

        # Stack the 8 passes -> shape: (8, 1, C, H*scale, W*scale)
        outs = torch.stack(outs)
        
        # Final SR is the mean of the ensemble
        sr = outs.mean(dim=0)
        
        # Variance `v` is the disagreement across the 8 passes (mean across color channels)
        v = outs.var(dim=0).mean(dim=1, keepdim=True)

        # ---------------------------------------------------------
        # 2. DATA CONSISTENCY (Signal `c` and `res` stack)
        # Forces the output to answer for what the satellite actually saw.
        # ---------------------------------------------------------
        res_list = []
        for i in range(K):
            # Degrade our 2.5m guess back down to 10m using frame i's shift
            lr_est = degrade(sr, tile.shifts[:, i], scale=scale)
            
            # Difference from actual satellite observation (mean across color channels)
            diff = (lr_est - tile.lr[:, i]).abs().mean(dim=1, keepdim=True)
            res_list.append(diff)
            
        # The full residual stack: (1, K, 1, H, W)
        res = torch.stack(res_list, dim=1)

        # Mask out clouds so they don't count against consistency
        res_masked = res.masked_fill(tile.masks < 0.5, float('nan'))

        # Median across K frames (ignoring NaNs/clouds)
        # We use median so one missed cloud shadow doesn't ruin the pixel's score
        c_lr = torch.nanmedian(res_masked, dim=1).values
        c_lr = torch.nan_to_num(c_lr, nan=1.0) # Fill totally cloudy pixels with high error

        # Nearest-neighbor upsample consistency map back to SR resolution
        c = F.interpolate(c_lr, scale_factor=scale, mode='nearest')

    return AuditResult(
        sr=sr.squeeze(0),         # (C, H*s, W*s)
        c=c.squeeze(0),           # (1, H*s, W*s)
        v=v.squeeze(0),           # (1, H*s, W*s)
        res=res.squeeze(0),       # (K, 1, H, W)
        masks=tile.masks.squeeze(0), # (K, 1, H, W)
        meta=tile.meta
    )