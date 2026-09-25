import math
import torch
import torch.nn.functional as F

def gauss2d(sigma: float, device: torch.device) -> torch.Tensor:
    """Generates a 1D Gaussian and computes the 2D outer product."""
    r = int(math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    g = torch.exp(-x**2 / (2 * sigma**2))
    g = g / g.sum()
    return g[:, None] * g[None, :]

def degrade(sr: torch.Tensor, shift_lr: torch.Tensor, scale: int = 4, sigma_hr: float = 2.1) -> torch.Tensor:
    """
    Simulates the Sentinel-2 sensor forward model: Blur -> Sub-pixel Shift -> Downsample.
    
    Args:
        sr: High-res input tensor (B, C, H, W).
        shift_lr: Sub-pixel shifts in LR pixel units (B, 2) where dim 1 is (dy, dx).
        scale: Downsampling factor (default: 4).
        sigma_hr: Gaussian blur standard deviation in HR pixel units (default: 2.1).
        
    Returns:
        LR tensor (B, C, H//scale, W//scale).
    """
    B, C, H, W = sr.shape
    
    # ---------------------------------------------------------
    # 1. BLUR (Sensor MTF simulation)
    # ---------------------------------------------------------
    k = gauss2d(sigma_hr, sr.device)[None, None].repeat(C, 1, 1, 1)
    p = k.shape[-1] // 2
    # Reflect padding prevents edge darkening artifacts
    x = F.conv2d(F.pad(sr, (p, p, p, p), mode='reflect'), k, groups=C)
    
    # ---------------------------------------------------------
    # 2. SUB-PIXEL SHIFT
    # ---------------------------------------------------------
    # Scale LR shifts into HR pixel space
    dy = shift_lr[:, 0] * scale
    dx = shift_lr[:, 1] * scale
    
    th = torch.zeros(B, 2, 3, device=sr.device)
    th[:, 0, 0] = 1.0  # Scale X
    th[:, 1, 1] = 1.0  # Scale Y
    # Translate X and Y (grid_sample uses coordinate range [-1, 1])
    th[:, 0, 2] = 2 * dx / W
    th[:, 1, 2] = 2 * dy / H
    
    grid = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    
    # ---------------------------------------------------------
    # 3. DOWNSAMPLE
    # ---------------------------------------------------------
    # Avg_pool2d perfectly models a standard sensor pixel bucket integration
    # Do NOT use F.interpolate here!
    return F.avg_pool2d(x, scale)