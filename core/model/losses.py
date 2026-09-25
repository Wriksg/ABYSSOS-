import torch
import torch.nn.functional as F
from .degrade import degrade

# ---------------------------------------------------------
# 1. THE AUGMENTATION FIX (Vital for training and testing)
# ---------------------------------------------------------
def rot_shifts(shifts, k, flip):
    """
    Rotates/flips the sub-pixel shift vectors to match image augmentations.
    shifts: (..., 2) tensor of (dy, dx)
    k: number of 90-degree rotations
    flip: boolean
    """
    dy, dx = shifts[..., 0], shifts[..., 1]
    for _ in range(k):
        # 90 deg rotation of vector (dy, dx) -> (-dx, dy)
        dy, dx = -dx, dy
    if flip:
        # Horizontal flip inverts dx
        dx = -dx
    return torch.stack([dy, dx], dim=-1)

# ---------------------------------------------------------
# 2. SHIFT-TOLERANT L1 (Prevents blurring from misaligned GT)
# ---------------------------------------------------------
def shift_tolerant_l1(sr, hr, m=3):
    """
    Searches a local m x m window to find the best-aligned patch 
    between SR and HR before computing L1, ignoring global brightness bias.
    """
    B, C, H, W = sr.shape
    # Center crop of SR
    s = sr[..., m:H-m, m:W-m]
    best = None
    
    for dy in range(-m, m+1):
        for dx in range(-m, m+1):
            h = hr[..., m+dy:H-m+dy, m+dx:W-m+dx]
            bias = (h - s).mean(dim=(-2,-1), keepdim=True)   # brightness offset
            l = (s + bias - h).abs().mean(dim=(-2,-1))
            best = l if best is None else torch.minimum(best, l)
            
    return best.mean()

# ---------------------------------------------------------
# 3. CONSISTENCY LOSS (The physics check)
# ---------------------------------------------------------
def consistency_loss(sr, lrs, shifts, masks, scale=4, sigma_hr=2.1):
    """
    Forces the SR output to match the observed LR frames when re-degraded,
    computed only over clear pixels (ignoring clouds).
    
    lrs: (B, K, C, h, w)
    shifts: (B, K, 2)
    masks: (B, K, 1, h, w) -> 1 is clear, 0 is cloud
    """
    B, K, C, h, w = lrs.shape
    total_loss = 0.0
    
    for i in range(K):
        # 1. Degrade SR guess using the known shift for frame i
        lr_est = degrade(sr, shifts[:, i, :], scale=scale, sigma_hr=sigma_hr)
        
        # 2. Difference from actual observation
        diff = (lr_est - lrs[:, i]).abs()
        
        # 3. Mask out clouds
        masked_diff = diff * masks[:, i]
        
        # Normalize by number of clear pixels so cloudy tiles don't have artificially low loss
        clear_px_count = masks[:, i].sum() * C
        # add epsilon to avoid div by zero if tile is 100% cloud
        total_loss += masked_diff.sum() / (clear_px_count + 1e-8)
        
    return total_loss / K

# ---------------------------------------------------------
# 4. SPECTRAL ANGLE MAPPER (SAM) - Preserves agriculture/NDVI
# ---------------------------------------------------------
def spectral_angle_loss(sr, hr, eps=1e-8):
    """Computes spectral angle between SR and HR."""
    # Dot product along channel dimension
    dot = (sr * hr).sum(dim=1)
    # L2 norms
    norm_sr = torch.clamp(torch.norm(sr, dim=1), min=eps)
    norm_hr = torch.clamp(torch.norm(hr, dim=1), min=eps)
    
    cos_theta = torch.clamp(dot / (norm_sr * norm_hr), -1.0 + eps, 1.0 - eps)
    sam = torch.acos(cos_theta)
    return sam.mean()

# ---------------------------------------------------------
# 5. GRADIENT L1 LOSS (Protects thin roads)
# ---------------------------------------------------------
def gradient_loss(sr, hr):
    """Standard Sobel edge loss."""
    device = sr.device
    C = sr.shape[1]
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device)
    
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    
    # Compute edges
    sr_dx = F.conv2d(sr, sobel_x, groups=C, padding=1)
    sr_dy = F.conv2d(sr, sobel_y, groups=C, padding=1)
    hr_dx = F.conv2d(hr, sobel_x, groups=C, padding=1)
    hr_dy = F.conv2d(hr, sobel_y, groups=C, padding=1)
    
    return F.l1_loss(sr_dx, hr_dx) + F.l1_loss(sr_dy, hr_dy)