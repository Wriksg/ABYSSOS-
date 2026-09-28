import numpy as np

def validate_tile(path):
    """Validates the exact Backend-1 Harvester contract."""
    data = np.load(path, allow_pickle=True)
    
    lr = data['lr']       # (K, 4, H, W)
    mask = data['mask']   # (K, H, W)
    shifts = data['shifts'] # (K, 2)
    sar = data['sar']     # (2, H, W)
    
    # Handle meta dictionary (numpy saves dicts as object arrays)
    meta = data['meta'].item() if data['meta'].shape == () else data['meta'][0]
    
    K, C, H, W = lr.shape
    ref_idx = meta.get('ref_idx', 0)
    dates = meta.get('dates', [])

    # 1. Channels
    assert C == 4, f"Expected 4 channels, got {C}."
    
    # 2. Shapes
    assert mask.shape == (K, H, W), f"Mask shape {mask.shape} mismatch with LR {(K, H, W)}"
    assert shifts.shape == (K, 2), f"Shifts shape {shifts.shape} mismatch with K={K}"
    assert sar.shape == (2, H, W), f"SAR shape {sar.shape} mismatch with H,W {(H, W)}"
    
    # 3. Shift Constraints
    valid_shifts = shifts[~np.isnan(shifts).any(axis=1)]
    if len(valid_shifts) > 0:
        max_shift = np.abs(valid_shifts).max()
        assert max_shift <= 3.0, f"Max shift exceeded 3.0px limit. Found {max_shift:.2f}px."
        
    assert np.allclose(shifts[ref_idx], [0.0, 0.0], equal_nan=False), \
        f"Shift at ref_idx {ref_idx} must be [0,0], got {shifts[ref_idx]}"
        
    # 4. Radiometry
    assert lr.min() >= 0.0 and lr.max() <= 1.5, \
        f"LR Reflectance out of bounds [0, 1.5]. Min: {lr.min():.2f}, Max: {lr.max():.2f}"
        
    # 5. Metadata
    assert len(dates) == K, f"Found {len(dates)} dates for K={K} frames."
    
    print(f"✅ Tile {path} is fully contract-compliant.")
    return True