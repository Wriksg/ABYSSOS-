import torch

# UI Classes
VERIFIED = 0
SAR_SUPPORTED = 1
PRIOR_ONLY = 2

def classify(c, v, a, taus):
    """
    Translates raw Truth Auditor signals into the 3 UI Classes.
    c: Data Consistency (H, W)
    v: Ensemble Disagreement (H, W)
    a: SAR Agreement (H, W) or None (Stopgap)
    taus: Dictionary of calibrated thresholds
    """
    # 1. Default everything to PRIOR_ONLY (Red/Hallucination)
    out = torch.full_like(c, PRIOR_ONLY, dtype=torch.uint8)
    
    # 2. Add SAR support (if the signal exists. If cut, it safely passes through)
    if a is not None and 'tau_a' in taus:
        out[a > taus['tau_a']] = SAR_SUPPORTED
        
    # 3. Override with VERIFIED (Green) if optical physics agree and model is stable
    verified_mask = (c < taus['tau_c']) & (v < taus['tau_v'])
    out[verified_mask] = VERIFIED
    
    return out