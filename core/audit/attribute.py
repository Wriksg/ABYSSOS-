import torch
from .evidence import classify

def attribute(ar, x, y, taus, radius=8):
    """
    Given a user click at (x,y), determines which satellite passes support the object.
    ar: AuditResult dataclass from week 3.
    """
    K = ar.res.shape[0]
    scale = ar.sr.shape[-1] // ar.res.shape[-1] # SR to LR scale factor (e.g. 4)
    
    # STOPGAP: Map HR click to LR grid using a hardcoded radius
    # (Srijoni will replace this with U-Net object segmentation masks)
    lr_x, lr_y = x // scale, y // scale
    r_lr = max(1, radius // scale)
    
    supporting_dates = []
    missing_dates = {}
    
    dates = ar.meta.get("dates", [f"Pass_2026-09-0{i+1}" for i in range(K)])

    # Check the physical residual stack frame by frame
    for k in range(K):
        date_str = dates[k]
        
        # STOPGAP: Bounding box region extraction
        H_lr, W_lr = ar.masks.shape[-2:]
        min_y, max_y = max(0, lr_y - r_lr), min(H_lr, lr_y + r_lr + 1)
        min_x, max_x = max(0, lr_x - r_lr), min(W_lr, lr_x + r_lr + 1)
        
        local_mask = ar.masks[k, 0, min_y:max_y, min_x:max_x]
        local_res = ar.res[k, 0, min_y:max_y, min_x:max_x]
        
        # Check cloud cover
        if local_mask.mean().item() < 0.5:
            missing_dates[date_str] = "Cloud Covered"
            continue
            
        # Check physical consistency (is the error below tau_c?)
        mean_res = local_res[local_mask > 0.5].mean().item()
        if mean_res < taus.get("tau_c", float('inf')):
            supporting_dates.append(date_str)
        else:
            missing_dates[date_str] = "Not Consistent (Physical deviation)"
            
    # Calculate Evidence Class
    # STOPGAP: SAR 'a' is set to None.
    ev_map = classify(ar.c, ar.v, None, taus)
    ev_class = int(ev_map[y, x].item())
    class_names = ["VERIFIED", "SAR_SUPPORTED", "PRIOR_ONLY"]
    
    return {
        "evidence_class": class_names[ev_class],
        "supporting_dates": supporting_dates,
        "missing_dates": missing_dates
    }