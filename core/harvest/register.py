import numpy as np
import warnings
import cv2
from skimage.registration import phase_cross_correlation
from scipy.ndimage import sobel, gaussian_filter
from scipy.optimize import least_squares

def sobel_mag(img):
    dx = sobel(img, axis=1)
    dy = sobel(img, axis=0)
    return np.hypot(dx, dy)

def estimate_shifts(stack, masks, ref_idx, band=3, use_old_bugged_method=False):
    K, C, H, W = stack.shape
    shifts = np.full((K, 2), np.nan, dtype=np.float32)
    ref_img = stack[ref_idx, band]
    ref_mask = masks[ref_idx].astype(bool)
    
    if not use_old_bugged_method: 
        ref_img = sobel_mag(ref_img)
    else: 
        ref_img = ref_img * ref_mask
        
    all_tiny = True
    c = 16 # 16-px interior crop to strip border reflection artifacts
    
    for i in range(K):
        if i == ref_idx:
            shifts[i] = [0.0, 0.0]
            continue
        overlap = (ref_mask & masks[i].astype(bool)).sum() / (H * W)
        if overlap < 0.4: continue
        
        mov_img, mov_mask = stack[i, band], masks[i].astype(bool)
        
        if not use_old_bugged_method:
            mov_img = sobel_mag(mov_img)
            
            # Crop to pure interior
            r_c = ref_img[c:-c, c:-c]
            m_c = mov_img[c:-c, c:-c]
            mask_c = (ref_mask & mov_mask)[c:-c, c:-c].astype(float)
            
            # Soften mask edges and subtract mean
            blend = gaussian_filter(mask_c, sigma=1.5)
            mu_r = np.sum(r_c * mask_c) / max(mask_c.sum(), 1)
            mu_m = np.sum(m_c * mask_c) / max(mask_c.sum(), 1)
            
            s, _, _ = phase_cross_correlation((r_c - mu_r) * blend, (m_c - mu_m) * blend, upsample_factor=100)
        else:
            mov_img = mov_img * mov_mask
            s, _, _ = phase_cross_correlation(ref_img, mov_img, upsample_factor=100)
            
        if abs(s[0]) > 3.0 or abs(s[1]) > 3.0: continue
        shifts[i] = s
        if abs(s[0]) >= 0.05 or abs(s[1]) >= 0.05: all_tiny = False
        
    if K > 1 and all_tiny and not use_old_bugged_method:
        warnings.warn("🚨 All valid shifts are under 0.05 px! Phase correlation failed.")
    return shifts

_ecc_calibrated = False
_ecc_sign = np.array([1.0, 1.0])

def _get_ecc_shift(ref_sobel, mov_sobel, mask):
    global _ecc_calibrated, _ecc_sign
    if not _ecc_calibrated:
        img = gaussian_filter(np.random.randn(64, 64), 2.0).astype(np.float32)
        img -= img.mean()
        from scipy.ndimage import shift
        mov = shift(img, (2.0, -3.0), order=3).astype(np.float32)
        
        s_A, _, _ = phase_cross_correlation(img[16:-16, 16:-16], mov[16:-16, 16:-16], upsample_factor=100)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cv2.findTransformECC(img, mov, warp, cv2.MOTION_TRANSLATION, 
                                 (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001), 
                                 np.ones_like(img, dtype=np.uint8), 5)
            s_B_raw = np.array([warp[1, 2], warp[0, 2]])
            _ecc_sign[0] = np.sign(s_A[0]) * np.sign(s_B_raw[0]) if s_B_raw[0] != 0 else 1.0
            _ecc_sign[1] = np.sign(s_A[1]) * np.sign(s_B_raw[1]) if s_B_raw[1] != 0 else 1.0
        except cv2.error: pass
        _ecc_calibrated = True

    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cv2.findTransformECC(ref_sobel.astype(np.float32), mov_sobel.astype(np.float32), warp, 
                             cv2.MOTION_TRANSLATION, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001), 
                             mask.astype(np.uint8), 5)
        return np.array([warp[1, 2] * _ecc_sign[0], warp[0, 2] * _ecc_sign[1]])
    except cv2.error:
        return None

def estimate_shifts_v2(stack, masks, ref_idx, band=3):
    K, C, H, W = stack.shape
    pairs = []
    sobel_stack = np.array([sobel_mag(stack[k, band]) for k in range(K)])
    c = 16 # 16-px interior crop
    
    for i in range(K):
        for j in range(i+1, K):
            overlap = masks[i].astype(bool) & masks[j].astype(bool)
            frac = overlap.sum() / (H * W)
            if frac < 0.4: continue
            
            # Estimator A (Interior Crop)
            r_c = sobel_stack[i, c:-c, c:-c]
            m_c = sobel_stack[j, c:-c, c:-c]
            mask_c = overlap[c:-c, c:-c].astype(float)
            
            blend = gaussian_filter(mask_c, sigma=1.5)
            mu_i = np.sum(r_c * mask_c) / max(mask_c.sum(), 1)
            mu_j = np.sum(m_c * mask_c) / max(mask_c.sum(), 1)
            
            s_A, _, _ = phase_cross_correlation((r_c - mu_i) * blend, (m_c - mu_j) * blend, upsample_factor=100)
            
            # Estimator B (ECC)
            s_B = _get_ecc_shift(sobel_stack[i], sobel_stack[j], overlap)
            
            if s_B is None or abs(s_A[0] - s_B[0]) > 0.15 or abs(s_A[1] - s_B[1]) > 0.15: 
                continue
                
            s_final = (s_A + s_B) / 2.0
            pairs.append((i, j, s_final[0], s_final[1], frac))
            
    def residuals(params):
        shifts = np.zeros((K, 2))
        idx = 0
        for k in range(K):
            if k == ref_idx: continue
            shifts[k] = params[idx:idx+2]
            idx += 2
        res = []
        for i, j, dy, dx, w in pairs:
            res.append(w * (shifts[j, 0] - shifts[i, 0] - dy))
            res.append(w * (shifts[j, 1] - shifts[i, 1] - dx))
        return np.array(res) if len(res) > 0 else np.array([0.0])

    initial_guess = np.zeros((K - 1) * 2)
    opt_params = least_squares(residuals, initial_guess, loss="huber").x if pairs else initial_guess
        
    opt_shifts = np.zeros((K, 2))
    idx = 0
    for k in range(K):
        if k == ref_idx: continue
        opt_shifts[k] = opt_params[idx:idx+2]
        idx += 2
        
    res_per_frame = np.zeros(K)
    counts = np.zeros(K)
    for i, j, dy, dx, w in pairs:
        err_sq = (opt_shifts[j, 0] - opt_shifts[i, 0] - dy)**2 + (opt_shifts[j, 1] - opt_shifts[i, 1] - dx)**2
        res_per_frame[i] += err_sq; res_per_frame[j] += err_sq
        counts[i] += 1; counts[j] += 1
        
    final_shifts = np.full((K, 2), np.nan, dtype=np.float32)
    conf_dict = {}
    
    print("\n  [V2 JOINT SOLVE RESULTS]")
    print(f"  {'Frame':<5} | {'Shift (dy, dx)':<16} | {'RMS Res (px)':<12} | {'Pairs':<5} | {'Status':<10}")
    for k in range(K):
        rms = np.sqrt(res_per_frame[k] / max(counts[k], 1)) if (counts[k] > 0 or k == ref_idx) else np.nan
        conf_dict[k] = rms
        status, s_str = "OK", f"[{opt_shifts[k,0]:6.3f}, {opt_shifts[k,1]:6.3f}]"
        rms_str = f"{rms:.4f}" if not np.isnan(rms) else "N/A"
        
        if k != ref_idx and counts[k] == 0: status = "DROPPED (0 pairs)"
        elif not np.isnan(rms) and rms > 0.1 and k != ref_idx: status = "DROPPED (RMS>0.1)"
        elif abs(opt_shifts[k,0]) > 3.0 or abs(opt_shifts[k,1]) > 3.0: status = "DROPPED (|s|>3)"
        else: final_shifts[k] = opt_shifts[k]
            
        if status != "OK": s_str = "[   NaN,    NaN]"
        if k == ref_idx: status = "REF"
        print(f"  {k:<5} | {s_str:<16} | {rms_str:<12} | {int(counts[k]):<5} | {status}")
        
    return final_shifts, conf_dict