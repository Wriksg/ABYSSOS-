import numpy as np
from scipy.ndimage import binary_erosion
from skimage.transform import resize

def clear_mask(scl_20m, target_shape):
    """
    Converts a Sentinel-2 Scene Classification (SCL) map into a dilated cloud mask.
    scl_20m: 2D array of SCL classes (at 20m resolution)
    target_shape: (H, W) of the target 10m LR grid
    """
    # Keep SCL classes {4: Vegetation, 5: Not Vegetated, 6: Water, 7: Unclassified}
    is_clear = np.isin(scl_20m, [4, 5, 6, 7])
    
    # Erode the CLEAR mask by 3 iterations (this dilates the clouds/shadows)
    # This prevents thin cloud edges from bleeding into our super-resolution math
    eroded_clear = binary_erosion(is_clear, iterations=3)
    
    # Nearest neighbor upsample to 10m target shape
    mask_10m = resize(
        eroded_clear.astype(np.uint8), 
        target_shape, 
        order=0,               # Nearest neighbor
        preserve_range=True, 
        anti_aliasing=False
    )
    
    return mask_10m.astype(np.uint8)