import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation


def create_pos_mask(
    mask: np.ndarray, 
    erosion_iter: int = 1,
    struct_erosion: np.ndarray = np.ones((3, 3, 3), dtype=bool)
) -> np.ndarray:
    """
    Create a positive mask by eroding the original mask.

    Args:
        mask: A boolean array of shape (H, W, D) where True values indicate the presence of the lesion (foreground) and False values indicate the background.
        erosion_iter: The number of iterations to apply for erosion.
        struct_erosion: The structuring element used for erosion. Default is a 3x3x3 cube.
    
    Returns:
        pos_mask: A boolean array of shape (H, W, D) where True values indicate the foreground (positive mask) and False values indicate the background.
    """
    pos_mask = np.zeros_like(mask, dtype=bool)

    # Erode label mask
    binary_erosion(
        mask,
        structure=struct_erosion,
        iterations=erosion_iter,
        output=pos_mask
    )

    return pos_mask


def create_neg_mask_from_pos_mask(
    pos_mask: np.ndarray, 
    dilation_iter: int = 3,
    struct_dil: np.ndarray = np.ones((3, 3, 3), dtype=bool)
) -> np.ndarray:
    """
    Create a negative mask by dilating the positive mask and inverting it.
    If dilation_iter is set to 0, the negative mask will be empty (all False). Otherwise, 
    the negative mask will be a ring around the positive mask.

    Args:
        pos_mask: A boolean array of shape (H, W, D) where True values indicate the presence of the lesion (foreground) and False values indicate the background.
        dilation_iter: The number of iterations to apply for dilation.
    Returns:
        neg_mask: A boolean array of shape (H, W, D) where True values indicate the background (negative mask) and False values indicate the presence of the lesion (foreground).
    """ 
    neg_mask = np.zeros_like(pos_mask, dtype=bool)

    if dilation_iter > 0:
        tmp_mask = np.zeros_like(pos_mask, dtype=bool)

        # 1. dilate label mask
        binary_dilation(
            pos_mask,
            structure=struct_dil,
            iterations=dilation_iter,
            output=neg_mask
        )

        # 2. create ring mask by subtracting the original positive mask from the dilated mask
        np.logical_not(pos_mask, out=tmp_mask)  # Invert the label mask to get the background
        np.logical_and(neg_mask, tmp_mask, out=neg_mask)  # Create a ring mask by keeping only the dilated region that is not part of the original positive mask

    return neg_mask


def create_surface_band_mask(
    pos_mask: np.ndarray,
    dilation_iter: int = 0,
    struct: np.ndarray = np.ones((3, 3, 3), dtype=bool)
) -> np.ndarray:
    """
    Creates a surface band around a binary mask.

    The surface is defined as the foreground voxels that disappear after one
    erosion. Optionally, the surface is dilated to create a thicker band.

    Args:
        pos_mask: Boolean foreground mask.
        dilation_iter: Number of dilation iterations. If 0, only the
            one-voxel-thick surface is returned.
        struct: Structuring element used for erosion and dilation.

    Returns:
        Boolean surface band mask.
    """

    # One-voxel-thick foreground surface: voxels present in the original
    # mask but removed by a single erosion step (i.e. they touch the
    # background and form the outer shell).
    surface = pos_mask & ~binary_erosion(pos_mask, structure=None)

    if dilation_iter > 0:
        # Thicken the one-voxel surface into a band by dilating it outward.
        surface = binary_dilation(
            surface,
            structure=struct,
            iterations=dilation_iter,
        )

    return surface
