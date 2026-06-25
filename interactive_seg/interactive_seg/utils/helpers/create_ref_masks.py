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
    dilation_iter: int = 1,
    struct_dil: np.ndarray = np.ones((3, 3, 3), dtype=bool)
) -> np.ndarray:
    """
    Detect the change of the label mask along the x, y, and z axes to create a boundary mask.
    Then dilate the surface band mask to create a thicker boundary region. This is the reference 
    for creating the reference positive and negative masks for the prompt generation.
    If dilation_iter is set to 0, the surface band mask will be the boundary of the label mask.
    Otherwise, the surface band mask will be a thicker boundary region around the label mask.

    Args:
        pos_mask: A boolean array of shape (H, W, D) where True values indicate the presence of the lesion (foreground) and False values indicate the background.
        dilation_iter: The number of iterations to apply for dilation.
        struct_dil: The structuring element used for dilation. Default is a 3x3x3 cube.
    Returns:
        boundary_mask: A boolean array of shape (H, W, D) where True values indicate the boundary region of the lesion and False values indicate the background.
    """
    # Detect the change of the label mask along the x, y, and z axes to create a boundary mask
    boundary_mask = np.zeros_like(pos_mask, dtype=bool)
    boundary_mask[:-1, :, :] |= pos_mask[:-1, :, :] != pos_mask[1:, :, :]
    boundary_mask[:, :-1, :] |= pos_mask[:, :-1, :] != pos_mask[:, 1:, :]
    boundary_mask[:, :, :-1] |= pos_mask[:, :, :-1] != pos_mask[:, :, 1:]

    # Dilate the boundary mask to create a thicker boundary region
    dilated_boundary_mask = np.zeros_like(boundary_mask, dtype=bool)
    if dilation_iter > 0:
        binary_dilation(
            boundary_mask,
            structure=struct_dil,
            iterations=dilation_iter,
            output=dilated_boundary_mask
        )
    else:
        dilated_boundary_mask = boundary_mask

    return dilated_boundary_mask
