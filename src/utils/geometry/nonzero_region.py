import numpy as np
import torch

def get_non_zero_region(
        image: np.ndarray | torch.Tensor,
        offset: int = 0
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    Get the bounding box coordinates of the non-zero region of a 3D image.

    Returns half-open intervals [[x1, x2], [y1, y2], [z1, z2]] in original-image
    coordinates, i.e. each axis's upper bound is exclusive, so the region can be
    used directly for slicing/cropping:

        image[x1:x2, y1:y2, z1:z2]

    This matches the bounding-box convention used by nnInteractive (and nnU-Net),
    where crop/bbox coordinates are stored as half-open [start, stop) ranges per
    axis rather than inclusive min/max indices.

    Args:
        image: Input 3D image (numpy array or torch tensor). Format uint8, shape (H, W, D).
        offset: Offset to apply to the bounding box. Format int, shape (6,).

    Returns:
        The bounding box of the non-zero region as half-open intervals
        ((x1, x2), (y1, y2), (z1, z2)), where x2/y2/z2 are exclusive.
    """
    if not isinstance(image, (np.ndarray, torch.Tensor)):
        raise ValueError("Input image must be a numpy array or a torch tensor.")
    if image.ndim != 3:
        raise ValueError("Input image must be a 3D array.")

    is_torch = isinstance(image, torch.Tensor)
    nonzero = image != 0

    if is_torch:
        if not nonzero.any():
            return ((0, image.shape[0]), (0, image.shape[1]), (0, image.shape[2]))
        coords = torch.nonzero(nonzero, as_tuple=False)
        mins = coords.min(dim=0).values.tolist()
        maxs = coords.max(dim=0).values.tolist()
    else:
        if not nonzero.any():
            return ((0, image.shape[0]), (0, image.shape[1]), (0, image.shape[2]))
        coords = np.nonzero(nonzero)
        mins = [int(c.min()) for c in coords]
        maxs = [int(c.max()) for c in coords]

    x1, y1, z1 = mins
    x2, y2, z2 = maxs

    # Maxs above are inclusive indices of the last nonzero voxel -> make exclusive
    x2 += 1
    y2 += 1
    z2 += 1

    # Apply offset (expand the region symmetrically, clipped to image bounds)
    x1 = max(0, x1 - offset)
    y1 = max(0, y1 - offset)
    z1 = max(0, z1 - offset)

    x2 = min(image.shape[0], x2 + offset)
    y2 = min(image.shape[1], y2 + offset)
    z2 = min(image.shape[2], z2 + offset)

    return ((x1, x2), (y1, y2), (z1, z2))
