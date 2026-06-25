import numpy as np
import torch

def compute_nonzero_bbox(
        image: np.ndarray | torch.Tensor,
        margin: int = 0
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """
    Compute the bounding box of the non-zero region in a 3D image, with an optional margin.

    Returns half-open intervals [[x1, x2], [y1, y2], [z1, z2]] in original-image
    coordinates, i.e. each axis's upper bound is exclusive, so the region can be
    used directly for slicing/cropping:

        image[x1:x2, y1:y2, z1:z2]

    This matches the bounding-box convention used by nnInteractive (and nnU-Net),
    where crop/bbox coordinates are stored as half-open [start, stop) ranges per
    axis rather than inclusive min/max indices.

    Args:
        image: Input 3D image (numpy array or torch tensor). Format uint8, shape (H, W, D).
        margin: Margin to apply to the bounding box. Format int, shape (6,).

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

    # Apply margin (expand the region symmetrically, clipped to image bounds)
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    z1 = max(0, z1 - margin)

    x2 = min(image.shape[0], x2 + margin)
    y2 = min(image.shape[1], y2 + margin)
    z2 = min(image.shape[2], z2 + margin)

    return ((x1, x2), (y1, y2), (z1, z2))

def crop_nonzero_bbox_with_margin(
        image: np.ndarray | torch.Tensor,
        margin: int = 0
    ) -> tuple[np.ndarray | torch.Tensor, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """
    Crop a 3D image to its non-zero region.

    Args:
        image: Input 3D image (numpy array or torch tensor). Format uint8, shape (H, W, D).
        margin: Margin to apply to the bounding box. Format int, shape (6,).

    Returns:
        A tuple containing:
            - The cropped image as a numpy array or torch tensor.
            - The bounding box of the non-zero region as half-open intervals
              ((x1, x2), (y1, y2), (z1, z2)), where x2/y2/z2 are exclusive.
    """
    bbox = compute_nonzero_bbox(image, margin)
    (x1, x2), (y1, y2), (z1, z2) = bbox
    cropped_image = image[x1:x2, y1:y2, z1:z2]
    return cropped_image, bbox

def crop_image_with_bbox(
        image: np.ndarray | torch.Tensor,
        bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ) -> np.ndarray | torch.Tensor:
    """
    Crop a 3D image using a specified bounding box.

    Args:
        image: Input 3D image (numpy array or torch tensor). Format uint8, shape (H, W, D).
        bbox: Bounding box as half-open intervals ((x1, x2), (y1, y2), (z1, z2)), where x2/y2/z2 are exclusive.

    Returns:
        The cropped image as a numpy array or torch tensor.
    """
    (x1, x2), (y1, y2), (z1, z2) = bbox
    cropped_image = image[x1:x2, y1:y2, z1:z2]
    return cropped_image
