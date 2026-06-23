import numpy as np

from utils.helpers.create_ref_masks import create_neg_mask_from_pos_mask
from utils.helpers.sample_slices import sample_slices_from_mask

def generate_pts_prompts(
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    num_pts_pos: int = 0,
    num_pts_neg: int = 0,
    num_slices: int = 1,
    neg_pts_dilation_iter: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """Generate positive and/or negative point prompts from a 3D mask.

    Foreground slices are first sampled along the axial direction with a
    bias towards the center of the foreground extent. Positive points are
    then sampled uniformly from foreground voxels within the selected
    slices. Negative points are sampled from a background region obtained
    by dilating the foreground mask and inverting the result.

    Args:
        pos_mask: Binary 3D mask of shape ``(H, W, D)`` where foreground voxels
            are represented by ``True`` (or ``1``) and background voxels by
            ``False`` (or ``0``).
        num_pts_pos: Number of positive points to sample per selected slice.
        num_pts_neg: Number of negative points to sample per selected slice.
        neg_pts_dilation_iter: Number of binary dilation iterations used to
            create the negative sampling region from the foreground mask.
        num_slices: Number of axial slices to sample from the foreground
            extent of the mask.
        rng: NumPy random number generator. If ``None``, a new generator is
            created via ``np.random.default_rng()``.

    Returns:
        A tuple ``(positive_points, negative_points)`` where each element is
        a tuple of point coordinates in ``(x, y, z)`` format. Either tuple
        may be empty if the corresponding number of requested points is
        zero.

    Raises:
        ValueError: If the foreground or background mask is not a NumPy array.
        ValueError: If the foreground or background mask is not 3-dimensional.
        ValueError: If the foreground or background mask is not binary.
        ValueError: If the number of positive and negative points to generate
            is not positive.
        ValueError: If the number of slices to sample is not positive.
        ValueError: If the number of dilation iterations for negative point
            generation is less than 1.
        ValueError: If the foreground mask contains no foreground voxels.
    """
    # Input mask validation
    if not isinstance(pos_mask, np.ndarray) or not isinstance(neg_mask, np.ndarray):
        raise ValueError("Foreground and background masks must be numpy arrays.")
    if pos_mask.ndim != 3 or neg_mask.ndim != 3:
        raise ValueError("Foreground and background masks must be 3D arrays.")
    
    is_binary_pos_mask = (pos_mask.dtype == bool) or (pos_mask.max() <= 1 and pos_mask.min() >= 0)
    if not is_binary_pos_mask:
        raise ValueError("Foreground mask must be binary (values 0/1 or bool).")
    is_binary_neg_mask = (neg_mask.dtype == bool) or (neg_mask.max() <= 1 and neg_mask.min() >= 0)
    if not is_binary_neg_mask:
        raise ValueError("Background mask must be binary (values 0/1 or bool).")
    
    # Parameters validation
    if num_pts_pos <= 0 and num_pts_neg <= 0:
        raise ValueError("Number of positive or negative points to generate must be greater than zero.")
    if num_slices <= 0:
        raise ValueError("Number of slices to sample must be greater than zero.")
    if neg_pts_dilation_iter < 1:
        raise ValueError("Number of dilation iterations for negative point generation must be at least 1.")

    # Create a random generator if not provided
    if rng is None:
        rng = np.random.default_rng()
    
    # Select the reference mask for slice sampling based on the requested number of positive and negative points
    if num_pts_pos > 0 and np.any(pos_mask):
        ref_mask = pos_mask
    elif num_pts_neg > 0 and np.any(neg_mask):
        ref_mask = neg_mask
    else:
        raise ValueError(
            "No valid reference mask found. "
            "Expected either pos_mask or neg_mask with non-empty content."
        )
        
    # Sample slice indices from the mask with a bias towards the center of the foreground extent along the axial direction
    sampled_slices_idx = sample_slices_from_mask(
        ref_mask,
        num_slices=num_slices, 
        axis=2, # Sample slices along the axial direction
        center_bias=2.0, 
        rng=rng
    )
    
    # Output containers: sampled (x, y, slice_idx) coordinates for positive and negative points
    sampled_slice_idx_pos, sampled_slice_idx_neg = [], []

    # TODO: Maybe a bit redundant logic of checking for foreground voxels in the selected slice, but it ensures that we don't attempt to sample from an empty slice.
    # Iterate over selected slices (sampling done per 2D slice)
    for slice_idx in sampled_slices_idx:

        # Foreground sampling
        if num_pts_pos > 0:
            # Extract 2D foreground mask for current slice
            slice_pos_mask = pos_mask[:, :, slice_idx]

            # Check if there are any foreground voxels in the selected slice
            if np.any(slice_pos_mask):
                # Get all foreground coordinates (y, x)
                coords_pos = np.argwhere(slice_pos_mask)

                # Randomly select up to num_pts_pos points
                idx = rng.choice(
                    coords_pos.shape[0],
                    size=min(num_pts_pos, coords_pos.shape[0]),
                    replace=False,
                )

                # Store sampled points as (x, y, slice_idx)
                for x, y in coords_pos[idx]:
                    sampled_slice_idx_pos.append((int(x), int(y), slice_idx))
            else:
                raise ValueError(
                    f"No foreground voxels found in the selected slice {slice_idx} for positive point sampling."
                )

        # Background sampling
        if num_pts_neg > 0:
            # Extract 2D background mask for current slice
            slice_neg_mask = neg_mask[:, :, slice_idx]

            # Check if there are any background voxels in the selected slice
            if np.any(slice_neg_mask):
                coords_neg = np.argwhere(slice_neg_mask)

                # Randomly select up to num_pts_neg points
                idx = rng.choice(
                    coords_neg.shape[0],
                    size=min(num_pts_neg, coords_neg.shape[0]),
                    replace=False,
                )

                # Store sampled points as (x, y, slice_idx)
                for x, y in coords_neg[idx]:
                    sampled_slice_idx_neg.append((int(x), int(y), slice_idx))
            else:
                raise ValueError(
                    f"No background voxels found in the selected slice {slice_idx} for negative point sampling."
                )
        
    return tuple(sampled_slice_idx_pos), tuple(sampled_slice_idx_neg)
