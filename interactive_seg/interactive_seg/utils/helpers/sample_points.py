import numpy as np
from scipy.ndimage import distance_transform_edt

def sample_point_with_bias(
    mask: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
    num_pts_sampled: int = 1,
) -> tuple[tuple[int, ...], ...]:
    """
    Sample multiple pixels/voxels according to:
        p(x) ∝ D(x)^alpha

    which means that the probability of sampling a pixel/voxel is proportional to the distance to the nearest background pixel/voxel raised to the power of alpha. This biases the sampling towards the center of the mask.

    Works for both 2D and 3D masks.

    Args:
        mask: Binary mask of shape (H, W) or (D, H, W).
        alpha: Exponent for the distance transform. Higher values bias sampling towards the center of the mask. If alpha=0, sampling is uniform and alpha>0 biases towards the center.
        rng: Random number generator for reproducibility.
        num_pts_sampled: Number of points to sample.

    Returns:
        Tuple of (x, y) or (x, y, z) coordinates.
    """
    if mask.sum() == 0:
        raise ValueError("Cannot sample from empty mask.")

    if mask.ndim not in (2, 3):
        raise ValueError("Mask must be 2D or 3D.")

    # Distance transform
    dist = distance_transform_edt(mask).astype(np.float32)
    dist += 1e-8

    # Probability distribution
    probs = dist ** alpha
    probs = probs.ravel()
    probs /= probs.sum()

    # Sample indices
    idxs = rng.choice(
        probs.size,
        size=num_pts_sampled,
        p=probs,
        replace=True,
    )

    # Convert to coordinates
    coords = np.array(np.unravel_index(idxs, mask.shape)).T

    return tuple(tuple(int(c) for c in point) for point in coords)
