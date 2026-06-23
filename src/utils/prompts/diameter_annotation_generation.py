import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import cdist
from skimage.draw import line
from skimage.measure import find_contours

from utils.helpers.sample_slices import sample_slices_from_mask


def generate_diameter_annotation_prompts(
    pos_mask: np.ndarray,
    num_slices: int = 1, # TODO: This parameter is currently not used, as diameter annotation is typically generated on a single slice. It can be extended in the future to generate diameter annotations on multiple slices if needed.
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Generate a diameter annotation prompt for a binary lesion mask.

    A single axial slice is sampled with probability biased toward the
    lesion center. On that slice, the two contour points with the largest
    Euclidean separation are connected with a straight line.

    Args:
        pos_mask: Boolean 3D lesion mask of shape (H, W, D) where foreground voxels are represented by True (or 1) 
            and background voxels by False (or 0).
        num_slices: Number of axial slices to sample from the foreground extent of the mask (default is 1, as diameter annotation is typically generated on a single slice).
        rng: NumPy random generator.

    Returns:
        diameter_mask: boolean 3D mask containing the diameter annotation line.

    Raises:
        ValueError: If the foreground mask is not a NumPy array.
        ValueError: If the foreground mask is not 3-dimensional.
        ValueError: If the foreground mask is not binary.
        ValueError: If the foreground mask contains no foreground voxels.
    """

    # Input mask validation
    if not isinstance(pos_mask, np.ndarray):
        raise ValueError("Foreground mask must be a numpy array.")
    if pos_mask.ndim != 3:
        raise ValueError("Foreground mask must be a 3D array.")

    is_binary = (pos_mask.dtype == bool) or (pos_mask.max() <= 1 and pos_mask.min() >= 0)
    if not is_binary:
        raise ValueError("Foreground mask must be binary (values 0/1 or bool).")
    if not np.any(pos_mask):
        raise ValueError("Foreground mask contains no foreground voxels to generate prompts from.")

    # Parameters validation
    if num_slices <= 0:
        raise ValueError("Number of slices to sample must be greater than zero.")

    # Create a random generator if not provided
    if rng is None:
        rng = np.random.default_rng()

    # Sample an axial slice with probability biased toward the lesion center
    z = sample_slices_from_mask(pos_mask, num_slices=1, axis=2, center_bias=2.0, rng=rng)[0]

    # Find contours at the half-maximum of the binary mask (0.5 is a common choice for binary masks)
    contours = find_contours(image=pos_mask[..., z], level=0.5)
    if not contours:
        raise ValueError(
            "No contours found on the selected slice."
        )

    # Stack all contour points into a single array for distance computation
    contour_points = np.vstack(contours)
    if len(contour_points) < 2:
        raise ValueError(
            "Not enough contour points found on the selected slice."
        )

    # Compute pairwise distances between contour points and find the two points with the largest separation
    distances = cdist(contour_points, contour_points)
    i, j = np.unravel_index(np.argmax(distances), distances.shape)

    p1 = np.round(contour_points[i]).astype(int)
    p2 = np.round(contour_points[j]).astype(int)

    rr, cc = line(p1[0], p1[1], p2[0], p2[1])

    # Rounding contour coordinates can occasionally push a pixel just
    # outside the array bounds, so clip defensively before indexing.
    valid = (
        (rr >= 0)
        & (rr < pos_mask.shape[0])
        & (cc >= 0)
        & (cc < pos_mask.shape[1])
    )
    rr = rr[valid]
    cc = cc[valid]

    diameter_mask = np.zeros_like(pos_mask, dtype=bool)
    diameter_mask[rr, cc, z] = True

    # TODO: Consider adding a parameter to control the number of dilation iterations for the diameter annotation line. This would allow users to adjust the thickness of the diameter annotation based on their specific needs or preferences.
    # TODO: Check if this necessart as it may disconnect the diameter annotation from the lesion, but it can make the diameter annotation more visible
    # Dilate the diameter annotation line to make it more visible.
    # Dilate into a temp buffer since binary_dilation does not safely
    # support output aliasing input, then re-apply the label value.
    binary_dilation(diameter_mask[..., z], iterations=1, output=diameter_mask[..., z])  

    return diameter_mask