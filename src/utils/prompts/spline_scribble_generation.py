import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import binary_dilation
from skimage.draw import line

from utils.helpers.sample_slices import sample_slices_from_mask


def _select_spread_control_points(slice_coords, num_control_points, rng):
    """
    Select control points spread across the lesion in one slice.

    Picks the spanning axis (the one with greater extent in this slice),
    then chooses points near low percentiles (5-15th), high percentiles
    (80-95th), and equally-spaced percentiles in between, to avoid
    clustering all control points into a small region.

    Args:
        slice_coords: 2D coordinates (row, col) of the foreground pixels
            in the slice.
        num_control_points: Number of control points to select.
        rng: NumPy random generator.

    Returns:
        control_points: Selected control points, sorted along the
            spanning axis, in (row, col) format.
    """
    # Pick the axis with the larger spread to spread points along
    extent_0 = np.ptp(slice_coords[:, 0])
    extent_1 = np.ptp(slice_coords[:, 1])
    axis = 0 if extent_0 >= extent_1 else 1

    axis_vals = slice_coords[:, axis]
    order = np.argsort(axis_vals)
    sorted_coords = slice_coords[order]
    sorted_vals = axis_vals[order]

    n = num_control_points

    # Build target percentiles: low end, high end, equally spaced middle
    low_p = rng.uniform(5, 15)
    high_p = rng.uniform(80, 95)

    if n == 1:
        target_percentiles = np.array([50.0])
    elif n == 2:
        target_percentiles = np.array([low_p, high_p])
    else:
        target_percentiles = np.concatenate(
            [[low_p], np.linspace(low_p, high_p, n)[1:-1], [high_p]]
        )

    selected_idx = []
    used = np.zeros(len(sorted_coords), dtype=bool)

    for p in target_percentiles:
        target_val = np.percentile(sorted_vals, p)

        # Find closest available point to this target value along axis
        diffs = np.abs(sorted_vals - target_val)
        diffs[used] = np.inf  # exclude already-picked points

        idx = np.argmin(diffs)
        used[idx] = True
        selected_idx.append(idx)

    control_points = sorted_coords[np.sort(selected_idx)]
    return control_points


def generate_spline_scribble_prompts(
    pos_mask: np.ndarray,
    num_scribble_spline: int = 1,
    num_slices: int = 1,
    num_control_points: int = 3,
    rng: np.random.Generator | None = None
) -> np.ndarray:
    """
    Generate spline scribbles inside a binary lesion mask.

    Args:
        pos_mask: Boolean 3D lesion mask of shape (H, W, D) where foreground voxels are represented by True (or 1) 
            and background voxels by False (or 0).
        num_scribble_spline: Number of independent spline scribbles.
        num_control_points: Number of spline control points.
        rng: NumPy random generator.

    Returns:
        scribble_mask: boolean 3D mask containing the spline scribbles.

    Raises:
        ValueError: If the positive mask is not a NumPy array.
        ValueError: If the positive mask is not 3-dimensional.
        ValueError: If the positive mask is not binary.
        ValueError: If the number of spline scribbles is not positive.
        ValueError: If the number of control points is not greater than 2.
        ValueError: If the positive mask contains no foreground voxels.
    """

    # Input mask validation
    if not isinstance(pos_mask, np.ndarray):
        raise ValueError("Positive mask must be a numpy array.")
    if pos_mask.ndim != 3:
        raise ValueError("Positive mask must be a 3D array.")

    if not pos_mask.dtype == bool:
        raise ValueError("Positive mask must be binary (values 0/1 or bool).")
    if not np.any(pos_mask):
        raise ValueError("Positive mask contains no foreground voxels to generate prompts from.")

    # Parameters validation
    if num_scribble_spline <= 0:
        raise ValueError("Number of spline scribbles to generate must be greater than zero.")
    if num_control_points <= 2:
        raise ValueError("Number of control points must be greater than 2 to create a valid spline.")

    # Create a random generator if not provided
    if rng is None:
        rng = np.random.default_rng()

    scribble_mask = np.zeros_like(pos_mask, dtype=bool)
    min_control_points = 3  # Minimum number of control points required for a valid spline

    # Sample slices (z-axis)
    selected_slices = sample_slices_from_mask(
        pos_mask,
        num_slices=num_slices,
        axis=2,
        center_bias=2.0,
        rng=rng,
    )

    for _ in range(num_scribble_spline):

        for z_idx in selected_slices:

            slice_mask = pos_mask[:, :, z_idx]
            slice_coords = np.argwhere(slice_mask)

            if len(slice_coords) < num_control_points:
                continue

            control_points = _select_spread_control_points(
                slice_coords, num_control_points, rng
            )

            # Already sorted along the spanning axis inside the helper,
            # which gives a smoother, more spread-out spline than sorting
            # by column alone.

            try:
                k = min(min_control_points, len(control_points) - 1)
                tck, _ = splprep(
                    [control_points[:, 0], control_points[:, 1]],
                    s=0,
                    k=k,
                )
                num_samples = max(50, 20 * num_control_points)
                spline_points = np.array(
                    splev(np.linspace(0, 1, num_samples), tck)
                ).T
            except Exception:
                continue

            spline_points = np.round(spline_points).astype(int)

            # Draw spline in 2D slice
            for p1, p2 in zip(spline_points[:-1], spline_points[1:]):

                rr, cc = line(p1[0], p1[1], p2[0], p2[1])

                valid = (
                    (rr >= 0)
                    & (rr < pos_mask.shape[0])
                    & (cc >= 0)
                    & (cc < pos_mask.shape[1])
                )

                rr = rr[valid]
                cc = cc[valid]

                # Keep only inside lesion
                inside = slice_mask[rr, cc]

                scribble_mask[rr[inside], cc[inside], z_idx] = True

    # TODO: Consider adding a parameter to control the number of dilation iterations for the spline scribble. This would allow users to adjust the thickness of the spline scribble based on their specific needs or preferences.
    # But thickness > 1 can be too much for a single slice prompt, so we keep it at 1 for now.
    # Dilate scribble mask to increase visibility (optional, can be
    # commented out if not desired). Use a temp buffer since
    # binary_dilation does not safely support output aliasing input,
    # and dedupe slice indices in case sampling returned repeats.
    for z_idx in set(selected_slices):
        binary_dilation(scribble_mask[..., z_idx], iterations=1, output=scribble_mask[..., z_idx])

    return scribble_mask
