from typing import Literal

import numpy as np

OffsetMode = Literal["max_relative", "per_axis_relative", "clamped_max_relative"]


def compute_max_relative_margin(
    bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    fraction: float = 1 / 3,
) -> int:
    """
    Compute an margin based on the maximum extent of the bounding box.

    Args:
        bbox: A tuple representing the bounding box in the format ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
        fraction: A float representing the fraction of the maximum extent to use as the margin. Default is 1/3.

    Returns:
        An integer representing the computed margin based on the maximum extent of the bounding box.
    """
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = bbox
    extents = (xmax - xmin, ymax - ymin, zmax - zmin)
    max_extent = max(extents)
    margin = int(round(max_extent * fraction))
    return margin


def compute_per_axis_margin(
    bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    fraction: float = 1 / 3,
) -> tuple[int, int, int]:
    """
    Compute margins for each axis based on the extents of the bounding box.

    Args:
        bbox: A tuple representing the bounding box in the format ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
        fraction: A float representing the fraction of the extent to use as the margin. Default is 1/3.

    Returns:
        A tuple of margins for each axis (ox, oy, oz) based on the extents of the bounding box.
    """
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = bbox
    extents = np.array([xmax - xmin, ymax - ymin, zmax - zmin])
    margins = np.round(extents * fraction).astype(int)
    return tuple(int(o) for o in margins)  # Return as a tuple of ints


def compute_clamped_max_relative_margin(
    bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    fraction: float = 1 / 3,
    min_margin: int = 5,
    max_margin: int = 64,
) -> int:
    """
    Compute an margin based on the maximum extent of the bounding box, constrained by minimum and maximum values.

    Args:
        bbox: A tuple representing the bounding box in the format ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
        fraction: A float representing the fraction of the maximum extent to use as the margin. Default is 1/3.
        min_margin: An integer representing the minimum allowed margin. Default is 5.
        max_margin: An integer representing the maximum allowed margin. Default is 64.

    Returns:
        An integer representing the computed margin based on the maximum extent of the bounding box, constrained by the specified minimum and maximum values.
    """
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = bbox
    max_extent = max(xmax - xmin, ymax - ymin, zmax - zmin)
    margin = int(round(max_extent * fraction))
    return max(min_margin, min(margin, max_margin))


def compute_margin(
    mode: OffsetMode,
    bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    fraction: float = 1 / 3,
    min_margin: int = 5,
    max_margin: int = 64,
) -> int | tuple[int, int, int]:
    """
    Dispatch to the margin computation matching `mode`.

    Args:
        mode: Which computation to use:
            - "absolute": fixed margin (int).
            - "max_relative": fraction of the largest axis extent, unclamped (int).
            - "per_axis_relative": fraction applied independently per axis (tuple of 3 ints).
            - "clamped_max_relative": fraction of the largest axis extent, clamped to
              [min_margin, max_margin] (int).
        bbox: ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
        fraction: Fraction of the relevant extent to use as the margin.
        min_margin: Lower clamp bound. Only used by "clamped_max_relative".
        max_margin: Upper clamp bound. Only used by "clamped_max_relative".

    Returns:
        An int for "max_relative"/"clamped_max_relative", or a tuple[int, int, int]
        (oz, oy, ox) for "per_axis_relative".

    Raises:
        ValueError: If `mode` is not one of the supported OffsetMode values.
    """
    if mode == "max_relative":
        return compute_max_relative_margin(bbox=bbox, fraction=fraction)

    if mode == "per_axis_relative":
        return compute_per_axis_margin(bbox=bbox, fraction=fraction)

    if mode == "clamped_max_relative":
        return compute_clamped_max_relative_margin(
            bbox=bbox,
            fraction=fraction,
            min_margin=min_margin,
            max_margin=max_margin,
        )

    raise ValueError(
        f"Unknown margin mode: {mode!r}. "
        f"Expected one of: 'max_relative', 'per_axis_relative', 'clamped_max_relative'."
    )
