import numpy as np


def sample_slices_from_mask(
    pos_mask: np.ndarray,
    num_slices: int = 1,
    axis: int = 0,
    center_bias: float = 2.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample foreground-containing slices from a mask.

    Slice sampling is biased towards the center of the foreground extent
    along the specified axis. The weighting strategy is adapted from
    nnInteractive:
    https://github.com/MIC-DKFZ/nnInteractive/blob/fca24ab26dbc5d4a4a276fefc5e1d87631c325f3/nnInteractive/supervoxel/src/supervoxel.py#L136-L139

    Args:
        pos_mask: Boolean mask containing the foreground region.
        num_slices: Number of slices to sample.
        axis: Axis along which slices are sampled.
        center_bias: Strength of the bias towards the center of the
            foreground extent. A value of 0 corresponds to uniform
            sampling.

    Returns:
        Sampled slice indices along ``axis``.

    Raises:
        ValueError: If the foreground mask contains no foreground voxels.
    """
    
    fg_slices = np.where(pos_mask.any(axis=tuple(i for i in range(pos_mask.ndim) if i != axis)))[0]

    if len(fg_slices) == 0:
        raise ValueError("Foreground mask contains no foreground voxels.")

    probabilities = np.exp(
        -np.linspace(-1, 1, len(fg_slices)) ** 2 * center_bias
    )
    probabilities /= probabilities.sum()

    if rng is None:
        rng = np.random.default_rng()

    sampled_positions = rng.choice(
        len(fg_slices),
        size=num_slices,
        p=probabilities,
    )

    return tuple(int(fg_slices[pos]) for pos in sampled_positions)