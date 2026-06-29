from __future__ import annotations

import numpy as np

from surface_distance.metrics import (
    compute_dice_coefficient,
    compute_robust_hausdorff,
    compute_surface_distances,
    compute_average_surface_distance,
)


def compute_surface_dist_metrics(
    prediction: np.ndarray, 
    target: np.ndarray, 
    space_mm: tuple[float, float, float]
) -> dict[str, float]:
    
    """
    Computes surface distance metrics between the predicted and target binary masks.
    
    Args:
        prediction: A binary bool numpy array representing the predicted segmentation mask.
        target: A binary bool numpy array representing the ground truth segmentation mask.
        space_mm: A tuple of three floats representing the voxel spacing in millimeters along each axis (x, y, z).

    Returns:
        A dictionary containing the computed surface distance metrics:
            - "dice": Dice coefficient between the prediction and target masks.
            - "hd95": 95th percentile Hausdorff distance in millimeters.
            - "hd99": 99th percentile Hausdorff distance in millimeters.
            - "avg_surface_dist_gt_to_pred": Average surface distance from ground truth to prediction in millimeters.
            - "avg_surface_dist_pred_to_gt": Average surface distance from prediction to ground truth in millimeters.

    """
    # Compute the Dice coefficient
    dice = compute_dice_coefficient(target, prediction)

    # Compute surface distances from ground truth to prediction and vice versa
    dict_surface_distances = compute_surface_distances(
        target, prediction, spacing_mm=space_mm
    )

    # Compute the 95th and 99th percentile Hausdorff distances
    hd95 = compute_robust_hausdorff(dict_surface_distances, percent=95)
    hd99 = compute_robust_hausdorff(dict_surface_distances, percent=99)

    # Compute the average surface distances from ground truth to prediction and from prediction to ground truth
    avg_dist_gt_pred, avg_dist_pred_to_gt = compute_average_surface_distance(dict_surface_distances)

    return {
        "dice": np.round(dice, 4),
        "hd95": np.round(hd95, 4),
        "hd99": np.round(hd99, 4),
        "avg_surface_dist_gt_to_pred": np.round(avg_dist_gt_pred, 4),
        "avg_surface_dist_pred_to_gt": np.round(avg_dist_pred_to_gt, 4),
    }
