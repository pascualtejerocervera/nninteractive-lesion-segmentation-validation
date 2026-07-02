import numpy as np
from collections.abc import Mapping

Point3D = tuple[tuple[int, int, int], ...]
BBox3D = tuple[tuple[int, int, int, int, int, int], ...]

def extract_labels_from_prompts(
    prompts_dict: dict[str, dict[int, Point3D] | dict[int, BBox3D] | np.ndarray]
) -> list[int]:
    """
    Extract unique label IDs from heterogeneous prompt inputs.

    Args:
        prompts_dict: Dictionary mapping prompt names to their content.
            Supported formats:
            - Keys containing "pts" or "bboxes":
                dict[int, Point3D] or dict[int, BBox3D]
            - Keys containing "scribble_diameter_ann" or "scribble_spline":
                np.ndarray of integer labels (0 is ignored)

    Returns:
        list[int]: Sorted tuple of unique label IDs found across all prompts.
    """

    labels_detected: set[int] = set()

    for prompt_name, prompt_content in prompts_dict.items():

        # routing based on naming convention (fragile but simple heuristic)
        if "pts" in prompt_name or "bboxes" in prompt_name:
            if isinstance(prompt_content, Mapping):
                labels_detected.update(prompt_content.keys())  # extract dict keys as labels

        elif "scribble_diameter_ann" in prompt_name or "scribble_spline" in prompt_name:
            if isinstance(prompt_content, np.ndarray):
                # extract unique non-zero labels from array-like input
                labels_detected.update(int(x) for x in np.unique(prompt_content) if int(x) != 0)
                # labels_detected.update(np.nonzero(np.bincount(prompt_content.ravel()))[0][1:].tolist())
            elif isinstance(prompt_content, Mapping):
                # If the content is a dictionary, extract labels from the keys
                labels_detected.update(prompt_content.keys())

    # deterministic output
    return sorted(labels_detected)

def has_labels(
    prompts_dict: dict[str, dict[int, object] | np.ndarray]
) -> bool:
    """
    Check whether any recognized prompt contains at least one non-zero label.

    Args:
        prompts_dict:
            Dictionary mapping prompt names to prompt content.

            Supported formats:
            - "pts" / "bboxes": dict[int, ...] (labels in keys)
            - "scribble_diameter_ann" / "scribble_spline": np.ndarray (labels in values)

    Returns:
        bool:
            True if at least one non-zero label exists, False otherwise.
    """

    for prompt_name, prompt_content in prompts_dict.items():

        # dict-based prompts (labels stored in keys)
        if "pts" in prompt_name or "bboxes" in prompt_name:
            if any(k != 0 for k in prompt_content.keys()):
                return True

        # array-based prompts (labels stored in values)
        elif "scribble_diameter_ann" in prompt_name or "scribble_spline" in prompt_name:
            if isinstance(prompt_content, np.ndarray):
                if np.any(prompt_content != 0):
                    return True
            elif isinstance(prompt_content, Mapping):
                if any(k != 0 for k in prompt_content.keys()):  
                    return True
    return False
