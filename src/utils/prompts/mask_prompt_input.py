from dataclasses import dataclass

import numpy as np

from config.nninteractive.prompt_generation_config import NNInteractivePromptGenerationConfig
from utils.helpers.create_ref_masks import create_surface_band_mask, create_neg_mask_from_pos_mask

@dataclass
class MaskPromptInput:
    """
    Dataclass to hold the input for mask-based prompt generation.
    
    Attributes:
        config: An instance of NNInteractivePromptGenerationConfig containing the configuration parameters for 
            prompt generation.
        pos_mask: A boolean array of shape (H, W, D) where True values indicate the foreground (positive mask) and
            False values indicate the background. It is used to generate positive prompts such as positive points, bounding boxes, diameter annotations, and spline scribbles.
        neg_mask: A boolean array of shape (H, W, D) where True values indicate the background (negative mask) 
            and False values indicate the foreground. It is used to generate negative prompts such as negative points.
        surface_band_mask: A boolean array of shape (H, W, D) where True values indicate the boundary region of the 
            foreground mask and False values indicate the background. It is used to create a surface band mask which is the boundary region of the lesion. This area is uncertain and no prompt
        """
    pos_mask: np.ndarray
    neg_mask: np.ndarray
    surface_band_mask: np.ndarray
    label: int

def create_mask_prompt_input(
    config: NNInteractivePromptGenerationConfig | dict,
    mask_label: np.ndarray,
    label: int
) -> MaskPromptInput:
    """
    Create a MaskPromptInput dataclass instance from the given configuration and binary mask.

    Args:
        config: An instance of NNInteractivePromptGenerationConfig or a dictionary containing the 
            configuration parameters for prompt generation.
        mask_label: A binary 3D numpy array representing the segmentation mask for a specific label, where 
            True values indicate the presence of the lesion (foreground) and False values indicate the background.
        label: The integer label corresponding to the foreground region in the mask.

    Returns:
        An instance of MaskPromptInput containing the configuration, foreground mask, background mask, and image shape.
    """
    # Validate the config if it is provided as a dictionary and convert it to an instance of NNInteractivePromptGenerationConfig
    if isinstance(config, dict):
        config = NNInteractivePromptGenerationConfig.model_validate(config)
    elif not isinstance(config, NNInteractivePromptGenerationConfig):
        raise ValueError("Config must be an instance of NNInteractivePromptGenerationConfig or a dictionary.")
    
    # Validate the label mask
    if not isinstance(mask_label, np.ndarray):
        raise ValueError("Input mask must be a numpy array.")
    if mask_label.ndim != 3:
        raise ValueError("Input mask must be a 3D array.")
    if not mask_label.dtype == bool:        
        raise ValueError("Input mask must be binary (values 0/1 or bool).")
    if not np.any(mask_label):
        raise ValueError("Input mask contains no foreground voxels to generate prompts from.")

    # Validate the label
    if not isinstance(label, int) or label <= 0:
        raise ValueError("Label must be a positive integer.")
    
    # Create the surface band mask for the current label. It detects the boundary region of the foreground mask and dilates it to create a thicker boundary region. This is used to avoid generating prompts in uncertain areas.
    surface_band_mask = create_surface_band_mask(
        pos_mask=mask_label,
        dilation_iter=config.surface_band_mask_iter,
    )

    # Creating the positive mask by subtracting the surface band mask from the original label mask to avoid overlap with the uncertain boundary region
    pos_mask = np.zeros_like(mask_label, dtype=bool)
    np.logical_and(mask_label, np.logical_not(surface_band_mask), out=pos_mask)

    # Creating the negative mask by combining the positive mask and surface band mask to create a reference mask for negative sampling, then dilating it and subtracting the original positive mask to create a ring-shaped negative sampling region around the lesion
    tmp_mask = np.zeros_like(mask_label, dtype=bool)

    # Combine the positive mask and surface band mask to create a reference mask for negative sampling
    np.logical_or(pos_mask, surface_band_mask, out=tmp_mask)  

    # Create the negative mask by dilating the reference mask and subtracting the original positive mask to create a ring-shaped negative sampling region around the lesion
    neg_mask = create_neg_mask_from_pos_mask(
        pos_mask=tmp_mask,
        dilation_iter=config.neg_mask_dilation_iter
    )

    return MaskPromptInput(
        pos_mask=pos_mask,
        neg_mask=neg_mask,
        surface_band_mask=surface_band_mask,
        label=label
    )
