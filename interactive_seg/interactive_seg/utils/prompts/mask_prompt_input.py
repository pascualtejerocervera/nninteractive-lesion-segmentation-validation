from dataclasses import dataclass

import numpy as np

from interactive_seg.config.prompt_generation_config import NNInteractivePromptGenerationConfigBase
from interactive_seg.utils.geometry.crop_nonzero_bbox import compute_nonzero_bbox, crop_nonzero_bbox_with_margin, crop_image_with_bbox
from interactive_seg.utils.helpers.create_ref_masks import create_surface_band_mask, create_neg_mask_from_pos_mask

@dataclass
class MaskPromptInput:
    """
    Dataclass to hold the input for mask-based prompt generation.
    
    Attributes:
        config: An instance of NNInteractivePromptGenerationConfigBase containing the configuration parameters for 
            prompt generation.
        pos_mask: A boolean array of shape (H, W, D) where True values indicate the foreground (positive mask) and
            False values indicate the background. It is used to generate positive prompts such as positive points, bounding boxes, diameter annotations, and spline scribbles.
        neg_mask: A boolean array of shape (H, W, D) where True values indicate the background (negative mask) 
            and False values indicate the foreground. It is used to generate negative prompts such as negative points.
        surface_band_mask: A boolean array of shape (H, W, D) where True values indicate the boundary region of the 
            foreground mask and False values indicate the background. It is used to create a surface band mask which is the boundary region of the lesion. This area is uncertain and no prompts should be generated in this region.
        label: An integer representing the label of the foreground region in the mask. It is used to identify 
            the specific label for which prompts are being generated.
        crop_mask: An optional boolean array of shape (H, W, D) representing the cropped version of the union 
            of the positive, surface band, and negative masks. It is used to create a bounding box around the lesion and its surrounding region for prompt generation of scribble prompts (diameter and spline). If cropping is not applied, this attribute will be None.
        crop_bbox: An optional tuple of tuples representing the bounding box coordinates of the cropped region.
            The format is ((x_min, x_max), (y_min, y_max), (z_min, z_max)). If cropping is not applied, this attribute will be None.
        crop_pos_mask: An optional array of shape (H, W, D) representing the cropped version of the positive mask. 
            It is used after the creation of the prompts to ensure the scribble prompts are within the positive mask. If cropping is not applied, this attribute will be None. It is necessary to have this attribute to match the shapes of the cropped scribble prompts and the cropped positive mask for further processing.
    """
    pos_mask: np.ndarray
    neg_mask: np.ndarray
    surface_band_mask: np.ndarray
    label: int
    crop_mask: np.ndarray | None = None  
    crop_bbox: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None = None  
    crop_pos_mask: np.ndarray | None = None

def create_mask_prompt_input(
    config: NNInteractivePromptGenerationConfigBase | dict,
    mask_label: np.ndarray,
    label: int
) -> MaskPromptInput:
    """
    Create a MaskPromptInput dataclass instance from the given configuration and binary mask.

    Args:
        config: An instance of NNInteractivePromptGenerationConfigBase or a dictionary containing the
            configuration parameters for prompt generation.
        mask_label: A binary 3D numpy array representing the segmentation mask for a specific label, where 
            True values indicate the presence of the lesion (foreground) and False values indicate the background.
        label: The integer label corresponding to the foreground region in the mask.

    Returns:
        An instance of MaskPromptInput containing the configuration, foreground mask, background mask, and image shape.
    """
    # Validate the config if it is provided as a dictionary and convert it to an instance of NNInteractivePromptGenerationConfig
    if isinstance(config, dict):
        config = NNInteractivePromptGenerationConfigBase.model_validate(config)
    elif not isinstance(config, NNInteractivePromptGenerationConfigBase):
        raise ValueError("Config must be an instance of NNInteractivePromptGenerationConfigBase or a dictionary.")

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
        dilation_iter=config.prompt_generation_config.surface_band_mask_iter
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
        dilation_iter=config.prompt_generation_config.neg_mask_dilation_iter
    )

    # Compute the cropped version of the union of the positive, surface band, and negative masks to create a bounding box around the lesion and its surrounding region. This is used to crop the input image and masks for prompt generation of scribble prompts (diameter and spline) to reduce the search space and improve prompt generation efficiency.
    crop_mask, crop_bbox, crop_pos_mask = None, None, None
    if config.crop_roi_config.enable:
        margin = config.crop_roi_config.compute_margin(
            bbox=compute_nonzero_bbox(
                image=np.logical_or.reduce([pos_mask, surface_band_mask, neg_mask]),
                margin=0  # No margin for computing the bounding box, as we will apply the margin later)
            )
        )
        crop_mask, crop_bbox = crop_nonzero_bbox_with_margin(mask_label, margin=margin)
        crop_pos_mask = crop_image_with_bbox(pos_mask, crop_bbox)

    return MaskPromptInput(
        pos_mask=pos_mask,
        neg_mask=neg_mask,
        surface_band_mask=surface_band_mask,
        label=label,
        crop_mask=crop_mask,
        crop_bbox=crop_bbox,
        crop_pos_mask=crop_pos_mask,
    )
