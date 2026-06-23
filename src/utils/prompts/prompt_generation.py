from logging import config
from pathlib import Path

import numpy as np

from config.nninteractive_prompt_generation_config import NNInteractivePromptGenerationConfig

from utils.prompts.bbox_generation import generate_bbox_prompts
from utils.prompts.diameter_annotation_generation import generate_diameter_annotation_prompts
from utils.prompts.mask_prompt_input import create_mask_prompt_input
from utils.prompts.pts_generation import generate_pts_prompts
from utils.prompts.spline_scribble_generation import generate_spline_scribble_prompts

Point3D = tuple[tuple[int, int, int], ...]
BBox3D = tuple[tuple[int, int, int, int, int, int], ...]

def generate_nninteractive_prompts(
    config: NNInteractivePromptGenerationConfig | dict,
    mask: np.ndarray,
    affine, 
    header
) -> dict[str,  dict[int, Point3D] | dict[int, BBox3D] | np.ndarray]:
    """
    Generates prompts for the nnInteractive model.

    Args:
        config: An instance of NNInteractivePromptGenerationConfig or a dictionary containing the configuration parameters for prompt generation
        mask: A 3D numpy array representing the segmentation mask, where each unique positive integer value corresponds to a different label (e.g., 1 for label A, 2 for label B, etc.), and 0 represents the background.
        rng: A numpy random generator for reproducibility.
    Returns:
        prompts_dict: A dictionary containing the generated prompts for each label in the mask. The keys of the dictionary are:
            - "pos_pts": A dictionary where each key is a label and the value is a tuple of positive point prompts for that label.
            - "neg_pts": A dictionary where each key is a label and the value is a tuple of negative point prompts for that label
            - "pos_bboxes": A dictionary where each key is a label and the value is a tuple of bounding box prompts for that label.
            - "diameter_annotations": A 3D numpy array representing the diameter annotation prompts for all labels combined. The array has the same shape as the input mask, with positive values indicating the presence of diameter annotations.
            - "spline_scribbles": A 3D numpy array representing the spline scribble prompts for all labels combined.
    """
    # Validate the config if it is provided as a dictionary and convert it to an instance of NNInteractivePromptGenerationConfig
    if isinstance(config, dict):
        config = NNInteractivePromptGenerationConfig.model_validate(config)
    elif not isinstance(config, NNInteractivePromptGenerationConfig):
        raise ValueError("Config must be an instance of NNInteractivePromptGenerationConfig or a dictionary.")

    rng = np.random.default_rng(config.seed)  # Ensure reproducibility by using the provided seed
    labels_unique = tuple(int(label) for label in np.unique(mask) if int(label) != 0)  # Exclude background label (0)
    if len(labels_unique) == 0:
        raise ValueError("The mask does not contain any positive labels to generate prompts from.")
    
    prompts_dict = {}
    mask_label = np.zeros_like(mask, dtype=bool)  # Initialize mask_label to avoid potential reference before assignment

    for target_label in labels_unique:
        np.equal(mask, target_label, out=mask_label)  # In-place comparison to create a binary mask for the current label

        # Create the mask prompt input for the current label
        mask_prompt_input = create_mask_prompt_input(
            config=config,
            mask_label=mask_label,
            label=target_label
        )

        # Generate positive or negative from the original mask depending on the requested prompt types in the config (if only negative points are requested, the positive mask is not needed and vice versa)
        if config.num_pts_pos > 0 or config.num_pts_neg > 0:
            pos_pts, neg_pts = generate_pts_prompts(
                pos_mask=mask_prompt_input.pos_mask,
                neg_mask=mask_prompt_input.neg_mask,
                num_pts_pos=config.num_pts_pos,
                num_pts_neg=config.num_pts_neg, 
                num_slices=config.num_slices_pts, 
                neg_pts_dilation_iter=config.neg_pts_dilation_iter, # Ignored if num_pts_neg is 0
                rng=rng
            )
            if pos_pts:
                prompts_dict.setdefault("pos_pts", {})[target_label] = pos_pts 
            if neg_pts:
                prompts_dict.setdefault("neg_pts", {})[target_label] = neg_pts

        if config.num_bbox_pos > 0:
            # Generate bounding box prompts
            pos_bboxes = generate_bbox_prompts(
                pos_mask=mask_prompt_input.pos_mask,
                num_bboxes=config.num_bbox_pos, 
                num_slices=config.num_slices_bbox, 
                rng=rng
            )
            prompts_dict.setdefault("pos_bboxes", {})[target_label] = pos_bboxes

        if config.diameter_annotation:  
            # Generate diameter annotation prompts 
            diameter_annotation = generate_diameter_annotation_prompts(
                pos_mask=mask_prompt_input.pos_mask,
                num_slices=config.num_slices_diameter,  # Diameter annotation is typically generated on a single slice 
                rng=rng
            )
            
            # TODO: Check if this is necessary as the dilation can make the diameter annotation extend beyond the positive mask, but it may disconnect the diameter annotation from
            # Ensure diameter annotations are within the positive mask
            # np.logical_and(diameter_annotation, mask_prompt_input.pos_mask, out=diameter_annotation)

            # Assign the target label to the diameter annotation mask
            diameter_annotation = diameter_annotation.astype(np.uint8, copy=False)
            np.multiply(diameter_annotation, target_label, out=diameter_annotation)

            # Add the diameter annotation mask to the prompts_dict, combining with existing annotations if necessary
            if "diameter_annotations" not in prompts_dict:
                prompts_dict.setdefault("diameter_annotations", diameter_annotation)
            else:
                # Element-wise maximum to combine diameter annotations for different labels (labels should not overlap)
                np.maximum(
                    prompts_dict["diameter_annotations"],
                    diameter_annotation,
                    out=prompts_dict["diameter_annotations"]
                )

        if config.spline_scribble:
            # Generate spline scribble prompts 
            spline_scribble = generate_spline_scribble_prompts(
                pos_mask=mask_prompt_input.pos_mask,
                num_spline_scribbles=config.num_spline_scribbles,
                num_slices=config.num_slices_spline_scribble,  # It is generated on a single slice
                num_control_points=config.num_control_points_spline_scribble,
                rng=rng
            )

            # Ensure spline scribbles are within the positive mask
            np.logical_and(spline_scribble, mask_prompt_input.pos_mask, out=spline_scribble)  

            # Assign the target label to the spline scribble mask
            spline_scribble = spline_scribble.astype(np.uint8, copy=False)
            np.multiply(spline_scribble, target_label, out=spline_scribble)

            # Add the spline scribble mask to the prompts_dict, combining with existing scribbles if necessary
            if "spline_scribbles" not in prompts_dict:
                prompts_dict.setdefault("spline_scribbles", spline_scribble)
            else:                
                # Element-wise maximum to combine spline scribbles for different labels (labels should not overlap)
                np.maximum(
                    prompts_dict["spline_scribbles"],
                    spline_scribble,
                    out=prompts_dict["spline_scribbles"]
                )

    return prompts_dict
