import numpy as np

from interactive_seg.config.prompt_generation_config import NNInteractivePromptGenerationConfigBase
from interactive_seg.utils.prompts.bbox_generation import generate_bbox_prompts
from interactive_seg.utils.prompts.diameter_annotation_generation import generate_diameter_annotation_prompts
from interactive_seg.utils.prompts.mask_prompt_input import create_mask_prompt_input
from interactive_seg.utils.prompts.pts_generation import generate_pts_prompts
from interactive_seg.utils.prompts.spline_scribble_generation import generate_spline_scribble_prompts
from interactive_seg_nninteractive_v1.config import NNInteractiveV1ModelConfig
from interactive_seg_nninteractive_v2.config import NNInteractiveV2ModelConfig

Point3D = tuple[tuple[int, int, int], ...]  
BBox3D = tuple[tuple[int, int, int, int, int, int], ...]
PromptScribbleLasso = np.ndarray | dict[int, tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]  

def generate_nninteractive_prompts(
    config: NNInteractivePromptGenerationConfigBase | dict,
    model_config: NNInteractiveV1ModelConfig | NNInteractiveV2ModelConfig | dict,
    mask: np.ndarray,
) -> dict[str,  dict[int, Point3D] | dict[int, BBox3D] | PromptScribbleLasso]:
    """
    Generates prompts for the nnInteractive model.

    Args:
        config: An instance of NNInteractivePromptGenerationConfigBase or a dictionary containing the 
            configuration parameters for prompt generation
        mask: A 3D numpy array representing the segmentation mask, where each unique positive integer value 
            corresponds to a different label (e.g., 1 for label A, 2 for label B, etc.), and 0 represents the background.
        rng: A numpy random generator for reproducibility.
    Returns:
        prompts_dict: A dictionary containing the generated prompts for each label in the mask. The keys of the dictionary are:
            - "pt_pos": A dictionary where each key is a label and the value is a tuple of positive point prompts for that label.
            - "pt_neg": A dictionary where each key is a label and the value is a tuple of negative point prompts for that label
            - "bbox_pos": A dictionary where each key is a label and the value is a tuple of bounding box prompts for that label.
            - "scribble_diameter_ann": A 3D numpy array representing the diameter annotation prompts for all labels combined. The array has the same shape as the input mask, with positive values indicating the presence of diameter annotations.
            - "scribble_spline": A 3D numpy array representing the spline scribble prompts for all labels combined.
    """
    # Validate the config if it is provided as a dictionary and convert it to an instance of NNInteractivePromptGenerationConfig
    if isinstance(config, dict):
        config = NNInteractivePromptGenerationConfigBase.model_validate(config)
    elif not isinstance(config, NNInteractivePromptGenerationConfigBase):
        raise ValueError("Config must be an instance of NNInteractivePromptGenerationConfigBase or a dictionary.")
    
    # Validate the model_config if it is provided as a dictionary and convert it to an instance of the appropriate model config class
    if isinstance(model_config, dict):
        if model_config.get("model_version") == "v1":
            model_config = NNInteractiveV1ModelConfig.model_validate(model_config)
        elif model_config.get("model_version") == "v2":
            model_config = NNInteractiveV2ModelConfig.model_validate(model_config)
        else:
            raise ValueError("model_config must be an instance of NNInteractiveV1ModelConfig or NNInteractiveV2ModelConfig or a dictionary with a valid model_version.")
    
    # If the model version is v1 and ROI cropping is enabled, raise an error since ROI cropping is not supported in v1
    if isinstance(model_config, NNInteractiveV1ModelConfig) and config.crop_roi_config.enable:
        raise ValueError("ROI cropping is not supported in NNInteractiveV1ModelConfig. Please set crop_roi_config.enable to False for model version v1.")
    
    # Extract unique labels from the mask, excluding the background label (0)
    labels_unique = tuple(int(label) for label in np.unique(mask) if int(label) != 0)
    if len(labels_unique) == 0:
        raise ValueError("The mask does not contain any positive labels to generate prompts from.")
    
    # Create a random generator using the provided seed for reproducibility. If None is provided, a new generator is created for non-deterministic behavior.
    rng = np.random.default_rng(config.seed)  

    # Initialize the prompts dictionary and a boolean mask for label extraction
    prompts_dict = {}
    mask_label = np.zeros_like(mask, dtype=bool)  # Initialize mask_label to avoid potential reference before assignment
    roi_cropping = config.crop_roi_config.enable  # Check if cropping is enabled in the configuration

    for target_label in labels_unique:

        np.equal(mask, target_label, out=mask_label)  # In-place comparison

        # Create the mask prompt input for the current label
        mask_prompt_input = create_mask_prompt_input(
            config=config,
            mask_label=mask_label,
            label=target_label
        )

        # Generate positive or negative from the original mask depending on the requested prompt types in the config (if only negative points are requested, the positive mask is not needed and vice versa)
        if config.prompt_generation_config.num_pt_pos > 0 or config.prompt_generation_config.num_pt_neg > 0:
            pt_pos, pt_neg = generate_pts_prompts(
                pos_mask=mask_prompt_input.pos_mask,
                neg_mask=mask_prompt_input.neg_mask,
                num_pt_pos=config.prompt_generation_config.num_pt_pos,
                num_pt_neg=config.prompt_generation_config.num_pt_neg, 
                num_slices=config.prompt_generation_config.num_slices_pts, 
                pt_neg_dilation_iter=config.prompt_generation_config.pt_neg_dilation_iter, # Ignored if num_pt_neg is 0
                rng=rng
            )
            if pt_pos:
                prompts_dict.setdefault("pt_pos", {})[target_label] = pt_pos 
            if pt_neg:
                prompts_dict.setdefault("pt_neg", {})[target_label] = pt_neg

        if config.prompt_generation_config.num_bbox_pos > 0:
            # Generate bounding box prompts
            bbox_pos = generate_bbox_prompts(
                pos_mask=mask_prompt_input.pos_mask,
                num_bboxes=config.prompt_generation_config.num_bbox_pos, 
                num_slices=config.prompt_generation_config.num_slices_bbox, 
                rng=rng
            )
            prompts_dict.setdefault("bbox_pos", {})[target_label] = bbox_pos

        if config.prompt_generation_config.scribble_diameter_ann:
            # Generate diameter annotation prompts 
            diameter_annotation = generate_diameter_annotation_prompts(
                pos_mask=mask_prompt_input.pos_mask if not roi_cropping else mask_prompt_input.crop_mask,  # Use the cropped mask if cropping is enabled
                num_slices=config.prompt_generation_config.num_slices_diameter,  # Diameter annotation is typically generated on a single slice 
                rng=rng
            )
            
            # TODO: Diameter annotation we dont need as it needs to go until the margin of the lesion
            # Ensure diameter annotations are within the positive mask
            # np.logical_and(
            #     diameter_annotation, 
            #     mask_prompt_input.pos_mask if not roi_cropping else mask_prompt_input.crop_pos_mask,  # Use the cropped positive mask if cropping is enabled
            #     out=diameter_annotation
            # )

            # Add the diameter annotation mask to the prompts_dict, combining with existing annotations if necessary
            if roi_cropping:
                # Multiple arrays with different labels are stored in a dictionary, so we have boolean masks for each label, and the bounding box is also stored for each label to know where the diameter annotation is located in the original mask.
                prompts_dict.setdefault("scribble_diameter_ann", {})[target_label] = (diameter_annotation, mask_prompt_input.crop_bbox)

            else:
                # Since it mask of shape equal to the original mask, we need to have different labels for each diameter annotation, so we multiply the mask by the target label to ensure that each label has a unique value in the diameter annotation mask.
                diameter_annotation = diameter_annotation.astype(np.uint8, copy=False)
                np.multiply(diameter_annotation, target_label, out=diameter_annotation)
                if "scribble_diameter_ann" not in prompts_dict:
                    prompts_dict.setdefault("scribble_diameter_ann", diameter_annotation)
                else:
                    # Element-wise maximum to combine diameter annotations for different labels (labels should not overlap)
                    np.maximum(
                        prompts_dict["scribble_diameter_ann"],
                        diameter_annotation,
                        out=prompts_dict["scribble_diameter_ann"]
                    )

        if config.prompt_generation_config.scribble_spline:
            # Generate spline scribble prompts 
            spline_scribble = generate_spline_scribble_prompts(
                pos_mask=mask_prompt_input.pos_mask if not roi_cropping else mask_prompt_input.crop_pos_mask,  # Use the cropped positive mask if cropping is enabled
                num_scribble_spline=config.prompt_generation_config.num_scribble_spline,
                num_slices=config.prompt_generation_config.num_slices_scribble_spline,  # It is generated on a single slice
                num_control_points=config.prompt_generation_config.num_control_points_scribble_spline,
                rng=rng
            )

            # Ensure spline scribbles are within the positive mask
            np.logical_and(
                spline_scribble, 
                mask_prompt_input.pos_mask if not roi_cropping else mask_prompt_input.crop_pos_mask,  # Use the cropped positive mask if cropping is enabled
                out=spline_scribble
            )  

            if roi_cropping:
                # Multple arrays with different labels are stored in a dictionary, so we have boolean masks for each label, and the bounding box is also stored for each label to know where the scribble is located in the original mask.
                prompts_dict.setdefault("scribble_spline", {})[target_label] = (spline_scribble, mask_prompt_input.crop_bbox)
            else:
                # Since it mask of shape equal to the original mask, we need to have different labels for each scribble, so we multiply the mask by the target label to ensure that each label has a unique value in the scribble mask.
                spline_scribble = spline_scribble.astype(np.uint8, copy=False)
                np.multiply(spline_scribble, target_label, out=spline_scribble)
                # Add the spline scribble mask to the prompts_dict, combining with existing scribbles if necessary
                if "scribble_spline" not in prompts_dict:
                    prompts_dict.setdefault("scribble_spline", spline_scribble)
                else:                
                    # Element-wise maximum to combine spline scribbles for different labels (labels should not overlap)
                    np.maximum(
                        prompts_dict["scribble_spline"],
                        spline_scribble,
                        out=prompts_dict["scribble_spline"]
                    )

    return prompts_dict
