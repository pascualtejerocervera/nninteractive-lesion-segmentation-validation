import time

import numpy as np

from interactive_seg.seg_model.interactive_seg_model import InteractiveSegmentationModel
from interactive_seg.device.device import get_device, sync_device
from interactive_seg.utils.helpers.extract_prompt_labels import extract_labels_from_prompts, has_labels
from interactive_seg_nninteractive_v2.config import NNInteractiveV2ModelConfig
from interactive_seg_nninteractive_v2.model import NNInteractiveV2Model

Point3D = tuple[tuple[int, int, int], ...]
BBox3D = tuple[tuple[int, int, int, int, int, int], ...]
PromptVolume = np.ndarray | dict[int, tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]  

VALID_PROMPT_KEYS = (
    "pts_pos",
    "pts_neg",
    "bboxes_pos",
    "scribble_diameter_ann",
    "scribble_spline",
)

class NNInteractiveV2InferenceSession(InteractiveSegmentationModel):
    def __init__(
        self,
        config: NNInteractiveV2ModelConfig | dict,
    ) -> None:
        """
        Initialize the nnInteractive inference session.

        Args:
            config: An instance of NNInteractiveV2ModelConfig or a dictionary containing the model configuration.
        """
        # Initialize the base class
        super().__init__(
            valid_prompt_keys=VALID_PROMPT_KEYS
        )

        # Validate the model_config if it is provided as a dictionary and convert it to an instance of NNInteractiveV2ModelConfig
        if isinstance(config, dict):
            config = NNInteractiveV2ModelConfig.model_validate(config)

        # Get model device and sync_device flag from the configuration
        self.device = config.device or get_device()  # Use the specified device or auto-detect the best available device
        self.sync_device = config.sync_device  # Store the sync_device flag from the configuration

        # Initialize the model predictor
        self.model = NNInteractiveV2Model(
            device=self.device,
            use_memory=config.use_memory,
            download_dir=config.download_dir,
        )

        # Other parameters can be initialized here if needed
        self.input_image = None  # Placeholder for the input image

    def set_image(self, image: np.ndarray) -> None:
        """
        Set the 3D image for the inference session. And then, in this function, the image dimension is expanded to (1, H, W, D) to match the expected input shape for the model.
        Additionally, this function initializes the target buffer with zeros of the same shape as the input image. The target buffer is used for inference and should be a 3D numpy or torch tensor (H, W, D) for the model.
        
        Args:
            image: A 3D numpy array representing the image to be set for inference.
        """
        self.model.set_image(image)
        self.input_image = image  # Store the original input image
        self.model.set_target_buffer(np.zeros_like(image, dtype=np.uint8))  # Initialize target buffer with zeros

    def run(
        self,
        prompts_dict: dict[str,  dict[int, Point3D] | dict[int, BBox3D] | PromptVolume],
        labels: list[int] | None,
    ) -> None:
        """
        Run the inference session with the provided prompts and labels. The parameter prompts_dict is a dictionary that can contain various types of prompts, including points, bounding boxes, diameter annotations, and spline scribbles. The labels parameter is an optional list of label IDs to filter the prompts.
        If the labels parameter is provided, the function will only consider prompts that match the specified label IDs and make sure that those labels are present in the prompts_dict. If the labels parameter is None, all prompts in the prompts_dict will be considered for inference.

        Args:
            prompts_dict: A dictionary containing various types of prompts for inference.
            labels: An optional list of label IDs to filter the prompts. If provided, only prompts matching these
                labels will be considered. Otherwise, all prompts in the prompts_dict will be used for inference.

        Raises:
            ValueError: If prompts_dict is not a dictionary or if it does not contain any valid labels
            ValueError: If the labels parameter is provided but does not match any labels in the prompts_dict
        """
        if not isinstance(prompts_dict, dict):
            raise ValueError("prompts_dict must be a dictionary.")
        if not has_labels(prompts_dict):
            raise ValueError("No valid labels found in prompts_dict. Please provide at least one non-zero label.")

        # Extract unique labels from the prompts_dict and filter them based on the provided labels parameter
        extracted = extract_labels_from_prompts(prompts_dict)
        labels_extracted = sorted(set(labels) & set(extracted)) if labels is not None else extracted

        # Create a temporary mask for label extraction if diameter or spline prompts are present
        keys_scribble = tuple(key for key in self.valid_prompt_keys if "scribble" in key)
        keys_present = any(key in prompts_dict for key in keys_scribble)
        keys_are_ndarray = any(isinstance(prompts_dict.get(key), np.ndarray) for key in keys_scribble)
        if keys_present and keys_are_ndarray:
            label_mask = np.zeros_like(self.input_image, dtype=np.uint8)  

        # Create final result array
        output = np.zeros_like(self.input_image, dtype=np.uint8)  

        for label in labels_extracted:

            # Synchronize the device and start timing if sync_device is enabled
            if self.sync_device:
                sync_device(self.device)  # Synchronize the device before inference for accurate timing
                start_time = time.perf_counter()  # Start timing

            for prompt_name, prompt_content in prompts_dict.items():
                if "pt" in prompt_name:
                    if label in prompt_content:
                        points = prompt_content[label]
                        if "pos" in prompt_name:
                            self.model.add_interaction(pts_pos=points)
                        elif "neg" in prompt_name:
                            self.model.add_interaction(pts_neg=points)

                elif "bbox" in prompt_name:
                    if label in prompt_content:
                        bbox = prompt_content[label]
                        if "pos" in prompt_name:
                            self.model.add_interaction(bboxes_pos=bbox)
                        elif "neg" in prompt_name:
                            self.model.add_interaction(bboxes_neg=bbox)

                elif "scribble" in prompt_name:
                    if isinstance(prompt_content, np.ndarray):
                        np.equal(prompt_content, label, out=label_mask)
                        self.model.add_interaction(scribble_pos=label_mask)
                    elif isinstance(prompt_content, dict) and label in prompt_content:
                        self.model.add_interaction(scribble_pos=prompt_content[label])

            if self.sync_device:
                sync_device(self.device)  # Synchronize the device after inference for accurate timing
                elapsed = time.perf_counter() - start_time  # Calculate elapsed time
                self._inference_time_per_label[label] = round(elapsed, 2)  # Store the inference time for the current label, rounded to 2 decimal places

            # Update the output array (binary mask) for the current label
            output[self.model.target_buffer.view(np.bool_)] = label  # Use boolean indexing

        # Store the final output in the predictor's target buffer
        self._output = output
            
    def reset_session(self) -> None:
        """
        Reset the session
        """
        self.model.reset_session()

    def reset(self) -> None:
        """
        Reset the session and clear the input image and output buffer.
        """
        self.reset_session()
        self.input_image = None
        self._output = None
