import numpy as np

from interactive_seg.seg_model.interactive_seg_model import InteractiveSegmentationModel
from interactive_seg_nninteractive_v1.model import NNInteractiveV1Model
from interactive_seg.device.get_device import get_device
from interactive_seg.utils.helpers.extract_prompt_labels import extract_labels_from_prompts, has_labels
from interactive_seg_nninteractive_v1.config import NNInteractiveV1ModelConfig

Point3D = tuple[tuple[int, int, int], ...]
BBox3D = tuple[tuple[int, int, int, int, int, int], ...]
PromptScribbleLasso = np.ndarray | dict[int, tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]  

class NNInteractiveV1InferenceSession(InteractiveSegmentationModel):
    def __init__(
        self,
        config: NNInteractiveV1ModelConfig | dict,
    ) -> None:
        """
        Initialize the nnInteractive inference session.

        Args:
            config: An instance of NNInteractiveV1ModelConfig or a dictionary containing the model configuration.
        """
        # Initialize the base class
        super().__init__()

        # Validate the model_config if it is provided as a dictionary and convert it to an instance of NNInteractiveV1ModelConfig
        if isinstance(config, dict):
            model_config = NNInteractiveV1ModelConfig.model_validate(config)

        # Initialize the model predictor
        self.model = NNInteractiveV1Model(
            device=model_config.device or get_device(),
            use_memory=model_config.use_memory,
            download_dir=model_config.download_dir
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
        prompts_dict: dict[str,  dict[int, Point3D] | dict[int, BBox3D] | PromptScribbleLasso],
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

        # Create a temporary mask for label extraction
        label_mask = np.zeros_like(self.input_image, dtype=np.uint8) 

        # Create final result array
        output = np.zeros_like(self.input_image, dtype=np.uint8)  

        for label in labels_extracted:
            for prompt_name, prompt_content in prompts_dict.items():
                if "pt" in prompt_name:
                    if label in prompt_content:
                        points = prompt_content[label]
                        if "pos" in prompt_name:
                            self.model.add_interaction(pt_pos=points)
                        elif "neg" in prompt_name:
                            self.model.add_interaction(pt_neg=points)

                elif "bbox" in prompt_name:
                    if label in prompt_content:
                        bbox = prompt_content[label]
                        if "pos" in prompt_name:
                            self.model.add_interaction(bbox_pos=bbox)
                        elif "neg" in prompt_name:
                            self.model.add_interaction(bbox_neg=bbox)

                elif "scribble" in prompt_name:
                    np.equal(prompt_content, label, out=label_mask)
                    self.model.add_interaction(scribble_pos=label_mask)

            # Update the output array (binary mask) for the current label
            output = np.where(self.model.target_buffer, label, output)  

        # Store the final output in the predictor's target buffer
        self._output = output
            
    def reset_session(self) -> None:
        """
        Reset the session
        """
        self.model.reset_session()
