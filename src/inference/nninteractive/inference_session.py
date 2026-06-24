import numpy as np

from inference.interactive_seg_model import InteractiveSegmentationModel
from inference.nninteractive.model import NNInteractiveModel
from utils.device.get_device import get_device
from utils.helpers.extract_prompt_labels import extract_labels_from_prompts, has_labels

Point3D = tuple[tuple[int, int, int], ...]
BBox3D = tuple[tuple[int, int, int, int, int, int], ...]

class NNInteractiveInferenceSession(InteractiveSegmentationModel):
    def __init__(
        self,
        device: str | None = None,
        use_memory: bool = True,
        download_dir: str | None = None
    ) -> None:
        """
        Initialize the nnInteractive inference session.

        Args:
            device: A string specifying the device to run the model on (e.g., 'cpu', 'cuda', 'mps'). 
                If None, the device will be auto-detected.
            use_memory: A boolean indicating whether to use memory for the inference session.
            download_dir: A string specifying the directory to download model weights. If None, uses default cache directory.
        """
        # Initialize the base class
        super().__init__()

        # Initial parameters
        self.device = device or get_device()  # Use provided device or auto-detect
        self.use_memory = use_memory 

        # Initialize the model predictor
        self.predictor = NNInteractiveModel(
            device=self.device,
            use_memory=self.use_memory,
            download_dir=download_dir
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
        self.predictor.set_image(image)
        self.input_image = image  # Store the original input image
        self.predictor.set_target_buffer(np.zeros_like(image, dtype=np.uint8))  # Initialize target buffer with zeros

    def run(
        self,
        prompts_dict: dict[str,  dict[int, Point3D] | dict[int, BBox3D] | np.ndarray],
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

        # Create a boolean temporary mask for label extraction if diameter or spline prompts are present
        keys = ("scribble_diameter_ann", "scribble_spline")
        # if ("scribble_diameter_ann" in prompts_dict or "scribble_spline" in prompts_dict) and (isinstance(prompts_dict.get("scribble_diameter_ann"), np.ndarray) or isinstance(prompts_dict.get("scribble_spline"), np.ndarray)):
        if any(key in prompts_dict for key in keys) and any(isinstance(prompts_dict.get(key), np.ndarray) for key in keys):
            label_mask = np.zeros_like(self.input_image, dtype=np.uint8)  # Temporary mask for label extraction

        for label in labels_extracted:
            for prompt_name, prompt_content in prompts_dict.items():
                if "pt" in prompt_name:
                    if label in prompt_content:
                        points = prompt_content[label]
                        if "pos" in prompt_name:
                            self.predictor.add_interaction(pt_pos=points)
                        elif "neg" in prompt_name:
                            self.predictor.add_interaction(pt_neg=points)

                elif "bbox" in prompt_name:
                    if label in prompt_content:
                        bbox = prompt_content[label]
                        if "pos" in prompt_name:
                            self.predictor.add_interaction(bbox_pos=bbox)
                        elif "neg" in prompt_name:
                            self.predictor.add_interaction(bbox_neg=bbox)

                elif "scribble_diameter_ann" in prompt_name:
                    if isinstance(prompt_content, np.ndarray):
                        np.equal(prompt_content, label, out=label_mask)
                        self.predictor.add_interaction(scribble_pos=label_mask)
                    elif isinstance(prompt_content, dict) and label in prompt_content:
                        self.predictor.add_interaction(scribble_pos=prompt_content[label])

                elif "scribble_spline" in prompt_name:
                    if isinstance(prompt_content, np.ndarray):
                        np.equal(prompt_content, label, out=label_mask)
                        self.predictor.add_interaction(scribble_pos=label_mask)
                    elif isinstance(prompt_content, dict) and label in prompt_content:
                        self.predictor.add_interaction(scribble_pos=prompt_content[label])
            
    def reset_session(self) -> None:
        """
        Reset the session
        """
        pass
