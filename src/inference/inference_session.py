import numpy as np


from inference.model import NNInteractiveModel
from utils.device.get_device import select_device
from utils.helpers.extract_prompt_labels import extract_labels_from_prompts, has_nonzero_labels

Point3D = tuple[tuple[int, int, int], ...]
BBox3D = tuple[tuple[int, int, int, int, int, int], ...]

class NNInteractiveInferenceSession():
    def __init__(
        self,
        device: str | None,
        use_memory: bool = True,
    ):
        self.device = device if device is not None else select_device()
        self.use_memory = use_memory 
        self._output = None

        # Initialize the model predictor
        self.predictor = NNInteractiveModel(
            device=device,
            use_memory=use_memory
        )

    def set_image(self, image: np.ndarray) -> None:
        """Set the 3D image for the inference session."""
        self.predictor.set_image(image)

    def run(
        self,
        prompts_dict: dict[str,  dict[int, Point3D] | dict[int, BBox3D] | np.ndarray],
        labels: list[int] | None,
    ) -> None:
        
        extracted = set(extract_labels_from_prompts(prompts_dict))
        labels_extracted = sorted(set(labels) & extracted) if labels is not None else extracted

        for label in labels_extracted:
            for prompt_name, prompt_content in prompts_dict.items():
                if "pts" in prompt_name:
                    pass
                elif "bboxes" in prompt_name:
                    pass
                elif "diameter_annotations" in prompt_name:
                    pass
                elif "spline_scribbles" in prompt_name:
                    pass
            
    @property
    def output(self):
        if self._output is None:
            raise ValueError("Output has not been computed yet.")
        if not isinstance(self._output, np.ndarray):
            raise TypeError("Output must be a numpy array.")
        return self._output

            


    def reset_session(self) -> None:
        """
        Reset the session
        """
        pass
    