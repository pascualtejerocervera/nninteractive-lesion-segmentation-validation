from abc import ABC, abstractmethod

import numpy as np

VALID_PROMPT_KEYS = {
    "pt_pos",
    "pt_neg",
    "bbox_pos",
    "scribble_diameter_ann",
    "scribble_spline",
}

Point3D = tuple[tuple[int, int, int], ...]
BBox3D = tuple[tuple[int, int, int, int, int, int], ...]
PromptScribbleLasso = np.ndarray | dict[int, tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]  

class InteractiveSegmentationModel(ABC):
    """
    Base class for interactive segmentation models.
    """

    def __init__(self) -> None:
        """
        Initialize the interactive segmentation model.
        """
        self.valid_prompt_keys = VALID_PROMPT_KEYS
        self._output: np.ndarray | None = None

    @abstractmethod
    def run(
        self,
        image: np.ndarray,
        prompts: dict[str,  dict[int, Point3D] | dict[int, BBox3D] | PromptScribbleLasso],
    ) -> None:
            
        """
        Predict the segmentation mask based on the input image and prompts.

        Args:
            image: The input image for segmentation.
            prompts: The prompts provided for interactive segmentation.

        Returns:
            The predicted segmentation mask.
        """
        pass

    @abstractmethod
    def reset_session(self):
        """
        Reset the model's session, clearing any internal state.
        """
        pass

    @property
    def output(self):
        if self._output is None:
            raise ValueError("Output has not been computed yet. Please run the model with valid inputs first.")
        if not isinstance(self._output, np.ndarray):
            raise TypeError("Output must be a numpy array.")
        return self._output   
            
