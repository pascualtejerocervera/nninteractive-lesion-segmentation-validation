from typing import TYPE_CHECKING

import os
from pathlib import Path

import torch
import numpy as np

if TYPE_CHECKING:
    from inference.nninteractive.inference_session import nnInteractiveInferenceSession

REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future

PromptScribbleLasso = np.ndarray | dict[int, tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]]
class NNInteractiveModel():
    def __init__(
        self,
        device: str,
        use_memory: bool,
        download_dir: str | None = None
    ):
        # Initialize the model
        self.is_initialized = False
        self.device = device
        self.use_memory = use_memory
        self.download_dir = download_dir

        # Initialize the predictor
        self.is_initialized = False
        self.predictor = self._initialize_model(
            device=self.device, 
            use_memory=self.use_memory, 
            download_dir=self.download_dir
        )

    def _initialize_model(
        self, 
        device: str, 
        use_memory: bool = True,
        download_dir: str | None = None
    ) -> "nnInteractiveInferenceSession":
        """
        Initialize the nnInteractive model by downloading it from Hugging Face Hub and loading it into an inference session.
        
        Args:
            device: A string specifying the device to run the model on (e.g., 'cpu', 'cuda', 'mps').
            use_memory: A boolean indicating whether to use memory for the inference session.
            download_dir: A string specifying the directory to download model weights. If None, it creates a 
                'model_weights' directory in the current working directory.
        Returns:
            nnInteractiveInferenceSession: An instance of the nnInteractive inference session with the model loaded.
            """
        from huggingface_hub import snapshot_download  # Install huggingface_hub if not already installed
        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

        # Ensure the download directory exists
        if download_dir is not None:
            download_dir = Path(download_dir)
        else:
            download_dir = Path(__file__).parent / "model_weights"  # Default download directory
        download_dir.mkdir(parents=True, exist_ok=True) # Create the directory if it doesn't exist

        # Download the model from Hugging Face Hub
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=[f"{MODEL_NAME}/*"],
            local_dir=download_dir
        )

        # Initialize the nnInteractive inference session
        predictor = nnInteractiveInferenceSession(
            device=torch.device(device),
            use_torch_compile=False, 
            verbose=False,
            torch_n_threads=os.cpu_count(),  # Use available CPU cores
            do_autozoom=True,  # Enables AutoZoom for better patching
        )

        # Load the trained model
        model_path = os.path.join(str(download_dir), MODEL_NAME)
        predictor.initialize_from_trained_model_folder(model_path)
        self.is_initialized = True
        
        return predictor
    
    def set_image(self, image: np.ndarray) -> None:
        """
        Set the 3D image for the inference session. The image should be a 3D numpy array (H, W, D). Then, in this function, the image dimension is expanded to (1, H, W, D) to match the expected input shape for the model.
        
        Args:
            image: A 3D numpy array representing the image to be set for inference.

        Raises:
            ValueError: If the input image is not a numpy array.
            ValueError: If the input image is not 3D.
        """
        if not self.is_initialized:
            raise RuntimeError("Model is not initialized. Please initialize the model before setting the image.")
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array.")
        if image.ndim != 3:
            raise ValueError("Input image must be a 3D array.")
        
        # Expand dimensions to match the expected input shape for the model
        self.predictor.set_image(np.expand_dims(image, axis=0))  # Shape: (1, H, W, D)

    def set_target_buffer(self, target_buffer: np.ndarray | torch.Tensor) -> None:
        """
        Set the target buffer for the inference session. The target buffer it is the output array that will be used for inference. The target buffer should be a 3D numpy array (H, W, D) for the model.
        
        Args:
            target_buffer: A 3D numpy array representing the target buffer for inference.

        Raises:
            ValueError: If the model is not initialized before setting the target buffer.
            ValueError: If the input target buffer is not a numpy array.
            ValueError: If the input target buffer is not 3D.
        """
        if not self.is_initialized:
            raise RuntimeError("Model is not initialized. Please initialize the model before setting the target buffer.")
        if not isinstance(target_buffer, np.ndarray) and not isinstance(target_buffer, torch.Tensor):
            raise ValueError("Target buffer must be a numpy array or a torch tensor.")
        if target_buffer.ndim != 3:
            raise ValueError("Target buffer must be a 3D array.")
        
        # Set the target buffer in the predictor
        self.predictor.set_target_buffer(target_buffer)

    @property
    def target_buffer(self) -> np.ndarray:          
        """
        Get the target buffer from the inference session. The target buffer is the output array that will be used for inference. The target buffer should be a 3D numpy array (H, W, D) for the model.
        
        Returns:
            np.ndarray: The current target buffer used for inference. Format is a 3D numpy array (H, W, D) 
                representing the segmentation output with 0 for background and 1 for foreground (dtype uint8).

        Raises:
            ValueError: If the model is not initialized before getting the target buffer.
        """
        if not self.is_initialized:
            raise RuntimeError("Model is not initialized. Please initialize the model before getting the target buffer.")
        target_buffer = self.predictor.target_buffer
        if isinstance(target_buffer, torch.Tensor):
            return target_buffer.clone().cpu().numpy()  # Return a copy of the target buffer as a numpy array
        return target_buffer

    def add_interaction(
        self,
        pt_pos: tuple[tuple[int, int, int], ...] | None = None,
        pt_neg: tuple[tuple[int, int, int], ...] | None = None,
        bbox_pos: tuple[tuple[int, int, int, int, int, int], ...] | None = None,
        bbox_neg: tuple[tuple[int, int, int, int, int, int], ...] | None = None,
        scribble_pos: PromptScribbleLasso | None = None,
        scribble_neg: PromptScribbleLasso | None = None,
        lasso_pos: PromptScribbleLasso | None = None,
        lasso_neg: PromptScribbleLasso | None = None,
        mask_pos: np.ndarray | None = None
    ) -> None:
        if not self.is_initialized:
            raise RuntimeError("Model is not initialized. Please initialize the model before adding interactions.")
        
        if pt_pos is not None:
            for pt in pt_pos:
                self.predictor.add_point_interaction(pt, include_interaction=True, run_prediction=True)

        if pt_neg is not None:
            for pt in pt_neg:
                self.predictor.add_point_interaction(pt, include_interaction=False, run_prediction=True)

        if bbox_pos is not None:
            for bbox in bbox_pos:
                self.predictor.add_bbox_interaction(bbox, include_interaction=True, run_prediction=True)

        if bbox_neg is not None:
            for bbox in bbox_neg:
                self.predictor.add_bbox_interaction(bbox, include_interaction=False, run_prediction=True)

        if scribble_pos is not None:
            self.predictor.add_scribble_interaction(
                scribble_pos if isinstance(scribble_pos, np.ndarray) else scribble_pos[0],
                include_interaction=True, 
                run_prediction=True,
                interaction_bbox=None if isinstance(scribble_pos, np.ndarray) else scribble_pos[1]
            )

        if scribble_neg is not None:
            self.predictor.add_scribble_interaction(
                scribble_neg if isinstance(scribble_neg, np.ndarray) else scribble_neg[0],
                include_interaction=False, 
                run_prediction=True,
                interaction_bbox=None if isinstance(scribble_neg, np.ndarray) else scribble_neg[1]
            )

        if lasso_pos is not None:
            self.predictor.add_lasso_interaction(
                lasso_pos if isinstance(lasso_pos, np.ndarray) else lasso_pos[0],
                include_interaction=True, 
                run_prediction=True,
                interaction_bbox=None if isinstance(lasso_pos, np.ndarray) else lasso_pos[1]
            )

        if lasso_neg is not None:
            self.predictor.add_lasso_interaction(
                lasso_neg if isinstance(lasso_neg, np.ndarray) else lasso_neg[0],
                include_interaction=False, 
                run_prediction=True,
                interaction_bbox=None if isinstance(lasso_neg, np.ndarray) else lasso_neg[1]
            )

        if mask_pos is not None:
            self.predictor.add_initial_seg_interaction(
                mask_pos,
                run_prediction=True,
            )           

    def reset_session(self) -> None:
        if not self.is_initialized:
            raise RuntimeError("Model is not initialized. Please initialize the model before resetting the session.")
        self.predictor.reset_session()
