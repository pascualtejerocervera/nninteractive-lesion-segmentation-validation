import os
from pathlib import Path

import numpy as np

REPO_ID = "nnInteractive/nnInteractive"
MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future
DOWNLOAD_DIR = Path(__file__).parent / "models"  # Directory to store downloaded modelss

class NNInteractiveModel():
    def __init__(
        self,
        device: str,
        use_memory: bool
    ):
        # Initialize the model
        self.predictor = self._initialize_model(device)

    def _initialize_model(self, device: str):
    
        # Initialize Inference Session
        from huggingface_hub import snapshot_download  # Install huggingface_hub if not already installed
        from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

        # Ensure the download directory exists
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # Download the model from Hugging Face Hub
        snapshot_download(
            repo_id=REPO_ID,
            allow_patterns=[f"{MODEL_NAME}/*"],
            local_dir=DOWNLOAD_DIR
        )

        predictor = nnInteractiveInferenceSession(
            device=device,
            use_torch_compile=False, 
            verbose=False,
            torch_n_threads=os.cpu_count(),  # Use available CPU cores
            do_autozoom=True,  # Enables AutoZoom for better patching
        )

        # Load the trained model
        model_path = os.path.join(str(DOWNLOAD_DIR), MODEL_NAME)
        predictor.initialize_from_trained_model_folder(model_path)
        return predictor
    
    def set_image(image: np.ndarray) -> None:
        pass
    
    def add_interaction(
        pts_pos: tuple[tuple[float, float, float], ...] | None = None,
        pts_neg: tuple[tuple[float, float, float], ...] | None = None,
        bboxes_pos: tuple[tuple[float, float, float, float, float, float], ...] | None = None,
        bboxes_neg: tuple[tuple[float, float, float, float, float, float], ...] | None = None,
        scribble_pos: np.ndarray | None = None,
        scribble_neg: np.ndarray | None = None,
        lasso_pos: np.ndarray | None = None,
        lasso_neg: np.ndarray | None = None,
        mask_pos: np.ndarray | None = None,
    ) -> np.ndarray:
        pass

    def reset_session(self) -> None:
        self.predictor.reset_session()