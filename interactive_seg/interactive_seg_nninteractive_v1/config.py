from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class NNInteractiveV1ModelConfig(BaseModel):
    """
    Configuration for interactive segmentation prompt generation.
    """

    model_config = ConfigDict(extra="forbid")  # Forbid extra fields to ensure strict validation
    
    # Model version
    model_version: Literal["v1"] = Field(
        default="v1",
        description="Version of the nnInteractive model to use for prompt generation."
    )
    # Runtime
    device: Literal["cpu", "cuda", "mps"] | None = Field(
        default=None,
        description="Device to run the model on (e.g., 'cpu', 'cuda', 'mps'). If None, auto-detects best available device."
    )
    use_memory: bool = Field(
        default=True,
        description="Whether to use memory-efficient attention mechanisms (if supported by the model)"
    )
    download_dir: str | None = Field(
        default=None,
        description="Directory to download model weights. If None, it creates a 'model_weights' directory in the current working directory."
    )
