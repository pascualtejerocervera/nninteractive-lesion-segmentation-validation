from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator


DeviceType = Literal["cpu", "cuda", "mps"]

class NNInteractiveModelConfig(BaseModel):
    """
    Configuration for interactive segmentation prompt generation.
    """
    # Runtime
    device: DeviceType | None = Field(
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
