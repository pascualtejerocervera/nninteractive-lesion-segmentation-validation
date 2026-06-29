from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class NNInteractiveV2ModelConfig(BaseModel):
    """
    Configuration for interactive segmentation prompt generation.
    """

    model_config = ConfigDict(extra="forbid")  # Forbid extra fields to ensure strict validation
    
    # Model name
    model_name: Literal["nninteractive_v2"] = Field(
        default="nninteractive_v2",
        description="Name of the nnInteractive model to use for prompt generation."
    )

    # Model version
    model_version: Literal["v2"] = Field(
        default="v2",
        description="Version of the nnInteractive model to use for prompt generation."
    )
    # Runtime
    device: Literal["cpu", "cuda", "mps"] | None = Field(
        default=None,
        description="Device to run the model on (e.g., 'cpu', 'cuda', 'mps'). If None, auto-detects best available device."
    )
    sync_device: bool = Field(
        default=False,
        description="Whether to synchronize the device before and after model inference for computing benchmarking metrics. If False, the dictionary of metrics will not include the inference time."
    )
    use_memory: bool = Field(
        default=True,
        description="Whether to use memory-efficient attention mechanisms (if supported by the model)"
    )
    download_dir: str | None = Field(
        default=None,
        description="Directory to download model weights. If None, it creates a 'model_weights' directory in the current working directory."
    )
