from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator


DeviceType = Literal["cpu", "cuda", "mps"]


class PromptNoiseConfig(BaseModel):
    """
    Controls stochasticity in simulated radiologist prompts.
    """

    point_jitter_std: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation of Gaussian noise added to point prompts (in voxels)"
    )
    bbox_jitter_std: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation of Gaussian noise added to bounding box coordinates (in voxels)"
    )


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
