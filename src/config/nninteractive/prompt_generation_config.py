from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


DeviceType = Literal["cpu", "cuda", "mps"]

class PromptNoiseConfig(BaseModel):
    """
    Controls stochasticity in simulated radiologist prompts.
    """
    model_config = ConfigDict(extra="forbid")  # Forbid extra fields to ensure strict validation

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

class NNInteractivePromptGenerationConfig(BaseModel):
    """
    Configuration for interactive segmentation prompt generation.
    """
    model_config = ConfigDict(extra="forbid")  # Forbid extra fields to ensure strict validation

    # General settings
    seed: int | None = Field(
        default=None,
        ge=0,
        description="Random seed for reproducibility; None = non-deterministic"
    )
    # Input prompt mask creation
    surface_band_mask_iter: int = Field(
        default=1,
        ge=0,
        description="Iterations of dilation to create a surface band mask (the boundary region of the lesion) for structured prompts (diameter/spline). If 0, no surface band mask is created and structured prompts will be generated directly from the original lesion mask."
    )
    neg_mask_dilation_iter: int = Field(
        default=3,
        ge=1,
        description="Iterations of dilation to create a negative mask from the positive mask for point prompts. This ensures that negative points are sampled from a background region that is sufficiently separated from the lesion."
    )
    # Point prompting constraints
    num_pt_pos: int = Field(
        default=0, 
        ge=0, 
        le=3, 
        description="Number of positive point prompts per label and per slice (0-3)"
    )
    num_pt_neg: int = Field(
        default=0, 
        ge=0, 
        le=3, 
        description="Number of negative point prompts per label and per slice (0-3)"
    )
    num_slices_pts: int = Field(
        default=1,
        ge=1,
        description="Number of slices to consider for point prompting"
    )
    pt_neg_dilation_iter: int = Field(
        default=3,
        ge=1,
        description="Number of dilation iterations to apply to the positive mask when generating negative point prompts (to ensure separation from the lesion)"
    )

    # Bounding box prompting constraints
    num_bbox_pos: int = Field(
        default=0, 
        ge=0, 
        le=1, 
        description="Number of bounding box prompts per label (0-1)"
    )
    num_slices_bbox: int = Field(
        default=1,
        ge=1,
        description="Number of slices to consider for bounding box prompting"
    )

    # Diameter annotation prompting constraints
    scribble_diameter_ann: bool = Field(
        default=False,
        description="Enable diameter-based annotation mode (e.g., simulate lesion diameter prompts)"
    )
    num_slices_diameter: int = Field(
        default=1,
        ge=1,
        description="Number of slices to consider for diameter annotation prompting (typically 1, as diameter annotation is generated on a single slice)"
    )

    # Spline scribble mode
    scribble_spline: bool = Field(
        default=False,
        description="Enable spline-based scribble annotations instead of point-based prompts"
    )
    num_slices_scribble_spline: int = Field(
        default=1,
        ge=1,
        description="Number of slices to consider for spline scribble prompting (typically 1, as scribble annotation is generated on a single slice)"
    )
    num_scribble_spline: int = Field(
        default=1,
        ge=1,
        description="Number of spline scribble prompts per label (0-1) and per slice (typically 1)"
    )
    num_control_points_scribble_spline: int = Field(
        default=3,
        ge=3,
        description="Number of control points to define the spline scribble (minimum 3 for a valid spline)"
    )
    # Noise model
    noise: PromptNoiseConfig | None = Field(
        default=None,
        description="Configuration for noise to add to prompts to simulate variability in radiologist annotations"
    )

    @model_validator(mode="after")
    def validate_consistency(self) -> "NNInteractivePromptGenerationConfig":
        """
        Validate that the configuration is consistent and that at least one prompt type is enabled.
        Also checks for mutual exclusivity between diameter annotation and spline scribble modes.
        """

        # Ensure noise is always usable downstream
        if self.noise is None:
            self.noise = PromptNoiseConfig()

        has_points = (self.num_pt_pos > 0) or (self.num_pt_neg > 0)

        has_bbox = self.num_bbox_pos > 0

        has_structured_prompt = (
            self.scribble_diameter_ann
            or self.scribble_spline
        )

        # At least one prompt source must exist
        if not (has_points or has_bbox or has_structured_prompt):
            raise ValueError(
                "At least one prompt type must be enabled: "
                "point prompts, bounding boxes, or diameter/spline annotations"
            )

        # Mutual exclusivity
        if self.scribble_diameter_ann and self.scribble_spline:
            raise ValueError(
                "Diameter annotation and spline scribble modes are mutually exclusive. "
                "Please enable only one of them."
            )

        return self
