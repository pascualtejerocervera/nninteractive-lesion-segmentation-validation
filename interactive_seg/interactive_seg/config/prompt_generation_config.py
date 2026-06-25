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


class NNInteractiveCropROIConfig(BaseModel):
    """
    Configuration for generating crop bounding boxes around lesions.

    Supports four margin modes:
      - "absolute": always pad by a fixed number of voxels (``margin``).
      - "max_relative": pad by ``fraction`` of the lesion's largest axis extent,
        unclamped (see ``compute_max_relative_margin``).
      - "per_axis_relative": pad each axis independently by ``fraction`` of that
        axis's own extent (see ``compute_per_axis_margin``). Returns a per-axis
        tuple rather than a single voxel count.
      - "clamped_max_relative": same as "max_relative" but constrained to
        [``min_margin``, ``max_margin``] (see ``compute_clamped_max_relative_margin``).
        This is the recommended default for relative margins, since it avoids
        degenerate (too small) or runaway (too large) padding.
    """
    model_config = ConfigDict(extra="forbid")  # Forbid extra fields to ensure strict validation

    enable: bool = Field(
        default=False,
        description="Whether to generate interaction bounding boxes around lesions"
    )
    margin_mode: Literal[
        "absolute",
        "max_relative",
        "per_axis_relative",
        "clamped_max_relative",
    ] = Field(
        default="absolute",
        description=(
            "How to compute the margin: a fixed voxel count ('absolute'); "
            "a fraction of the lesion's largest axis extent, unclamped ('max_relative'); "
            "a fraction applied independently per axis ('per_axis_relative', returns a "
            "tuple instead of a single int); or a fraction of the largest axis extent "
            "clamped to [min_margin, max_margin] ('clamped_max_relative')."
        ),
    )
    margin: int = Field(
        default=5,
        ge=0,
        description="Margin (in voxels) to add around the lesion when margin_mode='absolute'."
    )
    fraction: float = Field(
        default=1 / 3,
        gt=0.0,
        description="Fraction of the lesion's extent to use as margin. Ignored when margin_mode='absolute'."
    )
    min_margin: int = Field(
        default=5,
        ge=0,
        description="Minimum margin (in voxels) allowed when margin_mode='clamped_max_relative'."
    )
    max_margin: int = Field(
        default=64,
        ge=0,
        description="Maximum margin (in voxels) allowed when margin_mode='clamped_max_relative'."
    )

    @model_validator(mode="after")
    def validate_margins(self) -> "NNInteractiveCropROIConfig":
        if self.min_margin > self.max_margin:
            raise ValueError(
                f"min_margin ({self.min_margin}) cannot exceed max_margin ({self.max_margin})"
            )
        return self

    def compute_margin(
        self, 
        bbox: tuple[int, int, int, int, int, int]
    ) -> int | tuple[int, int, int]:
        """
        Resolve the margin to use for a given bounding box based on the configured margin_mode.

        Returns:
            An int for margin_mode in {"absolute", "max_relative", "clamped_max_relative"}.
            A tuple[int, int, int] (oz, oy, ox) for margin_mode == "per_axis_relative".
        """
        if self.margin_mode == "absolute":
            return self.margin

        # Local import to avoid a hard dependency / circular import at module load time
        from interactive_seg.interactive_seg.utils.geometry.compute_bbox_margin import compute_margin

        return compute_margin(
            mode=self.margin_mode,
            bbox=bbox,
            fraction=self.fraction,
            min_margin=self.min_margin,
            max_margin=self.max_margin,
        )


class NNInteractivePromptGenerationConfigBase(BaseModel):
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

    # Prompt generation settings
    prompt_generation_config: NNInteractivePromptGenerationConfig = Field(
        default_factory=NNInteractivePromptGenerationConfig,
        description="Configuration for generating prompts (points, bounding boxes, diameter/spline annotations)"
    )

    # Interaction bounding box settings
    crop_roi_config: NNInteractiveCropROIConfig = Field(
        default_factory=NNInteractiveCropROIConfig,
        description="Configuration for generating crop bounding boxes around lesions"
    )
