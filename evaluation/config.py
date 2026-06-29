from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationModelRunConfig(BaseModel):
    """Configuration for a single model evaluation run.

    Each entry defines a model variant to be evaluated, including its
    identifier, backend implementation, and path to its configuration file.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        description="Human-readable model name used in the CSV output.",
    )
    config_path: str = Field(
        description="Path to the model configuration file.",
    )


class EvaluationRunConfig(BaseModel):
    """Top-level configuration for an evaluation pipeline.

    This class defines:
        - Dataset selection
        - Models to evaluate
        - Output locations
        - Execution behavior (error handling, saving outputs)
    """

    model_config = ConfigDict(extra="forbid")
    
    experiment_name: str = Field(
        default="default_experiment",
        required=True,
        description="Name of the experiment for organizing results.",
    )
    output_path: str | None = Field(
        default=None,
        description="Base directory for saving evaluation results. If None, defaults to the evaluation/ directory.",
    )
    dataset: str = Field(
        default="all",
        description="Dataset name, dataset config stem, dataset config path, or 'all'.",
    )
    model_runs: list[EvaluationModelRunConfig] = Field(
        default_factory=list,
        description="Model runs to compare in a single evaluation execution.",
    )
    save_predictions: bool = Field(
        default=False,
        description="Whether to write prediction masks to disk.",
    )
    continue_on_error: bool = Field(
        default=True,
        description="Continue processing later samples if one sample fails.",
    )

    @model_validator(mode="after")
    def validate_model_runs(self) -> "EvaluationRunConfig":
        """Ensures that at least one model run is defined.

        Raises:
            ValueError: If `model_runs` is empty.
        """
        if not self.model_runs:
            raise ValueError("At least one model run must be configured")
        return self
