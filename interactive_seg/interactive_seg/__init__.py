"""Top-level package for the interactive segmentation utilities.

Exports a small convenient surface for common configs and prompt generation.
"""
from .config.prompt_generation_config import (
    NNInteractivePromptGenerationConfig,
    NNInteractivePromptGenerationConfigBase,
)
from .prompt_generation.generate_nninteractive_prompts import (
    generate_nninteractive_prompts,
)

__all__ = [
    "NNInteractivePromptGenerationConfig",
    "NNInteractivePromptGenerationConfigBase",
    "generate_nninteractive_prompts",
]
