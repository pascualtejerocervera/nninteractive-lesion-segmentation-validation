from __future__ import annotations

from typing import Any

from interactive_seg_nninteractive_v1 import NNInteractiveV1InferenceSession
from interactive_seg_nninteractive_v2 import NNInteractiveV2InferenceSession


def build_predictor(
    model_name: str,
    model_config: dict[str, Any],
) -> NNInteractiveV1InferenceSession | NNInteractiveV2InferenceSession:
    """Factory function for creating inference sessions.

    Selects the appropriate nnInteractive backend implementation based on
    the provided model name.

    Args:
        model_name: Identifier of the model backend (e.g. "nninteractive_v1").
        model_config: Configuration dictionary passed to the session.

    Returns:
        An initialized inference session instance for the selected backend.

    Raises:
        ValueError: If the requested backend is not supported.
    """
    backend_key = model_name.lower()

    if backend_key == "nninteractive_v1":
        return NNInteractiveV1InferenceSession(model_config)

    if backend_key == "nninteractive_v2":
        return NNInteractiveV2InferenceSession(model_config)

    raise ValueError(f"Unsupported backend '{model_name}'")
