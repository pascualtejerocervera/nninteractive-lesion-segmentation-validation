"""Compatibility shim for nnInteractive v2 package exports."""
from .config import NNInteractiveV2ModelConfig
from .inference_session import NNInteractiveV2InferenceSession

__all__ = ["NNInteractiveV2ModelConfig", "NNInteractiveV2InferenceSession"]
