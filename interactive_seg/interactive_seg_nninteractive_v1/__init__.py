"""Compatibility shim for nnInteractive v1 package exports."""
from .config import NNInteractiveV1ModelConfig
from .inference_session import NNInteractiveV1InferenceSession

__all__ = ["NNInteractiveV1ModelConfig", "NNInteractiveV1InferenceSession"]
