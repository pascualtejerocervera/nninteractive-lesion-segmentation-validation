"""Geometry helper functions for bounding boxes and coordinate transforms."""
from .compute_bbox_margin import (
    compute_margin,
    compute_max_relative_margin,
    compute_per_axis_margin,
    compute_clamped_max_relative_margin,
)
from .voxel_transforms import voxel_to_world, world_to_voxel

__all__ = [
    "compute_margin",
    "compute_max_relative_margin",
    "compute_per_axis_margin",
    "compute_clamped_max_relative_margin",
    "voxel_to_world",
    "world_to_voxel",
]
