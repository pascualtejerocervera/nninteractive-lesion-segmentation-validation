from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from datasets.base import BaseMedicalSegmentationDataset

# Global registry mapping dataset names to dataset classes.
_DATASET_REGISTRY: dict[str, type[BaseMedicalSegmentationDataset]] = {}

# Type variable for dataset class types.
DatasetT = TypeVar("DatasetT", bound=BaseMedicalSegmentationDataset)


def register_dataset(
    name: str,
) -> Callable[[type[BaseMedicalSegmentationDataset]], type[BaseMedicalSegmentationDataset]]:
    """Decorator for registering a dataset class in the global registry.

    This enables dataset discovery by name at runtime, typically used in
    factory-based dataset construction pipelines.

    Dataset names are case-insensitive (normalized to lowercase).

    Args:
        name: Unique dataset identifier used for lookup.

    Returns:
        A decorator that registers the dataset class.

    Raises:
        ValueError: If a different dataset class is already registered
            under the same name.
    """
    key = name.lower()

    def decorator(
        dataset_cls: type[BaseMedicalSegmentationDataset],
    ) -> type[BaseMedicalSegmentationDataset]:
        """Registers the dataset class in the global registry.

        Args:
            dataset_cls: Dataset class to register.

        Returns:
            The same class, unmodified (enables decorator usage).
        """
        existing = _DATASET_REGISTRY.get(key)

        # Prevent silent overwrites of existing dataset registrations
        if existing is not None and existing is not dataset_cls:
            raise ValueError(
                f"Dataset '{name}' is already registered by {existing.__name__}"
            )

        _DATASET_REGISTRY[key] = dataset_cls
        return dataset_cls

    return decorator


def get_dataset(name: str) -> type[BaseMedicalSegmentationDataset]:
    """Retrieves a dataset class from the registry.

    Lookup is case-insensitive.

    Args:
        name: Dataset identifier.

    Returns:
        Registered dataset class.

    Raises:
        KeyError: If no dataset is registered under the given name.
    """
    key = name.lower()

    if key not in _DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'")

    return _DATASET_REGISTRY[key]
