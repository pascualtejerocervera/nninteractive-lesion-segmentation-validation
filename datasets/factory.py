from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

from datasets.base import BaseMedicalSegmentationDataset
from datasets.registry import get_dataset

# Keys used only for resolving the dataset class, not for constructor args.
_FACTORY_KEYS = {"name", "class", "module"}


def _load_config(yaml_path: str | Path) -> dict[str, Any]:
    """Loads a YAML configuration file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML file is invalid.
        ValueError: If the parsed YAML is not a dictionary.
    """
    path = Path(yaml_path).expanduser().resolve()

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected YAML mapping at {path}, got {type(config).__name__}"
        )

    return config


def _resolve_dataset_class(
    config: dict[str, Any],
) -> type[BaseMedicalSegmentationDataset]:
    """Resolves a dataset class from a configuration dictionary.

    Resolution order:
        1. Try registry lookup by dataset name.
        2. If not found and module is provided, import module and resolve class.

    Args:
        config: Dataset configuration containing at least:
            - name: Dataset identifier in registry.
            - class: Class name (fallback resolution).
            - module: Optional module path for dynamic import.

    Returns:
        A dataset class inheriting from BaseMedicalSegmentationDataset.

    Raises:
        KeyError: If dataset cannot be resolved from registry or module.
        TypeError: If resolved class is not a valid dataset subclass.
    """
    dataset_name = config["name"]
    dataset_class_name = config["class"]
    module_name = config.get("module")

    # First attempt: registry lookup (preferred path)
    try:
        return get_dataset(dataset_name)
    except KeyError:
        pass

    # Second attempt: dynamic import if module is provided
    if module_name:
        module = importlib.import_module(module_name)

        try:
            # Retry registry after module import (module may register dataset)
            return get_dataset(dataset_name)
        except KeyError:
            dataset_cls = getattr(module, dataset_class_name)

            if not issubclass(dataset_cls, BaseMedicalSegmentationDataset):
                raise TypeError(
                    f"{dataset_class_name} must inherit from "
                    "BaseMedicalSegmentationDataset"
                )

            return dataset_cls

    raise KeyError(f"Unable to resolve dataset '{dataset_name}'")


def build_dataset(yaml_path: str | Path) -> BaseMedicalSegmentationDataset:
    """Factory function to build a dataset from a YAML configuration.

    The configuration must define at least:
        - name: Dataset registry key
        - class: Dataset class name
        - module: Optional module path for dynamic import

    Remaining keys are passed directly to the dataset constructor.

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        Instantiated dataset object.

    Raises:
        KeyError: If dataset cannot be resolved.
        TypeError: If dataset class is invalid.
        ValueError: If configuration is malformed.
    """
    config = _load_config(yaml_path)
    dataset_cls = _resolve_dataset_class(config)

    # Filter out factory-only keys before instantiation
    kwargs = {
        key: value
        for key, value in config.items()
        if key not in _FACTORY_KEYS
    }

    return dataset_cls(**kwargs)
