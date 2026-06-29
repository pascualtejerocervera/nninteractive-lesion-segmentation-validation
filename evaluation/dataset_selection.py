from __future__ import annotations

from pathlib import Path

import yaml

# Directory containing dataset configuration yaml files.
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "datasets" / "configs"


def iter_config_paths() -> list[Path]:
    """Lists all dataset configuration files.

    Returns:
        Sorted list of YAML configuration file paths.

    Raises:
        FileNotFoundError: If the configuration directory does not exist.
    """
    if not CONFIGS_DIR.exists():
        raise FileNotFoundError(f"Config directory not found: {CONFIGS_DIR}")

    return sorted(
        path for path in CONFIGS_DIR.glob("*.yaml") if path.is_file()
    )


def load_config_name(config_path: Path) -> str | None:
    """Extracts the dataset name from a configuration file.

    Args:
        config_path: Path to a dataset configuration YAML file.

    Returns:
        The dataset name if present, otherwise None.
    """
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    name = config.get("name")
    return str(name) if name is not None else None

def save_yaml_file(path: Path, data: dict) -> None:
    """Saves a dictionary to a YAML file.

    Args:
        path: The path to the YAML file to save.
        data: The dictionary to save as YAML.
    """
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)


def resolve_dataset_config(dataset_spec: str) -> Path:
    """Resolves a dataset specification to a configuration file path.

    The resolution order is:
        1. Direct file path (if exists)
        2. Filename match inside CONFIGS_DIR
        3. Stem or `name` field match inside CONFIGS_DIR

    Args:
        dataset_spec: Dataset identifier, filename, or full path.

    Returns:
        Absolute path to the resolved configuration file.

    Raises:
        FileNotFoundError: If no matching configuration is found.
        ValueError: If multiple configurations match the specification.
    """
    candidate = Path(dataset_spec)

    # Case 1: direct path
    if candidate.exists():
        return candidate.resolve()

    # Case 2: direct filename inside configs directory
    if candidate.suffix == ".yaml":
        resolved = CONFIGS_DIR / candidate.name
        if resolved.exists():
            return resolved.resolve()

    # Case 3: search by stem or internal "name" field
    matching_paths: list[Path] = []

    for config_path in iter_config_paths():
        if config_path.stem == dataset_spec:
            matching_paths.append(config_path)
            continue

        config_name = load_config_name(config_path)
        if config_name == dataset_spec:
            matching_paths.append(config_path)

    if not matching_paths:
        raise FileNotFoundError(
            f"Could not resolve dataset '{dataset_spec}' "
            f"to a config file under {CONFIGS_DIR}"
        )

    if len(matching_paths) > 1:
        paths = ", ".join(str(path) for path in matching_paths)
        raise ValueError(
            f"Dataset '{dataset_spec}' matched multiple configs: {paths}"
        )

    return matching_paths[0].resolve()


def resolve_requested_dataset_configs(
    dataset_spec: str,
    all_configs: bool,
) -> list[Path]:
    """Resolves one or multiple dataset configuration files.

    Args:
        dataset_spec: Dataset identifier or configuration reference.
        all_configs: If True, returns all available configs instead of
            resolving a specific dataset.

    Returns:
        List of resolved configuration file paths.
    """
    if all_configs:
        return iter_config_paths()

    return [resolve_dataset_config(dataset_spec)]
