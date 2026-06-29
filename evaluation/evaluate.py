from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is importable when running as a script
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import build_dataset
from evaluation.dataset_selection import resolve_requested_dataset_configs


def build_parser() -> argparse.ArgumentParser:
    """Creates the command-line argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Inspect a medical segmentation dataset config"
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--dataset",
        help="Dataset name, dataset config stem, or path to a dataset JSON config",
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Inspect every dataset config found in datasets/configs",
    )

    return parser


def _print_dataset_summary(config_path: Path) -> bool:
    """Loads and prints a summary of a dataset configuration.

    Args:
        config_path: Path to dataset configuration file.

    Returns:
        True if dataset loaded successfully, False otherwise.
    """
    try:
        dataset = build_dataset(config_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Config: {config_path}")
        print(f"Error: {exc}")
        return False

    print(f"Config: {config_path}")
    print(f"Dataset class: {type(dataset).__name__}")
    print(f"Number of samples: {len(dataset)}")

    if len(dataset) == 0:
        print("Dataset is empty")
        return True

    sample = dataset[0]
    print("Sample keys:", ", ".join(sample.keys()))

    meta = sample.get("meta", {})
    if isinstance(meta, dict):
        print("Meta keys:", ", ".join(meta.keys()))

    return True


def main(argv: list[str] | None = None) -> int:
    """Entry point for dataset inspection CLI.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)

    failures = 0

    for index, config_path in enumerate(
        resolve_requested_dataset_configs(
            args.dataset,
            all_configs=args.all,
        )
    ):
        if index > 0:
            print()

        if not _print_dataset_summary(config_path):
            failures += 1

    # Only fail fast in single-dataset mode
    if failures and not args.all:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
