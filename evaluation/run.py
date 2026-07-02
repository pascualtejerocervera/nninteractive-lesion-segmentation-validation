from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# Repository path setup for local imports when running as script
REPO_ROOT = Path(__file__).resolve().parents[1]
INTERACTIVE_SEG_ROOT = REPO_ROOT / "interactive_seg"

for path in (REPO_ROOT, INTERACTIVE_SEG_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluation.config import EvaluationRunConfig
from evaluation.runner import EvaluationRunner


def build_parser() -> argparse.ArgumentParser:
    """Creates the CLI argument parser for evaluation runs.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Run nnInteractive evaluation over one or more datasets"
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the evaluation run YAML config",
    )
    return parser


def _resolve_run_config(raw_config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Resolves relative paths inside an evaluation configuration.

    This ensures all file references are absolute before validation.

    Args:
        raw_config: Parsed YAML configuration dictionary.
        base_dir: Directory containing the config file.

    Returns:
        Updated configuration dictionary with resolved paths.
    """
    resolved = dict(raw_config)

    # Resolve dataset reference (file path or dataset name)
    dataset = resolved.get("dataset")
    if isinstance(dataset, str) and dataset != "all":
        candidate = Path(dataset)

        if candidate.exists():
            resolved["dataset"] = str(candidate.resolve())
        else:
            dataset_candidate = (base_dir / candidate).resolve()
            if dataset_candidate.exists():
                resolved["dataset"] = str(dataset_candidate)

    if resolved["output_path"] is not None:
        resolved["output_path"] = str((base_dir / resolved["output_path"]).resolve())
    else:
        resolved["output_path"] = str(base_dir.resolve())

    # Resolve model run config paths
    model_runs: list[dict[str, Any]] = []
    for model_run in resolved.get("model_runs", []):
        item = dict(model_run)
        item["model_run_name"] = item.pop("name")
        item["config_path"] = str(Path(base_dir / item["config_path"]).resolve())
        model_runs.append(item)

    resolved["model_runs"] = model_runs

    return resolved


def main(argv: list[str] | None = None) -> int:
    """Entry point for evaluation CLI.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)

    raw_config = yaml.safe_load(
        args.config.read_text(encoding="utf-8")
    )

    config = EvaluationRunConfig.model_validate(
        _resolve_run_config(raw_config, args.config.parent.parent)
    )

    evalutation_runner = EvaluationRunner(
        config_path=args.config,
        config=config,
    )
    evalutation_runner.run()

    print(f"Evaluation run completed. Results written to {config.output_path}.")
    return 0
    


if __name__ == "__main__":
    raise SystemExit(main())
