from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import yaml
from time import perf_counter
from pathlib import Path
from typing import Any

import numpy as np

from datasets import build_dataset
from evaluation.config import EvaluationModelRunConfig, EvaluationRunConfig
from evaluation.dataset_selection import resolve_requested_dataset_configs
from evaluation.metrics import compute_surface_dist_metrics
from evaluation.predictor_factory import build_predictor
from interactive_seg import generate_nninteractive_prompts
from interactive_seg.io.image_io import save_nifti_image


@dataclass(slots=True)
class EvaluationRow:
    """A single result row produced by the evaluation pipeline.

    Each row corresponds to the metrics computed for one label value within
    one sample, for a given dataset/model combination. Rows with an "error"
    status represent a failure for that sample/dataset/model and will have
    most metric fields left as ``None``.

    Attributes:
        dataset_name: Human-readable name of the dataset.
        model_name: Name of the model run (from the evaluation config).
        sample_id: Unique identifier for the sample (e.g. case ID).
        sample_index: Index of the sample within the dataset.
        label: The integer label value for which metrics were computed.
            None if the row represents an error for this sample.
        status: "ok" if metrics were successfully computed, "error" if a
            failure occurred for this sample/dataset/model combination.
        inference_seconds: Time taken to run inference on this sample, in
            seconds. None if an error occurred or if timing was disabled.
        dice_score: Dice coefficient between prediction and ground truth for
            this label value. None if an error occurred or if no ground truth
            exists for this label value.
        hd95: 95th percentile Hausdorff distance between prediction and ground truth for this label value, 
            in millimeters. None if an error occurred or if no ground truth exists for this label value.
        hd99: 99th percentile Hausdorff distance between prediction and ground truth for this label value, 
            in millimeters. None if an error occurred or if no ground truth exists for this label value.
        avg_dist_gt_to_pred: Average distance from ground truth to prediction for this label value,
            in millimeters. None if an error occurred or if no ground truth exists for this label value.
        avg_dist_pred_to_gt: Average distance from prediction to ground truth for this label value,
            in millimeters. None if an error occurred or if no ground truth exists for this label value.
        error_message: Error message if an error occurred for this sample/dataset/model combination.
        sample_img_path: Path to the input image file for this sample, if available. None if not applicable.
        sample_gt_path: Path to the ground truth mask file for this sample, if available.
            None if not applicable or if no ground truth exists for this label value.
    """
    dataset_name: str
    model_name: str
    sample_id: str
    sample_index: int
    label: int | None
    status: str
    inference_seconds: float | None = None
    dice_score: float | None = None
    hd95: float | None = None
    hd99: float | None = None
    avg_dist_gt_to_pred: float | None = None
    avg_dist_pred_to_gt: float | None = None
    error_message: str | None = None
    sample_img_path: str | None = None
    sample_gt_path: str | None = None


class EvaluationRunner:
    """Runs evaluation of one or more models against one or more datasets.

    The runner iterates over all requested dataset configs and, for each
    dataset, evaluates every configured model run. For every sample in a
    dataset it generates interactive segmentation prompts, runs the model,
    computes metrics per label value, and optionally saves the predicted
    segmentation to disk. Results are collected into ``EvaluationRow``
    instances and written out to a CSV file.
    """

    # Column order used when writing results.csv. Note this intentionally
    # excludes "model_config_path" even though it exists on EvaluationRow.
    CSV_FIELDNAMES = [
        "dataset_name",
        "model_name",
        "sample_id",
        "sample_index",
        "label",
        "status",
        "inference_seconds",
        "dice_score",
        "hd95",
        "hd99",
        "avg_dist_gt_to_pred",
        "avg_dist_pred_to_gt",
        "error_message",
        "sample_img_path",
        "sample_gt_path",
    ]

    def __init__(
        self, 
        config_path: Path | str,
        config: EvaluationRunConfig
    ) -> None:
        """Initializes the EvaluationRunner with the provided configuration.

        Args:
            config: An instance of EvaluationRunConfig containing the
                evaluation settings (datasets, model runs, output path,
                experiment name, error-handling behavior, etc.).
        """
        # Store the configuration for use in the evaluation process.
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = config

        # Set up output paths based on the configuration.
        self.output_path = (Path(config.output_path) / config.experiment_name).expanduser().resolve()
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Define paths for the output CSV and predictions directory.
        self.output_csv = self.output_path / "results.csv"
        if self.config.save_predictions:
            self.predictions_dir = self.output_path / "predictions"
            self.predictions_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.predictions_dir = None

        # Create a directory for saving the configuration used for this evaluation run, so it can be referenced later.
        self._save_yaml_file(
            path=self.output_path / "evaluation_config.yaml",
            data=self.config.model_dump(),
        )


    def run(self) -> list[EvaluationRow]:
        """Executes the evaluation process for the specified datasets and model runs.

        For each resolved dataset config, the dataset is built and then
        evaluated against every configured model run. Failures while
        building a dataset or running a model are recorded as error rows.
        If ``continue_on_error`` is False on the config, the first error
        encountered will be raised after writing out the results collected
        so far.

        Returns:
            A list of EvaluationRow instances containing the results of the
            evaluation (including any error rows).

        Raises:
            Exception: Re-raises the first encountered exception if
                ``self.config.continue_on_error`` is False.
        """
        rows: list[EvaluationRow] = []

        # Resolve which dataset config files to evaluate against (supports
        # evaluating against "all" known dataset configs).
        dataset_configs = resolve_requested_dataset_configs(
            self.config.dataset,
            all_configs=self.config.dataset == "all",
        )

        for dataset_config_path in dataset_configs:
            dataset_name = self._dataset_name(dataset_config_path)
            dataset = build_dataset(dataset_config_path)

            for model_run in self.config.model_runs:
                try:
                    self._run_model_on_dataset(
                        dataset=dataset,
                        dataset_name=dataset_name,
                        model_run=model_run,
                        rows=rows,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Catch-all for failures not already handled per-sample
                    # inside _run_model_on_dataset (e.g. predictor setup).
                    rows.append(
                        EvaluationRow(
                            dataset_name=dataset_name,
                            model_name=model_run.name,
                            sample_id="N/A",
                            sample_index=-1,
                            label=None,
                            status="error",
                            error_message=str(exc),
                            sample_img_path=None,
                            sample_gt_path=None,
                        )
                    )
                    if not self.config.continue_on_error:
                        self._write_csv(self.output_csv, rows)
                        raise

        # Always persist whatever results were gathered, even partial ones.
        self._write_csv(self.output_csv, rows)
        return rows

    def _run_model_on_dataset(
        self,
        dataset: Any,
        dataset_name: str,
        model_run: EvaluationModelRunConfig,
        rows: list[EvaluationRow],
    ) -> None:
        """Evaluates a single model run against every sample in a dataset.

        Loads the model's YAML config, builds the predictor, then iterates
        over every sample in the dataset: generates prompts from the ground
        truth mask, runs the predictor, computes per-label metrics, and
        optionally saves the predicted segmentation as a NIfTI file. Result
        rows (success or per-sample error) are appended to ``rows`` in place.

        Args:
            dataset: The dataset object to iterate over (supports ``len()``
                and indexing, returning dicts with "image", "mask", and
                "meta" keys).
            dataset_name: Human-readable name of the dataset, used for
                labeling rows and naming saved prediction files.
            model_run: Configuration describing which model to run and
                where to load its config from.
            rows: The list of EvaluationRow results to append to. Mutated
                in place so callers can observe partial progress on error.

        Raises:
            Exception: Re-raised for a given sample if
                ``self.config.continue_on_error`` is False, after the error
                row for that sample has already been recorded.
        """
        # Load the model's YAML config and split it into the pieces needed
        # for building the predictor and generating prompts.
        config = self._load_yaml_file(Path(model_run.config_path))
        self._save_yaml_file(
            path=self.output_path / f"{model_run.name}_config.yaml",
            data=config,
        )
        model_config = config["model_config"]
        prompt_config = config["prompt_generation"]

        predictor = build_predictor(model_run.name, model_config)

        # Check if the output CSV already exists and read it if so, to skip already processed samples.
        output_csv_existing = False
        if self.output_csv.exists():
            output_csv_existing = True
            output_csv = csv.DictReader(self.output_csv.open("r", newline="", encoding="utf-8"))

        for sample_index in range(len(dataset)):
            sample = dataset[sample_index]
            meta = sample["meta"]
            sample_id = str(meta["case_id"])

            # Check if the sample was already evaluated in a previous run (e.g. if the CSV exists and has a row for this sample/model/dataset).
            if output_csv_existing and self._check_sample_is_processed(output_csv, dataset_name, model_run.name, sample_id):
                print(f"Skipping already processed sample {sample_id} for dataset {dataset_name} and model {model_run.name}.")
                continue

            try:
                image = sample["image"]
                target = sample["mask"]

                # Generate interactive segmentation prompts (e.g. clicks,
                # bounding boxes) derived from the ground-truth mask.
                prompts = generate_nninteractive_prompts(
                    config=prompt_config,
                    model_config=model_config,
                    mask=target,
                )

                predictor.set_image(image)

                # Run the model using the generated prompts; no explicit
                # label filtering is applied here.
                predictor.run(prompts_dict=prompts, labels=None)
                prediction = predictor.output
                inference_time_per_label = predictor.inference_time_per_label if model_config["sync_device"] else None

                # Determine which non-background label values are present
                # in the ground truth, so metrics are computed per label.
                labels = self._labels(target)

                # Reset predictor state so the next sample starts clean.
                predictor.reset()

                prediction_path = None
                if self.config.save_predictions:
                    # Save the raw prediction volume to disk as NIfTI,
                    # reusing the original image's affine/header so it
                    # stays aligned with the source data.
                    prediction_path = self.predictions_dir / f"{dataset_name}__{model_run.name}__{self._safe_name(sample_id)}.nii.gz"
                    save_nifti_image(
                        image=prediction.astype(np.uint8, copy=False),
                        affine=sample["meta"]["affine"],
                        header=sample["meta"]["header"],
                        path=prediction_path,
                    )

                # Create temporary storage for the binary mask of the current label during prompt processing.
                tmp_pred = np.zeros_like(prediction, dtype=bool)
                tmp_gt = np.zeros_like(target, dtype=bool)

                # Compute and record metrics separately for each label value
                # found in the ground-truth mask.
                for label in labels:

                    # Compute metrics for the current label value, using the prediction and ground truth target, along with the voxel spacing from the image header.
                    metrics = compute_surface_dist_metrics(
                        np.equal(prediction, label, out=tmp_pred),
                        np.equal(target, label, out=tmp_gt),
                        space_mm=sample["meta"]["header"].get_zooms()
                    )

                    # Append a row for this label value, including metrics and inference time if available.
                    rows.append(
                        EvaluationRow(
                            dataset_name=dataset_name,
                            model_name=model_run.name,
                            sample_id=sample_id,
                            sample_index=sample_index,
                            label=label,
                            status="ok",
                            inference_seconds=inference_time_per_label.get(label) if inference_time_per_label else None,
                            dice_score=metrics["dice"],
                            hd95=metrics["hd95"],
                            hd99=metrics["hd99"],
                            avg_dist_gt_to_pred=metrics["avg_surface_dist_gt_to_pred"],
                            avg_dist_pred_to_gt=metrics["avg_surface_dist_pred_to_gt"],
                            error_message=None,
                            sample_gt_path=str(sample["meta"]["mask_path"]) if "mask_path" in sample["meta"] else None,
                            sample_img_path=str(sample["meta"]["image_path"]) if "image_path" in sample["meta"] else None
                        )
                    )

                    del tmp_pred, tmp_gt  # Clear temporary masks for the next label

            except Exception as exc:  # noqa: BLE001
                # Record the failure for this specific sample and either
                # continue to the next sample or propagate, depending on
                # configuration.
                rows.append(
                    EvaluationRow(
                        dataset_name=dataset_name,
                        model_name=model_run.name,
                        sample_id=sample_id,
                        sample_index=sample_index,
                        label=None,
                        status="error",
                        error_message=str(exc),
                        sample_img_path=str(sample["meta"]["image_path"]) if "image_path" in sample["meta"] else None,
                        sample_gt_path=str(sample["meta"]["mask_path"]) if "mask_path" in sample["meta"] else None
                    )
                )
                if not self.config.continue_on_error:
                    raise exc
            
        # Final reset in case the loop body didn't reach the per-sample
        # reset for the last sample (e.g. it raised before getting there).
        predictor.reset()

    @staticmethod
    def _load_yaml_file(path: Path) -> dict[str, Any]:
        """Loads and parses a YAML file.

        Args:
            path: Path to the YAML file to load.

        Returns:
            The parsed YAML content as a dictionary.
        """
        with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)
        
    @staticmethod
    def _save_yaml_file(path: Path, data: dict[str, Any]) -> None:
        """Saves a dictionary to a YAML file.

        Args:
            path: Path to the YAML file to save.
            data: The dictionary to save as YAML.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)

    @staticmethod
    def _labels(mask: np.ndarray) -> list[int]:
        """Extracts the non-background label values present in a mask.

        Args:
            mask: A numpy array containing integer label values, where 0
                is treated as background.

        Returns:
            A list of unique non-zero integer label values found in the mask.
        """
        return [int(value) for value in np.unique(mask) if int(value) != 0]

    @classmethod
    def _write_csv(cls, output_csv: Path, rows: list[EvaluationRow]) -> None:
        """Writes evaluation rows to a CSV file.

        Args:
            output_csv: Path to the CSV file to write. Parent directories
                are assumed to already exist.
            rows: The list of EvaluationRow instances to write, one per
                CSV row, in order.
        """
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=cls.CSV_FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    @staticmethod
    def _safe_name(value: str) -> str:
        """Sanitizes a string for safe use as part of a filename.

        Replaces path separator characters so the resulting string can be
        used as a filename component without creating unintended
        subdirectories.

        Args:
            value: The raw string (e.g. a sample id) to sanitize.

        Returns:
            The sanitized string with "/" and "\\" replaced by "__".
        """
        return value.replace("/", "__").replace("\\", "__")

    @staticmethod
    def _dataset_name(config_path: Path) -> str:
        """Resolves a human-readable dataset name from its config path.

        Args:
            config_path: Path to the dataset's config file.

        Returns:
            The dataset name declared in the config file, or the config
            file's stem if no name is declared.
        """
        from evaluation.dataset_selection import load_config_name

        return load_config_name(config_path) or config_path.stem

    @staticmethod
    def _check_sample_is_processed(
        output_csv: csv.DictReader, 
        dataset_name: str, 
        model_name: str, 
        sample_id: str,
    ) -> bool:
        for row in output_csv:
            if (
                row["dataset_name"] == dataset_name and
                row["model_name"] == model_name and
                row["sample_id"] == sample_id and
                row["status"] == "ok"
            ):
                return True
        return False
