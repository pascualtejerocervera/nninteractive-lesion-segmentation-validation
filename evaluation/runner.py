from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import yaml
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
from interactive_seg.io.prompt_io import save_prompts_dict


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
        sample_img_path: Path to the input image file for this sample, if available. None if not applicable.
        sample_gt_path: Path to the ground truth mask file for this sample, if available.
            None if not applicable or if no ground truth exists for this label value.
        sample_pred_path: Path to the predicted segmentation file for this sample, if available.
            None if not applicable.
        sample_prompts_path: Path to the generated prompts file for this sample, if available.
            None if not applicable.
        error_message: Error message if an error occurred for this sample/dataset/model combination.
            None if no error occurred.
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
    sample_img_path: str | None = None
    sample_gt_path: str | None = None
    sample_pred_path: str | None = None
    sample_prompts_path: str | None = None
    error_message: str | None = None

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
        "sample_img_path",
        "sample_gt_path",
        "sample_pred_path",
        "sample_prompts_path",
        "error_message",
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

        if self.config.save_predictions or self.config.save_generated_prompts:
            self.results_dir = self.output_path / "saved_results"
            self.results_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.results_dir = None

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

        Any rows already present in an existing results CSV (from a prior,
        possibly interrupted run) are loaded up front, so re-running the
        evaluation does not lose previously computed results.

        Returns:
            A list of EvaluationRow instances containing the results of the
            evaluation (including any error rows).

        Raises:
            Exception: Re-raises the first encountered exception if
                ``self.config.continue_on_error`` is False.
        """
        # Start from whatever was already persisted in a previous run (if
        # any), so resuming doesn't discard prior results when we rewrite
        # the CSV at the end.
        rows: list[EvaluationRow] = self._load_existing_rows(self.output_csv)
        
        # If the last row contains an error, remove it so we can re-run that sample and potentially recover from the error.
        if rows and rows[-1].status == "error":
            print(f"Removing last error row for sample {rows[-1].sample_id} to allow re-evaluation.")
            rows.pop()

        # Resolve which dataset config files to evaluate against (supports
        # evaluating against "all" known dataset configs).
        dataset_configs = resolve_requested_dataset_configs(
            self.config.dataset,
            all_configs=self.config.dataset == "all",
        )

        for dataset_config_path in dataset_configs:
            dataset_name = self._dataset_name(dataset_config_path)
            dataset = build_dataset(dataset_config_path)

            self._save_yaml_file(
                path=self.output_path / "dataset_configs" / f"{dataset_name}.yaml",
                data=self._load_yaml_file(dataset_config_path),
            )

            for model_run in self.config.model_runs:
                try:
                    self._run_model_on_dataset(
                        dataset=dataset,
                        dataset_name=dataset_name,
                        model_run=model_run,
                        rows=rows,
                    )
                except Exception as exc:  # noqa: BLE001
                    if not self.config.continue_on_error:
                        # If we're not continuing on error, re-raise the exception
                        # after writing out whatever results were gathered so far.
                        self._write_csv(self.output_csv, rows)
                        raise exc

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
                May already contain rows loaded from a previous run; these
                are used to determine which samples can be skipped.

        Raises:
            Exception: Re-raised for a given sample if
                ``self.config.continue_on_error`` is False, after the error
                row for that sample has already been recorded.
        """
        # Load the model's YAML config and split it into the pieces needed
        # for building the predictor and generating prompts.
        config = self._load_yaml_file(Path(model_run.config_path))
        model_name = config["model_config"]["model_name"]
        model_config = config["model_config"]
        prompt_config = config["prompt_generation"]

        # Save the model config used for this run to the output directory, so it can be referenced later.
        self._save_yaml_file(
            path=self.output_path / "model_configs" / f"{Path(model_run.config_path).stem}.yaml",
            data=config,
        )

        # Build the predictor for this model run, which will be used to run inference on each sample in the dataset.
        predictor = build_predictor(model_name, model_config)

        # Build a set of (dataset, model, sample_id) keys already marked
        # "ok" in the rows accumulated so far (including rows loaded from
        # a previous run), so we can skip re-processing them. Computed once
        # up front rather than re-scanning a CSV reader per sample.
        processed_keys = {
            (row.dataset_name, row.model_name, row.sample_id)
            for row in rows
            if row.status == "ok"
        }

        for sample_index in range(len(dataset)):
            # Extract the sample data and metadata for the current index.
            sample = dataset[sample_index]
            meta = sample["meta"]
            sample_id = str(meta["case_id"])

            # Initialize variables for the prediction, inference time, and prompts for this sample
            prediction = None
            inference_time_per_label = None
            prompts = None

            # Check if the sample was already evaluated in a previous run.
            if (dataset_name, model_name, sample_id) in processed_keys:
                print(f"Skipping already processed sample {sample_id} for dataset {dataset_name} and model {model_name}.")
                continue

            try:
                # Extract the image and ground truth mask from the sample
                image = sample["image"]
                target = sample["mask"]
                labels = self._labels(target) # Unique non-zero label values in the ground truth mask
                if not labels:
                    print(f"Sample {sample_id} has no foreground labels in the ground truth mask; skipping metric computation.")
                    continue

                # Generate interactive segmentation prompts (e.g. clicks,
                # bounding boxes) derived from the ground-truth mask.
                prompts = generate_nninteractive_prompts(
                    config=prompt_config,
                    model_config=model_config,
                    mask=target,
                )

                # Set the input image for the predictor, which will be used for inference.
                predictor.set_image(image)

                # Run the model using the generated prompts for all the labels in the ground truth mask, and measure inference time per label if available.
                predictor.run(prompts_dict=prompts, labels=None)
                prediction = predictor.output
                inference_time_per_label = predictor.inference_time_per_label.copy() if predictor.inference_time_per_label else None

                # Reset predictor state so the next sample starts clean.
                predictor.reset_session()

                if self.config.save_predictions:
                    prediction_path = (
                        self.results_dir
                        / dataset_name
                        / model_name
                        / self._safe_name(sample_id)
                        / f"{self._safe_name(sample_id)}_pred.nii.gz"
                    )
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    save_nifti_image(
                        image=image,
                        affine=sample["meta"]["affine"],    
                        header=sample["meta"]["header"],
                        file_path=prediction_path.parent / f"{self._safe_name(sample_id)}_image.nii.gz",
                    )
                    save_nifti_image(
                        image=target,
                        affine=sample["meta"]["affine"],
                        header=sample["meta"]["header"],
                        file_path=prediction_path.parent / f"{self._safe_name(sample_id)}_gt.nii.gz",
                    )
                    save_nifti_image(
                        image=prediction,
                        affine=sample["meta"]["affine"],
                        header=sample["meta"]["header"],
                        file_path=prediction_path,
                    )


                if self.config.save_generated_prompts:
                    prompts_path = (
                        self.results_dir
                        / dataset_name
                        / model_name
                        / self._safe_name(sample_id)
                        / f"{self._safe_name(sample_id)}_prompts"
                    )
                    prompts_path.mkdir(parents=True, exist_ok=True)
                    save_prompts_dict(
                        prompts_dict=prompts,
                        ref_affine=sample["meta"]["affine"],
                        ref_header=sample["meta"]["header"],
                        output_dir=prompts_path,
                    )

                # Create temporary storage for the binary mask of the current label during prompt processing.
                tmp_pred = np.zeros_like(prediction, dtype=bool)
                tmp_gt = np.zeros_like(target, dtype=bool)

                # Compute and record metrics separately for each label value
                # found in the ground-truth mask.
                for label in labels:

                    # Compute metrics for the current label value, using the prediction and ground truth target, along with the voxel spacing from the image header.
                    np.equal(prediction, label, out=tmp_pred)
                    np.equal(target, label, out=tmp_gt)
                    metrics = compute_surface_dist_metrics(
                        prediction=tmp_pred,
                        target=tmp_gt,
                        space_mm=sample["meta"]["header"].get_zooms()
                    )
                    print(f"Computed metrics for sample {sample_id}, label {label}: {metrics}")

                    # Append a row for this label value, including metrics and inference time if available.
                    rows.append(
                        EvaluationRow(
                            dataset_name=dataset_name,
                            model_name=model_name,
                            sample_id=sample_id,
                            sample_index=sample_index,
                            label=label,
                            status="ok",
                            inference_seconds=inference_time_per_label[label] if inference_time_per_label else None,
                            dice_score=metrics["dice"],
                            hd95=metrics["hd95"],
                            hd99=metrics["hd99"],
                            avg_dist_gt_to_pred=metrics["avg_surface_dist_gt_to_pred"],
                            avg_dist_pred_to_gt=metrics["avg_surface_dist_pred_to_gt"],
                            sample_gt_path=str(sample["meta"]["mask_path"]),
                            sample_img_path=str(sample["meta"]["image_path"]),
                            sample_pred_path=str(prediction_path) if self.config.save_predictions and prediction is not None else None,
                            sample_prompts_path=str(prompts_path) if self.config.save_generated_prompts and prompts is not None else None,
                            error_message=None,
                        )
                    )

                del tmp_pred, tmp_gt  # Clear temporary masks at the end of processing for this sample to free memory.

            except Exception as exc:  # noqa: BLE001

                # Record the failure for this specific sample and either
                # continue to the next sample or propagate, depending on
                # configuration.
                rows.append(
                    EvaluationRow(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        sample_id=sample_id,
                        sample_index=sample_index,
                        label=None,
                        status="error",
                        inference_seconds=None,
                        dice_score=None,
                        hd95=None,
                        hd99=None,
                        avg_dist_gt_to_pred=None,
                        avg_dist_pred_to_gt=None,
                        sample_img_path=str(sample["meta"]["image_path"]),
                        sample_gt_path=str(sample["meta"]["mask_path"]),
                        sample_pred_path=str(prediction_path) if self.config.save_predictions and prediction is not None else None,
                        sample_prompts_path=str(prompts_path) if self.config.save_generated_prompts and prompts is not None else None,
                        error_message=str(exc),
                    )
                )
                if not self.config.continue_on_error:
                    raise exc
            
        # Final reset in case the loop body didn't reach the per-sample
        # reset for the last sample (e.g. it raised before getting there).
        predictor.reset_session()

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
        # encoding="utf-8-sig" adds a UTF-8 BOM so Excel detects the file as
        # UTF-8 and correctly displays accented characters. The "sep=,"
        # directive on the first line tells Excel to use "," as the
        # delimiter regardless of the user's regional settings, ensuring
        # values with a "." decimal (e.g. 1.1245) are read correctly
        # instead of being misinterpreted as 11245.
        with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
            handle.write("sep=,\n")
            writer = csv.DictWriter(handle, fieldnames=cls.CSV_FIELDNAMES, delimiter=",")
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    @classmethod
    def _load_existing_rows(cls, output_csv: Path) -> list[EvaluationRow]:
        """Loads previously written rows from an existing results CSV.

        Used to resume an interrupted evaluation run without losing rows
        that were already computed and persisted in a prior invocation
        (since ``_write_csv`` overwrites the file each time it's called).

        Args:
            output_csv: Path to the CSV file to load, if it exists.

        Returns:
            A list of EvaluationRow instances reconstructed from the CSV,
            or an empty list if the file does not exist.
        """
        if not output_csv.exists():
            return []

        rows: list[EvaluationRow] = []
        with output_csv.open("r", newline="", encoding="utf-8-sig") as handle:
            # _write_csv prepends a "sep=," directive line (an Excel-only
            # convention, not part of the CSV spec) so that double-clicking
            # the file in Excel uses the correct delimiter regardless of
            # locale. csv.DictReader has no knowledge of this convention and
            # would otherwise treat "sep=," as the header row, so it must be
            # skipped explicitly before handing the stream to DictReader.
            first_line_pos = handle.tell()
            first_line = handle.readline()
            if not first_line.startswith("sep="):
                handle.seek(first_line_pos)

            for raw in csv.DictReader(handle, delimiter=","):
                rows.append(
                    EvaluationRow(
                        dataset_name=raw["dataset_name"],
                        model_name=raw["model_name"],
                        sample_id=raw["sample_id"],
                        sample_index=int(raw["sample_index"]),
                        label=int(raw["label"]) if raw["label"] not in ("", None) else None,
                        status=raw["status"],
                        inference_seconds=cls._to_float(raw["inference_seconds"]),
                        dice_score=cls._to_float(raw["dice_score"]),
                        hd95=cls._to_float(raw["hd95"]),
                        hd99=cls._to_float(raw["hd99"]),
                        avg_dist_gt_to_pred=cls._to_float(raw["avg_dist_gt_to_pred"]),
                        avg_dist_pred_to_gt=cls._to_float(raw["avg_dist_pred_to_gt"]),
                        error_message=raw["error_message"] or None,
                        sample_img_path=raw["sample_img_path"] or None,
                        sample_gt_path=raw["sample_gt_path"] or None,
                    )
                )
        return rows

    @staticmethod
    def _to_float(value: str | None) -> float | None:
        """Converts a CSV string field back to a float, treating empty/None as None."""
        return float(value) if value not in ("", None) else None

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
