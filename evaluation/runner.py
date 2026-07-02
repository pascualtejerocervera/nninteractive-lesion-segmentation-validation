from __future__ import annotations

import csv
import gc
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import yaml

from datasets import build_dataset
from evaluation.config import EvaluationModelRunConfig, EvaluationRunConfig
from evaluation.dataset_selection import resolve_requested_dataset_configs
from evaluation.metrics import compute_surface_dist_metrics
from evaluation.predictor_factory import build_predictor
from interactive_seg import generate_nninteractive_prompts
from interactive_seg.io.image_io import save_nifti_image
from interactive_seg.io.prompt_io import save_prompts_dict

# Key identifying a single (dataset, model, sample) combination, used for
# resume/dedup bookkeeping. Kept as a plain tuple of strings (rather than a
# full EvaluationRow) so the resume state stays cheap to hold in memory even
# for runs with a very large number of prior results.
SampleKey = tuple[str, str, str]


@dataclass(slots=True)
class EvaluationRow:
    """A single result row produced by the evaluation pipeline.

    Each row corresponds to the metrics computed for one label value within
    one sample, for a given dataset/model combination. Rows with an "error"
    status represent a failure for that sample/dataset/model and will have
    most metric fields left as ``None``.

    Attributes:
        model_run_name: Name of the model run (taken from the stem of the model config file path) used for this evaluation.
        dataset_name: Human-readable name of the dataset.
        model_name: Name of the model (from the evaluation config).
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
    model_run_name: str
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
    segmentation to disk.

    Memory behavior: result rows are streamed straight to the output CSV as
    they're produced (flushed immediately) rather than accumulated in an
    in-memory list for the whole run. Resume state from a prior, possibly
    interrupted run is likewise kept as a small set of
    ``(dataset, model, sample_id)`` keys rather than fully materialized
    ``EvaluationRow`` objects. This keeps peak memory roughly bounded by the
    volumes of the sample currently being processed, instead of growing with
    the total number of results produced over the run.
    """

    # Column order used when writing results.csv. Note this intentionally
    # excludes "model_config_path" even though it exists on EvaluationRow.
    CSV_FIELDNAMES = [
        "model_run_name",
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

    # How many samples to process between forced gc.collect() calls. This
    # doesn't change correctness; it just gives the allocator a chance to
    # actually release freed numpy buffers (which can otherwise linger due
    # to reference cycles created by exception tracebacks) back to the OS
    # sooner rather than later.
    GC_EVERY_N_SAMPLES = 5

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

    def run(self) -> None:
        """Executes the evaluation process for the specified datasets and model runs.

        For each resolved dataset config, the dataset is built and then
        evaluated against every configured model run. Failures while
        building a dataset or running a model propagate after the current
        results have been flushed to disk, if ``continue_on_error`` is
        False on the config.

        Result rows are streamed directly to the results CSV as they're
        computed (see ``_csv_row_writer``), so a crash or interruption never
        loses more than the row currently in flight. Resume state from any
        prior run present at ``self.output_csv`` is recovered up front as a
        lightweight set of keys (see ``_prepare_resume_state``) rather than
        loading the full prior result set into memory.

        Raises:
            Exception: Re-raises the first encountered exception if
                ``self.config.continue_on_error`` is False.
        """
        processed_keys = self._prepare_resume_state(self.output_csv)
        existing_row_count = self._count_csv_rows(self.output_csv)
        write_header = existing_row_count == 0

        dataset_configs = resolve_requested_dataset_configs(
            self.config.dataset,
            all_configs=self.config.dataset == "all",
        )

        with self._csv_row_writer(self.output_csv, write_header=write_header) as write_row:
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
                            write_row=write_row,
                            processed_keys=processed_keys,
                        )
                    except Exception as exc:  # noqa: BLE001
                        if not self.config.continue_on_error:
                            raise exc

                # Drop the dataset object before moving on to the next one
                # so any volumes it may be caching aren't held alongside
                # the next dataset's.
                del dataset
                gc.collect()

    def _run_model_on_dataset(
        self,
        dataset: Any,
        dataset_name: str,
        model_run: EvaluationModelRunConfig,
        write_row: Callable[[EvaluationRow], None],
        processed_keys: set[SampleKey],
    ) -> None:
        """Evaluates a single model run against every sample in a dataset.

        Loads the model's YAML config, builds the predictor, then iterates
        over every sample in the dataset: generates prompts from the ground
        truth mask, runs the predictor, computes per-label metrics, and
        optionally saves the predicted segmentation as a NIfTI file. Each
        result row (success or per-sample error) is written to disk via
        ``write_row`` as soon as it's produced, rather than being
        accumulated.

        Args:
            dataset: The dataset object to iterate over (supports ``len()``
                and indexing, returning dicts with "image", "mask", and
                "meta" keys).
            dataset_name: Human-readable name of the dataset, used for
                labeling rows and naming saved prediction files.
            model_run: Configuration describing which model to run and
                where to load its config from.
            write_row: Callable that appends a single EvaluationRow to the
                results CSV and flushes it to disk immediately.
            processed_keys: Set of (dataset_name, model_name, sample_id)
                keys already completed successfully in a prior run (or
                earlier in this run); samples matching a key are skipped.
                Mutated in place to add newly-completed samples.

        Raises:
            Exception: Re-raised for a given sample if
                ``self.config.continue_on_error`` is False, after the error
                row for that sample has already been written.
        """
        # Load the model's YAML config and split it into the pieces needed
        # for building the predictor and generating prompts.

        config_path = Path(model_run.config_path).expanduser().resolve()
        config = self._load_yaml_file(config_path)
        model_name = config["model_config"]["model_name"]
        model_config = config["model_config"]
        prompt_config = config["prompt_generation"]
        model_run_name = model_run.model_run_name

        # Save the model config used for this run to the output directory, so it can be referenced later.
        self._save_yaml_file(
            path=self.output_path / "model_configs" / f"{config_path.stem}.yaml",
            data=config,
        )

        # Build the predictor for this model run, which will be used to run inference on each sample in the dataset.
        predictor = build_predictor(model_name, model_config)
        n_samples = len(dataset)

        try:
            for sample_index, sample in enumerate(dataset, start=1):
                # Extract the sample data and metadata for the current index.
                meta = sample["meta"]
                sample_id = str(meta["case_id"])

                # Check if the sample was already evaluated in a previous run.
                if (model_run_name, dataset_name, model_name, sample_id) in processed_keys:
                    print(f"Sample {sample_index}/{n_samples}: Model run \"{model_run_name}\": Skipping sample {sample_id} for dataset {dataset_name} and model {model_name} (already completed in a prior run).")
                    break

                # Initialize variables for the large per-sample objects up
                # front (and to None) so the `finally` cleanup below is safe
                # to run no matter how far this iteration got before failing.
                image = None
                target = None
                prediction = None
                prediction_path = None
                prompts = None
                prompts_path = None
                inference_time_per_label = None

                try:
                    # Extract the image and ground truth mask from the sample
                    image = sample["image"]
                    target = sample["mask"]
                    labels = self._labels(target)  # Unique non-zero label values in the ground truth mask
                    if not labels:
                        print(f"Sample {sample_index}/{n_samples}: Sample {sample_id} has no foreground labels in the ground truth mask; skipping evaluation for this sample.")
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
                            affine=meta["affine"],
                            header=meta["header"],
                            file_path=prediction_path.parent / f"{self._safe_name(sample_id)}_image.nii.gz",
                        )
                        save_nifti_image(
                            image=target,
                            affine=meta["affine"],
                            header=meta["header"],
                            file_path=prediction_path.parent / f"{self._safe_name(sample_id)}_gt.nii.gz",
                        )
                        save_nifti_image(
                            image=prediction,
                            affine=meta["affine"],
                            header=meta["header"],
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
                            ref_affine=meta["affine"],
                            ref_header=meta["header"],
                            output_dir=prompts_path,
                        )

                    # Reusable scratch buffers for the per-label binary
                    # masks, allocated once per sample and overwritten in
                    # place (np.equal(..., out=...)) for each label rather
                    # than allocating a fresh array per label.
                    tmp_pred = np.zeros_like(prediction, dtype=bool)
                    tmp_gt = np.zeros_like(target, dtype=bool)

                    # Compute and record metrics separately for each label value
                    # found in the ground-truth mask.
                    for label in labels:
                        np.equal(prediction, label, out=tmp_pred)
                        np.equal(target, label, out=tmp_gt)
                        metrics = compute_surface_dist_metrics(
                            prediction=tmp_pred,
                            target=tmp_gt,
                            space_mm=meta["header"].get_zooms(),
                        )

                        write_row(
                            EvaluationRow(
                                model_run_name=model_run_name,
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
                                sample_gt_path=str(meta["mask_path"]),
                                sample_img_path=str(meta["image_path"]),
                                sample_pred_path=str(prediction_path) if self.config.save_predictions and prediction is not None else None,
                                sample_prompts_path=str(prompts_path) if self.config.save_generated_prompts and prompts is not None else None,
                                error_message=None,
                            )
                        )

                    processed_keys.add((dataset_name, model_name, sample_id))
                    print(f"Sample {sample_index}/{n_samples}: Model run \"{model_run_name}\": Completed sample {sample_id} for dataset {dataset_name} and model {model_name}.")

                    # Clear the per-label scratch buffers now that we're
                    # done with them for this sample.
                    del tmp_pred, tmp_gt

                except Exception as exc:  # noqa: BLE001
                    # Record the failure for this specific sample and either
                    # continue to the next sample or propagate, depending on
                    # configuration.
                    write_row(
                        EvaluationRow(
                            model_run_name=model_run_name,
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

                    print(f"Sample {sample_index}/{n_samples}: Model run \"{model_run_name}\": Error processing sample {sample_id} for dataset {dataset_name} and model {model_name}: {exc}")

                    if not self.config.continue_on_error:
                        raise
                finally:
                    # Explicitly release references to this sample's large
                    # volumes (image, mask, prediction, prompts) rather than
                    # waiting for the next iteration's reassignment to do it.
                    # This also breaks any reference the `exc` traceback
                    # above might otherwise hold onto this frame's locals.
                    sample = None
                    image = None
                    target = None
                    prediction = None
                    prompts = None
                    if sample_index % self.GC_EVERY_N_SAMPLES == 0:
                        gc.collect()

        finally:
            # Final reset in case the loop body didn't reach the per-sample
            # reset for the last sample (e.g. it raised before getting there).
            predictor.reset_session()

        print(f"Model run \"{model_run_name}\": Completed evaluation of {n_samples} samples for dataset {dataset_name} and model {model_name}.")

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

    @staticmethod
    def _skip_sep_line(handle) -> None:
        """Skips the leading "sep=," Excel directive line if present.

        ``_csv_row_writer`` prepends a "sep=," directive line (an
        Excel-only convention, not part of the CSV spec) so that
        double-clicking the file in Excel uses the correct delimiter
        regardless of locale. ``csv.DictReader`` has no knowledge of this
        convention and would otherwise treat "sep=," as the header row, so
        it must be skipped explicitly before handing the stream to
        DictReader.

        Args:
            handle: An open, readable file handle positioned at the start
                of the file. Left positioned right after the directive line
                if one was present, or rewound back to the start otherwise.
        """
        pos = handle.tell()
        line = handle.readline()
        if not line.startswith("sep="):
            handle.seek(pos)

    @classmethod
    def _count_csv_rows(cls, output_csv: Path) -> int:
        """Counts data rows in an existing results CSV without materializing them.

        Args:
            output_csv: Path to the CSV file to count, if it exists.

        Returns:
            The number of data rows (excluding the sep directive and
            header), or 0 if the file does not exist.
        """
        if not output_csv.exists():
            return 0
        with output_csv.open("r", newline="", encoding="utf-8-sig") as handle:
            cls._skip_sep_line(handle)
            return sum(1 for _ in csv.DictReader(handle, delimiter=","))

    @classmethod
    def _prepare_resume_state(cls, output_csv: Path) -> set[SampleKey]:
        """Recovers resume state from a prior run without loading it fully into memory.

        Streams through an existing results CSV (if any) to collect the set
        of (model_run_name, dataset_name, model_name, sample_id) keys that completed
        successfully ("ok") in a prior, possibly interrupted run. Only that
        small set of string tuples is kept in memory — not the full
        ``EvaluationRow`` data for every prior row.

        If the last row on disk belongs to a sample that ended in "error",
        all rows for that sample are dropped so it can be retried cleanly.
        This is done by streaming the old file into a temp file (skipping
        the failed sample's rows) and atomically replacing the original,
        so at most one row is held in memory at a time even during the
        rewrite.

        Args:
            output_csv: Path to a previous run's results CSV, if any.

        Returns:
            The set of (model_run_name, dataset_name, model_name, sample_id) keys that
            completed successfully in a prior run and can be skipped.
        """
        if not output_csv.exists():
            return set()

        processed_keys: set[SampleKey] = set()
        last_key: SampleKey | None = None
        last_status: str | None = None

        with output_csv.open("r", newline="", encoding="utf-8-sig") as handle:
            cls._skip_sep_line(handle)
            for raw in csv.DictReader(handle, delimiter=","):
                key = (raw["model_run_name"], raw["dataset_name"], raw["model_name"], raw["sample_id"])
                if raw["status"] == "ok":
                    processed_keys.add(key)
                last_key, last_status = key, raw["status"]

        if last_status != "error" or last_key is None:
            return processed_keys

        print(f"Model run \"{last_key[0]}\": Last row in {output_csv} was an error for model_run_name={last_key[0]}, dataset_name={last_key[1]}, model_name={last_key[2]}, sample_id={last_key[3]}. Dropping all rows for that sample so it can be retried cleanly.")
        processed_keys.discard(last_key)

        tmp_path = output_csv.with_suffix(output_csv.suffix + ".tmp")
        with (
            output_csv.open("r", newline="", encoding="utf-8-sig") as src,
            tmp_path.open("w", newline="", encoding="utf-8-sig") as dst,
        ):
            cls._skip_sep_line(src)
            dst.write("sep=,\n")
            writer = csv.DictWriter(dst, fieldnames=cls.CSV_FIELDNAMES, delimiter=",")
            writer.writeheader()
            for raw in csv.DictReader(src, delimiter=","):
                key = (raw["model_run_name"], raw["dataset_name"], raw["model_name"], raw["sample_id"])
                if key == last_key:
                    continue
                writer.writerow(raw)
        tmp_path.replace(output_csv)

        return processed_keys

    @contextmanager
    def _csv_row_writer(
        self, output_csv: Path, write_header: bool
    ) -> Iterator[Callable[[EvaluationRow], None]]:
        """Opens the results CSV for streaming, append-only writes.

        Rows are written and flushed to disk as soon as they're produced,
        instead of being accumulated in memory and rewritten in bulk on
        every checkpoint. This keeps peak memory bounded by the current
        sample rather than growing with the total number of results
        produced over the run, and means a crash never loses more than the
        row currently in flight.

        Args:
            output_csv: Path to the CSV file to append to (or create).
            write_header: Whether to write the "sep=," directive and column
                header before any rows (True for a fresh file).

        Yields:
            A callable that appends a single EvaluationRow to the file and
            flushes it to disk immediately.
        """
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if write_header else "a"
        handle = output_csv.open(mode, newline="", encoding="utf-8-sig")
        try:
            writer = csv.DictWriter(handle, fieldnames=self.CSV_FIELDNAMES, delimiter=",")
            if write_header:
                # "sep=," tells Excel to use "," as the delimiter regardless
                # of the user's regional settings, so values with a "."
                # decimal (e.g. 1.1245) aren't misread as 11245. utf-8-sig
                # adds a BOM so Excel detects UTF-8 and shows accented
                # characters correctly.
                handle.write("sep=,\n")
                writer.writeheader()

            def write_row(row: EvaluationRow) -> None:
                writer.writerow(asdict(row))
                handle.flush()

            yield write_row
        finally:
            handle.close()

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
