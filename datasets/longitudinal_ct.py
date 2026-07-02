from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from datasets.medical_segmentation import PairedNiftiMedicalSegmentationDataset
from datasets.nifti_utils import NiftiPair, is_nifti_path, read_nifti
from datasets.registry import register_dataset


@register_dataset("longitudinal_ct")
class LongitudinalCTDataset(PairedNiftiMedicalSegmentationDataset):
    """Longitudinal CT dataset based on paired NIfTI volumes.

    This dataset implementation wraps a paired image/mask structure stored
    in NIfTI format and provides compatibility with the medical segmentation
    pipeline defined in ``PairedNiftiMedicalSegmentationDataset``.

    The class is registered in the dataset registry under the key:
        "longitudinal_ct"
    """

    def __init__(
        self,
        *,
        root: str | Path,
        image_folder: str = "images",
        mask_folder: str = "masks",
        modality: str = "ct",
        dimension: str = "3d",
        preprocess: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the longitudinal CT dataset.

        Args:
            root: Root directory of the dataset.
            image_folder: Folder containing input CT images.
            mask_folder: Folder containing segmentation masks.
            modality: Imaging modality (default: "ct").
            dimension: Data dimensionality (default: "3d").
            preprocess: Optional preprocessing function applied to samples.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(
            root=root,
            image_folder=image_folder,
            mask_folder=mask_folder,
            modality=modality,
            dimension=dimension,
            preprocess=preprocess,
            **kwargs,
        )

        # Additional attributes specific to longitudinal CT datasets can be initialized here
        self.inputs_tr = kwargs.pop("inputs_tr_folder", "inputsTr")
        self.targets_tr = kwargs.pop("targets_tr_folder", "targetsTr")

        # Discover paired samples in the dataset
        self.samples = self._discover_pairs()

    # Filenames look like:
    #   {case_id}_{BL|FU}_img_{region:02d}.nii.gz
    #   {case_id}_{BL|FU}_mask_{region:02d}.nii.gz
    # BL image+mask both live in inputsTr. FU images live in inputsTr but
    # FU masks live in targetsTr. The trailing 2-digit suffix is a body
    # *region* index (not a timepoint index) -- a patient can have several
    # disjoint scan regions per phase (e.g. chest + abdomen), each with its
    # own image/mask pair.
    _FILENAME_RE = re.compile(
        r"^(?P<case_id>[0-9a-f]+)_(?P<phase>BL|FU)_(?P<kind>img|mask)_"
        r"(?P<region>\d+)\.nii(?:\.gz)?$"
    )

    @classmethod
    def _index_nifti_dir(cls, folder: Path) -> dict[tuple[str, str, str, str], Path]:
        """Indexes a directory of NIfTI files by (case_id, phase, region, kind)."""
        index: dict[tuple[str, str, str, str], Path] = {}
        for content in os.listdir(folder):
            path = folder / content
            if not is_nifti_path(path):
                continue
            match = cls._FILENAME_RE.match(path.name)
            if match is None:
                continue
            key = (
                match["case_id"],
                match["phase"],
                match["region"],
                match["kind"],
            )
            index[key] = path
        return index

    def _discover_pairs(self) -> list[NiftiPair]:
        """Scans inputsTr/targetsTr and builds image-mask pairs.

        BL pairs are resolved entirely within inputsTr. FU pairs are
        resolved by matching an FU image in inputsTr to the FU mask with
        the same (case_id, region) in targetsTr.
        """

        inputs_tr = self.root / self.inputs_tr   # inputsTr
        targets_tr = self.root / self.targets_tr   # targetsTr

        if not inputs_tr.exists():
            raise FileNotFoundError(f"Input folder not found: {inputs_tr}")
        if not targets_tr.exists():
            raise FileNotFoundError(f"Target folder not found: {targets_tr}")

        inputs_index = self._index_nifti_dir(inputs_tr)
        targets_index = self._index_nifti_dir(targets_tr)

        image_files: dict[tuple[str, str, str], Path] = {}
        mask_files: dict[tuple[str, str, str], Path] = {}

        for (case_id, phase, region, kind), path in inputs_index.items():
            if kind != "img":
                continue
            image_files[(case_id, phase, region)] = path

        for (case_id, phase, region, kind), path in inputs_index.items():
            if kind != "mask" or phase != "BL":
                continue
            mask_files[(case_id, phase, region)] = path

        for (case_id, phase, region, kind), path in targets_index.items():
            if kind != "mask" or phase != "FU":
                continue
            mask_files[(case_id, phase, region)] = path

        # Optional strict validation
        missing_images = sorted(set(mask_files) - set(image_files))
        missing_masks = sorted(set(image_files) - set(mask_files))

        if missing_images:
            raise ValueError(f"Missing images for cases: {missing_images[:5]}")
        if missing_masks:
            raise ValueError(f"Missing masks for cases: {missing_masks[:5]}")
        
        # Sort keys for deterministic ordering
        keys = sorted(image_files.keys())

        # ---------------------------------------------------------
        # Build dataset
        # ---------------------------------------------------------
        samples: list[NiftiPair] = []
        n_keys = len(keys)

        for idx, (case_id, phase, region) in enumerate(keys, start=1):
            image_path = image_files[(case_id, phase, region)]
            mask_path = mask_files[(case_id, phase, region)]

            # Load for validation (keep your strict checks)
            affine_img, header_img, shape_img = read_nifti(image_path)
            affine_mask, header_mask, shape_mask, unique_labels = read_nifti(mask_path, extract_unique_labels=True)

            if len(unique_labels) == 0:
                print(f"Not added sample {idx}/{n_keys}: {case_id}_{phase}_{region} because mask has no labels.")
                continue  # Skip masks with no labels

            if shape_img != shape_mask:
                raise ValueError(
                    f"Shape mismatch for {case_id}: "
                    f"{shape_img} vs {shape_mask}"
                )

            if not np.allclose(affine_img, affine_mask):
                raise ValueError(f"Affine mismatch for {case_id}")

            if header_img.get_zooms() != header_mask.get_zooms():
                raise ValueError(f"Voxel size mismatch for {case_id}")

            samples.append(
                NiftiPair(
                    case_id=f"{case_id}_{phase}_{region}",
                    image_path=image_path,
                    mask_path=mask_path,
                    unique_labels=unique_labels
                )
            )
            print(f"Added sample {idx}/{n_keys}: {case_id}_{phase}_{region} with labels {unique_labels}")

        return samples
