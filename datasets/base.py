from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

import numpy as np

SampleDict = dict[str, Any]
PreprocessHook = Callable[
    [np.ndarray, np.ndarray, Any, dict[str, Any]],
    tuple[np.ndarray, np.ndarray, dict[str, Any]],
]


class BaseMedicalSegmentationDataset(ABC):
    """Abstract base class for medical image segmentation datasets.

    This class defines the common interface and shared configuration for
    segmentation datasets. Concrete implementations are responsible for
    loading the dataset and returning samples in a consistent format.

    Attributes:
        root: Root directory containing the dataset.
        image_folder: Relative path to the image directory.
        mask_folder: Relative path to the segmentation mask directory.
        modality: Imaging modality (e.g., ``"ct"``, ``"mr"``).
        dimension: Dataset dimensionality (e.g., ``"2d"``, ``"3d"``).
        preprocess: Optional preprocessing function applied to each sample.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        image_folder: str = "images",
        mask_folder: str = "masks",
        modality: str = "ct",
        dimension: str = "3d",
        preprocess: PreprocessHook | None = None,
        **_: Any,
    ) -> None:
        """Initializes the dataset.

        Args:
            root: Root directory containing the dataset.
            image_folder: Name of the directory containing the images.
            mask_folder: Name of the directory containing the segmentation
                masks.
            modality: Imaging modality.
            dimension: Dataset dimensionality.
            preprocess: Optional preprocessing function applied to each
                sample.
            **_: Additional unused keyword arguments. Ignored to allow
                dataset-specific configuration dictionaries.
        """
        # Store the dataset root as an absolute path.
        self.root = Path(root).expanduser().resolve()

        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.modality = modality
        self.dimension = dimension
        self.preprocess = preprocess

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> SampleDict:
        """Returns a dataset sample.

        Args:
            index: Sample index.

        Returns:
            A dictionary containing the sample data and associated metadata.
        """
        raise NotImplementedError
