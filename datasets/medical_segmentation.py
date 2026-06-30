from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets.base import BaseMedicalSegmentationDataset, SampleDict
from interactive_seg.io.image_io import load_nifti_image

class PairedNiftiMedicalSegmentationDataset(BaseMedicalSegmentationDataset):
    """Medical segmentation dataset for paired NIfTI volumes.

    This dataset assumes a directory structure of the form:

        root/
            images/
            masks/

    or different directory names specified by ``image_folder`` and ``mask_folder``. kwargs: Additional keyword arguments passed to the base class.

    Each image must have a corresponding mask with the same case identifier.
    The dataset performs strict validation of:
        - Shape consistency
        - Affine matrix equality
        - Voxel spacing consistency
    """

    def __init__(
        self,
        *,
        root: str | Path,
        image_folder: str | None = None,
        mask_folder: str | None = None,
        modality: str = "ct",
        dimension: str = "3d",
        **kwargs: Any,
    ) -> None:
        """Initializes the paired NIfTI dataset.

        Args:
            root: Root directory of the dataset.
            image_folder: Subdirectory containing image volumes.
            mask_folder: Subdirectory containing segmentation masks.
            modality: Imaging modality (e.g., "ct", "mr").
            dimension: Data dimensionality (e.g., "3d").
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(
            root=root,
            image_folder=image_folder,
            mask_folder=mask_folder,
            modality=modality,
            dimension=dimension,
            **kwargs,
        )

        self.samples = None # To be populated by subclasses with a list of NiftiPair instances.
        

    def __len__(self) -> int:
        """Returns number of paired samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> SampleDict:
        """Loads a single image-mask pair.

        Args:
            index: Index of the sample.

        Returns:
            Dictionary containing:
                - image: np.ndarray volume
                - mask: np.ndarray segmentation mask
                - meta: metadata dictionary with case information
        """
        sample = self.samples[index]

        img_np, affine_img, header_img = load_nifti_image(
            sample.image_path, is_mask=False
        )
        mask_np, _, _ = load_nifti_image(sample.mask_path, is_mask=True)

        meta: dict[str, Any] = {
            "case_id": sample.case_id,
            "image_path": str(sample.image_path),
            "mask_path": str(sample.mask_path),
            "modality": self.modality,
            "dimension": self.dimension,
            "shape": img_np.shape,
            "affine": affine_img,
            "header": header_img,
        }

        return {
            "image": img_np,
            "mask": mask_np,
            "meta": meta,
        }
