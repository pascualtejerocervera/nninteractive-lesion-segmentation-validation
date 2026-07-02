from dataclasses import dataclass
from pathlib import Path

import numpy as np
import nibabel as nib
@dataclass(frozen=True, slots=True)
class NiftiPair:
    """Container for a paired image-mask sample.

    Attributes:
        case_id: Unique identifier for the case.
        image_path: Path to the image volume.
        mask_path: Path to the segmentation mask.
        unique_labels: List of unique non-zero label values in the mask.
    """
    case_id: str
    image_path: Path
    mask_path: Path
    unique_labels: list[int] 

def is_nifti_path(path: Path) -> bool:
    """Checks whether a file path corresponds to a NIfTI image.

    Supports both `.nii` and `.nii.gz` formats.

    Args:
        path: File path to check.

    Returns:
        True if the path is a NIfTI file, False otherwise.
    """
    return path.suffix == ".nii" or path.name.endswith(".nii.gz")

def strip_nifti_extension(path: Path) -> Path:
    """
    Strips the NIfTI file extension from a path.    

    Args:
        path: File path to strip.

    Returns:
        Path without the NIfTI extension.
    """
    name = path.name

    if name.endswith(".nii.gz"):
        return path.with_name(name[:-7])

    if name.endswith(".nii"):
        return path.with_name(name[:-4])

    return path

def read_nifti(path: Path, extract_unique_labels: bool = False):
    """
    Reads NIfTI metadata and optionally extracts unique non-zero labels.

    Args:
        path: Path to NIfTI file.
        extract_unique_labels: If True, also computes unique non-zero labels.

    Returns:
        affine, header, shape, (optional) unique_labels
    """
    img = nib.load(str(path), mmap=False)

    affine = img.affine
    header = img.header
    shape = img.shape

    if not extract_unique_labels:
        return affine, header, shape

    data = np.asanyarray(img.dataobj, dtype=np.uint8)

    # Compute unique non-zero labels in the mask
    labels = [int(x) for x in np.unique(data) if int(x) != 0]
    return affine, header, shape, labels
