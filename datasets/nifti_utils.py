from dataclasses import dataclass
from pathlib import Path

import nibabel as nib

@dataclass(frozen=True, slots=True)
class NiftiPair:
    """Container for a paired image-mask sample.

    Attributes:
        case_id: Unique identifier for the case.
        image_path: Path to the image volume.
        mask_path: Path to the segmentation mask.
    """
    case_id: str
    image_path: Path
    mask_path: Path

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

def read_nifti_meta(path):
    """
    Reads the affine and header metadata from a NIfTI file without loading the full image data
    
    Args:
        path: Path to the NIfTI file.

    Returns:
        Tuple of (affine, header, shape).
    """
    img = nib.load(str(path), mmap=False)  # does NOT load full array
    return img.affine, img.header, img.shape
