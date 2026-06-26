from pathlib import Path

import nibabel as nib
import numpy as np

def load_nifti_image(file_path: Path, is_mask: bool) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """
    Load a NIfTI image and return the image data, affine matrix, and header.s

    Args:
        file_path: Path to the NIfTI file.

    Returns:
        image_np: Numpy array of the image data.
        affine: Affine transformation matrix.
        header: NIfTI image header.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"The specified file does not exist: {file_path}")
    if not (file_path.name.endswith(".nii") or file_path.name.endswith(".nii.gz")):
        raise ValueError(f"The specified file is not a NIfTI file: {file_path}")
    if not isinstance(is_mask, bool):
        raise ValueError("The 'is_mask' parameter must be a boolean value.")

    # Load the NIfTI image using nibabel and convert it to canonical orientation
    # img = nib.as_closest_canonical(nib.load(file_path)) # Not recommended as it changes the affine
    img = nib.load(file_path)  # Reload to preserve original orientation in affine and header

    # Extract image data andaffine matrix
    image_np = np.asanyarray(img.dataobj)  # avoids float64 conversion
    affine = img.affine
    header = img.header

    image_np = image_np.astype(np.uint8 if is_mask else np.float32, copy=False)

    return image_np, affine, header

def save_nifti_image(
    image_np: np.ndarray, 
    affine: np.ndarray, 
    header: nib.Nifti1Header, 
    file_path: Path
) -> None:
    """
    Save a NIfTI image from a numpy array and affine matrix.

    Args:
        image_np: Numpy array of the image data.
        affine: Affine transformation matrix.
        header: NIfTI image header.
        file_path: Path to save the NIfTI file
    """
    if not isinstance(image_np, np.ndarray):
        raise ValueError("Input image must be a numpy array.")
    if image_np.ndim != 3:
        raise ValueError("Input image must be a 3D array.")
    if not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
        raise ValueError("Affine must be a 4x4 numpy array.")
    if not isinstance(header, nib.Nifti1Header):
        raise ValueError("Header must be a nibabel Nifti1Header object.")

    # Create a new NIfTI image using nibabel
    nifti_image = nib.Nifti1Image(image_np, affine, header)

    # Save the NIfTI image to the specified file path
    nib.save(nifti_image, file_path)
    
