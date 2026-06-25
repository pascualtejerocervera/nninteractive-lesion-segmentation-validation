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
    # Create a new NIfTI image using nibabel
    nifti_image = nib.Nifti1Image(image_np, affine, header)

    # Save the NIfTI image to the specified file path
    nib.save(nifti_image, file_path)
    
