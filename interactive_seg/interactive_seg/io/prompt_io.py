import json
import numpy as np
from pathlib import Path

import nibabel as nib

from interactive_seg.io.image_io import save_nifti_image
from interactive_seg.utils.geometry.voxel_transforms import voxel_to_world

SCHEMA = "https://raw.githubusercontent.com/Slicer/Slicer/main/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#"
Point3D = tuple[tuple[int, int, int], ...]
BBox3D = tuple[tuple[int, int, int, int, int, int], ...]

def save_prompts_dict(
    prompts_dict: dict[str, dict[int, Point3D] | dict[int, BBox3D] | np.ndarray], 
    ref_affine: np.ndarray,
    ref_header: nib.Nifti1Header,
    output_dir: Path
):
    """
    Save the generated prompts to disk in a structured format.

    Args:
        prompts_dict: Dictionary containing the generated prompts.
        ref_affine: Affine matrix to use for saving the prompts (can be the same as the original image or an 
            identity matrix).
        ref_header: NIfTI image header to use for saving the prompt masks (can be the same as the original image 
            header or a default header).
        output_dir: Directory where the prompts will be saved.
        ref_header: NIfTI image header to use for saving the prompt masks (can be the same as the original image 
            header or a default header).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    save_prompt_markups_json(prompts_dict, ref_affine, output_dir)  # Save prompts in JSON format for Slicer compatibility
    save_prompt_masks(prompts_dict, ref_affine, ref_header, output_dir)  # Save arrays (diameter annotations and spline scribbles) as NIfTI files


def save_prompt_masks(prompts_dict, ref_affine, ref_header, output_dir):
    """
    Save the arrays in the prompts_dict to disk as NIfTI files.

    Args:
        prompts_dict: Dictionary containing the generated prompts. Keys can include:
            - "pts_pos": dict[label, Point3D]
            - "pts_neg": dict[label, Point3D]
            - "bboxes_pos": dict[label, BBox3D]
            - "scribble_diameter_ann": np.ndarray of shape (D, H, W) and dtype uint8, where non-zero values indicate diameter annotation voxels.
            - "scribble_spline": np.ndarray of shape (D, H, W) and dtype uint8, where non-zero values indicate spline scribble voxels.
        ref_affine: Affine matrix to use for saving the prompts (can be the same as the original image or an 
            identity matrix).
        ref_header: NIfTI image header to use for saving the prompt masks (can be the same as the original 
            image header or a default header).
        output_dir: Directory where the prompts will be saved.
    """
    for prompt_type, content in prompts_dict.items():
        if prompt_type in ("scribble_diameter_ann", "scribble_spline") and isinstance(content, np.ndarray):
            file_path = output_dir / f"{prompt_type}.nii.gz"
            save_nifti_image(content, ref_affine, ref_header, file_path)  # Use the reference affine for consistency
        elif prompt_type in ("scribble_diameter_ann", "scribble_spline") and isinstance(content, dict):
            for label, mask in content.items():
                file_path = output_dir / f"{prompt_type}_label_{label}.nii.gz"
                save_nifti_image(mask, ref_affine, ref_header, file_path)  # Use the reference affine for consistency

def save_prompt_markups_json(prompts_dict, affine, out_path):

    if "pts_pos" not in prompts_dict and "pts_neg" not in prompts_dict and "bboxes_pos" not in prompts_dict:
        print("No point or bounding box prompts found. Skipping saving markups JSON.")
        return

    control_points = []

    def bbox_to_corner_points(bbox):
        (x0, x1), (y0, y1), (z0, z1) = bbox

        return [
            (x0, y0, z0),
            (x0, y0, z1),
            (x0, y1, z0),
            (x0, y1, z1),
            (x1, y0, z0),
            (x1, y0, z1),
            (x1, y1, z0),
            (x1, y1, z1),
        ]


    # -------------------------
    # POSITIVE POINTS
    # -------------------------
    for label, pts in (prompts_dict.get("pts_pos") or {}).items():
        for idx, pt in enumerate(pts): 
            control_points.append({
                "label": f"PT-POS-label={label}-idx={idx}",
                "position": [float(coord) for coord in voxel_to_world(pt, affine)]
            })

    # -------------------------
    # NEGATIVE POINTS
    # -------------------------
    for label, pts in (prompts_dict.get("pts_neg") or {}).items():
        for idx, pt in enumerate(pts):
            control_points.append({
                "label": f"PT-NEG-label={label}-idx={idx}",
                "position": [float(coord) for coord in voxel_to_world(pt, affine)]
            })

    # -------------------------
    # BOUNDING BOXES → 8 CORNERS
    # -------------------------
    for label, bboxes in (prompts_dict.get("bboxes_pos") or {}).items():
        for idx, bbox in enumerate(bboxes):
            for pt in bbox_to_corner_points(bbox):
                control_points.append({
                    "label": f"BOX-POS-label={label}-idx={idx}",
                    "position": [float(coord) for coord in voxel_to_world(pt, affine)]
                })

    # -------------------------
    # FINAL SLICER MARKUPS JSON
    # -------------------------
    output = {
        "@schema": SCHEMA,
        "markups": [
            {
                "type": "Fiducial",
                "coordinateSystem": "RAS",
                "controlPoints": control_points
            },
            
        ]
    }


    out_path = Path(out_path)
    with open(out_path / "prompts.json", "w") as f:
        json.dump(output, f, indent=2)
