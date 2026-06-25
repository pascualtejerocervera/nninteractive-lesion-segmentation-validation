import numpy as np


def voxel_to_world(voxel_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert voxel/pixel coordinates -> world coordinates.

    voxel_coords: (N, 3) or (3,)
    affine: (4, 4)

    returns: (N, 3)
    """
    voxel_coords = np.asarray(voxel_coords, dtype=float)

    single = False
    if voxel_coords.ndim == 1:
        voxel_coords = voxel_coords[None, :]
        single = True

    homo = np.concatenate([voxel_coords, np.ones((voxel_coords.shape[0], 1))], axis=1)
    world = homo @ affine.T

    result = world[:, :3]
    return result[0] if single else result


def world_to_voxel(world_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert world coordinates -> voxel/pixel coordinates.

    world_coords: (N, 3) or (3,)
    affine: (4, 4)

    returns: (N, 3)
    """
    world_coords = np.asarray(world_coords, dtype=float)

    single = False
    if world_coords.ndim == 1:
        world_coords = world_coords[None, :]
        single = True

    inv_affine = np.linalg.inv(affine)

    homo = np.concatenate([world_coords, np.ones((world_coords.shape[0], 1))], axis=1)
    voxel = homo @ inv_affine.T

    result = voxel[:, :3]
    return result[0] if single else result
