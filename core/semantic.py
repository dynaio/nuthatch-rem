import numpy as np


def sample_cell_xyz(cell, n_side=8):
    """
    Return (n_side^2, 3) unit vectors sampled uniformly in the cell's
    face UV rectangle.  Uses the SAME equal-area mapping as the corners,
    so the samples lie inside the cell on the sphere.
    """
    from core.projection import cube_face_to_sphere_equal_area

    uv = cell.uv_corners
    u_min, v_min = uv[0]
    u_max, v_max = uv[2]

    lin = np.linspace(0.0, 1.0, n_side)
    uu, vv = np.meshgrid(lin, lin)
    u = u_min + (u_max - u_min) * uu.ravel()
    v = v_min + (v_max - v_min) * vv.ravel()

    local = cube_face_to_sphere_equal_area(u, v)
    xyz = local @ cell.rot.T
    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz


def xyz_to_uv(xyz):
    """Unit vectors -> equirectangular UV in [0, 1] x [0, 1]."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lat = np.arcsin(np.clip(z, -1.0, 1.0))
    lon = np.arctan2(y, x)

    u = (lon + np.pi) / (2.0 * np.pi)
    v = (np.pi / 2.0 - lat) / np.pi
    return u, v


def sample_mask(mask, u, v):
    H, W = mask.shape
    px = np.clip((u * W).astype(int), 0, W - 1)
    py = np.clip((v * H).astype(int), 0, H - 1)
    return mask[py, px]


def compute_land_fraction(cell, mask, n_side=8):
    xyz = sample_cell_xyz(cell, n_side=n_side)
    u, v = xyz_to_uv(xyz)
    samples = sample_mask(mask, u, v)
    frac = float(samples.mean())
    cell.land_fraction = frac
    cell.sea_fraction  = 1.0 - frac
    return frac
