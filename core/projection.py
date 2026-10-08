import numpy as np


def local_frame(center_xyz):
    """Return an orthonormal tangent basis (e1, e2) at center_xyz."""
    c = center_xyz / np.linalg.norm(center_xyz)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, c)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(ref, c)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(c, e1)
    return e1, e2


def project_point(p, center_xyz, e1, e2):
    """Azimuthal equidistant projection of p onto the tangent plane at c."""
    c = center_xyz
    cos_c = float(np.clip(np.dot(p, c), -1.0, 1.0))
    c_ang = np.arccos(cos_c)

    if c_ang < 1e-12:
        return np.array([0.0, 0.0])

    tangent = p - cos_c * c
    x = float(np.dot(tangent, e1))
    y = float(np.dot(tangent, e2))
    n = np.hypot(x, y)

    if n < 1e-12:
        return np.array([0.0, 0.0])

    return c_ang * np.array([x / n, y / n])


def project_cell(cell):
    """Project the four corners of a cell using AEQD centred on its centroid."""
    c  = cell.centroid_xyz
    e1, e2 = local_frame(c)
    xy = np.array([project_point(p, c, e1, e2) for p in cell.corners_xyz])
    cell.local_frame  = (e1, e2)
    cell.projected_xy = xy
    return xy


# ---------------------------------------------------------------------
# Equal-area cube-sphere mapping (QSC family)
# ---------------------------------------------------------------------

def cube_face_to_sphere_equal_area(u, v):
    """
    Map (u, v) in [-1, 1]^2 on a cube face to a unit-sphere direction
    with EXACTLY uniform area density.

    Face centre -> face normal.
    Face corner -> equator.
    """
    r2 = u * u + v * v
    factor = np.sqrt(np.maximum(0.0, 1.0 - r2 / 4.0))
    X = u * factor
    Y = v * factor
    Z = 1.0 - r2 / 2.0
    n = np.sqrt(X * X + Y * Y + Z * Z)
    return np.stack([X / n, Y / n, Z / n], axis=-1)


def sphere_to_cube_face_equal_area(xyz):
    """Inverse of the above, per point."""
    X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    denom = np.maximum(1.0 + Z, 1e-12)
    scale = np.sqrt(2.0 / denom)
    return X * scale, Y * scale
