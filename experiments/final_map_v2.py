"""
Two visualisations of the REM world map:

  A) Cube-cross layout with no white gaps (the "true" REM map)
  B) Equirectangular world map showing the REM distortion field
     (familiar view, easier for readers)

Both use the same QSC equal-area mapping.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.projection import cube_face_to_sphere_equal_area


# ---------------------------------------------------------------------
# Face orientations — chosen so continents appear naturally
# Column 0 of R = world direction that local u maps to
# Column 1 of R = world direction that local v maps to
# Column 2 of R = world direction of face centre
# ---------------------------------------------------------------------
FACE_ROTATIONS = {
    0: np.array([[ 1, 0, 0],    # north pole face;  u east, v north-ish
                 [ 0, 1, 0],
                 [ 0, 0, 1]], dtype=float),
    1: np.array([[ 1, 0, 0],    # south pole face
                 [ 0,-1, 0],
                 [ 0, 0,-1]], dtype=float),
    2: np.array([[ 0, 1, 0],    # lon 0 face;  u = +Y east, v = +Z north
                 [ 0, 0, 1],
                 [ 1, 0, 0]], dtype=float),
    3: np.array([[ 0,-1, 0],    # lon 180 face
                 [ 0, 0, 1],
                 [-1, 0, 0]], dtype=float),
    4: np.array([[-1, 0, 0],    # lon +90 face
                 [ 0, 0, 1],
                 [ 0, 1, 0]], dtype=float),
    5: np.array([[ 1, 0, 0],    # lon -90 face
                 [ 0, 0, 1],
                 [ 0,-1, 0]], dtype=float),
}

LAND_RGB = np.array([210, 180, 130], dtype=np.uint8)
SEA_RGB  = np.array([160, 200, 230], dtype=np.uint8)


def face_rays(face_id, res):
    lin = np.linspace(-1.0, 1.0, res)
    U, V = np.meshgrid(lin, lin, indexing="xy")
    local = cube_face_to_sphere_equal_area(U.ravel(), V.ravel())
    R = FACE_ROTATIONS[face_id]
    xyz = local @ R           # already orthonormal rows -> same as R.T @ local.T
    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz


def sample_mask(xyz, mask):
    H, W = mask.shape
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lat = np.arcsin(np.clip(z, -1, 1))
    lon = np.arctan2(y, x)
    px = ((lon + np.pi) / (2 * np.pi) * W).astype(int) % W
    py = ((np.pi / 2 - lat) / np.pi * H).astype(int) % H
    return mask[py, px]


# ---------------------------------------------------------------------
# A) Cube cross layout
# ---------------------------------------------------------------------

def render_cross(mask, res=500):
    positions = {0: (1, 0), 5: (0, 1), 2: (1, 1), 4: (2, 1), 3: (3, 1), 1: (1, 2)}
    W = 4 * res
    H = 3 * res
    img = np.ones((H, W, 3), dtype=np.uint8) * 255

    for fid, (col, row) in positions.items():
        xyz = face_rays(fid, res)
        is_land = sample_mask(xyz, mask)
        face = np.zeros((res * res, 3), dtype=np.uint8)
        face[is_land]  = LAND_RGB
        face[~is_land] = SEA_RGB
        face = face.reshape(res, res, 3)
        y0 = row * res
        x0 = col * res
        img[y0:y0+res, x0:x0+res] = face

    # Thin face borders
    for fid, (col, row) in positions.items():
        y0 = row * res
        x0 = col * res
        img[y0, x0:x0+res] = 0
        img[y0+res-1, x0:x0+res] = 0
        img[y0:y0+res, x0] = 0
        img[y0:y0+res, x0+res-1] = 0

    return img


# ---------------------------------------------------------------------
# B) Equirectangular world view with REM distortion field
# ---------------------------------------------------------------------

def equirect_rem_field(res=1024):
    """
    For each pixel of an equirectangular grid, compute:
      - land/sea from the Natural Earth mask
      - the REM log area scale (layout * AEQD)
    """
    H = res
    W = 2 * res
    lat_edges = np.linspace(90, -90, H + 1)
    lon_edges = np.linspace(-180, 180, W + 1)
    lat = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    LON, LAT = np.meshgrid(lon, lat)

    lat_r = np.radians(LAT)
    lon_r = np.radians(LON)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    xyz = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)

    # Pick the cube face whose centre direction has the highest dot product
    # with each sample point, then compute AEQD log scale on that face.
    centres = np.stack([FACE_ROTATIONS[f][:, 2] for f in range(6)], axis=1)  # (3, 6)
    dots = xyz @ centres              # (N, 6)
    best = np.argmax(dots, axis=1)    # (N,)
    # Recompute the scale for the best face
    scale_out = np.empty(xyz.shape[0])
    for f in range(6):
        mask_f = (best == f)
        if not mask_f.any():
            continue
        local = xyz[mask_f] @ FACE_ROTATIONS[f].T
        z_local = np.clip(local[:, 2], -1.0, 1.0)
        rho = np.arccos(z_local)
        safe = rho > 1e-9
        scale_out[mask_f] = np.where(safe,
                                     rho / np.sin(np.where(safe, rho, 1.0)),
                                     1.0)
    log_scale = np.log(scale_out)
    return log_scale.reshape(H, W), LAT, LON


def main():
    mask = np.load("data/ne_land_mask.npy")

    print("Rendering cube cross layout ...")
    cross = render_cross(mask, res=500)
    Image.fromarray(cross).save("output/rem_cross_clean.png")
    print(f"  saved output/rem_cross_clean.png  ({cross.shape[1]}x{cross.shape[0]})")

    print("Rendering equirectangular REM field ...")
    field, LAT, LON = equirect_rem_field(res=1024)
    H, W = field.shape

    # Land/sea mask on the same equirectangular grid
    px = ((LON + 180) / 360 * mask.shape[1]).astype(int) % mask.shape[1]
    py = ((90 - LAT) / 180 * mask.shape[0]).astype(int) % mask.shape[0]
    land = mask[py, px]

    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[land]  = LAND_RGB
    rgb[~land] = SEA_RGB

    # Nuthatch-A heatmap on land only
    land_vals = field[land]
    nut_a_land = float(np.sqrt(((land_vals - land_vals.mean()) ** 2).mean()))
    print(f"  Nuthatch-A (land, equirect view): {nut_a_land:.6f}")
    print(f"  (reference depth-4 value       : 0.000385)")

    vmax = np.percentile(np.abs(field), 99)
    alpha = np.clip(np.abs(field) / max(vmax, 1e-9), 0, 1) * land
    overlay = rgb.astype(float) / 255.0
    tint = np.zeros_like(overlay); tint[..., 0] = 1.0
    blended = np.clip((1 - 0.6 * alpha[..., None]) * overlay
                       + 0.6 * alpha[..., None] * tint, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(22, 6.5))
    axes[0].imshow(rgb, extent=[-180, 180, -90, 90], aspect="auto")
    axes[0].set_title("Natural Earth land/sea mask")
    axes[0].set_xlabel("longitude"); axes[0].set_ylabel("latitude")

    axes[1].imshow(blended, extent=[-180, 180, -90, 90], aspect="auto")
    axes[1].set_title(f"REM Nuthatch-A heatmap  (land Nut-A = {nut_a_land:.6f})")
    axes[1].set_xlabel("longitude"); axes[1].set_ylabel("latitude")

    plt.tight_layout()
    plt.savefig("output/rem_equirect_heatmap.png", dpi=130)
    print("  saved output/rem_equirect_heatmap.png")

    # Combined figure: cross layout on top, equirect heatmap below
    fig = plt.figure(figsize=(20, 16))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(cross)
    ax1.set_title("REM world map — cube-cross layout  (6 faces, no gaps)")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.imshow(blended, extent=[-180, 180, -90, 90], aspect="auto")
    ax2.set_title("REM distortion heatmap — equirectangular view")
    ax2.set_xlabel("longitude"); ax2.set_ylabel("latitude")

    plt.tight_layout()
    plt.savefig("output/rem_final_comparison.png", dpi=130)
    print("  saved output/rem_final_comparison.png")


if __name__ == "__main__":
    main()
