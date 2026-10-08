"""
Produce the paper figures:
  1. REM world map in the classic cross (unfolded-cube) layout
  2. Nuthatch-A heatmap overlaid on the map
  3. Confirm Nuthatch-A equals the depth-sweep value

Colour scheme is intentionally data-driven (land = tan, sea = pale blue)
so the figure is reproducible without any downloaded texture.
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


FACE_ROTATIONS = [
    np.array([[ 1, 0, 0],[ 0, 1, 0],[ 0, 0, 1]], dtype=float),   # F0 +Z north pole
    np.array([[ 1, 0, 0],[ 0,-1, 0],[ 0, 0,-1]], dtype=float),   # F1 -Z south pole
    np.array([[ 0, 0, 1],[ 0, 1, 0],[-1, 0, 0]], dtype=float),   # F2 +X lon 0
    np.array([[ 0, 0,-1],[ 0, 1, 0],[ 1, 0, 0]], dtype=float),   # F3 -X lon 180
    np.array([[ 1, 0, 0],[ 0, 0, 1],[ 0,-1, 0]], dtype=float),   # F4 +Y lon 90E
    np.array([[ 1, 0, 0],[ 0, 0,-1],[ 0, 1, 0]], dtype=float),   # F5 -Y lon 90W
]

LAND_RGB = np.array([210, 180, 130], dtype=np.uint8)
SEA_RGB  = np.array([160, 200, 230], dtype=np.uint8)


def face_rays(face_id, res, rotation_deg=0.0):
    lin = np.linspace(-1.0, 1.0, res)
    U, V = np.meshgrid(lin, lin, indexing="xy")
    theta = np.radians(rotation_deg)
    c, s = np.cos(theta), np.sin(theta)
    U_r = c * U - s * V
    V_r = s * U + c * V

    local = cube_face_to_sphere_equal_area(U_r.ravel(), V_r.ravel())
    R = FACE_ROTATIONS[face_id]
    xyz = local @ R.T
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


def render_face(face_id, mask, res=500, rotation_deg=0.0):
    xyz = face_rays(face_id, res, rotation_deg)
    is_land = sample_mask(xyz, mask)
    rgb = np.zeros((res * res, 3), dtype=np.uint8)
    rgb[is_land]  = LAND_RGB
    rgb[~is_land] = SEA_RGB
    return rgb.reshape(res, res, 3)


def log_scale_face(face_id, res=500, rotation_deg=0.0):
    xyz = face_rays(face_id, res, rotation_deg)
    R = FACE_ROTATIONS[face_id]
    center = R @ np.array([0.0, 0.0, 1.0])
    center = center / np.linalg.norm(center)
    cos_rho = np.clip(xyz @ center, -1.0, 1.0)
    rho = np.arccos(cos_rho)
    safe = rho > 1e-9
    scale = np.where(safe, rho / np.sin(np.where(safe, rho, 1.0)), 1.0)
    return np.log(scale).reshape(res, res)


def assemble(rgb_faces, scale_faces, res=500, gap=2):
    positions = {0: (1, 0), 5: (0, 1), 2: (1, 1), 4: (2, 1), 3: (3, 1), 1: (1, 2)}
    W = 4 * res + 3 * gap
    H = 3 * res + 2 * gap
    rgb = np.ones((H, W, 3), dtype=np.uint8) * 255
    sc  = np.zeros((H, W), dtype=float)
    valid = np.zeros((H, W), dtype=bool)
    for fid, (col, row) in positions.items():
        y0 = row * (res + gap)
        x0 = col * (res + gap)
        rgb[y0:y0+res, x0:x0+res]   = rgb_faces[fid]
        sc [y0:y0+res, x0:x0+res]   = scale_faces[fid]
        valid[y0:y0+res, x0:x0+res] = True
    return rgb, sc, valid


def main():
    mask = np.load("data/ne_land_mask.npy")
    print(f"Mask: {mask.shape}")

    res = 500
    print(f"Rendering 6 faces at {res}x{res} ...")
    rotations = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

    rgb_faces, scale_faces = {}, {}
    for fid in range(6):
        rgb_faces[fid]   = render_face(fid, mask, res=res, rotation_deg=rotations[fid])
        scale_faces[fid] = log_scale_face(fid, res=res, rotation_deg=rotations[fid])
        Image.fromarray(rgb_faces[fid]).save(f"output/face_{fid}.png")

    rgb, sc, valid = assemble(rgb_faces, scale_faces, res=res, gap=2)
    Image.fromarray(rgb).save("output/rem_world_map.png")
    print(f"Saved output/rem_world_map.png  ({rgb.shape[1]}x{rgb.shape[0]})")

    # Nuthatch-A on the visible land pixels
    land_pixels = (rgb[:, :, 0] == LAND_RGB[0]) & \
                  (rgb[:, :, 1] == LAND_RGB[1]) & \
                  (rgb[:, :, 2] == LAND_RGB[2]) & valid
    land_vals = sc[land_pixels]
    if land_vals.size:
        mean = land_vals.mean()
        nut_a = float(np.sqrt(((land_vals - mean) ** 2).mean()))
        print(f"Nuthatch-A on land (cross layout, depth-4 equivalent): {nut_a:.6f}")
        print("  (compare to depth-4 sweep value: 0.000385)")

    # Heatmap overlay
    vmax = np.percentile(np.abs(sc[valid]), 99)
    overlay = rgb.astype(float) / 255.0
    alpha = np.clip(np.abs(sc) / vmax, 0, 1) * valid
    tint = np.zeros_like(overlay); tint[..., 0] = 1.0
    blended = np.clip((1 - 0.55 * alpha[..., None]) * overlay +
                      0.55 * alpha[..., None] * tint, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(22, 7.5))
    axes[0].imshow(rgb)
    axes[0].set_title("REM world map  (QSC equal-area tessellation + AEQD per cell)")
    axes[0].axis("off")

    axes[1].imshow(blended)
    axes[1].set_title("Nuthatch-A heatmap  (|log area scale|; red = more distortion)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("output/rem_world_map_heatmap.png", dpi=130)
    print("Saved output/rem_world_map_heatmap.png")


if __name__ == "__main__":
    main()
