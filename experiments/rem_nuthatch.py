"""
Compute Nuthatch of the REM map (raw, no correction).

Total local scale factor = layout_scale * aeqd_local_scale
  layout_scale = cell UV rect area / spherical area
  aeqd_local_scale = rho / sin(rho) at angular distance rho from centroid
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from config import MAX_DEPTH
from core.tessellation import TessellationEngine
from core.semantic import (
    compute_land_fraction, sample_cell_xyz, xyz_to_uv, sample_mask
)


def aeqd_local_scale(xyz, centroid):
    cos_rho = np.clip(xyz @ centroid, -1.0, 1.0)
    rho = np.arccos(cos_rho)
    safe = rho > 1e-9
    return np.where(safe, rho / np.sin(np.where(safe, rho, 1.0)), 1.0)


def wstd(x, w):
    total = w.sum()
    if total <= 0:
        return float("nan")
    mean = (x * w).sum() / total
    var = (((x - mean) ** 2) * w).sum() / total
    return float(np.sqrt(var))


def main():
    print(f"Building tessellation at depth {MAX_DEPTH}")
    leaves = TessellationEngine(max_depth=MAX_DEPTH).build()
    mask = np.load("data/ne_land_mask.npy")

    print("Computing per-cell land fraction ...")
    for c in leaves:
        compute_land_fraction(c, mask, n_side=8)

    n_side = 8
    log_scales, land_w, sea_w = [], [], []

    for c in leaves:
        uv = c.uv_corners
        uv_area = (uv[2, 0] - uv[0, 0]) * (uv[2, 1] - uv[0, 1])
        layout_scale = uv_area / c.spherical_area

        xyz = sample_cell_xyz(c, n_side=n_side)
        u, v = xyz_to_uv(xyz)
        land = sample_mask(mask, u, v)

        aeqd = aeqd_local_scale(xyz, c.centroid_xyz)
        log_s = np.log(layout_scale * aeqd)

        sw = c.spherical_area / (n_side ** 2)
        log_scales.append(log_s)
        land_w.append(sw * land.astype(float))
        sea_w.append(sw * (1.0 - land.astype(float)))

    log_scales = np.concatenate(log_scales)
    land_w = np.concatenate(land_w)
    sea_w = np.concatenate(sea_w)

    nut_land = wstd(log_scales, land_w)
    nut_sea  = wstd(log_scales, sea_w)
    nut_all  = wstd(log_scales, land_w + sea_w)

    print()
    print("=" * 55)
    print(f"REM depth {MAX_DEPTH}, NO correction:")
    print(f"  Nuthatch-A (land)  : {nut_land:.4f}")
    print(f"  Nuthatch-A (ocean) : {nut_sea:.4f}")
    print(f"  Nuthatch-A (all)   : {nut_all:.4f}")
    print("-" * 55)
    print("Reference values (from prior experiment):")
    print(f"  Web Mercator land  : 2.0152")
    print(f"  Equal Earth land   : 0.2276")
    print(f"  Robinson land      : 0.7518")
    print("=" * 55)


if __name__ == "__main__":
    main()
