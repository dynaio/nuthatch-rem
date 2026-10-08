"""
Sweep REM Nuthatch-A across depths to separate the two error channels.
  - AEQD distortion scales as rho^2  -> should shrink with depth
  - Layout distortion is scale-invariant -> should stay constant
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.tessellation import TessellationEngine
from core.semantic import (
    compute_land_fraction, sample_cell_xyz, xyz_to_uv, sample_mask
)
from experiments.rem_nuthatch import aeqd_local_scale, wstd


def run_depth(depth, mask, n_side=8):
    leaves = TessellationEngine(max_depth=depth).build()
    for c in leaves:
        compute_land_fraction(c, mask, n_side=8)

    log_layout, log_aeqd, log_total = [], [], []
    land_w, sea_w, all_w = [], [], []

    for c in leaves:
        uv = c.uv_corners
        uv_area = (uv[2, 0] - uv[0, 0]) * (uv[2, 1] - uv[0, 1])
        layout_scale = uv_area / c.spherical_area

        xyz = sample_cell_xyz(c, n_side=n_side)
        u, v = xyz_to_uv(xyz)
        land = sample_mask(mask, u, v)
        aeqd = aeqd_local_scale(xyz, c.centroid_xyz)

        sw = c.spherical_area / (n_side ** 2)
        log_layout.append(np.full(xyz.shape[0], np.log(layout_scale)))
        log_aeqd.append(np.log(aeqd))
        log_total.append(np.log(layout_scale * aeqd))
        land_w.append(sw * land.astype(float))
        sea_w.append(sw * (1.0 - land.astype(float)))
        all_w.append(np.full(xyz.shape[0], sw))

    log_layout = np.concatenate(log_layout)
    log_aeqd = np.concatenate(log_aeqd)
    log_total = np.concatenate(log_total)
    land_w = np.concatenate(land_w)
    sea_w = np.concatenate(sea_w)
    all_w = np.concatenate(all_w)

    return {
        "n_cells": len(leaves),
        "layout_land": wstd(log_layout, land_w),
        "aeqd_land":   wstd(log_aeqd, land_w),
        "total_land":  wstd(log_total, land_w),
        "layout_all":  wstd(log_layout, all_w),
        "aeqd_all":    wstd(log_aeqd, all_w),
        "total_all":   wstd(log_total, all_w),
    }


def main():
    mask = np.load("data/ne_land_mask.npy")
    print(f"{'depth':>5} {'cells':>7} {'AEQD_lnd':>10} {'Layout_lnd':>12} "
          f"{'Total_lnd':>11} {'Total_all':>11}")
    print("-" * 62)
    for d in [2, 3, 4, 5, 6]:
        r = run_depth(d, mask)
        print(f"{d:>5} {r['n_cells']:>7} {r['aeqd_land']:>10.5f} "
              f"{r['layout_land']:>12.5f} {r['total_land']:>11.5f} "
              f"{r['total_all']:>11.5f}")


if __name__ == "__main__":
    main()
