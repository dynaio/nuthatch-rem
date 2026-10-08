"""
Full Nuthatch comparison: Web Mercator, Equal Earth, Robinson,
and REM at depths 2-6. Reports both Nut-A and Nut-S on land and ocean.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Proj

from core.tessellation import TessellationEngine
from core.semantic import (
    compute_land_fraction, sample_cell_xyz, xyz_to_uv, sample_mask
)
from core.projection import cube_face_to_sphere_equal_area
from experiments.nuthatch import make_grid, load_land_mask, scale_factors, nuthatch
from experiments.rem_nuthatch import aeqd_local_scale, wstd


# ---------------------------------------------------------------------
# Classical projections
# ---------------------------------------------------------------------

def evaluate_classical(proj_str, lon_g, lat_g, land):
    proj = Proj(proj_str)
    log_area, log_shape = scale_factors(proj, lon_g, lat_g)

    out = {}
    for name, mask in [("land", land), ("ocean", ~land), ("all", np.ones_like(land))]:
        w = mask.astype(float)
        out[name] = {
            "nut_a": nuthatch(log_area,  w),
            "nut_s": nuthatch(log_shape, w),
        }
    return out


# ---------------------------------------------------------------------
# REM
# ---------------------------------------------------------------------

def evaluate_rem(depth, mask, n_side=8):
    leaves = TessellationEngine(max_depth=depth).build()
    for c in leaves:
        compute_land_fraction(c, mask, n_side=8)

    log_area, log_shape = [], []
    land_w, sea_w, all_w = [], [], []

    for c in leaves:
        uv = c.uv_corners
        uv_area = (uv[2, 0] - uv[0, 0]) * (uv[2, 1] - uv[0, 1])
        layout_scale = uv_area / c.spherical_area

        xyz = sample_cell_xyz(c, n_side=n_side)
        u, v = xyz_to_uv(xyz)
        land = sample_mask(mask, u, v)

        aeqd = aeqd_local_scale(xyz, c.centroid_xyz)
        # AEQD is radially stretched; approximate shape ratio h/k = 1/aeqd
        # (radial scale = 1, tangential scale = 1/aeqd along the radius).
        # This is the leading behaviour for small rho.
        log_s_a = np.log(layout_scale * aeqd)          # area scale
        log_s_s = np.log(1.0 / np.maximum(aeqd, 1e-12))  # shape ratio

        sw = c.spherical_area / (n_side ** 2)
        log_area.append(log_s_a)
        log_shape.append(log_s_s)
        land_w.append(sw * land.astype(float))
        sea_w.append(sw * (1.0 - land.astype(float)))
        all_w.append(np.full(xyz.shape[0], sw))

    log_area  = np.concatenate(log_area)
    log_shape = np.concatenate(log_shape)
    land_w = np.concatenate(land_w)
    sea_w  = np.concatenate(sea_w)
    all_w  = np.concatenate(all_w)

    return {
        "land":  {"nut_a": wstd(log_area, land_w), "nut_s": wstd(log_shape, land_w)},
        "ocean": {"nut_a": wstd(log_area, sea_w),  "nut_s": wstd(log_shape, sea_w)},
        "all":   {"nut_a": wstd(log_area, all_w),  "nut_s": wstd(log_shape, all_w)},
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("Building grid ...")
    lon_g, lat_g = make_grid(n_lon=720, n_lat=360)
    land = load_land_mask(lon_g, lat_g)
    print(f"  land fraction on grid: {land.mean():.4f}")

    print("\nClassical projections ...")
    rows = {}
    for name, proj in [("Web Mercator", "+proj=merc +R=1"),
                       ("Equal Earth",  "+proj=eqearth +R=1"),
                       ("Robinson",     "+proj=robin +R=1")]:
        rows[name] = evaluate_classical(proj, lon_g, lat_g, land)
        print(f"  {name:20s}  Nut-A_land={rows[name]['land']['nut_a']:.4f}")

    print("\nREM (QSC tessellation) ...")
    mask = np.load("data/ne_land_mask.npy")
    for depth in [2, 3, 4, 5, 6]:
        key = f"REM depth {depth}"
        rows[key] = evaluate_rem(depth, mask)
        print(f"  {key:20s}  Nut-A_land={rows[key]['land']['nut_a']:.6f}  "
              f"Nut-S_land={rows[key]['land']['nut_s']:.6f}")

    # ---- Table ----
    print()
    print("=" * 92)
    print(f"{'Projection':<20} {'Nut-A land':>13} {'Nut-A ocean':>13} "
          f"{'Nut-S land':>13} {'Nut-S ocean':>13}")
    print("-" * 92)
    for name, r in rows.items():
        print(f"{name:<20} "
              f"{r['land']['nut_a']:>13.6f} "
              f"{r['ocean']['nut_a']:>13.6f} "
              f"{r['land']['nut_s']:>13.6f} "
              f"{r['ocean']['nut_s']:>13.6f}")
    print("=" * 92)

    # ---- Save ----
    np.savez("output/nuthatch_full.npz",
             names=np.array(list(rows.keys())),
             nut_a_land =np.array([r['land']['nut_a']  for r in rows.values()]),
             nut_a_ocean=np.array([r['ocean']['nut_a'] for r in rows.values()]),
             nut_s_land =np.array([r['land']['nut_s']  for r in rows.values()]),
             nut_s_ocean=np.array([r['ocean']['nut_s'] for r in rows.values()]))
    print("  saved output/nuthatch_full.npz")

    # ---- Figure: two panels, log scale ----
    names = list(rows.keys())
    x = np.arange(len(names))
    w = 0.38

    nut_a_land  = [r['land']['nut_a']  for r in rows.values()]
    nut_a_ocean = [r['ocean']['nut_a'] for r in rows.values()]
    nut_s_land  = [r['land']['nut_s']  for r in rows.values()]
    nut_s_ocean = [r['ocean']['nut_s'] for r in rows.values()]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].bar(x - w/2, nut_a_land,  w, label="land")
    axes[0].bar(x + w/2, nut_a_ocean, w, label="ocean")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right")
    axes[0].set_ylabel("Nuthatch-A (log scale)")
    axes[0].set_title("Area uniformity — lower is better")
    axes[0].legend(); axes[0].grid(True, which="both", alpha=0.3)

    axes[1].bar(x - w/2, nut_s_land,  w, label="land")
    axes[1].bar(x + w/2, nut_s_ocean, w, label="ocean")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right")
    axes[1].set_ylabel("Nuthatch-S (log scale)")
    axes[1].set_title("Shape uniformity — lower is better")
    axes[1].legend(); axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/nuthatch_full_comparison.png", dpi=130)
    print("  saved output/nuthatch_full_comparison.png")


if __name__ == "__main__":
    main()
