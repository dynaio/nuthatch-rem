"""
Semantic correction: warp the coastline inside each coastal cell so that
the AEQD-weighted land fraction of the projected cell equals the
spherical land fraction.

The AEQD projection stretches area away from the cell centre by a factor
J(rho) = rho / sin(rho).  We therefore weight each sample by J to
reproduce what a viewer of the projected map would measure.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

from config import MAX_DEPTH
from core.tessellation import TessellationEngine
from core.semantic import sample_cell_xyz, xyz_to_uv, sample_mask


def signed_distance(land_bool):
    """Positive inland, negative in sea."""
    d_in  = distance_transform_edt(land_bool)
    d_out = distance_transform_edt(~land_bool)
    return (d_in - d_out).astype(float)


def correct_weighted(land_bool, weights, target_frac):
    """
    Subpixel level-set correction.  Each pixel near the coastline receives
    a fractional land value, so the weighted land fraction varies
    continuously with the shift t.  Returns a soft mask in [0, 1].
    """
    w = weights.flatten().astype(float)
    w_total = w.sum()
    sd = signed_distance(land_bool).flatten()   # >0 inland, <0 at sea

    def soft_mask(t):
        # 1 well inland, 0 well at sea, linear ramp of width 1 pixel
        return np.clip(sd - t + 0.5, 0.0, 1.0)

    def frac(t):
        return float((w * soft_mask(t)).sum() / w_total)

    current = frac(0.0)
    if abs(current - target_frac) < 1e-12:
        return soft_mask(0.0).reshape(land_bool.shape), 0.0

    lo, hi = sd.min() - 1.0, sd.max() + 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if frac(mid) > target_frac:
            lo = mid   # too much land -> raise threshold
        else:
            hi = mid   # too little land -> lower threshold
    t_star = 0.5 * (lo + hi)
    return soft_mask(t_star).reshape(land_bool.shape), t_star


def analyse_cell(cell, mask, n_side=32):
    xyz = sample_cell_xyz(cell, n_side=n_side)
    u, v = xyz_to_uv(xyz)
    land = sample_mask(mask, u, v).reshape(n_side, n_side)

    cos_rho = np.clip(xyz @ cell.centroid_xyz, -1.0, 1.0)
    rho = np.arccos(cos_rho)
    J = np.where(rho < 1e-9, 1.0, rho / np.sin(rho))
    J = J.reshape(n_side, n_side)

    f_sphere = float(land.mean())
    f_plane  = float((land * J).sum() / J.sum())
    return land, J, f_sphere, f_plane


def main():
    print(f"Building tessellation at depth {MAX_DEPTH}")
    leaves = TessellationEngine(max_depth=MAX_DEPTH).build()

    print("Loading mask ...")
    mask = np.load("data/ne_land_mask.npy")

    print("Scanning cells for coastlines ...")
    coastal = []
    for c in leaves:
        land, J, f_s, f_p = analyse_cell(c, mask, n_side=32)
        if 0.05 < f_s < 0.95:
            coastal.append((c, land, J, f_s, f_p))
    print(f"  {len(coastal)} coastal cells / {len(leaves)} total")

    before, after, shifts = [], [], []
    for _, land, J, f_s, f_p in coastal:
        before.append(abs(f_p - f_s))
        corrected, t = correct_weighted(land, J, f_s)
        f_p_new = float((corrected * J).sum() / J.sum())
        after.append(abs(f_p_new - f_s))
        shifts.append(t)

    before = np.array(before); after = np.array(after); shifts = np.array(shifts)

    print()
    print("=== Correction results ===")
    print(f"  |f_plane - f_sphere|  before : mean={before.mean():.3e}  max={before.max():.3e}")
    print(f"  |f_corrected - f_sph| after  : mean={after.mean():.3e}  max={after.max():.3e}")
    print(f"  |t*| (pixels)                : mean={np.abs(shifts).mean():.2f}  max={np.abs(shifts).max():.2f}")

    n_show = min(8, len(coastal))
    idx = np.linspace(0, len(coastal) - 1, n_show).astype(int)
    fig, axes = plt.subplots(2, n_show, figsize=(2.2 * n_show, 5.2))
    for col, i in enumerate(idx):
        _, land, J, f_s, f_p = coastal[i]
        corrected, t = correct_weighted(land, J, f_s)
        f_p_new = float((corrected * J).sum() / J.sum())

        axes[0, col].imshow(land, cmap="Blues_r", vmin=0, vmax=1)
        axes[0, col].set_title(f"orig\nf_pl={f_p:.4f}\nf_sph={f_s:.4f}", fontsize=7)
        axes[0, col].axis("off")

        axes[1, col].imshow(corrected, cmap="Blues_r", vmin=0, vmax=1)
        axes[1, col].set_title(f"corrected\nf={f_p_new:.4f}\nt={t:.2f}", fontsize=7)
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig("output/correction_demo.png", dpi=120)
    print("  saved output/correction_demo.png")


if __name__ == "__main__":
    main()
