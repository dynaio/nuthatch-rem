"""
Confirm that AEQD is the optimal local projection for joint Nuthatch.
Compare three: AEQD, Lambert azimuthal equal-area (LAEA), Stereographic.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from core.tessellation import TessellationEngine
from core.semantic import compute_land_fraction, sample_cell_xyz, xyz_to_uv, sample_mask


def local_scale_aeqd(rho):
    safe = rho > 1e-9
    k = np.where(safe, rho / np.sin(np.where(safe, rho, 1.0)), 1.0)
    return 1.0, k                 # h=1, k=rho/sin(rho)

def local_scale_laea(rho):
    # LAEA: h = sqrt(2/(1+cos rho)) = 1/k,  k = sqrt(2/(1+cos rho))
    safe = rho > 1e-9
    c = np.cos(np.where(safe, rho, 0.0))
    k = np.sqrt(2.0 / (1.0 + c))
    return 1.0 / k, k             # h=1/k, k=sqrt(2/(1+cos))

def local_scale_stereo(rho):
    # Stereographic: h = k = 2/(1+cos rho)
    safe = rho > 1e-9
    c = np.cos(np.where(safe, rho, 0.0))
    k = 2.0 / (1.0 + c)
    return k, k


def wstd(x, w):
    tot = w.sum()
    if tot <= 0: return float("nan")
    m = (x * w).sum() / tot
    return float(np.sqrt((((x - m) ** 2) * w).sum() / tot))


def evaluate(depth, mask, model, n_side=8):
    leaves = TessellationEngine(max_depth=depth).build()
    for c in leaves:
        compute_land_fraction(c, mask, n_side=8)

    log_area, log_shape = [], []
    w_land, w_all = [], []

    for c in leaves:
        uv = c.uv_corners
        uv_area = (uv[2, 0] - uv[0, 0]) * (uv[2, 1] - uv[0, 1])
        layout = uv_area / c.spherical_area

        xyz = sample_cell_xyz(c, n_side=n_side)
        u, v = xyz_to_uv(xyz)
        land = sample_mask(mask, u, v)

        cos_rho = np.clip(xyz @ c.centroid_xyz, -1.0, 1.0)
        rho = np.arccos(cos_rho)

        h, k = model(rho)
        log_area.append(np.log(layout * h * k))
        log_shape.append(np.log(np.maximum(h, 1e-12) / np.maximum(k, 1e-12)))

        sw = c.spherical_area / (n_side ** 2)
        w_land.append(sw * land.astype(float))
        w_all.append(np.full(xyz.shape[0], sw))

    la = np.concatenate(log_area); ls = np.concatenate(log_shape)
    wl = np.concatenate(w_land);  wa = np.concatenate(w_all)

    nut_a = wstd(la, wl)
    nut_s = wstd(ls, wl)
    joint = float(np.sqrt(nut_a**2 + nut_s**2))
    return nut_a, nut_s, joint


def main():
    mask = np.load("data/ne_land_mask.npy")
    print(f"{'model':<15} {'Nut-A':>12} {'Nut-S':>12} {'joint':>12}")
    print("-" * 53)
    for name, model in [("AEQD", local_scale_aeqd),
                        ("LAEA", local_scale_laea),
                        ("Stereographic", local_scale_stereo)]:
        a, s, j = evaluate(4, mask, model)
        print(f"{name:<15} {a:>12.6f} {s:>12.6f} {j:>12.6f}")


if __name__ == "__main__":
    main()
