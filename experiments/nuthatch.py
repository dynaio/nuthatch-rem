"""
Nuthatch criterion evaluation on Web Mercator, Equal Earth, and Robinson.

Nuthatch-A = weighted std of log(area scale) — how uneven is area distortion
Nuthatch-S = weighted std of log(h/k)      — how uneven is shape distortion

For a projection P: S^2 -> R^2, at each point (lambda, phi):
    h = meridional scale = |dP/dphi| / R
    k = parallel scale   = |dP/dlambda| / (R cos phi)

With R = 1 and (lambda, phi) in radians, both h and k are dimensionless.

The Nuthatch of the projection, restricted to a weighted domain, is the
weighted standard deviation of log(area scale) or log(h/k).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Proj


# ----------------------------------------------------------------------
def make_grid(n_lon=720, n_lat=360):
    """Cell-centre equirectangular grid in lon/lat degrees."""
    lon_edges = np.linspace(-180.0, 180.0, n_lon + 1)
    lat_edges = np.linspace(-90.0, 90.0, n_lat + 1)
    lon = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    return np.meshgrid(lon, lat)


def load_land_mask(lon_g, lat_g, path="data/ne_land_mask.npy"):
    mask = np.load(path)
    H, W = mask.shape
    px = ((lon_g + 180.0) / 360.0 * W).astype(int) % W
    py = ((90.0 - lat_g) / 180.0 * H).astype(int) % H
    return mask[py, px].astype(bool)


# ----------------------------------------------------------------------
def scale_factors(proj, lon_g, lat_g, eps=1e-4):
    """
    Return (log_area, log_shape) at each grid point.

    log_area  = log(h * k)   — area scale factor in log space
    log_shape = log(h / k)   — shape scale ratio in log space

    Both are computed by finite differences in the projection.
    """
    x0, y0 = proj(lon_g, lat_g)
    x_dl, y_dl = proj(lon_g + eps, lat_g)
    x_dp, y_dp = proj(lon_g, lat_g + eps)

    dx_dl = (x_dl - x0) / eps
    dy_dl = (y_dl - y0) / eps
    dx_dp = (x_dp - x0) / eps
    dy_dp = (y_dp - y0) / eps

    mag_lambda = np.hypot(dx_dl, dy_dl)      # |dP/dlambda| per degree
    mag_phi    = np.hypot(dx_dp, dy_dp)      # |dP/dphi| per degree

    cos_lat = np.cos(np.radians(lat_g))

    h = mag_phi                               # per degree of latitude
    k = mag_lambda / np.maximum(cos_lat, 1e-12)

    log_area  = np.log(np.maximum(h * k, 1e-30))
    log_shape = np.log(np.maximum(h / k, 1e-30))
    return log_area, log_shape


def nuthatch(values, weights):
    """Weighted standard deviation."""
    w = weights.astype(float)
    total = w.sum()
    if total <= 0:
        return float("nan")
    mean = (values * w).sum() / total
    var = (((values - mean) ** 2) * w).sum() / total
    return float(np.sqrt(var))


def evaluate(proj_str, lon_g, lat_g, land_mask):
    proj = Proj(proj_str)
    log_area, log_shape = scale_factors(proj, lon_g, lat_g)

    regions = {
        "global": np.ones_like(land_mask, dtype=bool),
        "land":   land_mask,
        "ocean":  ~land_mask,
    }

    out = {}
    for name, mask in regions.items():
        w = mask.astype(float)
        out[name] = {
            "nut_area":  nuthatch(log_area,  w),
            "nut_shape": nuthatch(log_shape, w),
            "mean_log_area":  float((log_area  * w).sum() / max(w.sum(), 1)),
            "mean_log_shape": float((log_shape * w).sum() / max(w.sum(), 1)),
        }
    return out, log_area, log_shape


# ----------------------------------------------------------------------
def main():
    print("Building sampling grid ...")
    lon_g, lat_g = make_grid(n_lon=720, n_lat=360)
    print(f"  {lon_g.size} sample points")

    print("Loading land mask ...")
    land = load_land_mask(lon_g, lat_g)
    print(f"  land fraction on grid: {land.mean():.4f}")

    projections = {
        "Web Mercator": "+proj=merc +R=1",
        "Equal Earth":  "+proj=eqearth +R=1",
        "Robinson":     "+proj=robin +R=1",
    }

    print("\nEvaluating projections ...")
    all_res = {}
    all_fields = {}
    for name, proj_str in projections.items():
        try:
            res, la, ls = evaluate(proj_str, lon_g, lat_g, land)
            all_res[name] = res
            all_fields[name] = (la, ls)
            print(f"  {name:15s}  land_nut_A={res['land']['nut_area']:.4f}  "
                  f"land_nut_S={res['land']['nut_shape']:.4f}")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    # Table
    print()
    print("=" * 88)
    print(f"{'Projection':<15} {'Nut-A land':>12} {'Nut-A ocean':>12} "
          f"{'Nut-S land':>12} {'Nut-S ocean':>12}")
    print("-" * 88)
    for name, r in all_res.items():
        print(f"{name:<15} "
              f"{r['land']['nut_area']:>12.4f} "
              f"{r['ocean']['nut_area']:>12.4f} "
              f"{r['land']['nut_shape']:>12.4f} "
              f"{r['ocean']['nut_shape']:>12.4f}")
    print("=" * 88)

    # Heatmaps
    n = len(all_fields)
    fig, axes = plt.subplots(2, n, figsize=(6.0 * n, 7.5))
    for col, (name, (la, ls)) in enumerate(all_fields.items()):
        im0 = axes[0, col].imshow(la, extent=[-180, 180, -90, 90],
                                   aspect="auto", cmap="RdBu_r",
                                   vmin=-2.0, vmax=2.0)
        axes[0, col].set_title(f"{name}\nlog(area scale)")
        plt.colorbar(im0, ax=axes[0, col], shrink=0.75)

        im1 = axes[1, col].imshow(ls, extent=[-180, 180, -90, 90],
                                   aspect="auto", cmap="RdBu_r",
                                   vmin=-1.0, vmax=1.0)
        axes[1, col].set_title(f"{name}\nlog(shape ratio)")
        plt.colorbar(im1, ax=axes[1, col], shrink=0.75)

    plt.tight_layout()
    plt.savefig("output/nuthatch_maps.png", dpi=120)
    print("\n  saved output/nuthatch_maps.png")

    # Bar chart
    names = list(all_res.keys())
    x = np.arange(len(names))
    w = 0.35

    nut_a_land  = [all_res[n]["land"]["nut_area"]  for n in names]
    nut_a_ocean = [all_res[n]["ocean"]["nut_area"] for n in names]
    nut_s_land  = [all_res[n]["land"]["nut_shape"] for n in names]
    nut_s_ocean = [all_res[n]["ocean"]["nut_shape"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x - w/2, nut_a_land,  w, label="land")
    axes[0].bar(x + w/2, nut_a_ocean, w, label="ocean")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=15)
    axes[0].set_ylabel("Nuthatch-A   (std of log area scale)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Area uniformity")

    axes[1].bar(x - w/2, nut_s_land,  w, label="land")
    axes[1].bar(x + w/2, nut_s_ocean, w, label="ocean")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=15)
    axes[1].set_ylabel("Nuthatch-S   (std of log shape ratio)")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Shape uniformity")

    plt.tight_layout()
    plt.savefig("output/nuthatch_bars.png", dpi=120)
    print("  saved output/nuthatch_bars.png")

    # Save raw numbers
    np.savez("output/nuthatch_results.npz",
             names=np.array(names),
             nut_a_land=np.array(nut_a_land),
             nut_a_ocean=np.array(nut_a_ocean),
             nut_s_land=np.array(nut_s_land),
             nut_s_ocean=np.array(nut_s_ocean))
    print("  saved output/nuthatch_results.npz")


if __name__ == "__main__":
    main()
