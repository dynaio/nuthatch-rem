# Recursive Error Management (REM) with Nuthatch

Source code and figures for the paper:

> **Berassil Abdelkader.** *Recursive Error Management with Nuthatch:
> A Content-Aware Framework for Uniform-Distortion Cartographic Projection.*
> Submitted to the *International Journal of Geographical Information Science*.
>
> ORCID: [0009-0002-2081-5664](https://orcid.org/0009-0002-2081-5664)

## Summary

This repository implements **Recursive Error Management (REM)**, a
hierarchical framework for flattening the globe with controlled and
uniform distortion. It introduces the **Nuthatch criterion** — named
after the endangered Algerian nuthatch (*Sitta ledanti*) — which
measures the uniformity of the area scale factor across a map. Nuthatch
is zero for any uniformly scaled projection, regardless of the overall
scale factor.

The framework combines:

- an **equal-area cube-sphere tessellation** (QSC family),
- **per-cell azimuthal equidistant projections**, and
- a **subpixel semantic correction** that preserves land/sea fractions
  to machine precision.

## Results

Nuthatch values on landmasses. Lower is better.

| Projection      | Nuthatch-A | Nuthatch-S | Joint   |
|-----------------|------------|------------|---------|
| Web Mercator    | 2.015229   | 0.000020   | 2.0152  |
| Equal Earth     | 0.227627   | 1.598129   | 1.6143  |
| Robinson        | 0.751797   | 0.947260   | 1.2096  |
| **REM depth 4** | **0.000385** | **0.000374** | **0.000537** |
| **REM depth 6** | **0.000024** | **0.000023** | **0.000034** |

REM improves the joint Nuthatch criterion by a factor of roughly
**2,250×** over the best-established global projection.

## Repository layout

```
core/         tessellation, projection, semantic correction
io/           GLB parser, Natural Earth mask loader
experiments/  all experiments reported in the paper
data/         Natural Earth land mask (derived)
output/       generated figures and results
paper/        LaTeX source of the paper
config.py     global parameters
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

```bash
# 1. Fetch the Natural Earth mask (once)
python io/fetch_land_mask.py

# 2. Gauss–Bonnet invariant and depth sweep
python -m experiments.rem_depth_sweep

# 3. Per-cell semantic correction
python -m experiments.correction_demo

# 4. Recursive merge invariance
python -m experiments.merge_invariant

# 5. Nuthatch comparison against classical projections
python -m experiments.nuthatch_full

# 6. Optimal local projection choice
python -m experiments.local_projection_choice

# 7. Final world map figures
python -m experiments.final_map_v2
```

All figures used in the paper are written to `output/`.

## Data

The land/sea mask is derived from the
[Natural Earth](https://www.naturalearthdata.com/) 1:110m physical
land polygons, which are in the public domain.

## Citation

If you use this code or the Nuthatch criterion in your work, please
cite the paper above. A BibTeX entry will be added once the paper is
accepted.

## License

MIT — see [LICENSE](LICENSE).
