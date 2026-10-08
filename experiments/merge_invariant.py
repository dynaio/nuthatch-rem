"""
Verify that per-cell semantic correction preserves the global
area-weighted land fraction through every level of the quadtree.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.tessellation import TessellationEngine
from core.semantic import sample_cell_xyz, xyz_to_uv, sample_mask
from experiments.correction_demo import (
    analyse_cell, correct_weighted
)


def global_fraction(cells, use_corrected=False):
    """Area-weighted global land fraction across cells."""
    A = np.array([c.spherical_area for c in cells])
    if use_corrected:
        f = np.array([c.corrected_fraction for c in cells])
    else:
        f = np.array([c.land_fraction for c in cells])
    return float((A * f).sum() / A.sum())


def compute_per_cell(cells, mask, n_side=32, correct=True):
    """Attach spherical and (optionally corrected) projected fractions."""
    for c in cells:
        land, J, f_s, f_p = analyse_cell(c, mask, n_side=n_side)
        c.land_fraction = f_s
        c.projected_fraction = f_p
        if correct:
            corrected, t = correct_weighted(land, J, f_s)
            f_p_new = float((corrected * J).sum() / J.sum())
            c.corrected_fraction = f_p_new
            c.correction_shift = t
        else:
            c.corrected_fraction = f_p
            c.correction_shift = 0.0


def aggregate_to_parents(cells):
    """Group leaves by parent id (strip the last '.N' from cell_id)."""
    groups = {}
    for c in cells:
        parent_id = c.cell_id.rsplit(".", 1)[0]
        groups.setdefault(parent_id, []).append(c)
    return groups


def main():
    print("Loading mask ...")
    mask = np.load("data/ne_land_mask.npy")

    for depth in [2, 3, 4, 5]:
        print(f"\n=== Depth {depth} ===")
        engine = TessellationEngine(max_depth=depth)
        leaves = engine.build()
        print(f"  leaves: {len(leaves)}")

        compute_per_cell(leaves, mask, n_side=32, correct=False)
        f_before = global_fraction(leaves, use_corrected=False)

        compute_per_cell(leaves, mask, n_side=32, correct=True)
        f_after_leaf = global_fraction(leaves, use_corrected=True)

        print(f"  f_global (before correction)  : {f_before:.10f}")
        print(f"  f_global (after correction)   : {f_after_leaf:.10f}")
        print(f"  |drift| at leaf level         : {abs(f_after_leaf - f_before):.3e}")

        # Merge at parent level, verify invariant
        parents = aggregate_to_parents(leaves)
        parent_A = []
        parent_f = []
        for pid, children in parents.items():
            A = sum(c.spherical_area for c in children)
            f = sum(c.spherical_area * c.corrected_fraction
                    for c in children) / A
            parent_A.append(A)
            parent_f.append(f)

        parent_A = np.array(parent_A)
        parent_f = np.array(parent_f)
        f_parent = float((parent_A * parent_f).sum() / parent_A.sum())
        print(f"  f_global (after parent merge) : {f_parent:.10f}")
        print(f"  |drift| at parent merge       : {abs(f_parent - f_after_leaf):.3e}")

        shifts = np.array([c.correction_shift for c in leaves])
        print(f"  correction shift (px)         : "
              f"mean={shifts.mean():.4f}  std={shifts.std():.4f}")


if __name__ == "__main__":
    main()
