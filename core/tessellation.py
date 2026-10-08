import numpy as np
from core.cell import Cell


def _face_rotations():
    """
    Six rotation matrices, one per cube face. Each maps
    (u, v, 1) -> world direction (before normalising).
    """
    return [
        # +Z
        np.array([[ 1, 0, 0],
                  [ 0, 1, 0],
                  [ 0, 0, 1]], dtype=float),
        # -Z
        np.array([[ 1, 0, 0],
                  [ 0,-1, 0],
                  [ 0, 0,-1]], dtype=float),
        # +X
        np.array([[ 0, 0, 1],
                  [ 0, 1, 0],
                  [-1, 0, 0]], dtype=float),
        # -X
        np.array([[ 0, 0,-1],
                  [ 0, 1, 0],
                  [ 1, 0, 0]], dtype=float),
        # +Y
        np.array([[ 1, 0, 0],
                  [ 0, 0, 1],
                  [ 0,-1, 0]], dtype=float),
        # -Y
        np.array([[ 1, 0, 0],
                  [ 0, 0,-1],
                  [ 0, 1, 0]], dtype=float),
    ]


class TessellationEngine:
    def __init__(self, max_depth=4):
        self.max_depth  = max_depth
        self.root_cells = []
        self.leaf_cells = []

    def build(self):
        self.root_cells = []
        self.leaf_cells = []

        base_uv = np.array([[-1.0, -1.0],
                            [ 1.0, -1.0],
                            [ 1.0,  1.0],
                            [-1.0,  1.0]], dtype=float)

        for i, R in enumerate(_face_rotations()):
            root = Cell(cell_id=f"F{i}", uv_corners=base_uv,
                        rot_matrix=R, depth=0)
            self.root_cells.append(root)
            self._subdivide(root)

        return self.leaf_cells

    def _subdivide(self, cell):
        if cell.depth >= self.max_depth:
            self.leaf_cells.append(cell)
            return
        for child in cell.subdivide():
            self._subdivide(child)
