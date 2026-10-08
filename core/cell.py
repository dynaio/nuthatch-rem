import numpy as np


class Cell:
    """
    A quadrilateral cell on the unit sphere, defined by its four
    (u, v) corners in the local face coordinates and a 3x3 rotation
    matrix that maps (u, v, 1) -> world direction (before normalising).
    """

    def __init__(self, cell_id, uv_corners, rot_matrix, parent=None, depth=0):
        self.cell_id    = cell_id
        self.uv_corners = np.asarray(uv_corners, dtype=float)   # (4, 2)
        self.rot        = np.asarray(rot_matrix, dtype=float)   # (3, 3)
        self.parent     = parent
        self.children   = []
        self.depth      = depth

        # Derived geometry
        self.corners_xyz     = None
        self.centroid_xyz    = None
        self.angular_radius  = None
        self.spherical_area  = None

        # Semantic metadata
        self.land_fraction   = None
        self.sea_fraction    = None

        # Projection state
        self.local_frame     = None    # (e1, e2)
        self.projected_xy    = None    # (4, 2)

        # Error state
        self.internal_distortion = None
        self.seam_distortion     = None

        self._compute_geometry()

    # ------------------------------------------------------------------
    def _compute_geometry(self):
        from core.projection import cube_face_to_sphere_equal_area
        uv  = self.uv_corners

        # Corner directions on the sphere
        local = cube_face_to_sphere_equal_area(uv[:, 0], uv[:, 1])
        xyz = local @ self.rot.T
        xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
        self.corners_xyz = xyz

        # Centroid computed in UV space to avoid antipodal degeneracy.
        uv_centre = uv.mean(axis=0)
        c_local = cube_face_to_sphere_equal_area(
            np.array([uv_centre[0]]), np.array([uv_centre[1]])
        )[0]
        c = self.rot @ c_local
        self.centroid_xyz = c / np.linalg.norm(c)

        dots = xyz @ self.centroid_xyz
        self.angular_radius = float(np.arccos(np.clip(dots, -1.0, 1.0)).max())

        self.spherical_area = self._spherical_excess()

    def _spherical_excess(self):
        """Signed spherical excess of the quadrilateral on the unit sphere."""
        c = self.corners_xyz
        n = len(c)
        angles = []
        for i in range(n):
            v1 = c[(i - 1) % n]
            v2 = c[i]
            v3 = c[(i + 1) % n]
            t_in  = v1 - v2 * np.dot(v1, v2)
            t_out = v3 - v2 * np.dot(v3, v2)
            t_in  = t_in  / np.linalg.norm(t_in)
            t_out = t_out / np.linalg.norm(t_out)
            angles.append(np.arccos(np.clip(np.dot(t_in, t_out), -1.0, 1.0)))
        return float(sum(angles) - (n - 2) * np.pi)

    # ------------------------------------------------------------------
    def subdivide(self):
        """Return the four children covering this cell."""
        uv = self.uv_corners
        m01 = 0.5 * (uv[0] + uv[1])
        m12 = 0.5 * (uv[1] + uv[2])
        m23 = 0.5 * (uv[2] + uv[3])
        m30 = 0.5 * (uv[3] + uv[0])
        ctr = uv.mean(axis=0)

        quads = [
            [uv[0], m01, ctr, m30],
            [m01,  uv[1], m12, ctr],
            [ctr,  m12, uv[2], m23],
            [m30,  ctr, m23,  uv[3]],
        ]

        children = []
        for i, quad in enumerate(quads):
            child = Cell(
                cell_id    = f"{self.cell_id}.{i}",
                uv_corners = np.array(quad, dtype=float),
                rot_matrix = self.rot,
                parent     = self,
                depth      = self.depth + 1,
            )
            children.append(child)

        self.children = children
        return children

    # ------------------------------------------------------------------
    def __repr__(self):
        return (f"<Cell {self.cell_id} depth={self.depth} "
                f"rho={self.angular_radius:.4f} area={self.spherical_area:.4e}>")
