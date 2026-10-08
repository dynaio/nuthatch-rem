import numpy as np


def interior_distortion_estimate(cell):
    """
    Leading-order area distortion for AEQD inside a cell of angular radius rho:
        delta_int ~ rho^2 / 6.
    """
    rho = cell.angular_radius
    return rho * rho / 6.0


def total_spherical_area(cells):
    return float(sum(c.spherical_area for c in cells))


def check_gauss_bonnet(cells, expected=4.0 * np.pi):
    total = total_spherical_area(cells)
    return total, abs(total - expected) / expected


def tissot_area_scale(rho):
    """Area scale factor h*k for AEQD at angular radius rho from centre."""
    if rho < 1e-12:
        return 1.0
    return rho / np.sin(rho)
