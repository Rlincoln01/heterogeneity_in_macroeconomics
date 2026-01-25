"""Interpolation and lottery methods."""

from typing import Tuple

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


def interp_linear(x_grid: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Vectorized 1D linear interpolation.

    Parameters
    ----------
    x_grid : ndarray, shape (n,)
        Grid points (must be sorted).
    y : ndarray, shape (n,) or (m, n)
        Values on grid.
    x_new : ndarray
        Points to interpolate.
    """
    x_grid = np.asarray(x_grid)
    y = np.asarray(y)
    x_new = np.asarray(x_new)

    if y.ndim == 1:
        return np.interp(x_new, x_grid, y)

    out = np.empty((y.shape[0],) + x_new.shape)
    for i in range(y.shape[0]):
        out[i] = np.interp(x_new, x_grid, y[i])
    return out


def interp_bilinear(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation on a rectilinear grid."""
    x_grid = np.asarray(x_grid)
    y_grid = np.asarray(y_grid)
    z = np.asarray(z)
    x_new = np.asarray(x_new)
    y_new = np.asarray(y_new)

    ix = np.searchsorted(x_grid, x_new, side="right") - 1
    iy = np.searchsorted(y_grid, y_new, side="right") - 1
    ix = np.clip(ix, 0, len(x_grid) - 2)
    iy = np.clip(iy, 0, len(y_grid) - 2)

    x0 = x_grid[ix]
    x1 = x_grid[ix + 1]
    y0 = y_grid[iy]
    y1 = y_grid[iy + 1]

    wx = (x_new - x0) / (x1 - x0)
    wy = (y_new - y0) / (y1 - y0)

    z00 = z[iy, ix]
    z10 = z[iy, ix + 1]
    z01 = z[iy + 1, ix]
    z11 = z[iy + 1, ix + 1]

    return (1 - wx) * (1 - wy) * z00 + wx * (1 - wy) * z10 + (1 - wx) * wy * z01 + wx * wy * z11


def get_lottery(a_policy: np.ndarray, a_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices and probabilities for lottery-based interpolation."""
    a_grid = np.asarray(a_grid)
    a_policy = np.asarray(a_policy)

    a_i = np.searchsorted(a_grid, a_policy, side="right") - 1
    a_i = np.clip(a_i, 0, len(a_grid) - 2)
    a_pi = (a_grid[a_i + 1] - a_policy) / (a_grid[a_i + 1] - a_grid[a_i])
    return a_i, a_pi


def spline_coef(x: np.ndarray, y: np.ndarray, kind: str = "cubic"):
    """Compute spline coefficients using SciPy."""
    from scipy.interpolate import CubicSpline, make_interp_spline

    x = np.asarray(x)
    y = np.asarray(y)
    if kind == "cubic":
        return CubicSpline(x, y)
    return make_interp_spline(x, y, k=3)


def spline_eval(coef, x_new: np.ndarray) -> np.ndarray:
    """Evaluate a spline object returned by spline_coef."""
    return coef(x_new)


def cheb_nodes(n: int, a: float, b: float) -> np.ndarray:
    """Chebyshev nodes on [a, b]."""
    i = np.arange(1, n + 1)
    x = np.cos((2 * i - 1) * np.pi / (2 * n))
    return 0.5 * (a + b) + 0.5 * (b - a) * x


def cheb_basis(x: np.ndarray, n: int, a: float, b: float) -> np.ndarray:
    """Chebyshev basis matrix evaluated at x."""
    x = np.asarray(x)
    xt = (2 * x - (a + b)) / (b - a)
    xt = np.clip(xt, -1.0, 1.0)
    theta = np.arccos(xt)
    return np.column_stack([np.cos(k * theta) for k in range(n)])


def cheb_coef(f, n: int, a: float, b: float) -> np.ndarray:
    """Fit Chebyshev coefficients for f on [a, b]."""
    nodes = cheb_nodes(n, a, b)
    values = f(nodes)
    basis = cheb_basis(nodes, n, a, b)
    coef, *_ = np.linalg.lstsq(basis, values, rcond=None)
    return coef


def cheb_eval(coef: np.ndarray, x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Evaluate Chebyshev approximation given coefficients."""
    basis = cheb_basis(np.asarray(x), len(coef), a, b)
    return basis @ coef


if njit:
    @njit
    def _interp1d_numba(x_grid, y, x_new):  # pragma: no cover
        out = np.empty_like(x_new)
        for i in range(x_new.size):
            x = x_new[i]
            if x <= x_grid[0]:
                out[i] = y[0]
                continue
            if x >= x_grid[-1]:
                out[i] = y[-1]
                continue
            k = np.searchsorted(x_grid, x) - 1
            x0 = x_grid[k]
            x1 = x_grid[k + 1]
            w = (x - x0) / (x1 - x0)
            out[i] = (1 - w) * y[k] + w * y[k + 1]
        return out

