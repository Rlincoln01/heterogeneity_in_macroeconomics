"""Dynamic programming tools for HA models."""

from typing import Tuple

import numpy as np

try:
    from numba import njit, uint
except Exception:  # pragma: no cover
    njit = None
    uint = np.uint32

from .backward import backward_egm, backward_vfi, policy_iteration


def solve_policy_egm(
    Va: np.ndarray,
    Pi: np.ndarray,
    a_grid: np.ndarray,
    y: np.ndarray,
    r: float,
    beta: float,
    eis: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single EGM step wrapper."""
    return backward_egm(Va, Pi, a_grid, y, r, beta, eis)


def solve_policy_vfi(
    V: np.ndarray,
    Pi: np.ndarray,
    a_grid: np.ndarray,
    y: np.ndarray,
    r: float,
    beta: float,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single VFI step wrapper."""
    return backward_vfi(V, Pi, a_grid, y, r, beta, gamma)


def solve_steady_policy(
    Pi: np.ndarray,
    a_grid: np.ndarray,
    y: np.ndarray,
    r: float,
    beta: float,
    eis: float,
    method: str = "egm",
    tol: float = 1e-9,
    max_iter: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterate policy to convergence."""
    return policy_iteration(Pi, a_grid, y, r, beta, eis, method=method, tol=tol, max_iter=max_iter)


def interp_grid(v: float, x: np.ndarray, i_start: int = 0) -> Tuple[int, float]:
    """Return bracket index and interpolation weight for value ``v`` on increasing ``x``.

    The return pair ``(ind, weight)`` satisfies approximately:
    ``v = weight * x[ind] + (1 - weight) * x[ind + 1]``.
    """
    n = len(x)
    if n < 2:
        raise ValueError("x must contain at least two points.")
    i = int(np.clip(i_start, 0, n - 2))

    if v > x[i]:
        for j in range(i + 1, n):
            if x[j] > v:
                i = j - 1
                break
        else:
            i = n - 2
    else:
        for j in range(i - 1, -1, -1):
            if x[j] <= v:
                i = j
                break
        else:
            i = 0

    den = x[i + 1] - x[i]
    if den == 0:
        return i, 1.0
    weight = (x[i + 1] - v) / den
    return i, weight


def solve_euler_one_asset(c_foc: np.ndarray, budget: np.ndarray, a_grid: np.ndarray):
    """Solve one-asset Euler step with linear interpolation and borrowing constraints.

    Parameters
    ----------
    c_foc : ndarray, shape (n_e, n_a)
        Consumption implied by the Euler equation at ``a'`` grid points.
    budget : ndarray, shape (n_e, n_a)
        Cash-on-hand for each exogenous state and current asset point.
    a_grid : ndarray, shape (n_a,)
        Next-period asset grid.
    """
    ne, na = c_foc.shape
    a = np.zeros((ne, na))
    a_ind = np.zeros((ne, na), dtype=np.int64)
    a_ind_w = np.zeros((ne, na))

    for ie in range(ne):
        ia_start = 0
        for ia in range(na):
            iap = ia_start
            c_here = budget[ie, ia] - a_grid[iap]
            for iap in range(ia_start, na):
                c_here = budget[ie, ia] - a_grid[iap]
                if c_foc[ie, iap] > c_here:
                    break

            if iap == 0:
                a_ind[ie, ia] = 0
                a_ind_w[ie, ia] = 1.0
            elif (iap == na - 1) and (c_foc[ie, iap] <= c_here):
                a_ind[ie, ia] = na - 2
                a_ind_w[ie, ia] = 0.0
            else:
                c_prev = budget[ie, ia] - a_grid[iap - 1]
                y0 = c_prev - c_foc[ie, iap - 1]
                y1 = c_here - c_foc[ie, iap]
                a_ind[ie, ia] = iap - 1
                den = y1 - y0
                a_ind_w[ie, ia] = 0.5 if den == 0 else y1 / den

            i0 = a_ind[ie, ia]
            w0 = a_ind_w[ie, ia]
            a[ie, ia] = w0 * a_grid[i0] + (1.0 - w0) * a_grid[i0 + 1]
            ia_start = iap

    c = budget - a
    return c, a, a_ind.astype(np.int64), a_ind_w


if njit:
    solve_euler_one_asset = njit(solve_euler_one_asset)  # pragma: no cover

