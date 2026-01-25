"""Steady state solvers."""

from typing import Callable, Dict, Tuple

import numpy as np

from .backward import policy_iteration
from .forward import stationary_distribution
from .interpolation import get_lottery


def household_steady_state(
    Pi: np.ndarray,
    a_grid: np.ndarray,
    y: np.ndarray,
    r: float,
    beta: float,
    eis: float,
    method: str = "egm",
    tol: float = 1e-9,
) -> Dict[str, np.ndarray]:
    """Compute household steady state given prices."""
    Va, a_pol, c_pol = policy_iteration(Pi, a_grid, y, r, beta, eis, method=method, tol=tol)
    a_i, a_pi = get_lottery(a_pol, a_grid)
    D = stationary_distribution(Pi, a_pol, a_grid)

    return {
        "D": D,
        "Va": Va,
        "a": a_pol,
        "c": c_pol,
        "a_i": a_i,
        "a_pi": a_pi,
        "A": np.vdot(a_pol, D),
        "C": np.vdot(c_pol, D),
        "Pi": Pi,
        "a_grid": a_grid,
        "y": y,
        "r": r,
        "beta": beta,
        "eis": eis,
    }


def market_clearing(
    r_bounds: Tuple[float, float],
    household_ss_func: Callable[[float], Dict[str, np.ndarray]],
    K_demand_func: Callable[[float], float],
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Dict[str, np.ndarray]:
    """Solve for market clearing interest rate using bisection."""
    r_low, r_high = r_bounds
    if r_low >= r_high:
        raise ValueError("r_bounds must satisfy r_low < r_high")

    def excess(r):
        ss = household_ss_func(r)
        return ss["A"] - K_demand_func(r)

    f_low = excess(r_low)
    f_high = excess(r_high)
    if f_low * f_high > 0:
        raise ValueError("Excess demand has same sign at bounds")

    for _ in range(max_iter):
        r_mid = 0.5 * (r_low + r_high)
        f_mid = excess(r_mid)
        if abs(f_mid) < tol:
            return household_ss_func(r_mid)
        if f_low * f_mid <= 0:
            r_high = r_mid
            f_high = f_mid
        else:
            r_low = r_mid
            f_low = f_mid

    return household_ss_func(r_mid)

