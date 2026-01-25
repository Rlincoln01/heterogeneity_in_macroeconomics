"""Aiyagari model example using hetmacro tools."""

from typing import Dict, Tuple

import numpy as np

from ..grids import make_asset_grid
from ..markov import rouwenhorst, stationary_distribution
from ..steady_state import household_steady_state, market_clearing


def firm_k_demand(r: float, alpha: float, delta: float) -> float:
    """Cobb-Douglas capital demand with labor normalized to 1."""
    return (alpha / (r + delta)) ** (1.0 / (1.0 - alpha))


def firm_wage(K: float, alpha: float) -> float:
    """Cobb-Douglas wage with labor normalized to 1."""
    return (1.0 - alpha) * K**alpha


def solve_steady_state(
    beta: float = 0.96,
    eis: float = 1.0,
    rho: float = 0.9,
    sigma: float = 0.2,
    n_e: int = 7,
    amin: float = 0.0,
    amax: float = 50.0,
    n_a: int = 500,
    alpha: float = 0.36,
    delta: float = 0.08,
    r_bounds: Tuple[float, float] = (0.005, 0.04),
) -> Dict[str, np.ndarray]:
    """Solve Aiyagari steady state with idiosyncratic income risk."""
    e, Pi = rouwenhorst(n_e, rho, sigma)
    pi = stationary_distribution(Pi, method="iterate")
    e = e / np.vdot(pi, np.exp(e))
    a_grid = make_asset_grid(amin, amax, n_a, curvature=2.0)

    def hh_ss(r: float) -> Dict[str, np.ndarray]:
        K = firm_k_demand(r, alpha, delta)
        w = firm_wage(K, alpha)
        y = w * np.exp(e)
        return household_steady_state(Pi, a_grid, y, r, beta, eis, method="egm")

    def K_demand(r: float) -> float:
        return firm_k_demand(r, alpha, delta)

    ss = market_clearing(r_bounds, hh_ss, K_demand)
    ss["K"] = K_demand(ss["r"])
    ss["w"] = firm_wage(ss["K"], alpha)
    return ss

