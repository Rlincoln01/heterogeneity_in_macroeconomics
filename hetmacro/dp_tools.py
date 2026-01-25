"""Dynamic programming tools for HA models."""

from typing import Tuple

import numpy as np

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

