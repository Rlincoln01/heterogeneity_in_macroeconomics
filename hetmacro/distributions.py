"""Distribution iteration tools for HA models."""

from typing import Tuple

import numpy as np

from .forward import forward_iteration, stationary_distribution
from .interpolation import get_lottery


def forward_policy_step(D: np.ndarray, a_policy: np.ndarray, a_grid: np.ndarray) -> np.ndarray:
    """Forward step using policy and grid."""
    a_i, a_pi = get_lottery(a_policy, a_grid)
    return forward_iteration(D, np.eye(a_policy.shape[0]), a_i, a_pi)


def stationary_distribution_ha(
    Pi: np.ndarray,
    a_policy: np.ndarray,
    a_grid: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """Stationary distribution for HA models."""
    return stationary_distribution(Pi, a_policy, a_grid, tol=tol)


def lottery_indices(a_policy: np.ndarray, a_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return lottery indices and weights."""
    return get_lottery(a_policy, a_grid)

