"""Forward iteration for distributions."""

from typing import Tuple

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


def forward_iteration(D: np.ndarray, Pi: np.ndarray, a_i: np.ndarray, a_pi: np.ndarray) -> np.ndarray:
    """Single forward step for distribution."""
    Dend = forward_policy(D, a_i, a_pi)
    return Pi.T @ Dend


def stationary_distribution(
    Pi: np.ndarray,
    a_policy: np.ndarray,
    a_grid: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> np.ndarray:
    """Compute stationary distribution from policy function."""
    from .interpolation import get_lottery

    a_i, a_pi = get_lottery(a_policy, a_grid)
    pi = stationary_markov(Pi)
    D = pi[:, np.newaxis] * np.ones_like(a_grid) / len(a_grid)

    for _ in range(max_iter):
        D_new = forward_iteration(D, Pi, a_i, a_pi)
        if np.max(np.abs(D_new - D)) < tol:
            return D_new
        D = D_new
    return D


def expectation_iteration(X: np.ndarray, Pi: np.ndarray, a_i: np.ndarray, a_pi: np.ndarray) -> np.ndarray:
    """Single expectation step for SSJ computations."""
    Xend = Pi @ X
    return expectation_policy(Xend, a_i, a_pi)


def expectation_functions(X: np.ndarray, Pi: np.ndarray, a_i: np.ndarray, a_pi: np.ndarray, T: int) -> np.ndarray:
    """Compute sequence of expectation functions up to horizon T."""
    out = np.empty((T,) + X.shape)
    out[0] = X
    for t in range(1, T):
        out[t] = expectation_iteration(out[t - 1], Pi, a_i, a_pi)
    return out


def stationary_markov(Pi: np.ndarray, tol: float = 1e-14, max_iter: int = 10_000) -> np.ndarray:
    """Stationary distribution of Markov chain by power iteration."""
    n = Pi.shape[0]
    pi = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        pi_new = Pi.T @ pi
        if np.max(np.abs(pi_new - pi)) < tol:
            return pi_new
        pi = pi_new
    return pi


if njit:
    @njit
    def forward_policy(D: np.ndarray, a_i: np.ndarray, a_pi: np.ndarray) -> np.ndarray:  # pragma: no cover
        Dend = np.zeros_like(D)
        for e in range(a_i.shape[0]):
            for a in range(a_i.shape[1]):
                i = a_i[e, a]
                pi = a_pi[e, a]
                Dend[e, i] += pi * D[e, a]
                Dend[e, i + 1] += (1.0 - pi) * D[e, a]
        return Dend

    @njit
    def expectation_policy(Xend: np.ndarray, a_i: np.ndarray, a_pi: np.ndarray) -> np.ndarray:  # pragma: no cover
        X = np.zeros_like(Xend)
        for e in range(a_i.shape[0]):
            for a in range(a_i.shape[1]):
                i = a_i[e, a]
                pi = a_pi[e, a]
                X[e, a] = pi * Xend[e, i] + (1.0 - pi) * Xend[e, i + 1]
        return X
else:
    def forward_policy(D: np.ndarray, a_i: np.ndarray, a_pi: np.ndarray) -> np.ndarray:
        Dend = np.zeros_like(D)
        for e in range(a_i.shape[0]):
            for a in range(a_i.shape[1]):
                i = a_i[e, a]
                pi = a_pi[e, a]
                Dend[e, i] += pi * D[e, a]
                Dend[e, i + 1] += (1.0 - pi) * D[e, a]
        return Dend

    def expectation_policy(Xend: np.ndarray, a_i: np.ndarray, a_pi: np.ndarray) -> np.ndarray:
        X = np.zeros_like(Xend)
        for e in range(a_i.shape[0]):
            for a in range(a_i.shape[1]):
                i = a_i[e, a]
                pi = a_pi[e, a]
                X[e, a] = pi * Xend[e, i] + (1.0 - pi) * Xend[e, i + 1]
        return X

