"""Forward iteration for distributions."""

from typing import Tuple

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


def forward_iteration(D: np.ndarray, Pi: np.ndarray, a_i: np.ndarray, a_pi: np.ndarray) -> np.ndarray:
    """Single forward step for distribution."""
    Dend = update_policy_1a(a_i, a_pi, D)
    return update_exogenous_1a(Pi, Dend)


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


def stationary_eigenvector(
    Q, tol: float = 1e-12, max_iter: int = 50_000
) -> np.ndarray:
    """Stationary distribution from left unit-eigenvector of transition matrix.

    Tries eigenvector decomposition first, falls back to power iteration
    if that fails or produces an invalid result.
    """
    from scipy import sparse

    if Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be square.")

    n = Q.shape[0]

    # Try eigenvector method first
    try:
        vec = _eigenvector_solve(Q, tol)
        if vec is not None:
            return vec
    except Exception:
        pass

    # Fallback: power iteration on Q^T
    g = np.full(n, 1.0 / n)
    if sparse.issparse(Q):
        QT = Q.T.tocsr()
    else:
        QT = np.asarray(Q, dtype=float).T

    for _ in range(max_iter):
        g_new = QT @ g
        g_new = np.maximum(g_new, 0.0)
        s = g_new.sum()
        if s > 0:
            g_new /= s
        if np.max(np.abs(g_new - g)) < tol:
            return g_new
        g = g_new
    return g


def _eigenvector_solve(Q, tol: float):
    """Eigenvector solve; returns None if result is invalid."""
    from scipy import sparse
    from scipy.sparse.linalg import eigs

    QT = Q.T
    if sparse.issparse(QT):
        vals, vecs = eigs(QT, k=1, sigma=1.0, tol=tol)
        vec = np.real(vecs[:, 0])
    else:
        vals, vecs = np.linalg.eig(np.asarray(QT, dtype=float))
        idx = int(np.argmin(np.abs(vals - 1.0)))
        vec = np.real(vecs[:, idx])

    if np.sum(vec) < 0:
        vec = -vec
    vec = np.maximum(vec, 0.0)
    s = vec.sum()
    if s <= 0:
        return None
    return vec / s


def update_policy_1a(a_ind: np.ndarray, a_ind_w: np.ndarray, g_prev: np.ndarray) -> np.ndarray:
    """Lottery update from current asset bin to adjacent next-asset bins."""
    return forward_policy(g_prev, a_ind, a_ind_w)


def update_exogenous_1a(Pi: np.ndarray, g_prev: np.ndarray) -> np.ndarray:
    """Exogenous income transition update."""
    return Pi.T @ g_prev


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

