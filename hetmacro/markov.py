"""Markov chain discretization and analysis."""

from math import erf, sqrt
from typing import Tuple

import numpy as np


def rouwenhorst(n: int, rho: float, sigma: float, mu: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Rouwenhorst discretization of AR(1)."""
    if n < 2:
        raise ValueError("n must be >= 2")
    p = (1.0 + rho) / 2.0

    Pi = np.array([[p, 1.0 - p], [1.0 - p, p]])
    for m in range(3, n + 1):
        P_old = Pi
        Pi = np.zeros((m, m))
        Pi[:-1, :-1] += p * P_old
        Pi[:-1, 1:] += (1.0 - p) * P_old
        Pi[1:, :-1] += (1.0 - p) * P_old
        Pi[1:, 1:] += p * P_old
        Pi[1:-1, :] /= 2.0

    y_sd = sigma / sqrt(1.0 - rho * rho)
    psi = y_sd * sqrt(n - 1)
    states = np.linspace(-psi, psi, n) + mu / (1.0 - rho)
    return states, Pi


def tauchen(
    n: int,
    rho: float,
    sigma: float,
    mu: float = 0.0,
    n_std: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tauchen discretization of AR(1)."""
    std_y = sigma / sqrt(1.0 - rho * rho)
    x_max = n_std * std_y
    x_min = -x_max
    x = np.linspace(x_min, x_max, n)
    step = (x_max - x_min) / (n - 1)
    half_step = 0.5 * step

    P = np.empty((n, n))
    for i in range(n):
        P[i, 0] = _std_norm_cdf((x[0] - rho * x[i] + half_step) / sigma)
        P[i, n - 1] = 1.0 - _std_norm_cdf((x[-1] - rho * x[i] - half_step) / sigma)
        for j in range(1, n - 1):
            z = x[j] - rho * x[i]
            P[i, j] = _std_norm_cdf((z + half_step) / sigma) - _std_norm_cdf(
                (z - half_step) / sigma
            )

    states = x + mu / (1.0 - rho)
    return states, P


def stationary_distribution(P: np.ndarray, method: str = "eigen", tol: float = 1e-12) -> np.ndarray:
    """Compute stationary distribution of a Markov chain."""
    if method == "eigen":
        vals, vecs = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(vals - 1.0))
        dist = np.real(vecs[:, idx])
        dist = dist / dist.sum()
        return dist
    if method == "iterate":
        n = P.shape[0]
        pi = np.full(n, 1.0 / n)
        for _ in range(10_000):
            pi_new = P.T @ pi
            if np.max(np.abs(pi_new - pi)) < tol:
                return pi_new
            pi = pi_new
        return pi
    raise ValueError("method must be 'eigen' or 'iterate'")


def simulate_markov(P: np.ndarray, s0: int, T: int, random_state=None) -> np.ndarray:
    """Simulate Markov chain indices."""
    rng = np.random.default_rng(random_state)
    n = P.shape[0]
    s = np.empty(T, dtype=int)
    s[0] = s0
    for t in range(1, T):
        s[t] = rng.choice(n, p=P[s[t - 1]])
    return s


def simulate_mc(
    P: np.ndarray,
    z_grid: np.ndarray,
    T: int,
    start_state: int = None,
    burn_in: float = 0.0,
    seed=None,
) -> np.ndarray:
    """Simulate Markov chain values from transition matrix and state grid."""
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    if start_state is None:
        state = int(rng.integers(0, n))
    else:
        state = int(start_state)

    path = np.empty(T)
    for t in range(T):
        path[t] = z_grid[state]
        state = int(rng.choice(np.arange(n), p=P[state]))

    b = int((burn_in / 100.0) * T) if burn_in > 0 else int(burn_in)
    return path[b:]


def markov_chain_ar1(rho: float, sig: float, n: int):
    """Compact AR(1) Rouwenhorst discretization with output-normalized levels."""
    p = q = 0.5 * (1.0 + rho)
    Pi = np.array([[p, 1 - p], [1 - q, q]])
    for k in range(2, n):
        Pi_new = np.zeros((k + 1, k + 1))
        Pi_new[:-1, :-1] += p * Pi
        Pi_new[:-1, 1:] += (1 - p) * Pi
        Pi_new[1:, :-1] += (1 - q) * Pi
        Pi_new[1:, 1:] += q * Pi
        Pi = Pi_new / np.sum(Pi_new, axis=1)[:, np.newaxis]

    rho_abs = min(max(abs(rho), 1e-12), 1 - 1e-12)
    steps = max(1, int(np.log(1e-15) / np.log(rho_abs)))
    pi = np.linalg.matrix_power(Pi.T, steps)[:, 0]
    z = np.exp(sig * np.sqrt(n - 1) * np.linspace(-1, 1, n))
    z /= np.sum(pi * z)
    return z, Pi, pi


def _std_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

