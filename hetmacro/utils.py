"""Utility functions and numerical helpers."""

from typing import Callable, Tuple

import numpy as np

_tic_time = None


def crra_utility(c: np.ndarray, gamma: float) -> np.ndarray:
    """CRRA utility."""
    c = np.asarray(c)
    if gamma == 1.0:
        return np.log(c)
    return c ** (1.0 - gamma) / (1.0 - gamma)


def crra_marginal(c: np.ndarray, gamma: float) -> np.ndarray:
    """Marginal utility of CRRA."""
    c = np.asarray(c)
    return c ** (-gamma)


def crra_inverse_marginal(u: np.ndarray, gamma: float) -> np.ndarray:
    """Inverse marginal utility for CRRA."""
    u = np.asarray(u)
    return u ** (-1.0 / gamma)


def ces_aggregator(x: np.ndarray, y: np.ndarray, sigma: float, alpha: float = 0.5) -> np.ndarray:
    """CES aggregator."""
    if sigma == 1.0:
        return x**alpha * y ** (1.0 - alpha)
    rho = (sigma - 1.0) / sigma
    return (alpha * x**rho + (1.0 - alpha) * y**rho) ** (1.0 / rho)


def compute_fixed_point(
    T: Callable,
    v0: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1_000,
    verbose: bool = False,
):
    """Compute a fixed point of operator T via iteration."""
    v = np.array(v0, copy=True)
    for i in range(max_iter):
        v_new = T(v)
        err = np.max(np.abs(v_new - v))
        v = v_new
        if verbose and (i % 10 == 0 or err < tol):
            print(f"iter={i}, err={err:.3e}")
        if err < tol:
            break
    return v


def numerical_derivative(f: Callable, x: float, h: float = 1e-7, method: str = "central") -> float:
    """Numerical derivative of scalar function."""
    if method == "forward":
        return (f(x + h) - f(x)) / h
    if method == "backward":
        return (f(x) - f(x - h)) / h
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_jacobian(f: Callable, x: np.ndarray, h: float = 1e-7) -> np.ndarray:
    """Numerical Jacobian of vector function."""
    x = np.asarray(x, dtype=float)
    fx = f(x)
    J = np.zeros((fx.size, x.size))
    for i in range(x.size):
        x_step = x.copy()
        x_step[i] += h
        J[:, i] = (f(x_step) - fx) / h
    return J


def tic():
    """Start timing."""
    global _tic_time
    import time

    _tic_time = time.time()


def toc() -> float:
    """Return elapsed time since tic()."""
    global _tic_time
    import time

    if _tic_time is None:
        return 0.0
    return time.time() - _tic_time

