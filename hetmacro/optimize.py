"""Root-finding and optimization routines."""

from typing import Callable, Tuple

import numpy as np
from scipy import optimize


def bisect(f: Callable, a: float, b: float, args=(), tol: float = 1e-12, max_iter: int = 100) -> float:
    """Bisection root-finding."""
    fa = f(a, *args)
    fb = f(b, *args)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m, *args)
        if abs(fm) < tol:
            return m
        if fa * fm <= 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    return m


def brentq(f: Callable, a: float, b: float, args=(), tol: float = 1e-12) -> float:
    """Brent's method root-finding."""
    return optimize.brentq(f, a, b, args=args, xtol=tol, rtol=tol)


def newton(
    f: Callable,
    x0: float,
    fprime: Callable,
    args=(),
    tol: float = 1e-8,
    max_iter: int = 50,
) -> float:
    """Newton-Raphson root-finding."""
    return optimize.newton(f, x0, fprime=fprime, args=args, tol=tol, maxiter=max_iter)


def secant(f: Callable, x0: float, args=(), tol: float = 1e-8, max_iter: int = 50) -> float:
    """Secant method."""
    return optimize.newton(f, x0, fprime=None, args=args, tol=tol, maxiter=max_iter)


def broyden(f: Callable, x0: np.ndarray, args=(), tol: float = 1e-8) -> np.ndarray:
    """Broyden's method for systems of equations."""
    res = optimize.root(lambda x: f(x, *args), x0, method="broyden1", tol=tol)
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def golden_search(f: Callable, a: float, b: float, args=(), tol: float = 1e-8) -> float:
    """Golden section search for scalar minimization."""
    res = optimize.minimize_scalar(f, bracket=(a, b), args=args, tol=tol, method="golden")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def brent_min(f: Callable, a: float, b: float, args=(), tol: float = 1e-8) -> float:
    """Brent's method for scalar minimization."""
    res = optimize.minimize_scalar(f, bracket=(a, b), args=args, tol=tol, method="brent")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def nelder_mead(f: Callable, x0: np.ndarray, args=(), tol: float = 1e-8) -> Tuple[np.ndarray, float]:
    """Nelder-Mead simplex optimization."""
    res = optimize.minimize(lambda x: f(x, *args), x0, method="Nelder-Mead", tol=tol)
    if not res.success:
        raise RuntimeError(res.message)
    return res.x, res.fun

