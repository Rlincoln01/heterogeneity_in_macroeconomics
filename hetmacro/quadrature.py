"""Quadrature rules for numerical integration."""

from typing import Tuple

import numpy as np
from numpy.polynomial import chebyshev, hermite, legendre
from scipy import special

from .grids import gridmake


def qnwlege(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [a, b]."""
    x, w = legendre.leggauss(n)
    nodes = 0.5 * (b - a) * x + 0.5 * (b + a)
    weights = 0.5 * (b - a) * w
    return nodes, weights


def qnwcheb(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Chebyshev nodes and weights on [a, b]."""
    x, w = chebyshev.chebgauss(n)
    nodes = 0.5 * (b - a) * x + 0.5 * (b + a)
    weights = 0.5 * (b - a) * w
    return nodes, weights


def qnwnorm(n: int, mu: float = 0.0, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite quadrature for N(mu, sigma^2)."""
    x, w = hermite.hermgauss(n)
    nodes = np.sqrt(2.0) * sigma * x + mu
    weights = w / np.sqrt(np.pi)
    return nodes, weights


def qnwlogn(n: int, mu: float = 0.0, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes/weights for lognormal distribution."""
    nodes, weights = qnwnorm(n, mu, sigma)
    return np.exp(nodes), weights


def qnwunif(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """Quadrature for uniform distribution on [a, b]."""
    nodes, weights = qnwlege(n, a, b)
    weights = weights / (b - a)
    return nodes, weights


def qnwsimp(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """Simpson's rule nodes and weights on [a, b]."""
    if n % 2 == 0:
        n += 1
    nodes = np.linspace(a, b, n)
    dx = nodes[1] - nodes[0]
    weights = np.kron(np.ones((n + 1) // 2), np.array([2.0, 4.0]))[:n]
    weights[0] = weights[-1] = 1.0
    weights *= dx / 3.0
    return nodes, weights


def qnwtrap(n: int, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """Trapezoid rule nodes and weights on [a, b]."""
    nodes = np.linspace(a, b, n)
    dx = nodes[1] - nodes[0]
    weights = dx * np.ones(n)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return nodes, weights


def qnwbeta(n: int, a: float = 1.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes and weights for Beta(a, b) on [0, 1]."""
    x, w = special.roots_jacobi(n, a - 1.0, b - 1.0)
    nodes = 0.5 * (x + 1.0)
    weights = 0.5 * w
    return nodes, weights


def qnwgamma(n: int, a: float = 1.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes and weights for Gamma(a, b) with shape a and scale b."""
    x, w = special.roots_genlaguerre(n, a - 1.0)
    nodes = b * x
    weights = w / special.gamma(a)
    return nodes, weights


def qnwnorm_mv(n: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Multivariate normal quadrature via tensor product."""
    n = np.atleast_1d(n)
    mu = np.atleast_1d(mu)
    Sigma = np.asarray(Sigma)
    d = len(n)

    nodes_1d = []
    weights_1d = []
    for i in range(d):
        xi, wi = qnwnorm(int(n[i]), 0.0, 1.0)
        nodes_1d.append(xi)
        weights_1d.append(wi)

    nodes = gridmake(*nodes_1d)
    weights = weights_1d[0]
    for wi in weights_1d[1:]:
        weights = np.kron(weights, wi)

    L = np.linalg.cholesky(Sigma)
    nodes = nodes @ L.T + mu
    return nodes, weights

