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


def qnwcheb(
    n: int, a: float, b: float, kind: str = "gauss"
) -> Tuple[np.ndarray, np.ndarray]:
    """Chebyshev-based nodes and weights on [a, b].

    kind="gauss" returns Gauss-Chebyshev nodes/weights for the weight
    function 1/sqrt(1-x^2). kind="clenshaw_curtis" matches CompEcon's
    qnwcheb (Chebyshev nodes with Clenshaw-Curtis weights for unweighted
    integrals).
    """
    kind = kind.lower()
    if kind in ("gauss", "chebyshev"):
        x, w = chebyshev.chebgauss(n)
        nodes = 0.5 * (b - a) * x + 0.5 * (b + a)
        weights = 0.5 * (b - a) * w
        return nodes, weights
    if kind in ("clenshaw_curtis", "compecon"):
        k = np.linspace(0.5, n - 0.5, n)
        nodes = 0.5 * (b + a) - 0.5 * (b - a) * np.cos(np.pi / n * k)

        t1 = np.arange(1, n + 1) - 0.5
        t2 = np.arange(0.0, n, 2)
        t3 = np.concatenate(
            [
                np.array([1.0]),
                -2.0 / (np.arange(1.0, n - 1, 2) * np.arange(3.0, n + 1, 2)),
            ]
        )
        weights = ((b - a) / n) * np.cos(np.pi / n * np.outer(t1, t2)).dot(t3)
        return nodes, weights
    raise ValueError("kind must be 'gauss' or 'clenshaw_curtis'")


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
    # Map Beta(a,b) on [0,1] to Jacobi weights on [-1,1]
    # Jacobi uses (1-x)^alpha (1+x)^beta, which corresponds to
    # (1-t)^alpha t^beta after x = 2t - 1, so swap a and b.
    x, w = special.roots_jacobi(n, b - 1.0, a - 1.0)
    nodes = 0.5 * (x + 1.0)
    weights = w * (2.0 ** (-(a + b - 1.0))) / special.beta(a, b)
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

