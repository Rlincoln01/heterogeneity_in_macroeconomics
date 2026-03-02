"""Income-process abstractions for household problems."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .markov import rouwenhorst, tauchen
from .quadrature import qnwnorm


class IncomeProcess(Protocol):
    """Interface for income process blocks used by ``Household``."""

    z_grid: np.ndarray

    def expected(self, values: np.ndarray) -> np.ndarray:
        """Conditional expectation operator applied to ``values``."""


@dataclass
class DiscreteIncome:
    """Exogenous discrete income process with transition matrix."""

    z_grid: np.ndarray
    Pi: np.ndarray

    def expected(self, values: np.ndarray) -> np.ndarray:
        return self.Pi @ values


@dataclass
class RouwenhorstIncome(DiscreteIncome):
    """AR(1) discretized with Rouwenhorst."""

    @classmethod
    def from_ar1(
        cls, n: int, rho: float, sigma: float, mu: float = 0.0
    ) -> "RouwenhorstIncome":
        z, Pi = rouwenhorst(n=n, rho=rho, sigma=sigma, mu=mu)
        return cls(z_grid=np.exp(z), Pi=Pi)


@dataclass
class TauchenIncome(DiscreteIncome):
    """AR(1) discretized with Tauchen."""

    @classmethod
    def from_ar1(
        cls, n: int, rho: float, sigma: float, mu: float = 0.0, n_std: float = 3.0
    ) -> "TauchenIncome":
        z, Pi = tauchen(n=n, rho=rho, sigma=sigma, mu=mu, n_std=n_std)
        return cls(z_grid=np.exp(z), Pi=Pi)


@dataclass
class ContinuousQuadratureIncome:
    """Continuous AR(1) process approximated with Gaussian quadrature."""

    rho: float
    sigma_eps: float
    mu: float = 0.0
    n_quad: int = 7

    def __post_init__(self):
        self.eps_nodes, self.eps_weights = qnwnorm(
            self.n_quad, mu=0.0, sig2=self.sigma_eps**2
        )
        self.z_std = self.sigma_eps / np.sqrt(1.0 - self.rho**2)
        self.z_grid = np.linspace(
            self.mu - 3.0 * self.z_std, self.mu + 3.0 * self.z_std, 41
        )

    def next_z_nodes(self, z: np.ndarray) -> np.ndarray:
        """Return quadrature next-z nodes for current z."""
        z = np.asarray(z)
        return self.mu + self.rho * (z - self.mu)[..., None] + self.eps_nodes

    def expected(self, values: np.ndarray) -> np.ndarray:
        """Return E[v(z')] using linear interpolation onto default z_grid."""
        from scipy.interpolate import interp1d

        if values.ndim != 2:
            raise ValueError("values must be (n_z, n_a) for ContinuousQuadratureIncome.")
        nz, na = values.shape
        if nz != self.z_grid.size:
            raise ValueError("values first axis must match len(z_grid).")

        out = np.empty_like(values)
        for ia in range(na):
            f = interp1d(
                self.z_grid,
                values[:, ia],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            z_next = self.next_z_nodes(self.z_grid)
            out[:, ia] = f(z_next) @ self.eps_weights
        return out
