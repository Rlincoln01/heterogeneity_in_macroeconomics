"""High-level modular household class."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .distributions import compute_joint_transition_matrix, stationary_dist
from .income_process import ContinuousQuadratureIncome, DiscreteIncome, IncomeProcess
from .utils import crra_marginal


def resolve_expectation_mode(household: "Household", expectation: str = "auto") -> str:
    """Validate and resolve the expectation mode string.

    Returns ``"discrete"`` or ``"quadrature"``.
    """
    income = household.income_process
    if expectation == "auto":
        if isinstance(income, DiscreteIncome):
            return "discrete"
        if isinstance(income, ContinuousQuadratureIncome):
            return "quadrature"
        raise ValueError(f"Cannot infer expectation mode for income type {type(income)}.")
    if expectation == "discrete":
        if not isinstance(income, DiscreteIncome):
            raise ValueError("expectation='discrete' requires a DiscreteIncome process.")
        return "discrete"
    if expectation == "quadrature":
        if not isinstance(income, ContinuousQuadratureIncome):
            raise ValueError(
                "expectation='quadrature' requires a ContinuousQuadratureIncome process."
            )
        return "quadrature"
    raise ValueError(f"Unknown expectation mode '{expectation}'. Use 'auto', 'discrete', or 'quadrature'.")


@dataclass
class SolvedPolicy:
    """Solver output with a common evaluation interface."""

    policy_a: np.ndarray
    policy_c: np.ndarray
    a_grid: np.ndarray
    z_grid: np.ndarray
    value: Optional[np.ndarray] = None

    def evaluate(self, a_points: np.ndarray, z_points: np.ndarray) -> Dict[str, np.ndarray]:
        """Evaluate policy functions on arbitrary points."""
        pts = np.column_stack([np.asarray(z_points).reshape(-1), np.asarray(a_points).reshape(-1)])
        pa = RegularGridInterpolator(
            (self.z_grid, self.a_grid),
            self.policy_a,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )(pts)
        pc = RegularGridInterpolator(
            (self.z_grid, self.a_grid),
            self.policy_c,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )(pts)
        out = {"a": pa.reshape(np.asarray(z_points).shape), "c": pc.reshape(np.asarray(z_points).shape)}
        if self.value is not None:
            vv = RegularGridInterpolator(
                (self.z_grid, self.a_grid),
                self.value,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )(pts)
            out["v"] = vv.reshape(np.asarray(z_points).shape)
        return out


class Household:
    """Composable household problem with IncomeProcess and solver blocks."""

    def __init__(
        self,
        income_process: IncomeProcess,
        a_grid: np.ndarray,
        beta: float,
        gamma: float,
        r: float,
        w: float = 1.0,
        borrowing_limit: Optional[float] = None,
    ):
        self.income_process = income_process
        self.a_grid = np.asarray(a_grid)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.r = float(r)
        self.w = float(w)
        self.borrowing_limit = self.a_grid[0] if borrowing_limit is None else borrowing_limit

        self.solved_policy: Optional[SolvedPolicy] = None
        self.g: Optional[np.ndarray] = None
        self.aggregates: Optional[Dict[str, float]] = None

    @property
    def z_grid(self):
        return np.asarray(self.income_process.z_grid)

    @property
    def is_continuous(self) -> bool:
        return isinstance(self.income_process, ContinuousQuadratureIncome)

    @property
    def y_grid(self):
        """Income levels on the z_grid.

        For DiscreteIncome, z_grid stores productivity levels so y = w * z.
        For ContinuousQuadratureIncome, z_grid is in log space so y = exp(z).
        """
        if self.is_continuous:
            return np.exp(self.z_grid)
        return self.w * self.z_grid

    def resources(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Cash on hand: y(z) + (1+r)*a."""
        if self.is_continuous:
            return np.exp(z) + (1.0 + self.r) * a
        return self.w * z + (1.0 + self.r) * a

    def solve(self, solver, **kwargs) -> SolvedPolicy:
        """Solve household policy with the provided solver block."""
        self.solved_policy = solver.solve(self, **kwargs)
        return self.solved_policy

    def compute_ergodic(self) -> np.ndarray:
        """Compute stationary income-asset distribution under solved policy."""
        if self.solved_policy is None:
            raise RuntimeError("Call solve() before compute_ergodic().")

        if isinstance(self.income_process, DiscreteIncome):
            return self._compute_ergodic_discrete()
        if isinstance(self.income_process, ContinuousQuadratureIncome):
            return self._compute_ergodic_quadrature()
        raise NotImplementedError(
            f"compute_ergodic not implemented for {type(self.income_process)}."
        )

    def _compute_ergodic_discrete(self) -> np.ndarray:
        from .interpolation import get_lottery

        a_ind, a_w = get_lottery(self.solved_policy.policy_a, self.a_grid)
        T = compute_joint_transition_matrix(a_ind, a_w, self.income_process.Pi)
        g = stationary_dist(T).reshape(self.z_grid.size, self.a_grid.size)
        self.g = g
        return g

    def _compute_ergodic_quadrature(self) -> np.ndarray:
        from scipy import sparse

        from .forward import stationary_eigenvector
        from .interpolation import linear_basis_matrix, tensor_basis_matrix

        income = self.income_process
        a_grid = self.a_grid
        z_grid = self.z_grid
        nz, na = z_grid.size, a_grid.size
        N = nz * na
        zmin, zmax = float(z_grid[0]), float(z_grid[-1])

        a_pol_flat = self.solved_policy.policy_a.reshape(-1)
        Z_flat = np.repeat(z_grid, na)

        Phi_a = linear_basis_matrix(a_grid, np.clip(a_pol_flat, a_grid[0], a_grid[-1]))
        Q = sparse.csr_matrix((N, N))

        for eps, w in zip(income.eps_nodes, income.eps_weights):
            z_next = np.clip(
                income.mu + income.rho * (Z_flat - income.mu) + eps, zmin, zmax
            )
            Phi_z = linear_basis_matrix(z_grid, z_next)
            B = tensor_basis_matrix(Phi_a, Phi_z).tocsr()
            Q = Q + w * B

        # Normalize rows to sum to 1 (clipping can cause small deviations)
        row_sums = np.asarray(Q.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        Q = sparse.diags(1.0 / row_sums) @ Q

        g = stationary_eigenvector(Q).reshape(nz, na)
        self.g = g
        return g

    def compute_aggregates(self) -> Dict[str, float]:
        """Compute aggregate moments from ergodic distribution."""
        if self.g is None:
            self.compute_ergodic()
        g = self.g
        pol = self.solved_policy
        agg = {
            "A": float(np.sum(g * pol.policy_a)),
            "C": float(np.sum(g * pol.policy_c)),
            "Y": float(np.sum(g * self.y_grid[:, None])),
        }
        self.aggregates = agg
        return agg

    def euler_check(self) -> np.ndarray:
        """Euler residual on grid under solved policy.

        Works for both DiscreteIncome (Pi expectation) and
        ContinuousQuadratureIncome (quadrature expectation).
        """
        if self.solved_policy is None:
            raise RuntimeError("Call solve() before euler_check().")

        c = self.solved_policy.policy_c
        up = crra_marginal(c, self.gamma)
        income = self.income_process

        if isinstance(income, DiscreteIncome):
            emu = income.Pi @ up
        elif isinstance(income, ContinuousQuadratureIncome):
            emu = income.expected(up)
        else:
            raise NotImplementedError(
                f"Euler check not implemented for {type(income)}."
            )

        rhs = self.beta * (1.0 + self.r) * emu
        c_rhs = rhs ** (-1.0 / self.gamma)
        return (c_rhs - c) / np.maximum(c, 1e-12)

    def simulate(self, T: int, n_agents: int = 1_000, seed: int = 0) -> Dict[str, np.ndarray]:
        """Monte Carlo simulation from solved policy (discrete-income case)."""
        if self.solved_policy is None:
            raise RuntimeError("Call solve() before simulate().")
        if not isinstance(self.income_process, DiscreteIncome):
            raise NotImplementedError("Simulation currently supports DiscreteIncome only.")

        rng = np.random.default_rng(seed)
        nz, na = self.z_grid.size, self.a_grid.size
        Pi = self.income_process.Pi
        pa = self.solved_policy.policy_a
        pc = self.solved_policy.policy_c

        z_idx = rng.integers(0, nz, size=n_agents)
        a_idx = np.zeros(n_agents, dtype=int)
        a = np.full(n_agents, self.a_grid[0], dtype=float)

        y_path = np.zeros((T, n_agents))
        c_path = np.zeros((T, n_agents))
        a_path = np.zeros((T + 1, n_agents))
        a_path[0] = a

        for t in range(T):
            y_path[t] = self.y_grid[z_idx]
            c_path[t] = pc[z_idx, a_idx]
            a_next = pa[z_idx, a_idx]
            a_path[t + 1] = a_next

            # map next assets back to nearest index
            a_idx = np.clip(np.searchsorted(self.a_grid, a_next), 0, na - 1)
            z_new = np.empty_like(z_idx)
            for i in range(n_agents):
                z_new[i] = rng.choice(np.arange(nz), p=Pi[z_idx[i]])
            z_idx = z_new

        return {"y": y_path, "c": c_path, "a": a_path}
