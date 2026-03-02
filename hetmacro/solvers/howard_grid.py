"""Non-collocation Howard policy improvement (grid search + exact evaluation)."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..household import SolvedPolicy, resolve_expectation_mode
from ..interpolation import get_lottery, linear_basis_matrix, tensor_basis_matrix
from ..utils import crra_utility


@dataclass
class HowardGrid:
    """Howard iteration on a fixed asset grid.

    Combines grid search policy improvement (like GridVFI) with exact
    policy evaluation via a linear system solve (like PFI). Converges
    on the value function.

    Parameters
    ----------
    tol : float
        Convergence tolerance on max|V_new - V|.
    max_iter : int
        Maximum Howard outer iterations.
    expectation : str
        ``"auto"`` (default), ``"discrete"``, or ``"quadrature"``.
    verbose : bool
        If True, print per-iteration max|V_new - V| and convergence status.
    """

    tol: float = 1e-8
    max_iter: int = 200
    expectation: str = "auto"
    verbose: bool = False

    # ------------------------------------------------------------------
    # Policy improvement: grid search
    # ------------------------------------------------------------------

    @staticmethod
    def _policy_improvement_grid(V, EV, y, a_grid, beta, gamma, r):
        """One step of Bellman maximization via grid search over a'."""
        nz, na = V.shape
        a_pol_idx = np.empty((nz, na), dtype=int)
        c_pol = np.empty((nz, na))
        V_new = np.empty((nz, na))

        for iz in range(nz):
            for ia, a in enumerate(a_grid):
                c = y[iz] + (1.0 + r) * a - a_grid
                util = np.where(c > 0, crra_utility(c, gamma), -1e18)
                rhs = util + beta * EV[iz, :]
                j = int(np.argmax(rhs))
                V_new[iz, ia] = rhs[j]
                a_pol_idx[iz, ia] = j
                c_pol[iz, ia] = c[j]

        a_pol = a_grid[a_pol_idx]
        return a_pol, a_pol_idx, c_pol, V_new

    # ------------------------------------------------------------------
    # Policy evaluation: exact linear system solve
    # ------------------------------------------------------------------

    @staticmethod
    def _policy_evaluation_discrete(a_pol_idx, c_pol, Pi, a_grid, beta, gamma):
        nz, na = c_pol.shape
        n = nz * na
        utility = crra_utility(np.maximum(c_pol, 1e-10), gamma).reshape(-1)

        rows, cols, data = [], [], []
        for iz in range(nz):
            for ia in range(na):
                row = iz * na + ia
                j = int(a_pol_idx[iz, ia])
                for izp in range(nz):
                    p = float(Pi[iz, izp])
                    if p == 0.0:
                        continue
                    cols.append(izp * na + j)
                    rows.append(row)
                    data.append(p)

        T = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        lhs = sparse.eye(n, format="csr") - beta * T
        v = spsolve(lhs.tocsc(), utility)
        return np.asarray(v).reshape(nz, na)

    @staticmethod
    def _policy_evaluation_quadrature(a_pol, c_pol, income, a_grid, beta, gamma):
        nz, na = a_pol.shape
        N = nz * na
        z_grid = income.z_grid
        zmin, zmax = float(z_grid[0]), float(z_grid[-1])

        a_pol_flat = a_pol.reshape(-1)
        Z_flat = np.repeat(z_grid, na)
        utility = crra_utility(np.maximum(c_pol, 1e-10), gamma).reshape(-1)

        Phi_a = linear_basis_matrix(a_grid, np.clip(a_pol_flat, a_grid[0], a_grid[-1]))
        Q = sparse.csr_matrix((N, N))

        for eps, w in zip(income.eps_nodes, income.eps_weights):
            z_next = np.clip(
                income.mu + income.rho * (Z_flat - income.mu) + eps, zmin, zmax
            )
            Phi_z = linear_basis_matrix(z_grid, z_next)
            B = tensor_basis_matrix(Phi_a, Phi_z).tocsr()
            Q = Q + w * B

        lhs = sparse.eye(N, format="csr") - beta * Q
        v = spsolve(lhs.tocsc(), utility)
        return np.asarray(v).reshape(nz, na)

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self, household, **kwargs) -> SolvedPolicy:
        tol = kwargs.get("tol", self.tol)
        max_iter = kwargs.get("max_iter", self.max_iter)
        verbose = kwargs.get("verbose", self.verbose)
        mode = resolve_expectation_mode(household, self.expectation)

        y = household.y_grid
        a_grid = household.a_grid
        beta = household.beta
        gamma = household.gamma
        r = household.r
        income = household.income_process

        nz, na = y.size, a_grid.size
        V = np.zeros((nz, na))

        for it in range(max_iter):
            if mode == "discrete":
                EV = income.Pi @ V
            else:
                EV = income.expected(V)

            a_pol, a_pol_idx, c_pol, _ = self._policy_improvement_grid(
                V, EV, y, a_grid, beta, gamma, r
            )

            if mode == "discrete":
                V_new = self._policy_evaluation_discrete(
                    a_pol_idx, c_pol, income.Pi, a_grid, beta, gamma
                )
            else:
                V_new = self._policy_evaluation_quadrature(
                    a_pol, c_pol, income, a_grid, beta, gamma
                )

            err = float(np.max(np.abs(V_new - V)))
            if verbose:
                print(f"  HowardGrid iter {it + 1}: max|V_new - V| = {err:.4e}  (tol = {tol:.4e})")
            V = V_new
            if err < tol:
                if verbose:
                    print(f"  HowardGrid converged at iteration {it + 1}.")
                break
        else:
            if verbose:
                print(f"  HowardGrid stopped at max_iter={max_iter}; last err = {err:.4e} (did not converge).")

        return SolvedPolicy(
            policy_a=a_pol,
            policy_c=c_pol,
            a_grid=a_grid,
            z_grid=household.z_grid,
            value=V,
        )
