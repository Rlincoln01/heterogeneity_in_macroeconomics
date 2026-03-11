"""Howard policy improvement with continuous asset choice.

Combines golden-section policy improvement (continuous a') with exact
policy evaluation via a linear system solve.  Lottery weights split
mass between adjacent grid points for the transition matrix.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..household import SolvedPolicy, resolve_expectation_mode
from ..interpolation import get_lottery, linear_basis_matrix, tensor_basis_matrix
from ..optimize import golden_max_vec
from ..utils import crra_utility


@dataclass
class HowardGrid:
    """Howard iteration with continuous asset choice on a fixed grid.

    Uses golden-section search for policy improvement (continuous a')
    and exact linear-system policy evaluation with lottery-based
    transitions between adjacent grid points.

    Parameters
    ----------
    tol : float
        Convergence tolerance on max|V_new - V|.
    max_iter : int
        Maximum Howard outer iterations.
    policy_tol : float
        Tolerance for the golden section policy improvement.
    c_floor : float
        Minimum consumption to avoid log(0).
    expectation : str
        ``"auto"`` (default), ``"discrete"``, or ``"quadrature"``.
    verbose : bool
        If True, print per-iteration max|V_new - V| and convergence status.
    """

    tol: float = 1e-8
    max_iter: int = 200
    policy_tol: float = 1e-8
    c_floor: float = 1e-10
    expectation: str = "auto"
    verbose: bool = False

    # ------------------------------------------------------------------
    # Policy improvement: golden section with interpolated continuation
    # ------------------------------------------------------------------

    @staticmethod
    def _expected_continuation_discrete(a_prime, V, a_grid, pi_row):
        """E[V(a', z')|z] with linear interpolation in assets, discrete z."""
        nz = V.shape[0]
        cont = np.zeros_like(a_prime, dtype=float)
        for izp in range(nz):
            cont += pi_row[izp] * np.interp(a_prime, a_grid, V[izp, :])
        return cont

    def _policy_improvement_discrete(self, V, y, Pi, a_grid, beta, gamma, r):
        nz, na = V.shape
        amin, amax = float(a_grid[0]), float(a_grid[-1])
        a_pol = np.empty((nz, na), dtype=float)
        c_pol = np.empty((nz, na), dtype=float)

        for iz in range(nz):
            m = y[iz] + (1.0 + r) * a_grid
            lo = np.full(na, amin, dtype=float)
            hi = np.minimum(amax, m - self.c_floor)
            hi = np.maximum(hi, lo + 1e-12)

            def obj(a_prime, _iz=iz, _m=m):
                a_prime = np.clip(a_prime, amin, amax)
                c = np.maximum(_m - a_prime, self.c_floor)
                cont = self._expected_continuation_discrete(a_prime, V, a_grid, Pi[_iz, :])
                return crra_utility(c, gamma) + beta * cont

            a_star, v_star = golden_max_vec(obj, lo, hi, tol=self.policy_tol)
            a_pol[iz, :] = a_star
            c_pol[iz, :] = np.maximum(m - a_star, self.c_floor)

        return a_pol, c_pol

    def _policy_improvement_quadrature(self, V, y, income, a_grid, beta, gamma, r):
        EV = income.expected(V)
        nz, na = V.shape
        amin, amax = float(a_grid[0]), float(a_grid[-1])
        a_pol = np.empty((nz, na), dtype=float)
        c_pol = np.empty((nz, na), dtype=float)

        for iz in range(nz):
            m = y[iz] + (1.0 + r) * a_grid
            lo = np.full(na, amin, dtype=float)
            hi = np.minimum(amax, m - self.c_floor)
            hi = np.maximum(hi, lo + 1e-12)

            def obj(a_prime, _iz=iz, _m=m):
                a_prime = np.clip(a_prime, amin, amax)
                c = np.maximum(_m - a_prime, self.c_floor)
                cont = np.interp(a_prime, a_grid, EV[_iz, :])
                return crra_utility(c, gamma) + beta * cont

            a_star, _ = golden_max_vec(obj, lo, hi, tol=self.policy_tol)
            a_pol[iz, :] = a_star
            c_pol[iz, :] = np.maximum(m - a_star, self.c_floor)

        return a_pol, c_pol

    # ------------------------------------------------------------------
    # Policy evaluation: exact linear system solve with lottery weights
    # ------------------------------------------------------------------

    @staticmethod
    def _policy_evaluation_discrete(a_pol, c_pol, Pi, a_grid, beta, gamma, c_floor):
        nz, na = a_pol.shape
        n = nz * na

        a_idx, a_w = get_lottery(a_pol, a_grid)
        utility = crra_utility(np.maximum(c_pol, c_floor), gamma).reshape(-1)

        rows, cols, data = [], [], []
        for iz in range(nz):
            for ia in range(na):
                row = iz * na + ia
                j = int(a_idx[iz, ia])
                w = float(a_w[iz, ia])
                for izp in range(nz):
                    p = float(Pi[iz, izp])
                    if p == 0.0:
                        continue
                    cols.extend([izp * na + j, izp * na + min(j + 1, na - 1)])
                    rows.extend([row, row])
                    data.extend([p * w, p * (1.0 - w)])

        T = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        lhs = sparse.eye(n, format="csr") - beta * T
        v = spsolve(lhs.tocsc(), utility)
        return np.asarray(v).reshape(nz, na)

    @staticmethod
    def _policy_evaluation_quadrature(a_pol, c_pol, income, a_grid, beta, gamma, c_floor):
        nz, na = a_pol.shape
        N = nz * na
        z_grid = income.z_grid
        zmin, zmax = float(z_grid[0]), float(z_grid[-1])

        a_pol_flat = a_pol.reshape(-1)
        Z_flat = np.repeat(z_grid, na)
        utility = crra_utility(np.maximum(c_pol, c_floor), gamma).reshape(-1)

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
        policy_tol = kwargs.get("policy_tol", self.policy_tol)
        c_floor = kwargs.get("c_floor", self.c_floor)
        verbose = kwargs.get("verbose", self.verbose)
        self.policy_tol = float(policy_tol)
        self.c_floor = float(c_floor)

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
            # Policy improvement (golden section, continuous a')
            if mode == "discrete":
                EV = income.Pi @ V
                a_pol, c_pol = self._policy_improvement_discrete(
                    V, y, income.Pi, a_grid, beta, gamma, r
                )
            else:
                a_pol, c_pol = self._policy_improvement_quadrature(
                    V, y, income, a_grid, beta, gamma, r
                )

            # Exact policy evaluation (linear system with lottery transitions)
            if mode == "discrete":
                V_new = self._policy_evaluation_discrete(
                    a_pol, c_pol, income.Pi, a_grid, beta, gamma, self.c_floor
                )
            else:
                V_new = self._policy_evaluation_quadrature(
                    a_pol, c_pol, income, a_grid, beta, gamma, self.c_floor
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
