"""Grid-based value function iteration solver."""

from dataclasses import dataclass

import numpy as np

from ..household import SolvedPolicy, resolve_expectation_mode
from ..utils import crra_utility


@dataclass
class GridVFI:
    """Vanilla VFI on an asset grid.

    Parameters
    ----------
    tol : float
        Convergence tolerance on max|V_new - V|.
    max_iter : int
        Maximum VFI iterations.
    expectation : str
        ``"auto"`` (default), ``"discrete"``, or ``"quadrature"``.
    verbose : bool
        If True, print per-iteration max|V_new - V| and convergence status.
    """

    tol: float = 1e-8
    max_iter: int = 2_000
    expectation: str = "auto"
    verbose: bool = False

    def solve(self, household, **kwargs) -> SolvedPolicy:
        tol = kwargs.get("tol", self.tol)
        max_iter = kwargs.get("max_iter", self.max_iter)
        verbose = kwargs.get("verbose", self.verbose)
        mode = resolve_expectation_mode(household, self.expectation)

        y = household.y_grid
        income = household.income_process
        a_grid = household.a_grid
        beta = household.beta
        gamma = household.gamma
        r = household.r

        nz, na = y.size, a_grid.size
        V = np.zeros((nz, na))
        a_pol = np.empty_like(V)
        c_pol = np.empty_like(V)

        for it in range(max_iter):
            if mode == "discrete":
                EV = income.Pi @ V
            else:
                EV = income.expected(V)

            V_new = np.empty_like(V)
            for iz in range(nz):
                for ia, a in enumerate(a_grid):
                    c = y[iz] + (1.0 + r) * a - a_grid
                    util = np.where(c > 0, crra_utility(c, gamma), -1e18)
                    rhs = util + beta * EV[iz, :]
                    j = int(np.argmax(rhs))
                    V_new[iz, ia] = rhs[j]
                    a_pol[iz, ia] = a_grid[j]
                    c_pol[iz, ia] = c[j]

            err = float(np.max(np.abs(V_new - V)))
            if verbose:
                print(f"  GridVFI iter {it + 1}: max|V_new - V| = {err:.4e}  (tol = {tol:.4e})")
            if err < tol:
                V = V_new
                if verbose:
                    print(f"  GridVFI converged at iteration {it + 1}.")
                break
            V = V_new
        else:
            if verbose:
                print(f"  GridVFI stopped at max_iter={max_iter}; last err = {err:.4e} (did not converge).")

        return SolvedPolicy(
            policy_a=a_pol,
            policy_c=c_pol,
            a_grid=a_grid,
            z_grid=household.z_grid,
            value=V,
        )
