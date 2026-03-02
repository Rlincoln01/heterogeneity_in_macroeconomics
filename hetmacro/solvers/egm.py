"""Endogenous-grid method solver."""

from dataclasses import dataclass, field

import numpy as np

from ..dp_tools import solve_euler_one_asset
from ..household import SolvedPolicy, resolve_expectation_mode
from ..utils import crra_marginal


@dataclass
class EGM:
    """One-asset EGM solver.

    Parameters
    ----------
    tol : float
        Convergence tolerance on max|c_new - c_old|.
    max_iter : int
        Maximum EGM iterations.
    expectation : str
        ``"auto"`` (default), ``"discrete"``, or ``"quadrature"``.
    verbose : bool
        If True, print per-iteration max|c_new - c_old| and convergence status.
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
        a_grid = household.a_grid
        beta = household.beta
        gamma = household.gamma
        r = household.r
        income = household.income_process

        budget = y[:, None] + (1.0 + r) * a_grid[None, :]
        c_old = budget.copy()

        for it in range(max_iter):
            if mode == "discrete":
                emuc = income.Pi @ crra_marginal(c_old, gamma)
            else:
                emuc = income.expected(crra_marginal(c_old, gamma))
            c_foc = (beta * (1.0 + r) * emuc) ** (-1.0 / gamma)
            c_new, a_pol, _, _ = solve_euler_one_asset(c_foc, budget, a_grid)
            err = float(np.max(np.abs(c_new - c_old)))
            if verbose:
                print(f"  EGM iter {it + 1}: max|c_new - c_old| = {err:.4e}  (tol = {tol:.4e})")
            if err < tol:
                c_old = c_new
                if verbose:
                    print(f"  EGM converged at iteration {it + 1}.")
                break
            c_old = c_new
        else:
            if verbose:
                print(f"  EGM stopped at max_iter={max_iter}; last err = {err:.4e} (did not converge).")

        return SolvedPolicy(
            policy_a=a_pol,
            policy_c=c_old,
            a_grid=a_grid,
            z_grid=household.z_grid,
            value=None,
        )
