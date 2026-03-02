"""Coefficient-space Howard policy-improvement solver."""

from dataclasses import dataclass

import numpy as np

from ..household import SolvedPolicy
from .collocation_vfi import _CollocationEngine


@dataclass
class HowardImprovement:
    """Howard iteration in coefficient space with spline basis.

    Algorithm (matching PSet 3):
      1. Warmup: run n_warmup VFI steps in coefficient space.
      2. Outer loop:
         - Policy improvement at collocation nodes (golden section).
         - Policy evaluation: solve linear system (Phi - beta * V_theta) theta = u_pol.
         - Check convergence: max|theta_new - theta_old|.

    Parameters
    ----------
    n_a_breaks : int
        Number of breakpoints for the asset spline.
    n_z_breaks : int
        Number of breakpoints for z (continuous branch only).
    a_power : float
        Power spacing for asset breakpoints.
    tol : float
        Convergence tolerance on coefficient change.
    max_iter : int
        Maximum Howard outer iterations.
    n_warmup : int
        Number of VFI warmup steps before switching to Howard.
    policy_tol : float
        Tolerance for golden-section policy improvement.
    verbose : bool
        Print iteration diagnostics.
    """

    n_a_breaks: int = 51
    n_z_breaks: int = 11
    a_power: float = 3.0
    tol: float = 1e-6
    max_iter: int = 120
    n_warmup: int = 2
    policy_tol: float = 1e-8
    verbose: bool = True

    def solve(self, household, **kwargs) -> SolvedPolicy:
        tol = kwargs.get("tol", self.tol)
        max_iter = kwargs.get("max_iter", self.max_iter)
        n_warmup = kwargs.get("n_warmup", self.n_warmup)

        engine = _CollocationEngine(
            household,
            basis="spline",
            n_a=self.n_a_breaks,
            n_z=self.n_z_breaks,
            a_power=self.a_power,
            policy_tol=self.policy_tol,
        )

        theta = engine.initial_theta()

        for w in range(n_warmup):
            theta, warmup_err, _, _ = engine.vfi_step(theta)
            if self.verbose:
                print(f"Howard warmup {w + 1:2d} | err = {warmup_err:.3e}")

        for it in range(1, max_iter + 1):
            a_pol, c_pol, _ = engine.policy_improvement_nodes(theta)
            theta_new, c_eval = engine.howard_evaluate(a_pol, theta)
            err = float(np.max(np.abs(theta_new - theta)))
            theta = theta_new

            if self.verbose and (it <= 10 or it % 10 == 0):
                print(f"Howard iter {it:4d} | err = {err:.3e}")
            if err < tol:
                if self.verbose:
                    print(f"Howard converged at iter {it} (err={err:.3e})")
                break
        else:
            if self.verbose:
                print(f"Howard reached max_iter={max_iter} (err={err:.3e})")

        return engine.evaluate_on_grid(theta, household.a_grid, household.z_grid)
