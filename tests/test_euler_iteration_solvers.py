"""Smoke tests for Euler-equation-based solvers."""

import numpy as np

from hetmacro.grids import make_grid_1d
from hetmacro.household import Household
from hetmacro.income_process import ContinuousQuadratureIncome
from hetmacro.solvers import EGM, HowardEulerIteration, NaiveEulerIteration


def _build_household():
    beta, r, rho, sigma, gamma = 0.95, 0.02, 0.90, 0.20, 1.5
    a_grid = make_grid_1d(0.0, 50.0, 121, spacing="power", power=2.0)
    income = ContinuousQuadratureIncome(rho=rho, sigma_eps=sigma, mu=0.0, n_quad=11)
    return Household(
        income_process=income,
        a_grid=a_grid,
        beta=beta,
        gamma=gamma,
        r=r,
    )


def test_euler_solvers_shape_and_feasibility():
    hh = _build_household()

    naive = NaiveEulerIteration(
        n_a_breaks=41,
        n_z_breaks=13,
        tol=1e-6,
        max_iter=300,
        root_method="broyden",
        verbose=False,
    )
    pol = hh.solve(naive)

    nz, na = hh.z_grid.size, hh.a_grid.size
    assert pol.policy_a.shape == (nz, na)
    assert pol.policy_c.shape == (nz, na)

    resources = hh.resources(hh.a_grid[None, :], hh.z_grid[:, None])
    assert np.all(pol.policy_a >= hh.a_grid[0] - 1e-10)
    assert np.all(pol.policy_a <= hh.a_grid[-1] + 1e-10)
    assert np.all(pol.policy_c > 0.0)
    assert np.all(pol.policy_c <= resources + 1e-8)


def test_euler_solvers_close_to_reference_egm():
    hh = _build_household()
    reference = hh.solve(EGM(tol=1e-8, max_iter=2_000, verbose=False))

    naive = NaiveEulerIteration(
        n_a_breaks=51,
        n_z_breaks=15,
        tol=1e-6,
        max_iter=400,
        root_method="broyden",
        verbose=False,
    )
    howard = HowardEulerIteration(
        n_a_breaks=51,
        n_z_breaks=15,
        tol=1e-6,
        max_iter=200,
        n_warmup=10,
        root_method="broyden",
        verbose=False,
    )

    pol_n = hh.solve(naive)
    pol_h = hh.solve(howard)

    # Smoke-test tolerances: methods should be close to existing policy solver.
    assert np.allclose(pol_n.policy_c, reference.policy_c, atol=5e-2, rtol=5e-2)
    assert np.allclose(pol_h.policy_c, reference.policy_c, atol=5e-2, rtol=5e-2)
    assert np.allclose(pol_n.policy_a, reference.policy_a, atol=1e-1, rtol=5e-2)
    assert np.allclose(pol_h.policy_a, reference.policy_a, atol=1e-1, rtol=5e-2)


def test_howard_requires_fewer_iterations_than_naive():
    hh = _build_household()

    naive = NaiveEulerIteration(
        n_a_breaks=51,
        n_z_breaks=15,
        tol=1e-6,
        max_iter=500,
        root_method="broyden",
        verbose=False,
    )
    howard = HowardEulerIteration(
        n_a_breaks=51,
        n_z_breaks=15,
        tol=1e-6,
        max_iter=200,
        n_warmup=10,
        root_method="broyden",
        verbose=False,
    )

    hh.solve(naive)
    hh.solve(howard)
    assert howard.n_iter_ < naive.n_iter_
