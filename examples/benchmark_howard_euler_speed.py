"""Benchmark NaiveEulerIteration vs HowardEulerIteration.

Run:
    python examples/benchmark_howard_euler_speed.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hetmacro.grids import make_grid_1d
from hetmacro.household import Household
from hetmacro.income_process import ContinuousQuadratureIncome
from hetmacro.solvers import EGM, HowardEulerIteration, NaiveEulerIteration


def build_household():
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


def main():
    hh = build_household()
    print("Calibration: beta=0.95, r=0.02, rho=0.9, sigma=0.2, gamma=1.5")
    print("Grid: a=121 eval points, spline collocation: n_a=51, n_z=15\n")

    t0 = time.perf_counter()
    ref = hh.solve(EGM(tol=1e-8, max_iter=2_000, verbose=False))
    print(f"EGM reference time: {time.perf_counter() - t0:.3f}s")

    naive = NaiveEulerIteration(
        n_a_breaks=51,
        n_z_breaks=15,
        tol=1e-6,
        max_iter=400,
        root_method="broyden",
        verbose=False,
    )
    t0 = time.perf_counter()
    pol_n = hh.solve(naive)
    t_naive = time.perf_counter() - t0

    howard = HowardEulerIteration(
        n_a_breaks=51,
        n_z_breaks=15,
        tol=1e-6,
        max_iter=200,
        n_warmup=10,
        root_method="broyden",
        verbose=False,
    )
    t0 = time.perf_counter()
    pol_h = hh.solve(howard)
    t_howard = time.perf_counter() - t0

    dc_n = np.abs(pol_n.policy_c - ref.policy_c)
    dc_h = np.abs(pol_h.policy_c - ref.policy_c)
    da_n = np.abs(pol_n.policy_a - ref.policy_a)
    da_h = np.abs(pol_h.policy_a - ref.policy_a)

    print("\nRuntime and convergence")
    print(f"{'solver':<12} {'time(s)':>10} {'iters':>8} {'last_err':>12}")
    print(
        f"{'NaiveEuler':<12} {t_naive:10.3f} {naive.n_iter_:8d} {naive.last_error_:12.3e}"
    )
    print(
        f"{'HowardEuler':<12} {t_howard:10.3f} {howard.n_iter_:8d} {howard.last_error_:12.3e}"
    )

    print("\nPolicy distance vs EGM")
    print(f"{'solver':<12} {'max|dc|':>12} {'mean|dc|':>12} {'max|da|':>12} {'mean|da|':>12}")
    print(
        f"{'NaiveEuler':<12} {dc_n.max():12.3e} {dc_n.mean():12.3e} {da_n.max():12.3e} {da_n.mean():12.3e}"
    )
    print(
        f"{'HowardEuler':<12} {dc_h.max():12.3e} {dc_h.mean():12.3e} {da_h.max():12.3e} {da_h.mean():12.3e}"
    )

    print("\nInternal timing breakdown")
    print("NaiveEuler timings:", naive.timings_)
    print("HowardEuler timings:", howard.timings_)


if __name__ == "__main__":
    main()
