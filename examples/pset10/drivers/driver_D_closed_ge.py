"""Milestone D driver: closed-economy GE at xi = 0.86 (Korean calibration).

Small-grid sanity run (na=101, nz=7, amax=500) intended to verify the GE
loop converges and lands near slide 21 column 1 numbers. The full na=501
run belongs in the notebook; here we just confirm the machinery works.

Target (slide 21, Korea ksi=0.86):
  TFP loss ≈ 0.22%, τ ≈ 1.02, W ≈ 1.00, r ≈ 0.017, K/Y ≈ 3.62

Tolerances are loose because of the coarse grid (slide values come from
na=501); we only insist on correct sign of comparative statics.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from hetmacro.models.midrigan_xu import solve_mx_closed


PARAMS = dict(
    beta=0.96, theta=2.0, alpha=1.0 / 3.0, eta=0.85, delta=0.06,
    rho_z=0.896, sigma_z=0.40,
    na=101, nz=7, amin=1e-2, amax=500.0, a_curvature=3.0,
    W_init=1.0, r_init=0.02,
    damping=0.85, tol=1e-5, max_iter=400,
    vfi_tol=1e-6, vfi_max_iter=2500, howard_steps=50,
    verbose=True,
)


def _check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main() -> int:
    print("\n==== xi = 0.86 (Korean calibration) ====")
    t0 = time.time()
    ss86 = solve_mx_closed(xi=0.86, **PARAMS)
    dt86 = time.time() - t0
    a86 = ss86.agg
    tfp_loss86 = 100 * (1 - a86["TFP"] / a86["TFP_star"])
    ky86 = a86["K"] / a86["Y"]
    print(f"\n  converged in {ss86.n_outer} iters, "
          f"resid = {ss86.final_resid:.2e}, {dt86:.1f}s")
    print(f"  W = {ss86.W:.4f}   r = {ss86.r:+.4f}")
    print(f"  TFP loss = {tfp_loss86:.3f}%   τ = {a86['tau']:.4f}   K/Y = {ky86:.3f}")

    # Also compute xi = 0.5 for comparative statics sign
    print("\n==== xi = 0.5 ====")
    t1 = time.time()
    ss50 = solve_mx_closed(xi=0.5, **PARAMS)
    dt50 = time.time() - t1
    a50 = ss50.agg
    tfp_loss50 = 100 * (1 - a50["TFP"] / a50["TFP_star"])
    ky50 = a50["K"] / a50["Y"]
    print(f"\n  converged in {ss50.n_outer} iters, "
          f"resid = {ss50.final_resid:.2e}, {dt50:.1f}s")
    print(f"  W = {ss50.W:.4f}   r = {ss50.r:+.4f}")
    print(f"  TFP loss = {tfp_loss50:.3f}%   τ = {a50['tau']:.4f}   K/Y = {ky50:.3f}")

    # ---------- Checks --------------------------------------------------
    print("\n==== Milestone D checks ====")
    results = []

    # 1. Outer loop converged
    results.append(_check(
        "xi=0.86 outer loop converged",
        ss86.final_resid < 1e-5 and ss86.n_outer < 300,
        f"n_outer = {ss86.n_outer}",
    ))
    results.append(_check(
        "xi=0.5 outer loop converged",
        ss50.final_resid < 1e-5 and ss50.n_outer < 300,
        f"n_outer = {ss50.n_outer}",
    ))

    # 2. Market clearing at xi=0.86: K_d should equal A at equilibrium
    mc_err = abs(a86["K"] - a86["A"]) / max(abs(a86["A"]), 1e-12)
    results.append(_check(
        "closed-econ clearing A == K_demand at xi=0.86",
        mc_err < 5e-3,
        f"A = {a86['A']:.3f}, K_d = {a86['K']:.3f}, rel = {mc_err:.2e}",
    ))
    mc_err50 = abs(a50["K"] - a50["A"]) / max(abs(a50["A"]), 1e-12)
    results.append(_check(
        "closed-econ clearing A == K_demand at xi=0.5",
        mc_err50 < 5e-3,
        f"A = {a50['A']:.3f}, K_d = {a50['K']:.3f}, rel = {mc_err50:.2e}",
    ))

    # 3. Labor clearing: L_d = 1
    results.append(_check(
        "labor clearing L_d = 1 at xi=0.86",
        abs(a86["L_d"] - 1.0) < 5e-3,
        f"L_d = {a86['L_d']:.5f}",
    ))

    # 4. Comparative statics: tighter constraint -> more TFP loss, larger τ, lower W, lower r, lower K/Y
    results.append(_check(
        "TFP loss increases as xi falls (0.86 -> 0.5)",
        tfp_loss50 > tfp_loss86,
        f"{tfp_loss86:.3f}% -> {tfp_loss50:.3f}%",
    ))
    results.append(_check(
        "aggregate wedge increases as xi falls",
        a50["tau"] > a86["tau"],
        f"{a86['tau']:.4f} -> {a50['tau']:.4f}",
    ))
    results.append(_check(
        "wage falls as xi falls",
        ss50.W < ss86.W,
        f"{ss86.W:.4f} -> {ss50.W:.4f}",
    ))
    results.append(_check(
        "interest rate falls as xi falls",
        ss50.r < ss86.r,
        f"{ss86.r:+.4f} -> {ss50.r:+.4f}",
    ))
    results.append(_check(
        "K/Y falls as xi falls",
        ky50 < ky86,
        f"{ky86:.3f} -> {ky50:.3f}",
    ))

    # 5. Slide-21 order of magnitude (loose, given coarse grid)
    results.append(_check(
        "xi=0.86 TFP loss in [0, 1]%",
        0.0 <= tfp_loss86 < 1.0,
        f"{tfp_loss86:.3f}%",
    ))
    results.append(_check(
        "xi=0.86 wage in [0.95, 1.05]",
        0.95 <= ss86.W <= 1.05,
        f"W = {ss86.W:.4f}",
    ))
    results.append(_check(
        "xi=0.86 r in [0.01, 0.03]",
        0.010 <= ss86.r <= 0.030,
        f"r = {ss86.r:+.4f}",
    ))

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n  Milestone D: {n_pass} / {n_total} checks passed.")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
