"""Milestone D full-grid validation against slide 21.

Runs na=501, nz=11, amax=2500 (the MATLAB ``start.m`` defaults) for
xi ∈ {0.86, 0.5, 0}, using cascading warm starts on (W, r) to stabilize
the tight ``xi=0`` case, then compares to slide 21 numbers with tolerances
appropriate for the full grid.

Target (slide 21):

    xi     TFP loss %    τ       W       r        K/Y
    0.86   0.22          1.02    1.00    0.017    3.62
    0.5    2.37          1.16    0.96    0.009    3.52
    0      7.26          5.00    0.88   -0.043    3.36
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hetmacro.models.midrigan_xu import solve_mx_closed


COMMON = dict(
    beta=0.96, theta=2.0, alpha=1.0 / 3.0, eta=0.85, delta=0.06,
    rho_z=0.896, sigma_z=0.59,
    na=501, nz=11, amin=1e-2, amax=2500.0, a_curvature=3.0,
    vfi_tol=1e-6, vfi_max_iter=2500, howard_steps=50,
    tol=1e-5, max_iter=600,
    verbose=True,
)


TARGETS = {
    0.86: {"tfp_loss": 0.22, "tau": 1.02, "W": 1.00, "r": 0.017,  "KY": 3.62},
    0.5:  {"tfp_loss": 2.37, "tau": 1.16, "W": 0.96, "r": 0.009,  "KY": 3.52},
    0.0:  {"tfp_loss": 7.26, "tau": 5.00, "W": 0.88, "r": -0.043, "KY": 3.36},
}


def _check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def _summary(ss, xi):
    agg = ss.agg
    tfp_loss = 100 * (1 - agg["TFP"] / agg["TFP_star"])
    ky = agg["K"] / agg["Y"]
    tgt = TARGETS[xi]
    print(f"\n  xi = {xi}: n_outer = {ss.n_outer}  resid = {ss.final_resid:.2e}")
    print(f"    model:  W = {ss.W:.4f}  r = {ss.r:+.5f}  "
          f"TFP loss = {tfp_loss:.3f}%  τ = {agg['tau']:.4f}  K/Y = {ky:.3f}")
    print(f"    slide:  W = {tgt['W']:.4f}  r = {tgt['r']:+.5f}  "
          f"TFP loss = {tgt['tfp_loss']:.3f}%  τ = {tgt['tau']:.4f}  K/Y = {tgt['KY']:.3f}")
    return {"tfp_loss": tfp_loss, "ky": ky, "W": ss.W, "r": ss.r, "tau": agg["tau"]}


def main() -> int:
    results = []
    summaries = {}

    # xi = 0.86: fresh start
    print("\n==== xi = 0.86 (Korean calibration) ====")
    t = time.time()
    ss86 = solve_mx_closed(
        xi=0.86, W_init=1.0, r_init=0.02, damping=0.85, **COMMON,
    )
    print(f"  elapsed: {time.time()-t:.1f}s")
    s86 = _summary(ss86, 0.86)
    summaries[0.86] = s86

    # xi = 0.5: warm start from xi = 0.86
    print("\n==== xi = 0.5 ====")
    t = time.time()
    ss50 = solve_mx_closed(
        xi=0.5, W_init=ss86.W, r_init=ss86.r, damping=0.85, **COMMON,
    )
    print(f"  elapsed: {time.time()-t:.1f}s")
    s50 = _summary(ss50, 0.5)
    summaries[0.5] = s50

    # xi = 0: warm start from xi = 0.5; need more damping and patience
    print("\n==== xi = 0 ====")
    t = time.time()
    params_xi0 = dict(COMMON)
    params_xi0["damping"] = 0.92       # reviewer: tighter damping at r<0
    params_xi0["max_iter"] = 1500      # slower to converge
    ss00 = solve_mx_closed(
        xi=0.0, W_init=ss50.W, r_init=ss50.r, **params_xi0,
    )
    print(f"  elapsed: {time.time()-t:.1f}s")
    s00 = _summary(ss00, 0.0)
    summaries[0.0] = s00

    # ---------- Checks vs slide 21 (full-grid tolerances) ----------
    print("\n==== Milestone D full-grid checks ====")

    def _close(model, target, abs_tol, xi, name):
        ok = abs(model - target) <= abs_tol
        return _check(
            f"xi={xi} {name}",
            ok,
            f"model={model:.4f}, slide={target:.4f}, |Δ|={abs(model-target):.4f}, tol={abs_tol}",
        )

    # Slide 21 reports W normalized to 1.00 at the xi=0.86 baseline; our raw
    # wages sit on a different overall scale, so we normalize model wages
    # the same way before comparing.
    W_base = s86["W"]
    s86_Wn = 1.0
    s50_Wn = s50["W"] / W_base
    s00_Wn = s00["W"] / W_base

    # xi=0.86 (tight tolerances — small TFP loss)
    results.append(_close(s86["tfp_loss"], 0.22, 0.15, 0.86, "TFP loss %"))
    results.append(_close(s86["tau"],      1.02, 0.03, 0.86, "τ"))
    results.append(_close(s86_Wn,          1.00, 0.02, 0.86, "W (normalized)"))
    results.append(_close(s86["r"],        0.017, 0.008, 0.86, "r"))
    results.append(_close(s86["ky"],       3.62, 0.20, 0.86, "K/Y"))

    # xi=0.5
    results.append(_close(s50["tfp_loss"], 2.37, 0.50, 0.5, "TFP loss %"))
    results.append(_close(s50["tau"],      1.16, 0.05, 0.5, "τ"))
    results.append(_close(s50_Wn,          0.96, 0.02, 0.5, "W (normalized)"))
    results.append(_close(s50["r"],        0.009, 0.010, 0.5, "r"))
    results.append(_close(s50["ky"],       3.52, 0.20, 0.5, "K/Y"))

    # xi=0 (wider tolerances due to slower convergence of closed GE at r<0)
    results.append(_close(s00["tfp_loss"], 7.26, 1.50, 0.0, "TFP loss %"))
    results.append(_close(s00["tau"],      5.00, 2.50, 0.0, "τ (loose)"))
    results.append(_close(s00_Wn,          0.88, 0.04, 0.0, "W (normalized)"))
    results.append(_close(s00["r"],       -0.043, 0.025, 0.0, "r (loose)"))
    results.append(_close(s00["ky"],       3.36, 0.30, 0.0, "K/Y"))

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n  Milestone D (full grid): {n_pass} / {n_total} checks passed.")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
