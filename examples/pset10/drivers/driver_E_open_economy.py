"""Milestone E: small open economy at fixed r = 0.017, varying xi.

Target (slide 22):

    xi     TFP loss %    τ       W       K/Y     A/Y
    0.86   0.22          1.02    1.00    3.62    3.62
    0.5    1.71          1.11    0.95    3.30    4.19
    0      3.43          1.25    0.88    2.95    4.89

Tolerances follow slide 21's pattern (tighter for τ; slightly looser for r
equivalents since W is the only free price).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hetmacro.models.midrigan_xu import solve_mx_open


R_FIXED = 0.017

COMMON = dict(
    beta=0.96, theta=2.0, alpha=1.0 / 3.0, eta=0.85, delta=0.06,
    rho_z=0.896, sigma_z=0.59,
    na=501, nz=11, amin=1e-2, amax=2500.0, a_curvature=3.0,
    vfi_tol=1e-6, vfi_max_iter=2500, howard_steps=50,
    r=R_FIXED,
    damping=0.7, tol=1e-5, max_iter=600,
    verbose=True,
)


TARGETS = {
    0.86: {"tfp_loss": 0.22, "tau": 1.02, "W": 1.00, "KY": 3.62, "AY": 3.62},
    0.5:  {"tfp_loss": 1.71, "tau": 1.11, "W": 0.95, "KY": 3.30, "AY": 4.19},
    0.0:  {"tfp_loss": 3.43, "tau": 1.25, "W": 0.88, "KY": 2.95, "AY": 4.89},
}


def _check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def _summary(ss, xi):
    agg = ss.agg
    tfp_loss = 100 * (1 - agg["TFP"] / agg["TFP_star"])
    ky = agg["K"] / agg["Y"]
    ay = agg["A"] / agg["Y"]
    tgt = TARGETS[xi]
    print(f"\n  xi = {xi}: n_outer = {ss.n_outer}  resid = {ss.final_resid:.2e}")
    print(f"    model:  W = {ss.W:.4f}  TFP loss = {tfp_loss:.3f}%  "
          f"τ = {agg['tau']:.4f}  K/Y = {ky:.3f}  A/Y = {ay:.3f}")
    print(f"    slide:  W = {tgt['W']:.4f}  TFP loss = {tgt['tfp_loss']:.3f}%  "
          f"τ = {tgt['tau']:.4f}  K/Y = {tgt['KY']:.3f}  A/Y = {tgt['AY']:.3f}")
    return {"tfp_loss": tfp_loss, "ky": ky, "ay": ay, "W": ss.W, "tau": agg["tau"]}


def main() -> int:
    results = []
    summaries = {}

    print("\n==== xi = 0.86 (SOE, r = 0.017) ====")
    t = time.time()
    ss86 = solve_mx_open(xi=0.86, W_init=1.0, **COMMON)
    print(f"  elapsed: {time.time()-t:.1f}s")
    s86 = _summary(ss86, 0.86)
    summaries[0.86] = s86

    print("\n==== xi = 0.5 (SOE) ====")
    t = time.time()
    ss50 = solve_mx_open(xi=0.5, W_init=ss86.W, **COMMON)
    print(f"  elapsed: {time.time()-t:.1f}s")
    s50 = _summary(ss50, 0.5)
    summaries[0.5] = s50

    print("\n==== xi = 0 (SOE) ====")
    t = time.time()
    ss00 = solve_mx_open(xi=0.0, W_init=ss50.W, **COMMON)
    print(f"  elapsed: {time.time()-t:.1f}s")
    s00 = _summary(ss00, 0.0)
    summaries[0.0] = s00

    # ---------- Checks vs slide 22 ----------
    print("\n==== Milestone E full-grid checks ====")

    def _close(model, target, abs_tol, xi, name):
        ok = abs(model - target) <= abs_tol
        return _check(
            f"xi={xi} {name}",
            ok,
            f"model={model:.4f}, slide={target:.4f}, |Δ|={abs(model-target):.4f}, tol={abs_tol}",
        )

    # Slide 22 normalizes W to 1.00 at xi = 0.86 baseline; our raw wages
    # are on a different overall scale, so we divide by the xi=0.86 wage
    # before comparing.
    W_base = summaries[0.86]["W"]
    for xi, s in summaries.items():
        t = TARGETS[xi]
        Wn = s["W"] / W_base
        results.append(_close(s["tfp_loss"], t["tfp_loss"], 0.5, xi, "TFP loss %"))
        results.append(_close(s["tau"],      t["tau"],      0.08, xi, "τ"))
        results.append(_close(Wn,            t["W"],        0.02, xi, "W (normalized)"))
        results.append(_close(s["ky"],       t["KY"],       0.25, xi, "K/Y"))
        results.append(_close(s["ay"],       t["AY"],       0.30, xi, "A/Y"))

    # Comparative economic logic: TFP loss in open < TFP loss in closed at same xi
    # (skipped here, verified in the notebook by running both)

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n  Milestone E (full grid): {n_pass} / {n_total} checks passed.")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
