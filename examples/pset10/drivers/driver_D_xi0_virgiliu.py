"""Milestone D, xi=0: replicate start.m exactly.

Virgiliu's start.m uses:
  - step = 1/2 in ``W = W + step * (Wnew - W)``, which in our convention
    (damping = weight on OLD) corresponds to ``damping = 0.5``.
  - Fresh start: ``W = 1.0, r = 0.02``. No warm start.
  - na = 501, nz = 11.
  - tol on ``norm(parnew - parold)`` of 1e-7.
  - max_iter = 2500.

We hope this matches slide 21's r = -0.043 at xi=0.
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


def main() -> int:
    t = time.time()
    ss = solve_mx_closed(
        xi=0.0,
        method="damped",
        damping=0.5,         # Virgiliu's step = 1/2
        W_init=1.0,          # start.m line 20
        r_init=0.02,         # start.m line 19
        tol=1e-7,            # start.m line 78
        max_iter=2500,       # start.m line 57

        beta=0.96, theta=2.0, alpha=1.0 / 3.0, eta=0.85, delta=0.06,
        rho_z=0.896, sigma_z=0.59,
        na=501, nz=11, amin=1e-2, amax=2500.0, a_curvature=3.0,
        vfi_tol=1e-6, vfi_max_iter=2500, howard_steps=50,
        verbose=True,
    )
    dt = time.time() - t
    print(f"\nDone in {dt:.1f}s (n_outer = {ss.n_outer}, resid = {ss.final_resid:.2e})")

    agg = ss.agg
    tfp_loss = 100 * (1 - agg["TFP"] / agg["TFP_star"])
    W86 = 1.1240  # baseline
    print(f"\n  model:  W = {ss.W:.4f}  (W/W_86 = {ss.W/W86:.4f})")
    print(f"          r = {ss.r:+.5f}")
    print(f"          TFP loss = {tfp_loss:.3f}%")
    print(f"          τ = {agg['tau']:.4f}   K/Y = {agg['K']/agg['Y']:.3f}")
    print(f"          |K_d - A| = {abs(agg['K']-agg['A']):.3e}  "
          f"L_d = {agg['L_d']:.5f}")
    print(f"  slide:  W = 0.88 (norm), r = -0.043, TFP loss = 7.26%, τ = 5.0, K/Y = 3.36")
    return 0


if __name__ == "__main__":
    sys.exit(main())
