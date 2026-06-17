"""Milestone D, xi=0 via Brent root-find on r.

Uses the new ``method='brentq'`` option in :func:`solve_mx_closed`. Each
Brent evaluation runs a small open economy at the trial r and reports
K_d − A; Brent iterates until the residual is zero.

Expected runtime ~30-60s total (8 evaluations × 3-7s each).
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
)


def main() -> int:
    print("Solving xi = 0 closed economy via method='brentq'...")
    t = time.time()
    ss = solve_mx_closed(
        xi=0.0,
        method="brentq",
        r_bracket=(-0.055, 0.02),
        r_root_tol=1e-5,
        W_init=1.0,
        verbose=True,
        **COMMON,
    )
    dt = time.time() - t
    print(f"\nDone in {dt:.1f}s")

    agg = ss.agg
    tfp_loss = 100 * (1 - agg["TFP"] / agg["TFP_star"])
    W86 = 1.1239   # from earlier full-grid run
    W_norm = ss.W / W86
    print(f"\n  model:  W = {ss.W:.4f}  (W/W_86 = {W_norm:.4f})")
    print(f"          r = {ss.r:+.5f}")
    print(f"          TFP loss = {tfp_loss:.3f}%")
    print(f"          τ = {agg['tau']:.4f}")
    print(f"          K/Y = {agg['K']/agg['Y']:.3f}")
    print(f"          |K_d - A| = {abs(agg['K']-agg['A']):.3e}  "
          f"(L_d = {agg['L_d']:.5f})")
    print(f"\n  slide:  W = 0.88  r = -0.043  TFP loss = 7.26%  τ = 5.0  K/Y = 3.36")
    return 0


if __name__ == "__main__":
    sys.exit(main())
