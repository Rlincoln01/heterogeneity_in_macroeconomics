"""Milestone D deeper convergence at xi = 0: tight tol, more iters.

Warm-starts from the xi = 0.5 solution, uses damping = 0.80 (lighter than the
earlier 0.92 — enough to stabilize from the warm start, fast enough to reach
the true fixed point), tol = 5e-7, max_iter = 3000.
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
    verbose=True,
)


def main() -> int:
    print("Solving xi = 0.5 first (needed as warm start)")
    t = time.time()
    ss_warm = solve_mx_closed(
        xi=0.5, W_init=1.07, r_init=0.009,
        damping=0.85, tol=1e-5, max_iter=400,
        **COMMON,
    )
    print(f"  xi=0.5 done in {time.time()-t:.1f}s: W={ss_warm.W:.4f}, r={ss_warm.r:+.4f}")

    print("\nSolving xi = 0 with tight tolerance...")
    t = time.time()
    ss0 = solve_mx_closed(
        xi=0.0, W_init=ss_warm.W, r_init=ss_warm.r,
        damping=0.80, tol=5e-7, max_iter=3000,
        **COMMON,
    )
    print(f"  xi=0 done in {time.time()-t:.1f}s")
    print(f"  n_outer = {ss0.n_outer}, resid = {ss0.final_resid:.2e}")

    agg = ss0.agg
    tfp_loss = 100 * (1 - agg["TFP"] / agg["TFP_star"])
    W_86 = 1.1239  # from previous full-grid run at sigma_z=0.59
    W_norm = ss0.W / W_86
    print(f"\n  model:  W = {ss0.W:.4f}  (normalized = {W_norm:.4f})  "
          f"r = {ss0.r:+.5f}  TFP loss = {tfp_loss:.3f}%")
    print(f"          τ = {agg['tau']:.4f}  K/Y = {agg['K']/agg['Y']:.3f}  "
          f"A = {agg['A']:.3f}  K_d = {agg['K']:.3f}")
    print(f"  slide:  W = 0.8800 (normalized)  r = -0.0430  TFP loss = 7.260%  "
          f"τ = 5.0000  K/Y = 3.360")
    return 0


if __name__ == "__main__":
    sys.exit(main())
