"""Milestone D, xi=0 FINAL: Virgiliu setup + max_iter=3500 + save result.

Extends the 2500-iter run (which reached r=-0.0412) to 3500 iters so the
drift to slide 21's r=-0.043 can complete. Pickles the resulting
``MXSteadyState`` to ``examples/pset10/notebooks/cache/ss_closed_xi0.pkl``
so notebook simulations can reuse it without resolving.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hetmacro.models.midrigan_xu import (
    save_steady_state,
    solve_mx_closed,
)


def main() -> int:
    cache_path = _HERE.parents[1] / "notebooks" / "cache" / "ss_closed_xi0.pkl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    t = time.time()
    ss = solve_mx_closed(
        xi=0.0, method="damped",
        damping=0.5,            # Virgiliu's step = 1/2
        W_init=1.0, r_init=0.02,
        tol=1e-7, max_iter=3500,

        beta=0.96, theta=2.0, alpha=1.0 / 3.0, eta=0.85, delta=0.06,
        rho_z=0.896, sigma_z=0.59,
        na=501, nz=11, amin=1e-2, amax=2500.0, a_curvature=3.0,
        vfi_tol=1e-6, vfi_max_iter=2500, howard_steps=50,
        verbose=True,
    )
    print(f"\nTotal wall time: {time.time()-t:.1f}s")

    agg = ss.agg
    tfp_loss = 100 * (1 - agg["TFP"] / agg["TFP_star"])
    W86 = 1.1240
    print(f"\n  model:  W = {ss.W:.4f}  (W/W_86 = {ss.W/W86:.4f})")
    print(f"          r = {ss.r:+.5f}")
    print(f"          TFP loss = {tfp_loss:.3f}%")
    print(f"          τ = {agg['tau']:.4f}   K/Y = {agg['K']/agg['Y']:.3f}")
    print(f"          |K_d - A| = {abs(agg['K']-agg['A']):.3e}  "
          f"L_d = {agg['L_d']:.5f}")
    print(f"  slide:  W = 0.88, r = -0.043, TFP loss = 7.26%, τ = 5.0, K/Y = 3.36")

    save_steady_state(ss, cache_path)
    print(f"\nSaved: {cache_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
