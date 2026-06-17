"""Probe: what r clears the capital market at xi=0 closed economy?

Runs SOE at xi=0 for a sweep of r values, reports (K_d - A) at each.
The closed-economy r is where K_d = A. If that r ~= -0.043 (slide), the
closed-economy damped iteration was merely slow; if not, there's a real
discrepancy between my map and slide 21 to diagnose.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import time

from hetmacro.models.midrigan_xu import solve_mx_open


COMMON = dict(
    beta=0.96, theta=2.0, alpha=1.0 / 3.0, eta=0.85, delta=0.06,
    rho_z=0.896, sigma_z=0.59,
    na=501, nz=11, amin=1e-2, amax=2500.0, a_curvature=3.0,
    vfi_tol=1e-6, vfi_max_iter=2500, howard_steps=50,
    damping=0.7, tol=1e-6, max_iter=400,
    verbose=False,
)


def main() -> int:
    # Sweep r around the slide target -0.043.
    # Avoid r = -delta = -0.06 where R=0 (degenerate).
    rs = [-0.055, -0.050, -0.043, -0.035, -0.028, -0.020]
    print(f"{'r':>8s}  {'W':>8s}  {'A':>9s}  {'K_d':>9s}  {'K_d-A':>10s}  {'tau':>7s}  "
          f"{'TFP_loss%':>10s}  {'L_d':>7s}")
    print("-" * 85)
    W_init = 1.0
    for r in rs:
        t = time.time()
        ss = solve_mx_open(xi=0.0, r=r, W_init=W_init, **COMMON)
        agg = ss.agg
        dt = time.time() - t
        tfp_loss = 100 * (1 - agg["TFP"] / agg["TFP_star"])
        print(
            f"{r:+8.4f}  {ss.W:8.4f}  {agg['A']:9.4f}  {agg['K']:9.4f}  "
            f"{agg['K'] - agg['A']:+10.4f}  {agg['tau']:7.4f}  "
            f"{tfp_loss:10.3f}  {agg['L_d']:7.5f}  ({dt:.0f}s)"
        )
        W_init = ss.W   # warm start for the next r
    return 0


if __name__ == "__main__":
    sys.exit(main())
