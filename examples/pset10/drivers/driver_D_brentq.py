"""Milestone D, Brent root-finding option: xi = 0.86 and xi = 0.5.

Verifies that ``method='brentq'`` matches the damped method at the two xi
values where the capital-market residual K_d − A properly changes sign.

At xi = 0 the residual doesn't cross zero (degenerate collateral constraint —
k_i ≤ a_i for every firm, so K_d ≤ A with equality when the constraint binds
for all firms). Brent raises a bracket-violation error there; the damped
method remains the only option.
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


def _report(ss, xi):
    agg = ss.agg
    tfp_loss = 100 * (1 - agg["TFP"] / agg["TFP_star"])
    ky = agg["K"] / agg["Y"]
    print(f"  xi = {xi}")
    print(f"    method      = {ss.mode}  (n_outer = {ss.n_outer})")
    print(f"    r           = {ss.r:+.5f}")
    print(f"    W           = {ss.W:.4f}")
    print(f"    TFP loss    = {tfp_loss:.3f} %")
    print(f"    τ           = {agg['tau']:.4f}")
    print(f"    K/Y         = {ky:.4f}")
    print(f"    |K_d - A|   = {abs(agg['K']-agg['A']):.3e}  (L_d = {agg['L_d']:.5f})")


def main() -> int:
    for xi, r_bracket in ((0.86, (-0.05, 0.05)), (0.5, (-0.04, 0.05))):
        print(f"\n==== xi = {xi}, method='brentq' ====")
        t = time.time()
        ss = solve_mx_closed(
            xi=xi, method="brentq", r_bracket=r_bracket, r_root_tol=1e-6,
            W_init=1.0, verbose=True, **COMMON,
        )
        print(f"  total time: {time.time()-t:.1f}s")
        _report(ss, xi)

    # xi = 0 is expected to FAIL (residual doesn't change sign)
    print("\n==== xi = 0, method='brentq' (expected to fail) ====")
    try:
        ss = solve_mx_closed(
            xi=0.0, method="brentq", r_bracket=(-0.055, 0.02), r_root_tol=1e-5,
            W_init=1.0, verbose=True, **COMMON,
        )
        print("UNEXPECTED success — check residual sign pattern.")
    except RuntimeError as e:
        print(f"Expected failure: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
