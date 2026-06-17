"""Milestone C driver: verify ``aggregate_mx`` accounting and TFP identities.

Runs VFI once at Korean calibration (xi = 0.86), computes the stationary
distribution, and checks:

1. Distribution sums to 1 and is nonnegative.
2. Savings-assets consistency: Σ D · a == Σ D · a_policy.
3. Goods market: Y - W·L_d - R·K == Σ D · π.
4. Wedge τ ≥ 1 (with equality only if no firm binds).
5. TFP ≤ TFP* (strict when any firm binds).
6. Wealth Gini in [0.4, 0.8] (basic distribution sanity).

No reference numbers (they come at Milestone D); this is accounting only.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from hetmacro.grids import make_asset_grid
from hetmacro.markov import rouwenhorst
from hetmacro.models.midrigan_xu import (
    aggregate_mx,
    entrepreneur_vfi,
    firm_static,
)


BETA = 0.96
THETA = 2.0
ALPHA = 1.0 / 3.0
ETA = 0.85
DELTA = 0.06
RHO_Z = 0.896
SIGMA_Z = 0.59
XI = 0.86
LAM = 1.0 / (1.0 - XI)

R_NET = 0.02
W = 1.0
R_USER = R_NET + DELTA

NA = 101
NZ = 7
AMIN, AMAX = 1e-2, 500.0


def _check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def _gini(a: np.ndarray, w: np.ndarray) -> float:
    """Weighted Gini of ``a`` with nonnegative weights ``w``.

    Integrates the Lorenz curve L(p) on [0, 1] via the trapezoidal rule,
    with the (0, 0) endpoint prepended so the lowest population slice is
    not dropped from the quadrature.
    """
    av = a.ravel()
    wv = w.ravel()
    order = np.argsort(av)
    av = av[order]
    wv = wv[order]
    wv = wv / wv.sum()
    cum_w = np.concatenate([[0.0], np.cumsum(wv)])
    cum_aw = np.concatenate([[0.0], np.cumsum(av * wv) / (av * wv).sum()])
    return float(1.0 - 2.0 * np.trapz(cum_aw, cum_w))


def main() -> int:
    a_grid = make_asset_grid(AMIN, AMAX, NA, curvature=3.0)
    z_log, Pi = rouwenhorst(NZ, RHO_Z, SIGMA_Z)
    z_grid = np.exp(z_log)

    fs = firm_static(a_grid, z_grid, W, R_USER, LAM, ALPHA, ETA)
    sol = entrepreneur_vfi(
        pi_mat=fs["pi"], a_grid=a_grid, z_grid=z_grid, Pi=Pi,
        r=R_NET, beta=BETA, theta=THETA,
        tol=1e-6, max_iter=2500, howard_steps=50, verbose=False,
    )
    agg = aggregate_mx(
        sol=sol, fs=fs, a_grid=a_grid, z_grid=z_grid, Pi=Pi,
        r=R_NET, delta=DELTA, alpha=ALPHA, eta=ETA, L_supply=1.0,
    )

    D = agg["D"]
    results = []

    # 1. Distribution
    results.append(_check(
        "Σ D == 1, D >= 0",
        abs(float(D.sum()) - 1.0) < 1e-8 and (D >= -1e-12).all(),
        f"sum = {D.sum():.10f}, min = {D.min():.2e}",
    ))

    # 2. Savings-assets consistency
    A = agg["A"]
    A_from_policy = float(np.sum(D * sol["a_policy"]))
    ac_err = abs(A_from_policy - A) / max(abs(A), 1e-12)
    results.append(_check(
        "A = Σ D·a equals Σ D·a'",
        ac_err < 5e-5,
        f"A={A:.4f}, Σ D·a'={A_from_policy:.4f}, rel err={ac_err:.2e}",
    ))

    # 3. Goods market: Y - W·L - R·K == aggregate profits
    R = R_NET + DELTA
    goods_resid = agg["Y"] - W * agg["L_d"] - R * agg["K"] - agg["Pi_agg"]
    rel = abs(goods_resid) / max(abs(agg["Y"]), 1e-12)
    results.append(_check(
        "Y - W·L - R·K == Σ D·π",
        rel < 1e-8,
        f"resid = {goods_resid:.3e}, rel = {rel:.2e}",
    ))

    # 4. Wedge τ ≥ 1 (and strict iff some firm binds)
    any_binding = (fs["mu"] > 0).any()
    tau_ok = agg["tau"] >= 1.0 - 1e-12
    strict_ok = (agg["tau"] > 1.0 + 1e-12) == any_binding
    results.append(_check(
        "τ ≥ 1 (aggregate wedge) and strict iff binding",
        tau_ok and strict_ok,
        f"τ = {agg['tau']:.5f}, binding mass = {float((fs['mu']>0).sum()):.0f}/{fs['mu'].size}",
    ))

    # 5. TFP ≤ TFP*, strict iff binding
    tfp_ok = agg["TFP"] <= agg["TFP_star"] + 1e-12
    strict_tfp_ok = (agg["TFP"] < agg["TFP_star"] - 1e-12) == any_binding
    tfp_loss_pct = (1.0 - agg["TFP"] / agg["TFP_star"]) * 100
    results.append(_check(
        "TFP ≤ TFP* and strict iff binding",
        tfp_ok and strict_tfp_ok,
        f"TFP = {agg['TFP']:.4f}, TFP* = {agg['TFP_star']:.4f}, "
        f"loss = {tfp_loss_pct:.3f}%",
    ))

    # 6. Wealth Gini sanity
    a_mat = np.broadcast_to(a_grid[:, None], D.shape)
    gini = _gini(a_mat, D)
    results.append(_check(
        "wealth Gini in [0.4, 0.8]",
        0.4 <= gini <= 0.8,
        f"Gini = {gini:.3f}",
    ))

    # Report price updates for visual sanity
    print(f"\n  Aggregates: A={agg['A']:.3f}  K={agg['K']:.3f}  "
          f"L_d={agg['L_d']:.4f}  Y={agg['Y']:.3f}  C={agg['C']:.3f}")
    print(f"  Updates:   W_new={agg['W_new']:.4f}  r_new={agg['r_new']:.4f}  "
          f"(guess W={W}, r={R_NET})")
    print(f"  TFP loss = {tfp_loss_pct:.3f}%  τ = {agg['tau']:.4f}")

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n  Milestone C: {n_pass} / {n_total} checks passed.")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
