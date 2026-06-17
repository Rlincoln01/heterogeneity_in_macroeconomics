"""Milestone A driver: verify ``firm_static`` against the plan's checklist.

Run from the repo root with ``python -m examples.pset10.drivers.driver_A_firm_static``
or directly with ``python examples/pset10/drivers/driver_A_firm_static.py``.

Checklist (plan section "Milestone A"):

1. At slack constraint, output matches ``unconstrained_firm`` exactly.
2. Monotonicity: ``dk/dz >= 0``, ``dk/da >= 0``, ``dmu/da <= 0``, ``dmu/dz >= 0``.
3. Continuity at the binding threshold (where ``lam*a == k*``).
4. Sanity at the slide-18 Korean calibration: ``ksi=0.86`` yields ``mu=0`` for
   most of the mass and ``k/y in [3, 5]`` for a median-productivity firm.

Prints PASS / FAIL per check and exits nonzero on any FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo importable when running as a plain script.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from hetmacro.grids import make_asset_grid
from hetmacro.markov import rouwenhorst
from hetmacro.models.midrigan_xu import firm_static, unconstrained_firm


# ── Calibration (slide 18) + a smallish grid for fast checks ────────────
ALPHA = 1.0 / 3.0
ETA = 0.85
DELTA = 0.06
RHO_Z = 0.896
SIGMA_Z = 0.59
R_BASE = 0.02               # start.m initial guess
W_BASE = 1.0                # start.m initial guess
R_USER = R_BASE + DELTA     # user cost

NA = 51
NZ = 7
AMIN, AMAX = 1e-2, 100.0


def _build_grids():
    a_grid = make_asset_grid(AMIN, AMAX, NA, curvature=3.0)
    z_log, Pi = rouwenhorst(NZ, RHO_Z, SIGMA_Z)
    z_grid = np.exp(z_log)
    return a_grid, z_grid, Pi


def _check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main() -> int:
    a_grid, z_grid, _ = _build_grids()

    results = []

    # ---------- 1. Slack constraint matches unconstrained formula ---------
    # Pick lam very large so constraint is always slack.
    lam_slack = 1e6
    fs_slack = firm_static(a_grid, z_grid, W_BASE, R_USER, lam_slack, ALPHA, ETA)
    ufs = unconstrained_firm(z_grid, W_BASE, R_USER, ALPHA, ETA)

    # At any (a, z), output/capital/labor should equal the z-only unconstrained value.
    # Pick the top row (largest a) to minimize any residual numerical effect.
    y_col = fs_slack["y"][-1, :]
    k_col = fs_slack["k"][-1, :]
    mu_top = fs_slack["mu"][-1, :]
    y_err = np.max(np.abs(y_col - ufs["y"]) / np.abs(ufs["y"]))
    k_err = np.max(np.abs(k_col - ufs["k"]) / np.abs(ufs["k"]))
    results.append(_check(
        "slack constraint recovers frictionless y",
        y_err < 1e-10, f"max rel err = {y_err:.2e}",
    ))
    results.append(_check(
        "slack constraint recovers frictionless k",
        k_err < 1e-10, f"max rel err = {k_err:.2e}",
    ))
    results.append(_check(
        "slack constraint gives mu == 0",
        np.allclose(mu_top, 0.0), f"max mu = {mu_top.max():.2e}",
    ))

    # ---------- 2. Monotonicity on realistic lam -------------------------
    # Pick a moderately tight constraint so we see both regions.
    lam_tight = 2.0   # xi = 0.5
    fs = firm_static(a_grid, z_grid, W_BASE, R_USER, lam_tight, ALPHA, ETA)

    dk_da = np.diff(fs["k"], axis=0)
    dk_dz = np.diff(fs["k"], axis=1)
    dmu_da = np.diff(fs["mu"], axis=0)
    dmu_dz = np.diff(fs["mu"], axis=1)

    results.append(_check(
        "dk/da >= 0 everywhere",
        (dk_da >= -1e-12).all(), f"min = {dk_da.min():.2e}",
    ))
    results.append(_check(
        "dk/dz >= 0 everywhere",
        (dk_dz >= -1e-12).all(), f"min = {dk_dz.min():.2e}",
    ))
    results.append(_check(
        "dmu/da <= 0 everywhere",
        (dmu_da <= 1e-12).all(), f"max = {dmu_da.max():.2e}",
    ))
    results.append(_check(
        "dmu/dz >= 0 everywhere",
        (dmu_dz >= -1e-12).all(), f"min = {dmu_dz.min():.2e}",
    ))

    # ---------- 3. Continuity at the binding threshold -------------------
    # For each z with a transition in the grid: last binding row should
    # satisfy k == lam*a exactly; the first slack row should satisfy
    # k <= lam*a (with near-equality at the transition).
    continuity_max_gap = 0.0
    for j_z in range(len(z_grid)):
        mu_col = fs["mu"][:, j_z]
        slack_mask = mu_col == 0.0
        if not slack_mask.any() or slack_mask.all():
            continue
        first_slack = int(np.argmax(slack_mask))
        last_binding = first_slack - 1
        # Last binding row: k must equal lam*a within FP tolerance
        gap_bind = abs(
            fs["k"][last_binding, j_z] - lam_tight * a_grid[last_binding]
        ) / (lam_tight * a_grid[last_binding])
        # First slack row: k must NOT exceed lam*a
        ratio_slack = fs["k"][first_slack, j_z] / (lam_tight * a_grid[first_slack])
        continuity_max_gap = max(continuity_max_gap, gap_bind)
        assert gap_bind < 1e-10, (
            f"z={z_grid[j_z]:.3f}: last binding row k/lam*a deviates "
            f"by {gap_bind:.2e}"
        )
        assert ratio_slack <= 1.0 + 1e-9, (
            f"z={z_grid[j_z]:.3f}: first slack row k exceeds lam*a "
            f"(ratio = {ratio_slack:.4f})"
        )
    results.append(_check(
        "k continuous at binding threshold (both sides)",
        True, f"max binding-row gap = {continuity_max_gap:.2e}",
    ))

    # Also verify: where mu > 0, k equals lam*a up to float tolerance.
    binding = fs["mu"] > 0
    if binding.any():
        k_binding = fs["k"][binding]
        a_binding = np.broadcast_to(a_grid[:, None], fs["k"].shape)[binding]
        ratio = k_binding / (lam_tight * a_binding)
        max_dev = np.max(np.abs(ratio - 1.0))
        results.append(_check(
            "when mu>0, k == lam*a",
            max_dev < 1e-10, f"max |ratio - 1| = {max_dev:.2e}",
        ))
    else:
        results.append(_check(
            "when mu>0, k == lam*a", True, "(no binding region)",
        ))

    # ---------- 4. Sanity at Korean calibration --------------------------
    lam_korea = 1.0 / (1.0 - 0.86)
    fs_k = firm_static(a_grid, z_grid, W_BASE, R_USER, lam_korea, ALPHA, ETA)
    # Median productivity, roughly median a (slack region for Korea)
    j_med = len(z_grid) // 2
    i_mid = len(a_grid) // 2
    ky_ratio = fs_k["k"][i_mid, j_med] / fs_k["y"][i_mid, j_med]
    # Cobb-Douglas with alpha*eta ~ 0.28 and unconstrained k/y = alpha*eta/R
    # Expected: 0.333*0.85 / 0.08 = ~3.54
    expected_ky = ALPHA * ETA / R_USER
    results.append(_check(
        "Korean calibration median k/y ~ alpha*eta/R",
        abs(ky_ratio - expected_ky) / expected_ky < 0.02,
        f"ky = {ky_ratio:.3f}, expected {expected_ky:.3f}",
    ))

    # ---------- Summary --------------------------------------------------
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n  Milestone A: {n_pass} / {n_total} checks passed.")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
