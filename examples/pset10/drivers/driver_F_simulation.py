"""Milestone F: simulate firms from the xi=0.86 closed-economy SS and check
output-moment targets from slide 19.

Targets (slide 19, "Model" column; Data is the external target):

    std(log y_it)                                  Data 1.31  Model 1.31
    std(Δ log y_it)                                Data 0.59  Model 0.59
    corr(log y_t, log y_{t-1})                     Data 0.90  Model 0.90
    corr(log y_t, log y_{t-3})                     Data 0.87  Model 0.73
    corr(log y_t, log y_{t-5})                     Data 0.85  Model 0.59

Note Virgiliu explicitly writes "Autocorrelations decay too quickly, so need
transitory shocks" (slide 19); we don't implement transitory shocks, so we
check the 1-period correlation and standard deviations but accept that lag-3
and lag-5 correlations will be below data.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from hetmacro.models.midrigan_xu import simulate_firms, solve_mx_closed


COMMON = dict(
    beta=0.96, theta=2.0, alpha=1.0 / 3.0, eta=0.85, delta=0.06,
    rho_z=0.896, sigma_z=0.59,
    na=501, nz=11, amin=1e-2, amax=2500.0, a_curvature=3.0,
    vfi_tol=1e-6, vfi_max_iter=2500, howard_steps=50,
    damping=0.85, tol=1e-5, max_iter=400,
    verbose=False,
)


def _check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def main() -> int:
    print("Solving xi=0.86 SS for simulation...")
    t = time.time()
    ss = solve_mx_closed(xi=0.86, W_init=1.0, r_init=0.02, **COMMON)
    print(f"  SS: W={ss.W:.4f}, r={ss.r:+.4f}, elapsed {time.time()-t:.1f}s")

    print("\nSimulating N=50_000 firms for T=250 periods...")
    t = time.time()
    sim = simulate_firms(ss, N=50_000, T=250, seed=0, burn=50)
    print(f"  elapsed {time.time()-t:.1f}s")

    print(f"\n  std(log y)              = {sim['std_log_y']:.3f}   (Data 1.31)")
    print(f"  std(Δlog y)             = {sim['std_dlog_y']:.3f}   (Data 0.59)")
    print(f"  corr(log y_t, log y_{{t-1}}) = {sim['corr_log_y'][1]:.3f}   (Data 0.90)")
    print(f"  corr(log y_t, log y_{{t-3}}) = {sim['corr_log_y'][3]:.3f}   (Data 0.87)")
    print(f"  corr(log y_t, log y_{{t-5}}) = {sim['corr_log_y'][5]:.3f}   (Data 0.85)")

    results = []
    results.append(_check("std(log y) ≈ 1.31 (±0.10)",
                          abs(sim["std_log_y"] - 1.31) < 0.10,
                          f"got {sim['std_log_y']:.3f}"))
    results.append(_check("std(Δlog y) ≈ 0.59 (±0.10)",
                          abs(sim["std_dlog_y"] - 0.59) < 0.10,
                          f"got {sim['std_dlog_y']:.3f}"))
    results.append(_check("corr at lag 1 ≈ 0.90 (±0.03)",
                          abs(sim["corr_log_y"][1] - 0.90) < 0.03,
                          f"got {sim['corr_log_y'][1]:.3f}"))
    # Lags 3 and 5: we match Virgiliu's "Model" column per slide 19
    # (no transitory shocks in our setup).
    results.append(_check("corr at lag 3 in [0.65, 0.85] (Virgiliu Model: 0.73)",
                          0.65 <= sim["corr_log_y"][3] <= 0.85,
                          f"got {sim['corr_log_y'][3]:.3f}"))
    results.append(_check("corr at lag 5 in [0.50, 0.70] (Virgiliu Model: 0.59)",
                          0.50 <= sim["corr_log_y"][5] <= 0.70,
                          f"got {sim['corr_log_y'][5]:.3f}"))

    # Save a slide-12-style 6-panel figure for one firm
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        ind = 5
        t_slice = slice(100, 200)
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        titles_data = [
            ("wealth", sim["a"][ind, t_slice]),
            ("productivity", np.log(sim["z"][ind, t_slice])),
            ("multiplier", sim["mu"][ind, t_slice]),
            ("profits", sim["pi"][ind, t_slice]),
            ("consumption", sim["c"][ind, t_slice]),
            ("capital", sim["k"][ind, t_slice]),
        ]
        for ax, (title, data) in zip(axes.flat, titles_data):
            ax.plot(data, lw=1.5)
            ax.set_title(title)
        fig.suptitle("Milestone F: single firm, t in [100, 200]")
        fig.tight_layout()
        out = _HERE.parent / "driver_F_single_firm.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"\n  Single-firm plot saved to {out.name}")
    except ImportError:
        pass

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n  Milestone F: {n_pass} / {n_total} checks passed.")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
