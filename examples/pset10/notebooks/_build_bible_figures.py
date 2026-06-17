"""Generate three figures for the Macro Bible Midrigan-Xu section.

Uses the cached steady states in ``cache/`` to produce:
  1. mx_decision_rules.pdf  -- 6-panel decision rules at xi=0.86 closed
  2. mx_wedge_qq.pdf         -- wedge-vs-logz QQ curve at xi=0.5 and xi=0 closed
  3. mx_single_firm.pdf      -- one exogenous z-path simulated in three
                                 closed-economy SSes (xi = 0.86, 0.5, 0)
                                 with the three trajectories overlaid.

All figures save to the bible's figures directory. Figures have no top
titles (the LaTeX caption already provides the title); panel subtitles and
axis labels are bumped up so multi-panel figures remain legible.

Run: ``python _build_bible_figures.py`` from ``examples/pset10/notebooks/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hetmacro.forward import stationary_markov
from hetmacro.models.midrigan_xu import (
    load_steady_state,
    simulate_firms,
)

BIBLE_FIG_DIR = Path(
    "/Users/rafaellincoln/Dropbox/Academia/Macroeconomics/Macro Bible/"
    "Parts_and_chapters/Heterogeneity/Firm_heterogeneity/figures"
)
BIBLE_FIG_DIR.mkdir(parents=True, exist_ok=True)

# Bigger font defaults so multi-panel figures stay legible at print size.
# No top titles (LaTeX caption supplies the overall title).
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "savefig.bbox": "tight",
})


# ============================================================
# Figure 1: decision rules at xi=0.86 (slide 11 analogue)
# ============================================================

def figure_decision_rules() -> None:
    ss = load_steady_state(HERE / "cache" / "ss_closed_xi086.pkl")
    a_grid = ss.a_grid
    z_grid = ss.z_grid
    fs = ss.fs
    sol = ss.sol

    j_plot = 3 * len(z_grid) // 4
    plot_cap = int(np.searchsorted(a_grid, 15))
    a_slice = a_grid[:plot_cap]

    savings = sol["a_policy"][:plot_cap, j_plot] - a_slice

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.3))
    panels = [
        ("profits",          fs["pi"][:plot_cap, j_plot]),
        ("output",           fs["y"][:plot_cap, j_plot]),
        ("capital",          fs["k"][:plot_cap, j_plot]),
        (r"multiplier $\mu$",     fs["mu"][:plot_cap, j_plot]),
        (r"savings $a' - a$",     savings),
        ("consumption",      sol["c_policy"][:plot_cap, j_plot]),
    ]
    for ax, (title, series) in zip(axes.flat, panels):
        ax.plot(a_slice, series, lw=2.0, color="C0")
        ax.set_title(title)
        ax.set_xlabel(r"net worth $a$")

    fig.tight_layout()
    out = BIBLE_FIG_DIR / "mx_decision_rules.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# Figure 2: wedge-productivity QQ curve (xi=0.5 and xi=0 closed)
# ============================================================

def _wquantile(values, weights, qs):
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    order = np.argsort(v)
    v_s, w_s = v[order], w[order]
    ws = w_s.sum()
    if ws <= 0:
        return np.full_like(np.asarray(qs, dtype=float), np.nan)
    cum = np.cumsum(w_s) / ws
    return np.interp(qs, cum, v_s)


def _logz_quantile_smoothed(marginal, z_grid, qs):
    m = np.asarray(marginal, dtype=float)
    m = m / m.sum()
    cum = np.cumsum(m)
    F_mid = cum - m / 2.0
    log_z = np.log(z_grid)
    return np.interp(qs, F_mid, log_z)


def _qq_curves(ss, p_grid):
    D = ss.agg["D"]
    tau_i = ss.agg["tau_i"]
    z_grid = ss.z_grid
    z_mat = np.broadcast_to(z_grid[None, :], D.shape)
    w_unw = D
    w_zw = D * z_mat
    mz_unw = D.sum(axis=0)
    mz_zw = (D * z_mat).sum(axis=0)
    return (
        _logz_quantile_smoothed(mz_unw, z_grid, p_grid),
        _wquantile(tau_i, w_unw, p_grid),
        _logz_quantile_smoothed(mz_zw, z_grid, p_grid),
        _wquantile(tau_i, w_zw, p_grid),
    )


def figure_wedge_qq() -> None:
    ss_50 = load_steady_state(HERE / "cache" / "ss_closed_xi050.pkl")
    ss_00 = load_steady_state(HERE / "cache" / "ss_closed_xi0.pkl")
    p_grid = np.linspace(0.01, 0.99, 199)
    mark_pcts = [0.10, 0.50, 0.75, 0.90, 0.99]

    styles = {
        0.5: {"color": "C0", "label": r"$\xi = 0.5$"},
        0.0: {"color": "C3", "label": r"$\xi = 0$"},
    }

    fig, ax = plt.subplots(figsize=(9, 5.8))
    for xi, ss in ((0.5, ss_50), (0.0, ss_00)):
        lz_u, t_u, lz_w, t_w = _qq_curves(ss, p_grid)
        c = styles[xi]["color"]
        lab = styles[xi]["label"]
        ax.plot(lz_u, t_u, color=c, linestyle="-",  lw=2.0, label=f"{lab}, unweighted")
        ax.plot(lz_w, t_w, color=c, linestyle="--", lw=2.0, label=f"{lab}, z-weighted")
        lz_mark, t_mark, _, _ = _qq_curves(ss, np.array(mark_pcts))
        ax.scatter(lz_mark, t_mark, color=c, s=32, zorder=5)
        for p, x, y in zip(mark_pcts, lz_mark, t_mark):
            ax.annotate(f"p={p:.2f}", (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=10, color=c)

    ax.axhline(1.0, color="k", lw=0.7, alpha=0.4)
    ax.set_xlabel(r"$\log z$ quantile at percentile $p$")
    ax.set_ylabel(r"capital wedge $\tau_i$ quantile at percentile $p$")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    out = BIBLE_FIG_DIR / "mx_wedge_qq.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}")


# ============================================================
# Figure 3: same z-path simulated in three closed-economy SSes
# ============================================================

def figure_single_firm() -> None:
    ss_86 = load_steady_state(HERE / "cache" / "ss_closed_xi086.pkl")
    ss_50 = load_steady_state(HERE / "cache" / "ss_closed_xi050.pkl")
    ss_00 = load_steady_state(HERE / "cache" / "ss_closed_xi0.pkl")

    # Simulate each economy separately with the SAME RNG seed. Because the
    # three economies share the same Pi (same Rouwenhorst discretization),
    # the productivity index sequence ``z_idx`` drawn inside simulate_firms
    # is identical across the three calls — so firm ``ind`` faces the same
    # exogenous productivity path under each ξ. What differs is the
    # endogenous state ``a``, which after the burn-in period reflects each
    # economy's own typical wealth level (its policy functions differ).
    T = 250
    N = 20_000
    burn = 50
    seed = 5

    sim_86 = simulate_firms(ss_86, N=N, T=T, seed=seed, burn=burn)
    sim_50 = simulate_firms(ss_50, N=N, T=T, seed=seed, burn=burn)
    sim_00 = simulate_firms(ss_00, N=N, T=T, seed=seed, burn=burn)

    # Sanity: z paths should match across the three sims (same seed + same Pi).
    assert np.array_equal(sim_86["z_idx"], sim_50["z_idx"])
    assert np.array_equal(sim_86["z_idx"], sim_00["z_idx"])

    # Pick a firm that (a) binds at ξ=0 during the display window (so the
    # multiplier panel is informative), and (b) has a visibly different
    # wealth level across the three economies.
    t_slice = slice(100, 200)
    mu_win_00 = sim_00["mu"][:, t_slice]
    binds = (mu_win_00 > 1e-6).sum(axis=1)
    candidates = np.where(binds >= 5)[0]
    if len(candidates) == 0:
        ind = int(np.argmax(mu_win_00.max(axis=1)))
    else:
        ind = int(candidates[0])
    print(f"    single-firm panel uses firm index {ind} "
          f"(bind-count at ξ=0 in window = {int(binds[ind])})")

    tt = np.arange(t_slice.start, t_slice.stop)
    log_z_slice = np.log(sim_86["z"][ind, t_slice])   # identical across ξ

    paths = {
        0.86: {k: sim_86[k][ind, t_slice] for k in ("a", "mu", "pi", "c", "k")},
        0.5:  {k: sim_50[k][ind, t_slice] for k in ("a", "mu", "pi", "c", "k")},
        0.0:  {k: sim_00[k][ind, t_slice] for k in ("a", "mu", "pi", "c", "k")},
    }
    styles = {
        0.86: {"color": "C2", "label": r"$\xi = 0.86$"},   # green = loose
        0.5:  {"color": "C0", "label": r"$\xi = 0.5$"},    # blue = medium
        0.0:  {"color": "C3", "label": r"$\xi = 0$"},      # red  = tight
    }

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4))
    # Productivity is common across ξ; plot it once in black.
    panels_series = [
        ("wealth $a$",       "a",  False),
        ("log productivity", None, True),     # common log z (black line)
        (r"multiplier $\mu$", "mu", False),
        (r"profits $\pi$",    "pi", False),
        ("consumption $c$",   "c",  False),
        ("capital $k$",       "k",  False),
    ]

    for ax, (title, key, is_z) in zip(axes.flat, panels_series):
        if is_z:
            ax.plot(tt, log_z_slice, color="k", lw=2.0)
        else:
            for xi, path in paths.items():
                ax.plot(tt, path[key], color=styles[xi]["color"],
                        lw=1.7, label=styles[xi]["label"])
        ax.set_title(title)
        ax.set_xlabel(r"time $t$")

    # Legend on the wealth panel (first panel with ξ-specific lines).
    axes[0, 0].legend(loc="best", frameon=False, ncol=1)

    fig.tight_layout()
    out = BIBLE_FIG_DIR / "mx_single_firm.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out}  (shared z-path, 3 economies overlaid)")


def main() -> int:
    print("Building bible figures...")
    figure_decision_rules()
    figure_wedge_qq()
    figure_single_firm()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
