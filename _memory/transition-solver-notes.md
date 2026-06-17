# Transition Path Solver — Notes

Migrated from `~/.claude/projects/-Users-rafaellincoln-Dropbox-Academia-Macroeconomics/memory/transition-solver.md` on 2026-05-06. Originally written 2026-03-11. Status: solver COMPLETE.

## Files created/modified

- `hetmacro/transition.py` — core solver (~400 lines): `compute_transition_path`, `_backward_path`, `_forward_path`, `diagnose_transition`, `make_backward_fn`
- `hetmacro/models/aiyagari.py` — added `aiyagari_transition_path()` wrapper
- `hetmacro/__init__.py` — added "transition" to `__all__`
- `examples/pset5/notebooks/pset5_transition_path.ipynb` — demo notebook

## Test results (2026-03-11)

1. **Flat-path (no shock, T=50):** max|K-K_ss| = 8e-6, RC = 6e-9
2. **TFP shock (T=500, shooting, damping=2/3):** Converged in 34 iters (50s). K range [58.80, 62.42]. RC = 2.6e-10.
3. **Root-finding (T=200, Broyden):** Converged in 35 evals (21s). Matches shooting to 4e-8 in K.

## Key fixes applied

- Income convention: always pass `y = w * z_grid` to `household_steady_state`
- Forward path returns capital (mean assets in `D_t`), not savings (`vdot(D, a_pol)`)
- `r` lower bound changed from -0.01 to 0.001 to avoid NaN in policy initialization
- RC diagnostic uses savings from policies directly (avoids terminal boundary contamination)

## Damping convention

- Our `damping` = weight on OLD guess: `K = damping * K_old + (1 - damping) * K_new`
- Virgiliu step=1/3 → our damping=2/3

## Calibration for pset5 Q3.1

`beta=0.99, gamma=1.5, rho=0.99, sigma_e=0.10, alpha=0.35, delta=0.02, ne=11, nk=501, a_max=2500`. Shock: `log_Z[1]=0.10`, `log_Z[t]=0.95*log_Z[t-1]`. T=500, damping=2/3.
SS: `r=0.004752, K=58.88, w=2.71, C=2.99, Y=4.16`.

## Live iteration plotting (`plot_iterations=True`)

- `_IterationPlotter` class manages Jupyter-aware live figure updates.
- Uses `IPython.display.clear_output(wait=True)` + `display(fig)` (NOT `plt.ion()` which fails in Jupyter inline backend).
- Verbose output buffered in list and re-printed after each `clear_output` to avoid losing iteration logs.
- Terminal fallback: `draw_idle()` + `flush_events()`.
- All plotting wrapped in try/except so failures never lose computed results.

## LaTeX documentation

- `docs/transition_path_methods.tex` — standalone document covering shooting + Broyden algorithms.
- Macro Bible style (algorithm environments, Palatino, motivating paragraphs).
- Includes EGM backward step detail with explicit interpolation formula (endogenous to exogenous grid).

## Status

COMPLETE. Future work: SSJ integration (separate task).
