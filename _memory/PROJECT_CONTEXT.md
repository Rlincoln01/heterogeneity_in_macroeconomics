# hetmacro — Project Context

> Last updated: 2026-03-11

## Purpose

Self-contained Python toolkit for solving heterogeneous agent macroeconomic models. Covers the full pipeline: grids, quadrature, income discretization, household DP (EGM/VFI/collocation/Howard/Euler), distribution iteration, market clearing, and sequence-space Jacobians.

## Architecture

Two-tier design:
- **Low-level utilities** (`grids`, `quadrature`, `interpolation`, `optimize`, `markov`, `utils`, `backward`, `forward`, `distributions`, `dp_tools`, `ssj`, `steady_state`) — standalone numerical routines.
- **High-level composable interface** (`household.py` + `income_process.py` + `solvers/`) — plug income process + solver into a `Household` object; call `.solve()`, `.compute_ergodic()`, `.compute_aggregates()`.

Dual-branch pattern throughout: discrete income (Markov Pi) vs. continuous income (Gauss-Hermite quadrature). Solvers and household methods branch on income type.

## Current state

- **Core package:** ~6,000 lines across 16 modules + 7 solvers + 1 model (Aiyagari).
- **New module:** `transition.py` (~550 lines) — MIT shock transition path solver (shooting + root-finding). Features: `_IterationPlotter` class for Jupyter-aware live iteration plotting (`plot_iterations=True`). Fully tested; demo notebook at `examples/pset5/notebooks/pset5_transition_path.ipynb`.
- **Examples:** 4 problem sets (pset1: quadrature/optimization, pset2: collocation/lifecycle, pset3: income fluctuation/Howard, pset5: Aiyagari GE multi-solver + transition path), plus benchmark and household test notebooks.
- **Tests:** 2 files (156 lines) covering basis functions and Euler iteration solvers. Coverage is thin.
- **Docs:** Codebook (76 pages, PDF from LaTeX), `docs/transition_path_methods.tex` (standalone LaTeX on shooting + Broyden algorithms), CompEcon comparison notes, `macro_agents.md` (500-line AI agent guide).
- **Git:** `dev/hetmacro-sync` branch, ahead of `main`. Working tree has uncommitted transition solver + docs.
- **README:** Up to date with current structure, all psets, solvers, and examples documented.

## Key modules (by size)

| Module | Lines | Role |
|--------|-------|------|
| optimize.py | 690 | Root-finding, Broyden, golden section, Nelder-Mead |
| collocation_vfi.py | 629 | Spline and Chebyshev collocation VFI |
| euler_iteration.py | 467 | Naive and Howard Euler solvers (continuous income) |
| interpolation.py | 453 | FunctionSpace, splines, Chebyshev basis, lotteries |
| grids.py | 324 | GridSpec, asset grids, Smolyak, concentration |
| household.py | 262 | Composable Household class |
| policy_iteration.py | 239 | PFI with golden section + linear interpolation |
| backward.py | 195 | EGM and VFI backward steps |
| howard_grid.py | 175 | Grid-based Howard improvement |
| forward.py | 148 | Stationary distribution iteration |
| dp_tools.py | 142 | DP wrappers and Euler equation solvers |
| markov.py | 138 | Rouwenhorst, Tauchen, stationary distribution |
| quadrature.py | 125 | Gaussian quadrature (Hermite, Legendre, Chebyshev) |
| income_process.py | 100 | Income process abstractions (discrete/continuous) |
| utils.py | 95 | CRRA utilities, fixed points, numerical derivatives |
| ssj.py | 93 | Sequence-space Jacobians (fake news algorithm) |
| steady_state.py | 79 | Household SS and market clearing |
| distributions.py | 77 | Joint transition matrices, lottery indices |
| aiyagari.py | 56 | Complete Aiyagari equilibrium solver |

## Immediate priorities

1. Transition path solver complete (shooting + Broyden + live plotting). Next: SSJ integration (future task).
2. Expand test coverage beyond 2 files.
3. Merge `dev/hetmacro-sync` into `main` when ready.
4. Keep codebook.tex in sync with future API changes.
5. Keep `docs/transition_path_methods.tex` in sync with transition.py changes.
