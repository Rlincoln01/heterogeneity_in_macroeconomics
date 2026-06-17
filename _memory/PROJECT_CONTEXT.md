# hetmacro — Project Context

> Last updated: 2026-04-21

## Purpose

Self-contained Python toolkit for solving heterogeneous agent macroeconomic models. Covers the full pipeline: grids, quadrature, income discretization, household DP (EGM/VFI/collocation/Howard/Euler), distribution iteration, market clearing, and sequence-space Jacobians.

## Architecture

Two-tier design:
- **Low-level utilities** (`grids`, `quadrature`, `interpolation`, `optimize`, `markov`, `utils`, `backward`, `forward`, `distributions`, `dp_tools`, `ssj`, `steady_state`) — standalone numerical routines.
- **High-level composable interface** (`household.py` + `income_process.py` + `solvers/`) — plug income process + solver into a `Household` object; call `.solve()`, `.compute_ergodic()`, `.compute_aggregates()`.

Dual-branch pattern throughout: discrete income (Markov Pi) vs. continuous income (Gauss-Hermite quadrature). Solvers and household methods branch on income type.

## Current state

- **Core package:** ~6,000 lines across 16 modules + 7 solvers + 3 models (Aiyagari, cm_rbc, krusell_smith).
- **New modules (Pset 6):**
  - `kronm.py` — fast Kronecker multiplication for 4D VFI.
  - `chebyshev.py` — 2D tensor-product Chebyshev projection for law of motion.
  - `models/solab.py` — QZ-based linear rational expectations solver.
  - `models/cm_rbc.py` — complete markets RBC (linear + nonlinear projection).
  - `models/krusell_smith.py` — SSJ GE IRFs + full KS algorithm (4D VFI, forward simulation, outer loop).
- **Phase status (Pset 6):** Phases 1-4 complete. Phase 5 (notebook) not started.
- **Pset 10 (Midrigan-Xu):** `models/midrigan_xu.py` complete (firm_static, entrepreneur_vfi with Howard, aggregate_mx, solve_mx_closed, solve_mx_open, simulate_firms). Notebook `examples/pset10/notebooks/pset10_midrigan_xu.ipynb` + six milestone driver scripts validated against slides 19-22 at full grid (na=501, nz=11).
- **transition.py** (~550 lines) — MIT shock transition path solver.
- **Examples:** 6 problem sets (pset1-3, pset5-8, pset10) + Bayesian HANK.
- **Tests:** 2 files (156 lines). Phase 1 infrastructure unit tested.
- **Git:** currently on `main`. Pre-pset10 WIP snapshot committed as `e2a8305` before pset10 module landed.
- **README:** Needs update to reflect Pset 6 modules.

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
