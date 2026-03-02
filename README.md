# Heterogeneity in Macroeconomics: hetmacro

Self-contained Python toolkit for solving macroeconomic models with heterogeneity. No external dependencies on QuantEcon/CompEcon; follows their algorithmic best practices.

The toolkit supports the full computational pipeline:
1. Build state grids and discretize shocks.
2. Solve household problems (EGM, VFI, collocation, Howard, Euler iteration, policy iteration).
3. Compute stationary distributions (forward iteration).
4. Clear markets and compute aggregates.
5. Compute sequence-space Jacobians for dynamics.

## Project Goals

1. General-purpose numerical foundation for computational macroeconomics.
2. Reusable tools for heterogeneous agent models with multiple solver backends.
3. Composable architecture: mix income processes and solvers via a `Household` class.
4. Clear documentation (codebook) and AI-agent guidance (`macro_agents.md`).

## Repository Structure

```
hetmacro/
  __init__.py
  grids.py              # Multi-dimensional grids, concentration, Smolyak
  quadrature.py         # Gaussian quadrature (Hermite, Legendre, Chebyshev)
  interpolation.py      # FunctionSpace, splines, Chebyshev basis, lotteries
  optimize.py           # Root-finding, Broyden, golden section, Nelder-Mead
  markov.py             # Rouwenhorst, Tauchen, stationary distributions
  utils.py              # CRRA utilities, fixed points, numerical derivatives
  backward.py           # EGM and VFI backward iteration steps
  forward.py            # Stationary distribution (forward iteration)
  dp_tools.py           # DP wrappers and Euler equation helpers
  distributions.py      # Joint transition matrices, lottery indices
  ssj.py                # Sequence-space Jacobians (fake news algorithm)
  steady_state.py       # Household steady state and market clearing
  household.py          # Composable Household class (high-level interface)
  income_process.py     # Income process abstractions (discrete/continuous)
  solvers/
    egm.py              # Endogenous grid method
    grid_vfi.py         # Grid-based value function iteration
    howard.py           # Coefficient-space Howard improvement
    howard_grid.py      # Grid-based Howard improvement
    euler_iteration.py  # Naive and Howard Euler iteration
    policy_iteration.py # Policy function iteration
    collocation_vfi.py  # Spline and Chebyshev collocation VFI
  models/
    aiyagari.py         # Aiyagari steady-state solver
  notebooks/
    test_grids.ipynb
    test_quadrature.ipynb
    test_compecon_comparison.ipynb

docs/
  codebook/
    codebook.pdf        # Canonical API reference
    codebook.tex
    figures/
  compecon_comparison.md

examples/
  README.md
  household_test.ipynb
  benchmark_howard_euler_speed.py
  pset1/                # Quadrature, nonlinear solvers, labor supply, portfolio
  pset2/                # Collocation, projection, lifecycle models
  pset3/                # Income fluctuation, VFI, Howard, ergodic distributions

tests/
  test_basis_and_stationary.py
  test_euler_iteration_solvers.py

macro_agents.md         # AI agent playbooks and workflow guidance
requirements.txt
```

## Core Modules

### Foundations
- `grids.py` — multi-dimensional grids with flexible spacing (power, double-exp, Chebyshev) and concentration points.
- `quadrature.py` — Gaussian quadrature (Legendre, Chebyshev, Hermite, Beta, Gamma) and Newton-Cotes rules.
- `interpolation.py` — `FunctionSpace` class, linear/cubic/Chebyshev basis matrices, `get_lottery()` for policy lotteries.
- `optimize.py` — bisection, Brent, Newton, secant, golden section, Nelder-Mead, and unified Broyden solver (compecon/scipy/componentwise).
- `markov.py` — Rouwenhorst and Tauchen discretization, stationary distribution computation.
- `utils.py` — CRRA utility functions, `compute_fixed_point()`, numerical differentiation.

### Heterogeneous Agent Tools
- `backward.py` / `dp_tools.py` — single-step EGM and VFI backward iteration.
- `forward.py` / `distributions.py` — stationary distributions via forward iteration and joint transition matrices.
- `steady_state.py` — household steady state and market clearing (bisection on interest rate).
- `ssj.py` — sequence-space Jacobians using the fake news algorithm.
- `models/aiyagari.py` — complete Aiyagari general equilibrium solver.

### Composable Household Interface
- `household.py` — `Household` class with `.solve()`, `.compute_ergodic()`, `.compute_aggregates()`, `.euler_check()`, `.simulate()`.
- `income_process.py` — `DiscreteIncome`, `RouwenhorstIncome`, `TauchenIncome`, `ContinuousQuadratureIncome` abstractions.

### Solvers

| Solver | Method | Best for |
|--------|--------|----------|
| `GridVFI` | Grid search VFI | Baseline validation; simplest |
| `HowardGrid` | Grid search + linear-system policy eval | Fast grid-based solving |
| `HowardImprovement` | Coefficient-space Howard | Fast collocation-based solving |
| `EGM` | Endogenous grid method | One-asset savings (fastest for simple problems) |
| `PolicyFunctionIteration` | Golden section + linear interp | Flexible hybrid approach |
| `CollocationVFI_Spline` | Cubic spline collocation | Smooth policies, few coefficients |
| `CollocationVFI_Chebyshev` | Chebyshev collocation | Smooth policies, spectral accuracy |
| `NaiveEulerIteration` | Euler equation iteration | Continuous income only |
| `HowardEulerIteration` | Howard-accelerated Euler | Continuous income, fast convergence |

All solvers implement `.solve(household) -> SolvedPolicy`.

## Examples

**Package tests** (`hetmacro/notebooks/`): visual tests for grids, quadrature, and CompEcon comparison.

**Problem sets** (`examples/`):
- **pset1/** — Quadrature, nonlinear solvers (Broyden, Nelder-Mead), two-period labor supply, portfolio choice. Includes report PDF.
- **pset2/** — Rouwenhorst diagnostics, cake-eating with collocation/projection, lifecycle model. Includes slides.
- **pset3/** — Income fluctuation problems (discrete + AR(1)), VFI, Howard improvement, ergodic distributions. Includes slides and TeX source.

**Benchmarks**: `examples/benchmark_howard_euler_speed.py` compares Howard vs Euler solver performance.

## Documentation

- `docs/codebook/codebook.pdf` — canonical API reference with figures and usage examples.
- `docs/compecon_comparison.md` — comparison notes with CompEcon (MATLAB/Python).
- `macro_agents.md` — AI agent playbooks: step-by-step workflows, solver selection, calibration templates.

## Getting Started

```bash
pip install -r requirements.txt
```

### Quick example (Aiyagari steady state)

```python
from hetmacro.models.aiyagari import solve_steady_state
ss = solve_steady_state()
```

### Composable interface

```python
from hetmacro.income_process import RouwenhorstIncome
from hetmacro.household import Household
from hetmacro.solvers import GridVFI
from hetmacro.grids import make_asset_grid

income = RouwenhorstIncome.from_ar1(n=7, rho=0.9, sigma=0.2)
hh = Household(income_process=income, a_grid=make_asset_grid(0, 50, 300),
               beta=0.96, gamma=2.0, r=0.04, w=1.0)
pol = hh.solve(GridVFI())
dist = hh.compute_ergodic()
agg = hh.compute_aggregates()
```

### Running notebooks

1. Create and activate a Python environment (3.11+ recommended).
2. Install dependencies from `requirements.txt`.
3. Run package tests from `hetmacro/notebooks/` or problem sets from `examples/psetN/notebooks/`.

### Running tests

```bash
pytest tests/
```
