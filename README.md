# Heterogeneity in Macroeconomics: hetmacro

This repository provides a foundational numerical toolkit (`hetmacro`) for solving macroeconomic models with heterogeneity. The package is designed to be **self-contained** (no external dependencies on QuantEcon/CompEcon), while still following their algorithmic best practices.

At a high level, the toolkit supports the standard computational pipeline:
1. Build state grids for continuous variables and discretize shocks.
2. Solve household problems via EGM or VFI (backward iteration).
3. Compute stationary distributions (forward iteration).
4. Clear markets and compute aggregates.
5. (Optional) Compute sequence-space Jacobians for dynamics.

## Project Goals

1. Build a general-purpose numerical foundation for computational macroeconomics.
2. Provide reusable tools for heterogeneous agent (HA) models (EGM, distribution iteration, SSJ).
3. Maintain clear documentation and a codebook for all functions.
4. Provide AI-agent guidance for using the toolkit in new model builds.

## Repository Structure

```
hetmacro/
  __init__.py
  grids.py
  quadrature.py
  interpolation.py
  optimize.py
  markov.py
  utils.py
  dp_tools.py
  distributions.py
  ssj.py
  steady_state.py
  backward.py
  forward.py
  notebooks/
    test_grids.ipynb
    test_quadrature.ipynb
    test_compecon_comparison.ipynb
  models/
    __init__.py
    aiyagari.py

docs/
  codebook/
    codebook.pdf
    figures/
  compecon_comparison.md

examples/
  README.md
  pset1/
    README.md
    notebooks/
      pset1_problem1.ipynb … pset1_problem5.ipynb
    report/
      pset1_report.pdf
      figures/

macro_agents.md
requirements.txt
```

## Core Modules

### Phase 1: Foundations
- `grids.py` — general multi-dimensional grids with flexible spacing and concentration.
- `quadrature.py` — Gaussian quadrature (Legendre/Chebyshev/Hermite) and Newton–Cotes rules.
- `interpolation.py` — linear, spline, and Chebyshev interpolation.
- `optimize.py` — root-finding and optimization routines.
- `markov.py` — discretization of AR(1) processes and Markov chain utilities.
- `utils.py` — economic utilities, numerical differentiation, fixed points, timing.

### Phase 2: Heterogeneous Agent Tools
- `backward.py` / `dp_tools.py` — VFI and EGM for policy functions.
- `forward.py` / `distributions.py` — stationary distributions and forward iteration.
- `steady_state.py` — household steady state and market clearing.
- `ssj.py` — sequence-space Jacobians (fake news algorithm).
- `models/aiyagari.py` — example Aiyagari steady-state solver.

## Examples and Notebooks

**Package tests (hetmacro/notebooks/):**
- `test_grids.ipynb` — visual tests for grid construction (asset grids, concentration, tensor vs Smolyak).
- `test_quadrature.ipynb` — validation and visual diagnostics for quadrature rules (Legendre, Chebyshev, Hermite, etc.).
- `test_compecon_comparison.ipynb` — comparison with CompEcon-style Broyden and quadrature.

**Examples (examples/):**
- `examples/pset1/` — Problem set 1 solutions: notebooks for each problem and report PDF (`report/pset1_report.pdf`). See `examples/pset1/README.md` for how to run.

## Documentation

- `docs/codebook/codebook.pdf` — codebook (PDF).
- `docs/compecon_comparison.md` — notes on CompEcon comparison.
- `macro_agents.md` — step-by-step guidance for AI agents and users.

## Getting Started

Install dependencies:
```
pip install -r requirements.txt
```

Example usage (Aiyagari steady state):
```python
from hetmacro.models.aiyagari import solve_steady_state
ss = solve_steady_state()
```

### Running the notebooks
1. Create and activate a Python environment (3.11+ recommended).
2. Install dependencies from `requirements.txt`.
3. Run package tests from `hetmacro/notebooks/` or full PSet 1 from `examples/pset1/notebooks/` (notebooks auto-detect the repo root).

