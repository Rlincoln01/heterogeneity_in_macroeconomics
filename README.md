# Heterogeneity in Macroeconomics: hetmacro

This repository provides a foundational numerical toolkit (`hetmacro`) for solving macroeconomic models with heterogeneity. The package is designed to be **self-contained** (no external dependencies on QuantEcon/CompEcon), while still following their algorithmic best practices.

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
  models/
    aiyagari.py

docs/
  codebook/
    codebook.tex
    codebook.pdf

codebook.md
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

## Documentation

- `codebook.md` — plain-text codebook.
- `docs/codebook/codebook.tex` — LaTeX codebook (compiled to PDF).
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

