## macro_agents.md

This file provides practical guidance for AI agents (and users) when using `hetmacro`.

### Core Principles

- Use `grids.py` to construct state grids before anything else.
- Use `markov.py` to discretize income or productivity processes.
- Use `backward.py` for policy iteration (EGM or VFI).
- Use `forward.py` or `distributions.py` to obtain stationary distributions.
- Use `steady_state.py` to assemble household steady state and market clearing.

---

## Step-by-step Workflow

### 1) Create grids
```python
from hetmacro.grids import GridSpec, make_grid_nd

specs = [
    GridSpec(0, 50, 500, spacing="power", power=2.0, concentration_points=[0, 3.5]),
    GridSpec(-0.2, 0.2, 5, spacing="linear")
]
grids, _ = make_grid_nd(specs, cartesian=False)
a_grid = grids[0]
```

### 2) Discretize income process
```python
from hetmacro.markov import rouwenhorst, stationary_distribution

e, Pi = rouwenhorst(n=7, rho=0.9, sigma=0.2)
pi = stationary_distribution(Pi, method="iterate")
```

### 3) Solve household policy functions
```python
from hetmacro.backward import policy_iteration

Va, a_pol, c_pol = policy_iteration(Pi, a_grid, y, r, beta, eis, method="egm")
```

### 4) Obtain stationary distribution
```python
from hetmacro.forward import stationary_distribution

D = stationary_distribution(Pi, a_pol, a_grid)
```

### 5) Compute steady state
```python
from hetmacro.steady_state import household_steady_state

ss = household_steady_state(Pi, a_grid, y, r, beta, eis)
```

### 6) General equilibrium (Aiyagari)
```python
from hetmacro.models.aiyagari import solve_steady_state

ss = solve_steady_state()
```

---

## Common Patterns

### Concentrating grids around kinks
```python
from hetmacro.grids import make_grid_1d

a_grid = make_grid_1d(0, 20, 100,
                      spacing="power",
                      power=2.0,
                      concentration_points=[0, 3.5],
                      concentration_weight=0.4)
```

### Switching between EGM and VFI
```python
Va, a_pol, c_pol = policy_iteration(Pi, a_grid, y, r, beta, eis, method="egm")
# or
Va, a_pol, c_pol = policy_iteration(Pi, a_grid, y, r, beta, eis, method="vfi")
```

---

## Notes for Agents

- If the model has multiple state variables, construct separate grids with `GridSpec` and assemble them with `make_grid_nd`.
- If accuracy is poor near kinks, increase grid density locally via `concentration_points`.
- When adding new models, place them under `hetmacro/models/` and reuse tools rather than re-implementing.

