## macro_agents.md

This file provides practical guidance for AI agents (and users) when using `hetmacro`.

### Core Principles

- Use `grids.py` to construct state grids before anything else.
- Use `quadrature.py` to compute expectations over continuous distributions.
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

### Computing expectations with quadrature

Use `quadrature.py` to approximate integrals/expectations over continuous distributions. Each function returns `(nodes, weights)` where `E[f(X)] ≈ sum(weights * f(nodes))`.

**Choosing the right rule:**

| Distribution | Function | Notes |
|--------------|----------|-------|
| Normal | `qnwnorm(n, mu, sigma)` | Gauss-Hermite; best for income shocks |
| Lognormal | `qnwlogn(n, mu, sigma)` | Exponentiated Hermite nodes |
| Uniform | `qnwunif(n, a, b)` | Normalized Legendre weights |
| Bounded interval | `qnwlege(n, a, b)` | Gauss-Legendre; general smooth functions |
| Beta | `qnwbeta(n, a, b)` | Gauss-Jacobi; bounded [0,1] |
| Gamma | `qnwgamma(n, a, b)` | Gauss-Laguerre; positive support |
| Multivariate normal | `qnwnorm_mv(n, mu, Sigma)` | Tensor product of Hermite |
| Pre-specified grid | `qnwsimp`, `qnwtrap` | Newton-Cotes; equally-spaced nodes |

**Example: E[exp(X)] where X ~ N(0, 0.1²)**
```python
from hetmacro.quadrature import qnwnorm
import numpy as np

nodes, weights = qnwnorm(n=5, mu=0.0, sigma=0.1)
approx = np.dot(weights, np.exp(nodes))
# True answer: exp(0.5 * 0.1^2) ≈ 1.00501
```

**Example: Multivariate expectation E[exp(x - 2y)]**
```python
from hetmacro.quadrature import qnwnorm_mv
import numpy as np

n = np.array([5, 5])
mu = np.array([0.0, 0.0])
Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
nodes, weights = qnwnorm_mv(n, mu, Sigma)

def f(X):
    return np.exp(X[:, 0] - 2 * X[:, 1])

approx = np.dot(weights, f(nodes))
```

**When to use which:**
- **Gauss rules** (Hermite, Legendre, etc.): Smooth functions; optimal accuracy with few nodes.
- **Newton-Cotes** (Simpson, Trapezoid): When function values are only available on a fixed grid.

---

## Notes for Agents

- If the model has multiple state variables, construct separate grids with `GridSpec` and assemble them with `make_grid_nd`.
- If accuracy is poor near kinks, increase grid density locally via `concentration_points`.
- For computing expectations over continuous shocks, use `quadrature.py`. Match the quadrature rule to the distribution: `qnwnorm` for normal, `qnwlege` for bounded intervals, etc.
- Beware the **curse of dimensionality**: tensor-product quadrature grows as n^d. For high dimensions, consider sparse grids or Monte Carlo.
- When adding new models, place them under `hetmacro/models/` and reuse tools rather than re-implementing.

