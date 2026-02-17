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

---

## Interpolation & approximation (practical notes)

### Chebyshev: **basis vs nodes**

- **Chebyshev basis**: a function class (polynomials \(T_0,\dots,T_{n-1}\) on \([-1,1]\)).
- **Chebyshev nodes**: a particular node set (roots / extrema mapped to \([a,b]\)) used to make polynomial interpolation **stable**.

You *can* fit a Chebyshev polynomial on any distinct node set (e.g. equispaced), but:
- **best practice** is to use **Chebyshev nodes** for Chebyshev interpolation, especially as \(n\) grows (conditioning/Runge issues).

### Multi-dimensional Chebyshev

- For tensor-product Chebyshev interpolation in \(d\) dimensions, use a **tensor product of 1D Chebyshev nodes** (one grid per dimension), then form the Cartesian product.

```python
from hetmacro.grids import make_grid_1d, gridmake

# 1D Chebyshev nodes per dimension (mapped to bounds)
a = make_grid_1d(0.0, 50.0, n, spacing="chebyshev")
z = make_grid_1d(1.0,  5.0, n, spacing="chebyshev")
e = make_grid_1d(1.0,  5.0, n, spacing="chebyshev")

# Nodes as (n^3, 3) array
nodes = gridmake(a, z, e)
```

### Grids: prefer `hetmacro.grids` helpers

- Use `make_grid_1d(...)` instead of raw `np.linspace(...)` when you want consistent spacing options (`linear`, `power`, `double_exp`, `chebyshev`, etc.).
- Use `gridmake(...)` (alias `cartesian_product`) instead of manual `meshgrid + column_stack` when building a Cartesian product of 1D grids.

### Splines: use a unified wrapper

- Use `hetmacro.interpolation.spline_fit/spline_eval` to standardize spline usage:
  - `method="linear"`: **linear spline** (degree \(k=1\))
  - `method="cubic"`: cubic spline (e.g. `bc_type="natural"`)
  - `method="pchip"`: shape-preserving (often good for monotone policy/value objects)
  - `method="akima"`: robust to local oscillations

### Implementation note (library boundaries)

- `hetmacro.interpolation.cheb_*` routines are implemented in **NumPy** (not a SciPy wrapper).
- `spline_fit` is a **thin SciPy-backed wrapper** that provides a stable, consistent API.

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
- **Documentation + GitHub policy**:
  - Treat `docs/codebook/codebook.tex` as the canonical user-facing documentation for the public API.
  - If you change any public function in `hetmacro/*.py`, update the relevant parts of `codebook.tex` (especially `\section{Public API Sync Checklist}`) and rebuild `codebook.pdf`.
  - Ensure code + codebook updates are committed and pushed to GitHub using the dev-branch workflow in `.cursor/rules/20-git-autopush.mdc` (never push from `main`).

## Broyden methods: when to use which

The package exposes a **unified Broyden solver** via `hetmacro.optimize.broyden(f, x0, method=..., ...)` with three methods. Use the one that matches your problem:

| Situation | Method | Call | Notes |
|-----------|--------|------|--------|
| **Full coupled system** \(F(h)=0\) in one shot (e.g. equilibrium residuals in all \(n\) unknowns) | `'compecon'` | `broyden(f, x0, method='compecon', tol=..., max_iter=...)` | CompEcon-style: inverse Jacobian, backstepping line search, restarts. Prefer this over `'scipy'` for nonlinear/ill-scaled systems; SciPy’s Broyden can diverge. |
| **Same full system** but you want a minimal wrapper | `'scipy'` | `broyden(f, x0, method='scipy', ...)` | Thin wrapper around `scipy.optimize.root(..., method='broyden1')`. Use only for well-behaved systems. |
| **Vectorized, largely componentwise** (e.g. many independent 1D FOCs with bounds \(a \le x \le b\)) | `'componentwise'` | `broyden(f, x0, method='componentwise', a=a, b=b, ...)` | MATLAB-style safeguarded componentwise Broyden; **requires** bounds `a` and `b`. Use for inner loop (e.g. household FOCs given wages) in a wage fixed-point iteration. |

**Rule of thumb:** For equilibrium systems that are strongly coupled and possibly ill-scaled (e.g. labor supply equilibrium \(F(h)=0\) over many nodes), use **`method='compecon'`**. For the inner step of a fixed-point iteration where you solve many independent equations with bounds (e.g. FOC for \(h_i\) given \(W\)), use **`method='componentwise'`** with `a`, `b` set to your bounds.

---

## Optimization Methods

### When to Use Each Optimizer

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| `nelder_mead` | Derivative-free, noisy objectives | Robust, no gradients needed | Slow, finds local minima |
| `golden_search` | 1D smooth functions | Fast, reliable | 1D only |
| `brent_min` | 1D with bounds | Superlinear convergence | 1D only |

### Nelder-Mead for Calibration

Best for:
- Moment matching (objective is simulation-based, noisy)
- Non-smooth or discontinuous loss functions
- When gradients are unavailable or expensive

**Workflow:**
```python
from hetmacro.optimize import nelder_mead
import numpy as np

def sse_loss(params):
    """Sum of squared errors between model and target moments."""
    model_moments = simulate_model(params)
    return np.sum((model_moments - target_moments)**2)

# Initial guess
x0 = np.array([0.9, 0.1, 0.05])

# Optimize with history tracking
params_opt, sse_opt, hist = nelder_mead(sse_loss, x0, return_info=True)
```

### Convergence Diagnostics

When `return_info=True`, `nelder_mead` returns a history dict with:
- `hist['f']`: objective value at each iteration (plot to check convergence)
- `hist['x']`: parameter vectors at each iteration

**Diagnostic checklist:**
1. Plot `hist['f']` to verify SSE decreases and stabilizes
2. Check that final simplex is small (parameters converged)
3. Restart from different initial points to check for local minima
4. If SSE is large, consider model misspecification

```python
import matplotlib.pyplot as plt

# Plot SSE trace
plt.plot(hist['f'])
plt.xlabel('Iteration')
plt.ylabel('SSE')
plt.yscale('log')
plt.title('Nelder-Mead Convergence')
plt.show()
```

---

## Calibration Workflows

### Method of Simulated Moments (MSM)

When analytical moments are unavailable, calibrate by simulation:

1. **Define target moments** from empirical data
2. **Write simulation function** that takes parameters and returns model moments
3. **Construct SSE loss** measuring distance to targets
4. **Optimize** with `nelder_mead`

### Example: Income Process Calibration

Target moments:
- sd(log y), sd(Δ log y)
- Autocorrelations at lags 1, 3, 5

Parameters to calibrate:
- ρ (persistence)
- σ_u (innovation std)
- σ_ε (measurement error std)

```python
import numpy as np
from hetmacro.optimize import nelder_mead

# Target moments from data
target = {
    'sd_logy': 0.92,
    'sd_dlogy': 0.35,
    'acf1': 0.89,
    'acf3': 0.78,
    'acf5': 0.71,
}

def simulate_income(rho, sigma_u, sigma_eps, T=10000, seed=42):
    """Simulate AR(1) with measurement error."""
    np.random.seed(seed)
    logy = np.zeros(T)
    for t in range(1, T):
        logy[t] = rho * logy[t-1] + sigma_u * np.random.randn()
    return logy + sigma_eps * np.random.randn(T)

def compute_moments(logy_obs):
    """Compute moments from simulated data."""
    dlogy = np.diff(logy_obs)
    return {
        'sd_logy': np.std(logy_obs),
        'sd_dlogy': np.std(dlogy),
        'acf1': np.corrcoef(logy_obs[1:], logy_obs[:-1])[0,1],
        'acf3': np.corrcoef(logy_obs[3:], logy_obs[:-3])[0,1],
        'acf5': np.corrcoef(logy_obs[5:], logy_obs[:-5])[0,1],
    }

def sse_loss(params):
    rho, sigma_u, sigma_eps = params
    if rho <= 0 or rho >= 1 or sigma_u <= 0 or sigma_eps <= 0:
        return 1e10  # penalty
    logy_obs = simulate_income(rho, sigma_u, sigma_eps)
    model = compute_moments(logy_obs)
    return sum((model[k] - target[k])**2 for k in target)

# Calibrate
x0 = np.array([0.9, 0.2, 0.1])
params_opt, sse_opt = nelder_mead(sse_loss, x0)
print(f"rho={params_opt[0]:.3f}, sigma_u={params_opt[1]:.3f}, sigma_eps={params_opt[2]:.3f}")
```

### Extension: Jump-Diffusion for Fat Tails

When Gaussian shocks don't match kurtosis in the data:

```python
def simulate_jump_diffusion(rho, sigma_u, sigma_eps, pi, mu_J, sigma_J, T=10000):
    """AR(1) with jump component for fat tails.
    
    J_t = b_t * kappa_t where:
      - b_t ~ Bernoulli(pi)
      - kappa_t ~ N(mu_J, sigma_J^2)
    """
    np.random.seed(42)
    logy = np.zeros(T)
    for t in range(1, T):
        jump = (np.random.rand() < pi) * (mu_J + sigma_J * np.random.randn())
        logy[t] = rho * logy[t-1] + sigma_u * np.random.randn() + jump
    return logy + sigma_eps * np.random.randn(T)
```

Ergodic variance formula for jump-diffusion:
```
Var(log y) = sigma_u^2 / (1 - rho^2) + pi * (mu_J^2 + sigma_J^2) / (1 - rho^2)
```

### Best Practices for Calibration

1. **Multiple restarts**: Run from 5-10 different initial guesses
2. **Check moment fit individually**: Which moments are hardest to match?
3. **Increase simulation length**: Reduces Monte Carlo noise (T=50000+)
4. **Parameter bounds**: Add penalty terms for invalid parameter regions
5. **Weight matrix**: Consider weighting moments by inverse variance for efficiency

---

## Agent Playbooks (Reusable Templates)

Copy-paste templates for common numerical tasks. For full theory, see Macro Bible Appendix B (Numerical Methods). For implementation details, see the codebook.

### Playbook 1: Solve equilibrium system (root finding)

**When:** Market clearing, FOC residuals, or any system \(F(x)=0\).

1. Define residual function \(F\) (vector-valued).
2. Choose initial guess (e.g. previous solution or simple heuristic).
3. Call `broyden(f, x0, method='compecon', tol=1e-10)`.
4. Check `info['converged']` and plot \(\|F(x^{(k)})\|\) vs iteration.

**Template:**
```python
from hetmacro.optimize import broyden
x_star, info = broyden(residual_fn, x0, method='compecon', tol=1e-10, return_info=True)
if not info.get('converged', True):
    # Try different x0 or componentwise with bounds
    x_star, _ = broyden(residual_fn, x0, method='componentwise', a=lb, b=ub)
```

### Playbook 2: Calibrate by simulated moments

**When:** Match model moments to data; no analytical moments.

1. Define target moments (from data).
2. Write `simulate(params)` returning model moments.
3. Define `sse(params) = sum((simulate(params) - target)**2)`.
4. Optimize with `nelder_mead(sse, x0, return_info=True)`.
5. Plot `hist['f']` for convergence; restart from several `x0` if needed.

**Template:**
```python
from hetmacro.optimize import nelder_mead
params_opt, sse_opt, hist = nelder_mead(sse_loss, x0, return_info=True)
# Optional: plot hist['f']; check final params and moment fit
```

### Playbook 3: Solve household dynamic programming

**When:** Value function or policy for consumption-savings / income fluctuation.

1. Build grids (`grids.py`) and income process (`markov.rouwenhorst`).
2. Call `policy_iteration(..., method='egm')` or `method='vfi'`.
3. Use `interp_linear` for off-grid evaluation; `get_lottery` for distribution iteration.

**Template:**
```python
from hetmacro.grids import make_grid_1d
from hetmacro.markov import rouwenhorst
from hetmacro.backward import policy_iteration
a_grid = make_grid_1d(0, 50, 200, spacing='power', power=2.0)
e, Pi = rouwenhorst(n=7, rho=0.9, sigma=0.2)
Va, a_pol, c_pol = policy_iteration(Pi, a_grid, np.exp(e), r, beta, eis, method='egm')
```

### Playbook 4: Euler equation time iteration and residuals

**When:** Solve or check policy via Euler equation (no value function).

1. Guess policy on grid; for each state, solve Euler residual = 0 for today’s choice (use `brentq` or `broyden` for 1D/vector).
2. Use quadrature (`qnwnorm`, `qnwnorm_mv`) for expectations.
3. Iterate until policy converges.
4. **Diagnostic:** Compute Euler residual (LHS − RHS of Euler equation) over state space; plot or report max residual (should be small in interior).

**Template:**
```python
from hetmacro.quadrature import qnwnorm_mv
from hetmacro.optimize import broyden  # or brentq per state
# ... define euler_residual(x, state); then solve per state
# Diagnostic: residual = euler_lhs - euler_rhs; np.max(np.abs(residual))
```

---

## CompEcon Comparison Notes- `qnwcheb` now supports `kind="clenshaw_curtis"` to match CompEcon’s `qnwcheb`. Default `kind="gauss"` retains Gauss‑Chebyshev weights.
- `qnwbeta` weights are normalized to the Beta pdf (expectations now match analytical moments).
- `qnwnorm` uses `sigma` (std) in hetmacro; CompEcon uses `sig2` (variance/covariance).
- `gridmake` output orientation differs: hetmacro returns `(N,d)` while CompEcon returns `(d,N)`. Use `.T` when needed for compatibility.