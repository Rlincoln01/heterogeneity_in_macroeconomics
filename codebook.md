# hetmacro Codebook

This codebook documents the functions available in the `hetmacro` package.

## grids.py

### `GridSpec`
Defines one grid dimension with flexible spacing and optional concentration.

**Parameters**
- `lower`, `upper` : bounds
- `n_points` : number of points
- `spacing` : `linear`, `power`, `log`, `double_exp`, `chebyshev`, `custom`
- `power` : power exponent for `power` spacing
- `concentration_points` : list of points to concentrate around
- `concentration_weight` : strength of concentration in `[0, 1]`
- `concentration_radius` : scale of concentration
- `custom_func` : map from `[0,1]` to `[lower,upper]` if `spacing='custom'`

### `make_grid_1d(lower, upper, n, spacing='linear', ...)`
Create a one-dimensional grid with flexible spacing and concentration.

**Example**
```python
from hetmacro.grids import make_grid_1d
a_grid = make_grid_1d(0, 20, 100, spacing="power", power=2.0,
                      concentration_points=[0, 3.5],
                      concentration_weight=0.4)
```

### `make_grid_nd(specs, cartesian=False)`
Build a multi-dimensional grid from a list of `GridSpec`.

### `cartesian_product(*grids, order='C')`, `gridmake(...)`
Tensor product of 1D grids into a 2D array.

### `make_asset_grid(amin, amax, n, curvature=2.0)`
Convenience wrapper for power-spaced asset grids.

### `double_exponential_grid(amin, amax, n)`
Double-exponential spacing commonly used for asset grids.

### `log_spaced_grid(amin, amax, n)`
Log-spaced grid on `(amin, amax]`.

---

## quadrature.py

Quadrature rules for numerical integration.

### Gaussian quadrature
- `qnwlege(n, a, b)` : Gauss-Legendre on `[a,b]`
- `qnwcheb(n, a, b)` : Gauss-Chebyshev on `[a,b]`
- `qnwnorm(n, mu, sigma)` : Gauss-Hermite for `N(mu, sigma^2)`
- `qnwlogn(n, mu, sigma)` : Lognormal
- `qnwunif(n, a, b)` : Uniform
- `qnwbeta(n, a, b)` : Beta
- `qnwgamma(n, a, b)` : Gamma
- `qnwnorm_mv(n, mu, Sigma)` : Multivariate normal

### Newton-Cotes
- `qnwsimp(n, a, b)` : Simpson's rule
- `qnwtrap(n, a, b)` : Trapezoid rule

---

## interpolation.py

### `interp_linear(x_grid, y, x_new)`
Vectorized 1D linear interpolation. Works for `y` of shape `(n,)` or `(m, n)`.

### `interp_bilinear(x_grid, y_grid, z, x_new, y_new)`
Bilinear interpolation on a rectilinear grid.

### `get_lottery(a_policy, a_grid)`
Compute lottery indices and probabilities for distribution iteration.

### Spline tools
- `spline_coef(x, y, kind='cubic')`
- `spline_eval(coef, x_new)`

### Chebyshev tools
- `cheb_nodes(n, a, b)`
- `cheb_basis(x, n, a, b)`
- `cheb_coef(f, n, a, b)`
- `cheb_eval(coef, x, a, b)`

---

## optimize.py

### Root-finding
- `bisect(f, a, b, ...)`
- `brentq(f, a, b, ...)`
- `newton(f, x0, fprime, ...)`
- `secant(f, x0, ...)`
- `broyden(f, x0, ...)`

### Optimization
- `golden_search(f, a, b, ...)`
- `brent_min(f, a, b, ...)`
- `nelder_mead(f, x0, ...)`

---

## markov.py

### `rouwenhorst(n, rho, sigma, mu=0)`
Rouwenhorst discretization of AR(1).

### `tauchen(n, rho, sigma, mu=0, n_std=3)`
Tauchen discretization of AR(1).

### `stationary_distribution(P, method='eigen'|'iterate')`
Compute stationary distribution of a Markov chain.

### `simulate_markov(P, s0, T)`
Simulate Markov chain indices.

---

## utils.py

### Utility functions
- `crra_utility(c, gamma)`
- `crra_marginal(c, gamma)`
- `crra_inverse_marginal(u, gamma)`
- `ces_aggregator(x, y, sigma, alpha=0.5)`

### Numerical tools
- `compute_fixed_point(T, v0, tol, max_iter)`
- `numerical_derivative(f, x, h, method)`
- `numerical_jacobian(f, x, h)`
- `tic()` / `toc()`

---

## backward.py

### `backward_egm(Va, Pi, a_grid, y, r, beta, eis)`
Endogenous Grid Method step.

### `backward_vfi(V, Pi, a_grid, y, r, beta, gamma)`
Value Function Iteration step.

### `policy_iteration(Pi, a_grid, y, r, beta, eis, method='egm')`
Policy convergence using EGM or VFI.

---

## forward.py

### `forward_iteration(D, Pi, a_i, a_pi)`
Single forward step for distributions.

### `stationary_distribution(Pi, a_policy, a_grid)`
Stationary distribution for a given policy.

### `expectation_iteration(X, Pi, a_i, a_pi)`
Expectation step for SSJ.

### `expectation_functions(X, Pi, a_i, a_pi, T)`
Compute sequence of expectation functions.

---

## steady_state.py

### `household_steady_state(Pi, a_grid, y, r, beta, eis)`
Solve household steady state for given prices.

### `market_clearing(r_bounds, household_ss_func, K_demand_func)`
Solve for general equilibrium interest rate by bisection.

---

## dp_tools.py

### `solve_policy_egm(...)`
Wrapper for a single EGM step.

### `solve_policy_vfi(...)`
Wrapper for a single VFI step.

### `solve_steady_policy(...)`
Iterate policy to convergence.

---

## distributions.py

### `stationary_distribution_ha(Pi, a_policy, a_grid)`
Stationary distribution for HA models.

### `lottery_indices(a_policy, a_grid)`
Return lottery indices and weights.

---

## ssj.py

### `jacobian(ss, shocks, T)`
Fake news algorithm Jacobians for aggregate outputs.

### `step1_backward(ss, shock, T, h=1e-4)`
Step 1 of the fake news algorithm.

### `J_from_F(F)`
Convert fake news matrix to Jacobian.

---

## models/aiyagari.py

### `solve_steady_state(...)`
Example Aiyagari model steady state using `hetmacro` tools.

