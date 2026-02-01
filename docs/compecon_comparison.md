# CompEcon vs hetmacro — Phase 1 Comparison Report

**Scope:** Foundational modules only (`grids.py`, `quadrature.py`, `interpolation.py`, `optimize.py`, `markov.py`, `utils.py`).  
**CompEcon reference:** Python package `compecon` (installed version 2024.5.19) and website docs.  
**Primary doc link:** https://randall-romero.github.io/CompEcon/about.html

## 1. Module Mapping (Function‑by‑Function)

### 1.1 `quadrature.py`

| hetmacro | CompEcon | Notes |
|---|---|---|
| `qnwlege(n,a,b)` | `compecon.quad.qnwlege(n,a,b)` | Same purpose; CompEcon handles multi‑dim by passing vector `n` and arrays `a,b`. |
| `qnwcheb(n,a,b)` | `compecon.quad.qnwcheb(n,a,b)` | Same name; normalization differs (see Section 2). |
| `qnwnorm(n,mu,sigma)` | `compecon.quad.qnwnorm(n,mu,sig2)` | CompEcon uses variance/cov (`sig2`) and allows multivariate; hetmacro uses `sigma` (std) and 1D only. |
| `qnwlogn(n,mu,sigma)` | `compecon.quad.qnwlogn(n,mu,sig2)` | Same mapping; CompEcon uses `sig2` and multivariate. |
| `qnwunif(n,a,b)` | `compecon.quad.qnwunif(n,a,b)` | Same purpose; CompEcon supports multi‑dim. |
| `qnwsimp(n,a,b)` | `compecon.quad.qnwsimp(n,a,b)` | Same rule; both enforce odd `n` and integrate on `[a,b]`. |
| `qnwtrap(n,a,b)` | `compecon.quad.qnwtrap(n,a,b)` | Same rule. |
| `qnwbeta(n,a,b)` | `compecon.quad.qnwbeta(n,a,b)` | Same name; hetmacro weights are not normalized to Beta pdf (see Section 2). |
| `qnwgamma(n,a,b)` | `compecon.quad.qnwgamma(n,a)` | CompEcon uses only `a` (shape‑like) with scale 1; hetmacro supports scale `b`. |
| `qnwnorm_mv(n,mu,Sigma)` | `compecon.quad.qnwnorm(n,mu,sig2)` | Both implement multivariate normal; CompEcon’s `qnwnorm` already handles multivariate. |
| — | `compecon.quad.qnwequi(...)` | Equidistributed sequences (no hetmacro counterpart). |
| — | `compecon.quad.quadrect(...)` | Generic rectangular integration wrapper (no hetmacro counterpart). |

### 1.2 `grids.py`

| hetmacro | CompEcon | Notes |
|---|---|---|
| `gridmake(*grids)` | `compecon.tools.gridmake(*arrays)` | Both form Cartesian products; **output shape differs** (hetmacro: `(N,d)`, CompEcon: `(d,N)`). |
| `cartesian_product(*grids)` | `compecon.tools.gridmake` | Same concept, different orientation. |
| `make_grid_1d(...)` | `compecon.tools.nodeunif(n,a,b)` | Similar to linear spacing only; CompEcon lacks power/log/chebyshev/concentration variants. |
| `smolyak_sparse_grid(...)` | `compecon.smolyak.SmolyakGrid(...)` | Both sparse grid; CompEcon uses Smolyak objects; conventions differ. |
| `make_grid_nd(...)` | — | No direct CompEcon analog; closest is multi‑dim basis grids. |

### 1.3 `interpolation.py`

| hetmacro | CompEcon | Notes |
|---|---|---|
| `interp_linear`, `interp_bilinear` | `compecon.basisLinear.BasisLinear` | CompEcon uses basis objects; interface differs. |
| `spline_coef`, `spline_eval` | `compecon.basisSpline.BasisSpline` | CompEcon uses basis objects; hetmacro is functional. |
| `cheb_nodes`, `cheb_basis`, `cheb_coef`, `cheb_eval` | `compecon.basisChebyshev.BasisChebyshev` | Same family; CompEcon is OOP and includes basis evaluation on grids. |
| `get_lottery` | — | No direct CompEcon counterpart. |

### 1.4 `optimize.py`

| hetmacro | CompEcon | Notes |
|---|---|---|
| `bisect`, `brentq`, `newton`, `secant`, `broyden` | — | CompEcon doesn’t expose these as standalone functions; it uses `OP`/`MLE` classes and `jacobian/hessian` utilities. |
| `golden_search`, `brent_min`, `nelder_mead` | — | No direct function match in CompEcon. |
| `numerical_jacobian` (utils) | `compecon.optimize.jacobian` | Similar intent; signatures differ. |

### 1.5 `markov.py`

| hetmacro | CompEcon | Notes |
|---|---|---|
| `tauchen`, `rouwenhorst` | — | Not present in CompEcon Python package. |
| `stationary_distribution(P,...)` | `compecon.tools.markov(P)` | CompEcon returns invariant distributions for each recurrent class; hetmacro returns a single stationary distribution (assumes irreducible). |
| `simulate_markov` | — | No direct CompEcon analog. |

### 1.6 `utils.py`

| hetmacro | CompEcon | Notes |
|---|---|---|
| `tic`, `toc` | `compecon.tools.tic`, `compecon.tools.toc` | Similar timer utilities. |
| `numerical_derivative` | `compecon.optimize.jacobian` (indirect) | CompEcon has jacobian/hessian but no direct derivative function. |
| CRRA/CES utilities | — | No direct CompEcon analog. |

## 2. Method‑Level Differences (Key Findings)

### Quadrature
- **Normalization differences**:
  - `qnwcheb`: hetmacro’s default uses Gauss‑Chebyshev weights (sum = π for the Chebyshev weight). CompEcon’s `qnwcheb` uses Clenshaw‑Curtis weights (sum = 2 on `[-1,1]`).
  - `qnwbeta`: **fixed** — weights are now normalized to integrate the Beta pdf to 1 (matches CompEcon).
- **Parameterization differences**:
  - `qnwnorm`/`qnwlogn`: CompEcon uses `sig2` (variance/covariance) and supports multivariate directly; hetmacro uses `sigma` (std) and 1D.
  - `qnwgamma`: CompEcon uses a single parameter `a` (shape‑like); hetmacro uses `(a,b)` where `b` is scale.
- **Multivariate support**:
  - CompEcon’s `qnw*` routines accept vector `n` and arrays `a,b,mu,sig2` for multi‑dimensional tensor product rules.
  - hetmacro’s `qnwnorm_mv` implements multivariate normal only; other rules are 1D.

### Grids
- **Grid orientation**: CompEcon’s `gridmake` returns `(d, N)` while hetmacro returns `(N, d)`. This affects downstream reshaping and matrix operations.
- **Spacing flexibility**: hetmacro supports power, log, double‑exp, chebyshev, and concentrated grids; CompEcon’s `nodeunif` is linear only.

### Markov
- **Recurrence classes**: CompEcon’s `markov(P)` returns one invariant distribution per recurrence class; hetmacro returns a single stationary distribution (eigen or iteration), best suited for irreducible chains.

## 3. Benchmark / Validation Checks (post‑fix)

### Quadrature node/weight comparison (1D)
Comparison uses identical inputs (mapping `sigma` ↔ `sig2 = sigma^2` for CompEcon).

| Rule | max|Δnodes| | max|Δweights| | sum(weights) hetmacro | sum(weights) CompEcon |
|---|---:|---:|---:|---:|
| `qnwlege(5,-1,1)` | 1.11e‑16 | 7.36e‑15 | 2.000000 | 2.000000 |
| `qnwcheb(5,-1,1)` (gauss) | 1.23e‑16 | 4.61e‑01 | 3.141593 | 2.000000 |
| `qnwcheb(5,-1,1, kind="clenshaw_curtis")` | 0.0 | 0.0 | 2.000000 | 2.000000 |
| `qnwnorm(7,mu,σ)` vs `qnwnorm(7,mu,σ²)` | 2.22e‑16 | 3.89e‑16 | 1.000000 | 1.000000 |
| `qnwlogn(7,mu,σ)` vs `qnwlogn(7,mu,σ²)` | 4.44e‑16 | 3.89e‑16 | 1.000000 | 1.000000 |
| `qnwunif(7,0,2)` | 0.0 | 3.86e‑15 | 1.000000 | 1.000000 |
| `qnwsimp(7,0,2)` | 0.0 | 0.0 | 2.000000 | 2.000000 |
| `qnwtrap(7,0,2)` | 0.0 | 0.0 | 2.000000 | 2.000000 |
| `qnwbeta(7,2,5)` | 1.11e‑16 | 4.77e‑12 | 1.000000 | 1.000000 |
| `qnwgamma(7,2,1)` vs `qnwgamma(7,2)` | 8.88e‑16 | 1.32e‑11 | 1.000000 | 1.000000 |

### Expectation checks
- **Beta(2,5)**: true mean = 2/7 ≈ 0.285714  
  - hetmacro: 0.285714 (fixed)  
  - CompEcon: 0.285714
- **Gamma(shape=2, scale=1)**: true mean = 2  
  - hetmacro: 2.000000  
  - CompEcon: 1.99999999998

### Markov stationary distribution
For a 2x2 irreducible chain `[[0.9,0.1],[0.2,0.8]]`, both:
- hetmacro `stationary_distribution` and CompEcon `tools.markov` produce `[0.6667, 0.3333]`.

### Grid orientation check
For `gridmake([1,2,3],[4,5])`:
- CompEcon: shape `(2,6)`  
- hetmacro: shape `(6,2)`  
Same points, different layout.

## 4. Issues / Gaps to Consider in Phase 2

1. **Clarify Chebyshev normalization** (`qnwcheb`) — hetmacro now supports `kind="clenshaw_curtis"` (CompEcon). Decide whether to keep default as Gauss‑Chebyshev or switch defaults.  
2. **Parameter conventions**: add optional `sig2`/covariance and multi‑dim support for `qnwnorm`/`qnwlogn`, or keep the existing 1D API and document differences.  
3. **Gamma parameterization**: decide whether to align to CompEcon’s single‑parameter (`a`) or keep `(a,b)`; possibly allow both.  
4. **Grid orientation**: decide whether to offer a CompEcon‑compatible layout option for `gridmake` or add a helper for `(d,N)` output.  
5. **Missing counterparts**: `qnwequi`, `quadrect`, and recurrence‑class invariant distributions are not in hetmacro. Should we add them or document as intentional gaps?

## 5. Technical Notes
- Importing `compecon.quadrature` raises an `IndentationError` in the installed package. All comparisons above use `compecon.quad` instead.

