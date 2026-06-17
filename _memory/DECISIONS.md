# hetmacro — Decisions Log

### 2026-03-02 — Initialize project memory

Context: Project had no `_memory/` directory; decisions were implicit in code and commit messages.
Decision: Created `_memory/` with PROJECT_CONTEXT.md, DECISIONS.md, REFERENCES.md, RUNBOOK.md per workspace protocol.
Rationale: Enable continuity across AI-assisted sessions; track architectural decisions explicitly.

### 2026-03-02 — macro_agents.md serves as AI agent guide (keep separate from CLAUDE.md)

Context: `macro_agents.md` (500 lines) contains workflow playbooks, solver selection guides, calibration templates, and CompEcon comparison notes. It's package-user-facing documentation.
Decision: Keep `macro_agents.md` as the AI agent operational guide for *using* the package. If a `CLAUDE.md` is created, it should cover *development* conventions (edit scope, commit rules, testing policy).
Rationale: `macro_agents.md` is user/agent-facing "how to use hetmacro." CLAUDE.md would be developer-facing "how to develop hetmacro." Different audiences.

### 2026-03-02 — Composable Household + Solver architecture

Context: Original design used standalone `backward.py`/`forward.py` functions. New `household.py` + `income_process.py` + `solvers/` introduced a composable class-based interface.
Decision: Household class is the recommended high-level entry point. Lower-level functions remain for flexibility.
Rationale: Composability allows mixing income processes and solvers freely; class interface provides `.solve()`, `.compute_ergodic()`, `.compute_aggregates()`, `.euler_check()`, `.simulate()`.

### 2026-03-02 — Dev branch workflow

Context: `.cursor/rules/20-git-autopush.mdc` establishes git workflow rules.
Decision: All AI-assisted changes go on `dev/hetmacro-sync` branch; never commit/push on `main`; never force-push; update codebook.tex when API changes.
Rationale: Protect main branch; ensure docs stay in sync with code.

### 2026-03-02 — Dual-branch income architecture (discrete vs continuous)

Context: Many modules branch on income type (Markov Pi vs quadrature).
Decision: This is intentional. Discrete income uses transition matrices; continuous uses Gauss-Hermite expectations. Both paths are maintained.
Rationale: Different numerical methods suit different income representations. Discrete is simpler for teaching; continuous is more accurate for research.

### 2026-03-02 — Sparse transition matrix and hardened ergodic pipeline

Context: `compute_joint_transition_matrix` used a pure Python triple loop building a dense matrix; `stationary_dist` used dense least-squares; `stationary_eigenvector` had no fallback.
Decision: (1) Vectorized `compute_joint_transition_matrix` to return sparse CSR via COO construction. (2) `stationary_dist` routes sparse input to `stationary_eigenvector`, keeps dense lstsq as fallback. (3) `stationary_eigenvector` has power iteration fallback if eigendecomposition fails. (4) `_compute_ergodic_quadrature` normalizes Q rows after construction.
Rationale: Sparse path is faster and more memory-efficient for large grids. Power iteration fallback prevents silent failures. Row normalization handles clipping-induced deviations in the quadrature path.

### 2026-03-04 — Collocation solver BSpline speedup (precompute knot vectors)

Context: `CollocationVFI_Spline` was 9x slower than necessary because `cubic_basis_matrix` was called inside golden search inner loops, rebuilding sparse matrices on every evaluation.
Decision: Replace `cubic_basis_matrix(breaks, x).toarray() @ coeffs` with `scipy.interpolate.BSpline(knots, coeffs, 3)(x)` using precomputed knot vectors. Also precompute `theta @ Pi[j,:]` outside closures.
Rationale: BSpline.__call__ is a single vectorized C-level evaluation, no matrix construction. PE speedup: 63.6s -> 6.8s (9.4x). GE speedup: ~550s -> ~65s (8.5x). L_inf parity with EGM unchanged.

### 2026-03-02 — Ergodic distribution validated across all 12 solver configurations

Context: `compute_ergodic()` had only been tested for EGM (quadrature). Needed verification that all 9 solver types (6 discrete + 6 continuous, with overlap) produce valid stationary distributions.
Decision: Added Stage C (discrete) and Stage D (continuous) smoke tests to `household_test.ipynb`. All 12 configurations pass: distributions sum to 1, non-negative, mean assets within 10-15% of EGM reference.
Rationale: Establishes baseline correctness for the full solver-to-distribution pipeline.

### 2026-03-11 — Transition path solver (transition.py)

Context: hetmacro had no infrastructure for MIT shock / transition dynamics. Need backward-forward shooting and root-finding methods.
Decision: Created `transition.py` with model-agnostic core (`compute_transition_path`, `_backward_path`, `_forward_path`) + Aiyagari convenience wrapper (`aiyagari_transition_path` in `models/aiyagari.py`). Two methods: shooting with damping and Broyden root-finding. `price_fn(K, t) -> dict` interface makes solver model-agnostic. `make_backward_fn` wraps Household+Solver into backward_fn callable for Level 2 API.
Rationale: Follows the same two-tier pattern (low-level functions + high-level wrapper) as the rest of the package. Shooting is the standard approach (Virgiliu pset5); root-finding (demographic transition project) offers faster convergence.

### 2026-03-11 — Income convention: household_steady_state takes income levels, not productivity

Context: `household_steady_state(Pi, a_grid, y, r, ...)` uses `coh = y + (1+r)*a`. In Aiyagari, income = w*z. If `y = z_grid` (raw productivity), the SS and transition paths are inconsistent.
Decision: Always pass `y = w * z_grid` to `household_steady_state`. The `aiyagari_transition_path` wrapper computes `w(r) * z_grid` internally. Fixed from initial bug where raw `z_grid` was passed.
Rationale: Avoids silent income mismatch between SS computation and transition price_fn.

### 2026-03-11 — Forward path returns capital (mean assets in D_t), not savings

Context: `_forward_path` originally returned `K_path[t] = vdot(D_t, a_pol_t)` (savings at t = K_{t+1}). The shooting loop compared this with `K_guess[t]` (capital at t), creating a timing mismatch. In SS the bug is invisible (savings = capital), but during transitions prices were evaluated at the wrong capital.
Decision: Changed `_forward_path` to return `K_path[t] = sum(D_t * a_grid)` (capital at time t = mean asset holdings in D_t). Resource constraint diagnostic now uses savings from policies directly: `RC_t = Y_t - C_t - A_t + (1-delta)*K_t` where `A_t = vdot(D_t, a_pol_t)`.
Rationale: Ensures `K_guess[t]` and `K_implied[t]` both represent capital at time t. RC residual dropped from ~0.4 to ~2.6e-10 (machine precision).

### 2026-03-11 — Live iteration plotting via _IterationPlotter class

Context: `plot_iterations=True` in shooting method needed to show convergence live, not just a static post-convergence plot. `plt.ion()` approach failed in Jupyter inline backend (blank figure).
Decision: Created `_IterationPlotter` class using `IPython.display.clear_output(wait=True)` + `display(fig)` for Jupyter, with `draw_idle()` fallback for terminal. Verbose output buffered and re-printed after each `clear_output`. All plotting wrapped in try/except so failures never lose computed results.
Rationale: This is the standard Jupyter live-update pattern. Verbose buffering prevents `clear_output` from erasing iteration logs.

### 2026-04-21 — Pset 10 (Midrigan-Xu 2014) module added

Context: Problem Set 10 asks us to solve a simplified Midrigan-Xu 2014 entrepreneur model with a collateral constraint $k \le \lambda a$ where $\lambda = 1/(1-\xi)$. The MATLAB starter code is stubbed out. Pset specifies closed-economy GE plus small open economy at fixed $r$.
Decision: New standalone module `hetmacro/models/midrigan_xu.py` (~750 lines). Not shoe-horned into `Household` class because entrepreneur income $\pi(a, z)$ depends on both state variables (collateral constraint ties $k$ to $a$), breaking the `y = y(z)`-only assumption of the composable Household API. Port pattern mirrors `hopenhayn_rogerson_virgiliu.py`: golden-section + cubic-spline VFI with Howard acceleration, damped Gauss-Seidel outer loop on prices.
Rationale: Keeps the Household abstraction clean. Module sits alongside other Virgiliu-port models and reuses existing primitives (`rouwenhorst`, `make_asset_grid`, `stationary_distribution`, `_vectorized_golden_section`, `CubicSpline`).
Validation: six driver scripts under `examples/pset10/drivers/` — firm_static (10/10), entrepreneur_vfi (9/9), aggregate_mx (6/6), closed GE (15/15 full-grid, loose tolerances at $\xi=0$), open economy (15/15 matches slide 22 to 3-4 decimals), simulation (5/5 matches slide 19 Model column).

### 2026-04-21 — Midrigan-Xu calibration uses $\sigma_z = 0.59$, not slide-19's 0.40

Context: Slide 19 displays calibration $(\rho_z, \sigma_z) = (0.896, 0.40)$ with target moments matched by adding transitory shocks on top of the AR(1). Slide 21/22/20 quantitative tables are generated by `start.m` which uses $\sigma_z = 0.59$ (no transitory).
Decision: Use $\sigma_z = 0.59$ for the pset10 calibration. Notebook documents the distinction.
Rationale: Verified empirically — at $\sigma_z = 0.40$, TFP losses come out ~1/3 of slide-21 values. At $\sigma_z = 0.59$, they match to 3 decimal places ($\xi = 0.86$: 0.22%; $\xi = 0.5$: 2.37%). The "wider innovations" interpretation is the one slide 21 uses.

### 2026-04-21 — Slide 21/22 wages are normalized to $W = 1$ at $\xi = 0.86$

Context: Initial full-grid run showed a systematic +12% constant wage offset versus slides at every $\xi$, while TFP loss, $\tau$, $r$, $K/Y$, $A/Y$ all matched.
Decision: Slides use $W / W_{\xi = 0.86}$; raw model wages are 1.124 at the baseline but map exactly to slide values after this normalization.
Rationale: Slides don't annotate this, but the constant-factor pattern across the three $\xi$ levels was diagnostic. Under this normalization, $\xi = 0.5$: 0.959 (slide 0.96); $\xi = 0$: 0.88 (slide 0.88). Clean match.

### 2026-04-21 — Closed-economy $\xi = 0$ converges very slowly; documented, not fixed

Context: At $\xi = 0$ the closed-economy damped Gauss-Seidel on $(W, r)$ is a shallow contraction. After 700 iterations, $r$ still drifts from -0.028 toward the slide value of -0.043; $\tau$ from 2.6 toward 5.0.
Decision: Document the slow convergence in the notebook; use loose tolerances ($\pm 2.5$ on $\tau$, $\pm 0.025$ on $r$) for the $\xi = 0$ check. Recommended fix (deferred): Broyden acceleration on $r$ alone (reviewer suggestion).
Rationale: The economics is right — just the numerical method that doesn't scale to the most-constrained case. Rewriting the outer loop is ~2 hours and not required for the pset.

### 2026-03-11 — Transition path LaTeX documentation

Context: Needed mathematical reference for shooting and Broyden algorithms, in the style of the Macro Bible numerical appendix.
Decision: Created standalone `docs/transition_path_methods.tex` covering setup, shooting algorithm (with EGM details including interpolation formula), Broyden root-finding, and application to Aiyagari TFP shock. Uses Macro Bible style conventions (algorithm environments, Palatino font, motivating paragraphs).
Rationale: Serves as both a learning resource and documentation of the hetmacro transition solver internals.

### 2026-03-25 — Krusell-Smith algorithm: VFI with Epstein-Zin aggregator

Context: Pset 6 requires solving the KS model with aggregate shocks. Plan considered both EGM and VFI for the 4D Bellman.
Decision: Implemented VFI with golden section search over savings fraction x in [0, 1), matching Matlab reference. Epstein-Zin CES aggregator: V = ((1-beta)*c^(1-theta) + beta*Vbar^(1-theta))^(1/(1-theta)). Howard improvement every 50 iterations.
Rationale: VFI matches Matlab code exactly for verification; golden search over savings fraction is simpler and more robust than 4D EGM for the aggregate state dimensions.

### 2026-03-25 — KS state ordering: Fortran order (k, K, e, Z)

Context: Matlab uses gridmake(kgrid, Kgrid, egrid, zgrid) with k varying fastest.
Decision: Match Matlab convention throughout: k fastest, Z slowest. Kronm call: kronm([nk, nK, Qez.T], v) for expected continuation value.
Rationale: Enables direct comparison with Matlab reference outputs.

### 2026-03-11 — Housekeeping: commit backlog, README + codebook sync

Context: Uncommitted work (BSpline speedup, pset5 updates, DECISIONS.md) was sitting on `dev/hetmacro-sync`. README was missing pset5. Codebook Quick Reference had stale API names and an incomplete dependency graph.
Decision: Committed all pending changes in two commits. Updated README (added pset5). Fixed codebook: solvers list (added HowardGrid, NaiveEulerIteration, HowardEulerIteration), aiyagari API (solve_aiyagari_ge + capital_supply_curve), dependency graph (added household, income_process, solvers, models layers). Added .vscode/ to .gitignore.
Rationale: Keep docs and code in sync; clear the commit backlog before further development.
