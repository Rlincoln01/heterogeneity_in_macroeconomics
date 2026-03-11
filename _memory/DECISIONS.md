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

### 2026-03-11 — Transition path LaTeX documentation

Context: Needed mathematical reference for shooting and Broyden algorithms, in the style of the Macro Bible numerical appendix.
Decision: Created standalone `docs/transition_path_methods.tex` covering setup, shooting algorithm (with EGM details including interpolation formula), Broyden root-finding, and application to Aiyagari TFP shock. Uses Macro Bible style conventions (algorithm environments, Palatino font, motivating paragraphs).
Rationale: Serves as both a learning resource and documentation of the hetmacro transition solver internals.

### 2026-03-11 — Housekeeping: commit backlog, README + codebook sync

Context: Uncommitted work (BSpline speedup, pset5 updates, DECISIONS.md) was sitting on `dev/hetmacro-sync`. README was missing pset5. Codebook Quick Reference had stale API names and an incomplete dependency graph.
Decision: Committed all pending changes in two commits. Updated README (added pset5). Fixed codebook: solvers list (added HowardGrid, NaiveEulerIteration, HowardEulerIteration), aiyagari API (solve_aiyagari_ge + capital_supply_curve), dependency graph (added household, income_process, solvers, models layers). Added .vscode/ to .gitignore.
Rationale: Keep docs and code in sync; clear the commit backlog before further development.
