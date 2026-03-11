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
