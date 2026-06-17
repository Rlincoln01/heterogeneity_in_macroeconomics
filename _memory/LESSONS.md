# hetmacro — Lessons

Project-specific lessons. Append-only. One line per lesson, prefixed with `[LEARN:tag]`.

## API gotchas (from `~/.claude/projects/.../MEMORY.md`, migrated 2026-05-06)

- `[LEARN:hetmacro]` `rouwenhorst(n, rho, sigma)` returns `(z_grid, Pi)` where `z_grid` is in **log space**. `RouwenhorstIncome.from_ar1()` exponentiates: `z_grid = np.exp(z_log)`.
- `[LEARN:hetmacro]` `household_steady_state(Pi, a_grid, y, r, beta, eis)` takes `y` as **income levels** (not raw productivity). For Aiyagari: pass `y = w * z_grid`.
- `[LEARN:hetmacro]` `backward_egm` uses `eis = 1/gamma`, not `gamma` directly. Easy to flip; check before debugging.
- `[LEARN:hetmacro]` `make_asset_grid(amin, amax, n)` — argument order is amin, amax, n.
- `[LEARN:hetmacro]` `policy_iteration` initial guess at very low `r` can produce NaN (`c_init = 0` at the borrowing limit). Use `r_low >= 0.001` in `brentq` brackets.

## Convergence

- `[LEARN:hetmacro]` 2026-03-31 — Boar-Midrigan SS with progressive HSV taxes (τ=0.45, ξ=0.20): the `(r, ι)` damped fixed-point **diverges** even with damping=0.95. Shooting works for the baseline (τ=0.25, ξ=0.05) but fails for the reform. *Why:* the mapping `(r, ι) → (r_new, ι_new)` has high sensitivity with progressive taxes; lottery-based distribution introduces non-smoothness. *How to apply:* use `method='root_finding'` (`scipy.optimize.root` with `'hybr'`) for the reform SS, starting from baseline SS values. Accept ~1e-4 residual tolerance. (K, L, ι) formulation or Anderson mixing could improve further. (Migrated from `~/.claude/projects/.../feedback_boar_midrigan_convergence.md`.)

## Damping convention

- `[LEARN:hetmacro]` Our `damping` = weight on **OLD** guess: `K = damping * K_old + (1 - damping) * K_new`. Virgiliu's "step" = weight on NEW: `step = 1 - damping`. So Virgiliu step=1/3 ↔ our damping=2/3.

## Solver defaults and convergence (added 2026-05-08)

- `[LEARN:hetmacro]` 2026-05-08 — Default grid resolution `n_z=101` over `n_z=51` for steady-state solvers. Coarse grids introduce ~9% wage error from quantization at exit thresholds in Hopenhayn (n_z=51 → 8.9% wage error vs reference). Set `n_z=101` as the default in `solve_hopenhayn_ss` and similar; users can override down for prototyping or up for production. *Evidence:* claude-mem #2068. [MIGRATED 2026-05-08 from claude-mem T5 backfill]

- `[LEARN:hetmacro]` 2026-05-08 — Two-tier API for transition-path solvers: provide a fast default (`backward_egm` directly, exact given continuation value) plus an optional `backward_fn` callable hook for custom backward iteration. Don't force one path on users. *Evidence:* claude-mem #942 (transition-path solver design, 2026-03-11). [MIGRATED 2026-05-08 from claude-mem T5 backfill]

- `[LEARN:hetmacro]` 2026-05-08 — `KL_iota` (quantity-space) iteration as alternative to `r_iota` (price-space) when the fixed-point is unstable under progressive HSV taxes. Add `method='KL_iota'` parameter to transition solvers. Extends the 2026-03-31 LESSONS entry on Boar-Midrigan SS divergence under progressive taxes. *Evidence:* claude-mem #1720. [MIGRATED 2026-05-08 from claude-mem T5 backfill]
