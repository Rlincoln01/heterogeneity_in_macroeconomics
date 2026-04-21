"""Howard's improvement solver with endogenous labor supply.

Solves the household problem for a Bewley economy with:
    - Separable utility: u(c) - v(h) = c^{1-theta}/(1-theta) - h^{1+gamma}/(1+gamma)
    - HSV progressive tax: T(y) = y - (1-tau)/(1-xi)*y^{1-xi} - iota
    - Budget constraint: c + a' = a + yhat(y), where y = W*e*h + r*a
    - Borrowing constraint: a' >= 0

Algorithm: Howard's improvement (policy iteration) combined with the Euler
equation for policy updates. For each state (a, e), the optimal hours
h in [h_bar, 1] are found by root-finding on the Euler residual. Once h
is found, c follows from the intratemporal FOC and a' from the budget.
Policy evaluation (Howard step) fixes policies and iterates the value
function and marginal value many times to accelerate convergence.

References
----------
Boar, Midrigan (2022). "Efficient Redistribution." JME.
Virgiliu Midrigan, Pset 7 — solve_dynamic.m, solveh.m, savings.m, euler.m
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..household import SolvedPolicyLabor
from ..kronm import kronm
from ..taxes import hsv_post_tax_and_marginal


# ── Vectorized bisection ─────────────────────────────────────────────


def _vectorized_bisect(f, lo, hi, args=(), tol=1e-12, max_iter=60):
    """Vectorized bisection on f(x, *args) = 0 for each element.

    Parameters
    ----------
    f : callable
        f(x, *args) -> ndarray, same shape as x.
    lo, hi : ndarray
        Lower and upper bounds, same shape.
    args : tuple
        Extra arguments passed to f.
    tol : float
        Absolute tolerance on the bracket width.
    max_iter : int
        Maximum bisection iterations.

    Returns
    -------
    ndarray
        Root estimates, same shape as lo.
    """
    lo = np.asarray(lo, dtype=float).copy()
    hi = np.asarray(hi, dtype=float).copy()

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid, *args)
        # Where f(mid) > 0, root is in [lo, mid]; else in [mid, hi]
        mask = f_mid > 0
        hi = np.where(mask, mid, hi)
        lo = np.where(mask, lo, mid)
        if np.max(hi - lo) < tol:
            break

    return 0.5 * (lo + hi)


# ── Helper functions (inner routines) ────────────────────────────────


def _savings(h, a, e, W, r, tau, xi, iota, theta, gamma):
    """Given hours h, compute consumption, savings, and tax outcomes.

    Uses the intratemporal FOC: h^gamma = c^{-theta} * (1 - T'(y)) * W * e
    to recover c from h.

    Parameters
    ----------
    h : ndarray
        Hours worked, shape (n,).
    a : ndarray
        Current assets, shape (n,).
    e : ndarray
        Productivity, shape (n,).
    W, r : float
        Wage and interest rate.
    tau, xi, iota : float
        HSV tax parameters.
    theta, gamma : float
        CRRA and inverse Frisch elasticity.

    Returns
    -------
    kprime : ndarray  — next-period assets
    c : ndarray       — consumption
    y : ndarray       — pre-tax income
    yhat : ndarray    — post-tax income
    mtax : ndarray    — marginal tax rate
    """
    y = W * e * h + r * a
    yhat, mtax = hsv_post_tax_and_marginal(y, tau, xi, iota)

    # Intratemporal FOC: h^gamma = c^{-theta} * (1 - mtax) * W * e
    # => c = ((1 - mtax) * W * e / h^gamma)^{1/theta}
    retention = np.maximum((1.0 - mtax) * W * e, 1e-15)
    h_safe = np.maximum(h, 1e-15)
    c = (retention / h_safe**gamma) ** (1.0 / theta)
    c = np.maximum(c, 1e-15)

    kprime = a + yhat - c
    return kprime, c, y, yhat, mtax


def _savings_residual(h, a, e, W, r, tau, xi, iota, theta, gamma):
    """Residual for finding h_bar: savings(h) = k' = 0.

    Returns k' so that root-finding on this gives the minimum hours
    consistent with the borrowing constraint a' >= 0.
    """
    kprime, _, _, _, _ = _savings(h, a, e, W, r, tau, xi, iota, theta, gamma)
    return kprime


def _euler_residual(h, EG_vals, a, e, W, r, tau, xi, iota, theta, gamma, beta):
    """Euler equation residual.

    residual = c^{-theta} - beta * (1 + (1 - T'(y')) * r) * EG(k', e') - chi

    When the agent is constrained (k' <= 0), the multiplier chi absorbs
    the gap, so the residual is defined for unconstrained agents.
    """
    kprime, c, _, _, mtax = _savings(h, a, e, W, r, tau, xi, iota, theta, gamma)
    lhs = c ** (-theta)
    rhs = beta * (1.0 + (1.0 - mtax) * r) * EG_vals
    return lhs - rhs


def _period_utility(c, h, theta, gamma):
    """Period utility: u(c, h) = c^{1-theta}/(1-theta) - h^{1+gamma}/(1+gamma)."""
    c_safe = np.maximum(c, 1e-15)
    h_safe = np.maximum(h, 1e-15)
    if abs(theta - 1.0) < 1e-12:
        uc = np.log(c_safe)
    else:
        uc = c_safe ** (1.0 - theta) / (1.0 - theta)
    vh = h_safe ** (1.0 + gamma) / (1.0 + gamma)
    return uc - vh


# ── Main solver ──────────────────────────────────────────────────────


def solve_household_labor(
    Pi, a_grid, e_grid, W, r, tau, xi, iota,
    beta, theta, gamma,
    h_init=None, g_init=None, v_init=None,
    n_warmup=10, n_howard=50, n_howard_inner=20, n_broyden=20,
    tol=1e-7, verbose=True,
):
    """Solve the household problem with endogenous labor and HSV taxes.

    Uses Howard's improvement (policy iteration) with the Euler equation
    for policy updates, following the three-stage algorithm of
    solve_dynamic.m.

    Parameters
    ----------
    Pi : ndarray, shape (n_e, n_e)
        Markov transition matrix for productivity.
    a_grid : ndarray, shape (n_a,)
        Asset grid.
    e_grid : ndarray, shape (n_e,)
        Productivity grid (levels, not logs).
    W : float
        Equilibrium wage.
    r : float
        Equilibrium interest rate.
    tau, xi, iota : float
        HSV tax parameters.
    beta : float
        Discount factor.
    theta : float
        CRRA parameter.
    gamma : float
        Inverse Frisch elasticity.
    h_init, g_init, v_init : ndarray, optional
        Initial guesses for hours, marginal value, and value function.
        Shape (n_e, n_a) each. If None, computed from h_bar.
    n_warmup : int
        Stage 1 warm-up iterations.
    n_howard : int
        Stage 2 outer iterations (Howard's improvement).
    n_howard_inner : int
        Stage 2 inner iterations per Howard step (policy evaluation).
    n_broyden : int
        Stage 3 Broyden/bisection refinement iterations.
    tol : float
        Convergence tolerance on relative change in h.
    verbose : bool
        Print convergence info.

    Returns
    -------
    SolvedPolicyLabor
        Solved policy functions and value/marginal-value arrays.
    """
    n_e, n_a = len(e_grid), len(a_grid)
    n_s = n_e * n_a

    # Build state arrays (Fortran order: a varies fastest within each e block)
    # s[:, 0] = assets, s[:, 1] = productivity
    a_flat = np.tile(a_grid, n_e)           # (n_s,)
    e_flat = np.repeat(e_grid, n_a)         # (n_s,)

    # Asset grid spacing for finite differences on v (used to compute g = V'(a))
    da_grid = np.diff(a_grid)  # (n_a-1,)

    # ── Compute h_bar: minimum hours consistent with a' >= 0 ──
    # Vectorized bisection on savings(h) = k' = 0
    def _hbar_residual(h_vec, a_vec, e_vec):
        kp, _, _, _, _ = _savings(h_vec, a_vec, e_vec, W, r, tau, xi, iota, theta, gamma)
        return kp

    lo_hbar = np.full(n_s, 1e-5)
    hi_hbar = np.full(n_s, 10.0)

    # Check if lower bound already gives k' > 0 (agent doesn't need to work much)
    kp_lo = _hbar_residual(lo_hbar, a_flat, e_flat)
    needs_bisect = kp_lo < 0  # only bisect where k'(h_lo) < 0

    hbar = lo_hbar.copy()
    if np.any(needs_bisect):
        hbar[needs_bisect] = _vectorized_bisect(
            _hbar_residual,
            lo_hbar[needs_bisect], hi_hbar[needs_bisect],
            args=(a_flat[needs_bisect], e_flat[needs_bisect]),
            tol=1e-14
        )

    # ── Initialize h, g, v ──
    if h_init is not None:
        h = h_init.ravel(order='C').copy()
    else:
        h = hbar.copy()

    if v_init is not None:
        v = v_init.ravel(order='C').copy()
    else:
        v = np.zeros(n_s)

    _, c, _, _, mtax = _savings(h, a_flat, e_flat, W, r, tau, xi, iota, theta, gamma)

    if g_init is not None:
        g = g_init.ravel(order='C').copy()
    else:
        # Initial guess: g = c^{-theta} * (1 + (1-mtax)*r)
        g = c ** (-theta) * (1.0 + (1.0 - mtax) * r)

    # Interpolation helper
    def _make_interpolant(vals):
        """Create interpolant over (a, e) grid from flat array."""
        return RegularGridInterpolator(
            (a_grid, e_grid),
            vals.reshape(n_e, n_a).T,  # transpose: interp expects (n_a, n_e)
            method='linear', bounds_error=False, fill_value=None
        )

    def _eval_interpolant(interp, kp, ep):
        """Evaluate interpolant at (kp, ep) points."""
        pts = np.column_stack([kp, ep])
        return interp(pts)

    def _compute_expectations(vals):
        """Compute E[vals(a, e')] = Pi @ vals reshaped appropriately.

        Uses kronm for efficiency: (I_a ⊗ Pi) @ vals
        """
        return kronm([n_a, Pi], vals)

    # ── Stage 1: Warm-up policy iteration ──
    if verbose:
        print("Stage 1: Warm-up policy iteration")
        print(f"{'iter':>4s} {'err_h':>10s} {'err_g':>10s} {'err_v':>10s}")

    for it in range(n_warmup):
        h_old = h.copy()
        g_old = g.copy()
        v_old = v.copy()

        # Expected continuation values
        gbar = _compute_expectations(g)
        vbar = _compute_expectations(v)
        EG_interp = _make_interpolant(gbar)
        EV_interp = _make_interpolant(vbar)

        # Solve for optimal hours
        h = _solve_hours(hbar, EG_interp, a_flat, e_flat,
                         W, r, tau, xi, iota, theta, gamma, beta)

        # Update value function
        kprime, c, y, yhat, mtax = _savings(
            h, a_flat, e_flat, W, r, tau, xi, iota, theta, gamma)
        kp_clipped = np.clip(kprime, a_grid[0], a_grid[-1])
        EV_vals = _eval_interpolant(EV_interp, kp_clipped, e_flat)
        v = _period_utility(c, h, theta, gamma) + beta * EV_vals

        # Update marginal value via envelope
        EG_vals = _eval_interpolant(EG_interp, kp_clipped, e_flat)
        chi = np.maximum(c ** (-theta) - beta * (1.0 + (1.0 - mtax) * r) * EG_vals, 0.0)
        g = c ** (-theta) * (1.0 + (1.0 - mtax) * r)  # simplified envelope for warm-up

        err_h = np.linalg.norm(h - h_old) / max(np.linalg.norm(h), 1e-15)
        err_g = np.linalg.norm(g - g_old) / max(np.linalg.norm(g), 1e-15)
        err_v = np.linalg.norm(v - v_old) / max(np.linalg.norm(v), 1e-15)

        if verbose:
            print(f"{it+1:4d} {err_h:10.2e} {err_g:10.2e} {err_v:10.2e}")

        if err_h < tol:
            break

    if verbose:
        print()

    # ── Stage 2: Howard's improvement ──
    if verbose:
        print("Stage 2: Howard's improvement")
        print(f"{'iter':>4s} {'err_h':>10s} {'err_g':>10s} {'err_v':>10s}")

    for it in range(n_howard):
        h_old = h.copy()
        g_old = g.copy()
        v_old = v.copy()

        # Expected continuation values
        gbar = _compute_expectations(g)
        vbar = _compute_expectations(v)
        EG_interp = _make_interpolant(gbar)
        EV_interp = _make_interpolant(vbar)

        # Policy update: solve for optimal hours
        h = _solve_hours(hbar, EG_interp, a_flat, e_flat,
                         W, r, tau, xi, iota, theta, gamma, beta)

        # Compute Euler residual (= constraint multiplier for constrained agents)
        kprime, c, y, yhat, mtax = _savings(
            h, a_flat, e_flat, W, r, tau, xi, iota, theta, gamma)
        kp_clipped = np.clip(kprime, a_grid[0], a_grid[-1])
        EG_vals = _eval_interpolant(EG_interp, kp_clipped, e_flat)
        chi = np.maximum(
            c ** (-theta) - beta * (1.0 + (1.0 - mtax) * r) * EG_vals, 0.0
        )

        # Value function update
        EV_vals = _eval_interpolant(EV_interp, kp_clipped, e_flat)
        v = _period_utility(c, h, theta, gamma) + beta * EV_vals

        # Howard's improvement: iterate v with fixed policies, compute g = dv/da
        for j in range(n_howard_inner):
            g_old_inner = g.copy()

            gbar = _compute_expectations(g)
            vbar = _compute_expectations(v)
            EG_interp = _make_interpolant(gbar)
            EV_interp = _make_interpolant(vbar)

            EG_vals = _eval_interpolant(EG_interp, kp_clipped, e_flat)
            EV_vals = _eval_interpolant(EV_interp, kp_clipped, e_flat)

            # Value function update
            v = _period_utility(c, h, theta, gamma) + beta * EV_vals

            # g = V'(a) via finite differences on v (stable, avoids noisy MPC)
            v_2d = v.reshape(n_e, n_a)
            g_2d = np.zeros_like(v_2d)
            # Central differences (interior)
            g_2d[:, 1:-1] = ((v_2d[:, 2:] - v_2d[:, :-2])
                             / (a_grid[2:] - a_grid[:-2]))
            # One-sided at boundaries
            g_2d[:, 0] = (v_2d[:, 1] - v_2d[:, 0]) / (a_grid[1] - a_grid[0])
            g_2d[:, -1] = (v_2d[:, -1] - v_2d[:, -2]) / (a_grid[-1] - a_grid[-2])
            g = g_2d.ravel()

            err_g_inner = np.linalg.norm(g - g_old_inner) / max(np.linalg.norm(g), 1e-15)
            if err_g_inner < 1e-9:
                break

        err_h = np.linalg.norm(h - h_old) / max(np.linalg.norm(h), 1e-15)
        err_g = np.linalg.norm(g - g_old) / max(np.linalg.norm(g), 1e-15)
        err_v = np.linalg.norm(v - v_old) / max(np.linalg.norm(v), 1e-15)

        if verbose:
            print(f"{it+1:4d} {err_h:10.2e} {err_g:10.2e} {err_v:10.2e}")

        if err_h < tol:
            break

    if verbose:
        print()

    # ── Stage 3: Broyden/bisection refinement ──
    if verbose:
        print("Stage 3: Broyden/bisection refinement")
        print(f"{'iter':>4s} {'err_h':>10s} {'err_g':>10s} {'err_v':>10s}")

    for it in range(n_broyden):
        h_old = h.copy()
        g_old = g.copy()
        v_old = v.copy()

        gbar = _compute_expectations(g)
        vbar = _compute_expectations(v)
        EG_interp = _make_interpolant(gbar)
        EV_interp = _make_interpolant(vbar)

        # Use Broyden/bisection for h (same as solveh but with tighter tol)
        h = _solve_hours(hbar, EG_interp, a_flat, e_flat,
                         W, r, tau, xi, iota, theta, gamma, beta)

        kprime, c, y, yhat, mtax = _savings(
            h, a_flat, e_flat, W, r, tau, xi, iota, theta, gamma)
        kp_clipped = np.clip(kprime, a_grid[0], a_grid[-1])

        EV_vals = _eval_interpolant(EV_interp, kp_clipped, e_flat)
        v = _period_utility(c, h, theta, gamma) + beta * EV_vals

        # g = V'(a) via finite differences on v
        v_2d = v.reshape(n_e, n_a)
        g_2d = np.zeros_like(v_2d)
        g_2d[:, 1:-1] = ((v_2d[:, 2:] - v_2d[:, :-2])
                         / (a_grid[2:] - a_grid[:-2]))
        g_2d[:, 0] = (v_2d[:, 1] - v_2d[:, 0]) / (a_grid[1] - a_grid[0])
        g_2d[:, -1] = (v_2d[:, -1] - v_2d[:, -2]) / (a_grid[-1] - a_grid[-2])
        g = g_2d.ravel()

        err_h = np.linalg.norm(h - h_old) / max(np.linalg.norm(h), 1e-15)
        err_g = np.linalg.norm(g - g_old) / max(np.linalg.norm(g), 1e-15)
        err_v = np.linalg.norm(v - v_old) / max(np.linalg.norm(v), 1e-15)

        if verbose:
            print(f"{it+1:4d} {err_h:10.2e} {err_g:10.2e} {err_v:10.2e}")

        if err_h < tol:
            break

    if verbose:
        print()

    # ── Final policy retrieval ──
    gbar = _compute_expectations(g)
    EG_interp = _make_interpolant(gbar)
    h = _solve_hours(hbar, EG_interp, a_flat, e_flat,
                     W, r, tau, xi, iota, theta, gamma, beta)
    kprime, c, y, yhat, mtax = _savings(
        h, a_flat, e_flat, W, r, tau, xi, iota, theta, gamma)

    # Clip and adjust c (ensure assets stay in grid)
    excess = kprime - np.clip(kprime, a_grid[0], a_grid[-1])
    c = c + excess
    kprime = np.clip(kprime, a_grid[0], a_grid[-1])

    # Reshape to (n_e, n_a)
    def _reshape(x):
        return x.reshape(n_e, n_a)

    return SolvedPolicyLabor(
        policy_a=_reshape(kprime),
        policy_c=_reshape(c),
        a_grid=a_grid,
        z_grid=e_grid,
        value=_reshape(v),
        policy_h=_reshape(h),
        policy_y=_reshape(y),
        policy_yhat=_reshape(yhat),
        policy_mtax=_reshape(mtax),
        marginal_value=_reshape(g),
    )


# ── Hours solver (solveh equivalent) ────────────────────────────────


def _solve_hours(hbar, EG_interp, a_flat, e_flat,
                 W, r, tau, xi, iota, theta, gamma, beta):
    """Find optimal hours for each state point (vectorized).

    For unconstrained agents: vectorized bisection on Euler residual in [hbar, hbar+5].
    For constrained agents: h = hbar (borrowing constraint binds).
    """
    h = hbar.copy()
    a_lo = EG_interp.grid[0][0]
    a_hi = EG_interp.grid[0][-1]

    # Check which agents are unconstrained
    kp_hbar, c_hbar, _, _, mtax_hbar = _savings(
        hbar, a_flat, e_flat, W, r, tau, xi, iota, theta, gamma)
    kp_clipped = np.clip(kp_hbar, a_lo, a_hi)
    EG_at_hbar = EG_interp(np.column_stack([kp_clipped, e_flat]))

    euler_at_hbar = (c_hbar ** (-theta)
                     - beta * (1.0 + (1.0 - mtax_hbar) * r) * EG_at_hbar)
    unconstrained = euler_at_hbar < 0
    idx = np.where(unconstrained)[0]

    if len(idx) == 0:
        return h

    # Vectorized Euler residual for bisection
    a_unc = a_flat[idx]
    e_unc = e_flat[idx]
    hbar_unc = hbar[idx]

    def _euler_vec(h_vec, a_vec, e_vec):
        kp, c, _, _, mtax = _savings(
            h_vec, a_vec, e_vec, W, r, tau, xi, iota, theta, gamma)
        kp_clip = np.clip(kp, a_lo, a_hi)
        EG_vals = EG_interp(np.column_stack([kp_clip, e_vec]))
        return c ** (-theta) - beta * (1.0 + (1.0 - mtax) * r) * EG_vals

    h_solved = _vectorized_bisect(
        _euler_vec,
        hbar_unc, hbar_unc + 5.0,
        args=(a_unc, e_unc),
        tol=1e-12
    )
    h[idx] = h_solved

    return h
