"""Single backward step for transition dynamics with endogenous labor supply.

This module provides the backward iteration step used by the transition
path solver for Bewley models with endogenous labor supply and HSV
progressive taxation. At each date t during the transition, given
expected continuation values E[g_{t+1}] and E[v_{t+1}], it solves for
the household's optimal policies (hours, consumption, savings) and
updates the marginal value of assets g_t and value function v_t.

References
----------
Boar, Midrigan (2022). "Efficient Redistribution." JME.
Virgiliu Midrigan, Pset 7 — transitions.m
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .kronm import kronm
from .solvers.howard_labor import (
    _period_utility,
    _savings,
    _vectorized_bisect,
)
from .taxes import hsv_post_tax_and_marginal


def backward_step_labor(
    gbar, vbar, hbar_next, Pi,
    a_grid, e_grid, W, r, tau, xi, iota,
    beta, theta, gamma,
    h_prev=None,
):
    """Single backward step for the Boar-Midrigan transition path.

    Given expected continuation values (already integrated over e'),
    solve for optimal household policies at date t.

    Parameters
    ----------
    gbar : ndarray, shape (n_e * n_a,)
        Expected marginal value of assets: E_t[g_{t+1}(a', e')].
        Already multiplied by Pi (expectation over e').
    vbar : ndarray, shape (n_e * n_a,)
        Expected value function: E_t[v_{t+1}(a', e')].
    hbar_next : ndarray, shape (n_e * n_a,)
        h_bar from the next period (used as initial guess for Broyden).
    Pi : ndarray, shape (n_e, n_e)
        Markov transition matrix for productivity.
    a_grid : ndarray, shape (n_a,)
        Asset grid.
    e_grid : ndarray, shape (n_e,)
        Productivity grid (levels).
    W, r : float
        Wage and interest rate at date t.
    tau, xi, iota : float
        HSV tax parameters at date t.
    beta, theta, gamma : float
        Preference parameters.
    h_prev : ndarray, optional, shape (n_e * n_a,)
        Previous period's hours (warm start for root-finding).

    Returns
    -------
    dict with keys:
        g : ndarray (n_e * n_a,) — marginal value of assets V'(a)
        v : ndarray (n_e * n_a,) — value function
        h : ndarray (n_e * n_a,) — hours worked
        c : ndarray (n_e * n_a,) — consumption
        kprime : ndarray (n_e * n_a,) — next-period assets
        y : ndarray (n_e * n_a,) — pre-tax income
        yhat : ndarray (n_e * n_a,) — post-tax income
        mtax : ndarray (n_e * n_a,) — marginal tax rate
        hbar : ndarray (n_e * n_a,) — minimum hours (borrowing constraint)
        gbar_out : ndarray (n_e * n_a,) — E[g_t] for use at t-1
        vbar_out : ndarray (n_e * n_a,) — E[v_t] for use at t-1
    """
    n_e, n_a = len(e_grid), len(a_grid)
    n_s = n_e * n_a

    a_flat = np.tile(a_grid, n_e)
    e_flat = np.repeat(e_grid, n_a)

    # Build interpolants for continuation values
    EG_interp = RegularGridInterpolator(
        (a_grid, e_grid),
        gbar.reshape(n_e, n_a).T,
        method='linear', bounds_error=False, fill_value=None
    )
    EV_interp = RegularGridInterpolator(
        (a_grid, e_grid),
        vbar.reshape(n_e, n_a).T,
        method='linear', bounds_error=False, fill_value=None
    )

    a_lo = a_grid[0]
    a_hi = a_grid[-1]

    # Compute h_bar: vectorized bisection
    def _hbar_residual(h_vec, a_vec, e_vec):
        kp, _, _, _, _ = _savings(h_vec, a_vec, e_vec, W, r, tau, xi, iota, theta, gamma)
        return kp

    lo_hbar = np.full(n_s, 1e-5)
    hi_hbar = np.full(n_s, 10.0)
    kp_lo = _hbar_residual(lo_hbar, a_flat, e_flat)
    needs_bisect = kp_lo < 0

    hbar = lo_hbar.copy()
    if np.any(needs_bisect):
        hbar[needs_bisect] = _vectorized_bisect(
            _hbar_residual,
            lo_hbar[needs_bisect], hi_hbar[needs_bisect],
            args=(a_flat[needs_bisect], e_flat[needs_bisect]),
            tol=1e-14
        )

    # Solve for optimal hours: vectorized bisection on Euler residual
    def _euler_vec(h_vec, a_vec, e_vec):
        kp, c, _, _, mtax = _savings(
            h_vec, a_vec, e_vec, W, r, tau, xi, iota, theta, gamma)
        kp_clip = np.clip(kp, a_lo, a_hi)
        EG_vals = EG_interp(np.column_stack([kp_clip, e_vec]))
        return c ** (-theta) - beta * (1.0 + (1.0 - mtax) * r) * EG_vals

    # Check which agents are unconstrained
    kp_hbar, c_hbar, _, _, mtax_hbar = _savings(
        hbar, a_flat, e_flat, W, r, tau, xi, iota, theta, gamma)
    kp_clipped = np.clip(kp_hbar, a_lo, a_hi)
    EG_at_hbar = EG_interp(np.column_stack([kp_clipped, e_flat]))
    euler_at_hbar = (c_hbar ** (-theta)
                     - beta * (1.0 + (1.0 - mtax_hbar) * r) * EG_at_hbar)
    unconstrained = euler_at_hbar < 0

    h = hbar.copy()
    idx = np.where(unconstrained)[0]
    if len(idx) > 0:
        h[idx] = _vectorized_bisect(
            _euler_vec,
            hbar[idx], hbar[idx] + 5.0,
            args=(a_flat[idx], e_flat[idx]),
            tol=1e-12
        )

    # Compute all policies from optimal h
    kprime, c, y, yhat, mtax = _savings(
        h, a_flat, e_flat, W, r, tau, xi, iota, theta, gamma)

    # Clip assets to grid
    excess = kprime - np.clip(kprime, a_lo, a_hi)
    c = c + excess
    kprime = np.clip(kprime, a_lo, a_hi)

    # Value function
    EV_vals = EV_interp(np.column_stack([kprime, e_flat]))
    v = _period_utility(c, h, theta, gamma) + beta * EV_vals

    # Marginal value of assets: g = dV/da via finite differences on V grid
    v_2d = v.reshape(n_e, n_a)
    g_2d = np.zeros_like(v_2d)
    g_2d[:, 1:-1] = (v_2d[:, 2:] - v_2d[:, :-2]) / (a_grid[2:] - a_grid[:-2])
    g_2d[:, 0] = (v_2d[:, 1] - v_2d[:, 0]) / (a_grid[1] - a_grid[0])
    g_2d[:, -1] = (v_2d[:, -1] - v_2d[:, -2]) / (a_grid[-1] - a_grid[-2])
    g = g_2d.ravel()

    # Compute expectations for use at t-1
    gbar_out = kronm([n_a, Pi], g)
    vbar_out = kronm([n_a, Pi], v)

    return {
        'g': g, 'v': v, 'h': h, 'c': c,
        'kprime': kprime, 'y': y, 'yhat': yhat, 'mtax': mtax,
        'hbar': hbar,
        'gbar_out': gbar_out, 'vbar_out': vbar_out,
    }
