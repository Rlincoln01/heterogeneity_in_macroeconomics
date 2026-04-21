"""Virgiliu Midrigan's simplified Hopenhayn-Rogerson closure (PS8, Inelastic Labor).

This module ports Virgiliu's MATLAB teaching code for PS8 firm dynamics to
Python on top of the `hetmacro` primitives. It is a *simplified* Hopenhayn
model used for building misallocation intuition, distinct from the canonical
free-entry Hopenhayn-Rogerson solver in :mod:`hopenhayn_rogerson`.

Model (PS8 Inelastic Labor)
---------------------------

Incumbent firms, unit mass, no entry/exit, no operating/entry cost. A linear
firing tax tau*max(0, n_prev - n) is the only friction::

    V(n_prev, z) = max_{n'} { z^(1-eta) * n'^eta   (revenue)
                              - w * n'             (wage bill)
                              - tau * max(0, n_prev - n')   (firing cost)
                              + beta * E[V(n', z') | z] }

with log z' = rho_z * log z + sigma_z * eps'. Two household closures are
available:

- ``labor_supply='inelastic'``: ``N^s = N_supply`` (default 1). The wage
  clears the labor market via ``ergodic.m`` TODO 7.
- ``labor_supply='separable'``: ``u(C, N) = C^{1-γ_c}/(1-γ_c) - ψ N^{1+γ_n}/(1+γ_n)``
  with FOC ``w · C^{-γ_c} = ψ · N^{γ_n}`` and ``C = Y`` (firing tax rebated).
  Mirrors Virgiliu's ``Elastic Labor`` folder.

Differences from the canonical free-entry closure in
:mod:`hopenhayn_rogerson`:

1.  **Revenue form.** PS8 uses ``z^(1-eta) * n^eta``, not ``z * n^eta``. This
    follows Virgiliu's start.m line 42 (the frictionless grid endpoint
    ``1/(1-eta)*log(eta/W) + log(z)`` is the log-FOC of the first form only).
2.  **No exit, no entry cost, no operating cost.** phi = kappa = 0 and firms
    never shut down.
3.  **Unit mass.** The ergodic distribution integrates to 1 by construction.
4.  **Labor-market-clearing wage.** w is updated by the damped fixed point
    ``w_new = eta * (E_lam[z] / N_s)^(1-eta)`` from ergodic.m, not by
    enforcing a free-entry condition.

Numerical approach
------------------

- z discretized by Rouwenhorst on the hetmacro convention.
- n grid log-spaced, endpoints derived from Virgiliu's start.m:42-43.
- VFI with Howard acceleration; the next-period employment choice ``n'`` is
  continuous, found by a vectorized golden-section search on a cubic-spline
  interpolation of the expected continuation value ``V-bar(n', z)``. This
  matches solve_golden.m and makes aggregate labor demand smooth in the wage,
  which the old discrete-grid implementation could not do.
- Stationary distribution solved by lottery-based power iteration on the
  joint ``(n_prev, z)`` grid; every non-grid ``n'`` is split linearly between
  the two bracketing grid points, matching findstationary.m.
- Outer wage loop is a damped fixed-point on the labor-market clearing
  formula. For the inelastic closure this matches ``start.m`` lines 64-80
  (constant-elasticity Newton step). For the separable closure it uses a
  Newton-style update derived from the household FOC and the (approximately
  frictionless) firm side, which has zero slope at the fixed point.

References
----------
Midrigan, V. "Heterogeneity in Macroeconomics," NYU PhD course. Problem Set 8
starter code, "Inelastic Labor" folder: start.m, solve_dynamic.m, bellman.m,
valfunc.m, solve_golden.m, findstationary.m, ergodic.m.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs as sparse_eigs

from ..markov import rouwenhorst, stationary_distribution
from .hopenhayn_rogerson import build_employment_grid


# ── Section A. Static formulas & grid construction ───────────────────────


def revenue(z: np.ndarray, n: np.ndarray, eta: float) -> np.ndarray:
    """PS8 revenue function ``z^(1-eta) * n^eta``.

    Broadcasts over ``z`` and ``n``. This is the revenue form used throughout
    Virgiliu's MATLAB starter code (see ``start.m`` line 42 for the
    frictionless grid-endpoint derivation).
    """
    return z ** (1.0 - eta) * n ** eta


def frictionless_n_star(z: np.ndarray, w: float, eta: float) -> np.ndarray:
    """Static FOC labor demand under the PS8 revenue form.

    From ``eta * z^(1-eta) * n^(eta-1) = w``::

        n*(z, w) = (eta / w)^(1/(1-eta)) * z
    """
    return (eta / w) ** (1.0 / (1.0 - eta)) * z


def _frictionless_wage(
    z_grid: np.ndarray,
    pi_z: np.ndarray,
    eta: Optional[float] = None,
    N_target: float = 1.0,
    theta: Optional[float] = None,
) -> float:
    """Inelastic labor-market-clearing wage under the PS8 revenue form.

    At the frictionless limit ``tau = 0`` the firm chooses
    ``n*(z) = (eta/w)^(1/(1-eta)) * z`` independent of history. Labor market
    clearing at aggregate supply ``N_target`` implies::

        sum_j pi_z[j] * (eta/w)^(1/(1-eta)) * z[j] = N_target
        =>  w = eta * (E[z] / N_target)^(1-eta)

    This is the formula used in ``ergodic.m`` TODO 7 for the inelastic case.

    Parameters
    ----------
    z_grid : (n_z,) array
        Productivity levels (not logs).
    pi_z : (n_z,) array
        Ergodic distribution over ``z_grid``.
    eta : float
        Revenue curvature in ``z^(1-eta) * n^eta``.
    N_target : float, default 1.0
        Target aggregate labor supply.
    theta : float, optional
        Deprecated alias for ``eta``.
    """
    eta = _resolve_eta(eta, theta)
    z_mean = float(np.sum(pi_z * z_grid))
    return eta * (z_mean / N_target) ** (1.0 - eta)


def _frictionless_wage_separable(
    z_grid: np.ndarray,
    pi_z: np.ndarray,
    eta: float,
    gamma_c: float,
    gamma_n: float,
    psi: float = 1.0,
) -> float:
    """Frictionless wage under separable utility ``u(C,N) = C^{1-γ_c}/(1-γ_c) - ψ N^{1+γ_n}/(1+γ_n)``.

    At ``tau = 0`` the firm side gives ``N(w) = (η/w)^{1/(1-η)} · E[z]`` and
    ``Y(w) = E[z]^{1-η} · N(w)^η``. The household FOC is
    ``w · C^{-γ_c} = ψ · N^{γ_n}``, with the resource constraint ``C = Y``
    (firing tax rebated as a transfer; firing flow vanishes at τ = 0). Taking
    logs and substituting yields a single linear equation in ``log w``::

        log w = (1-η) / (1-η+γ_n+η·γ_c) · [ log ψ + (γ_c+γ_n) · log E[z]
                                            + (γ_n+η·γ_c)/(1-η) · log η ]

    Returns the resulting frictionless wage. Used as the initial guess for the
    outer loop in the separable closure.
    """
    if not 0.0 < eta < 1.0:
        raise ValueError(f"eta must be in (0, 1), got {eta}")
    if gamma_c <= 0.0 or gamma_n <= 0.0:
        raise ValueError(f"gamma_c, gamma_n must be positive (got {gamma_c}, {gamma_n})")
    if psi <= 0.0:
        raise ValueError(f"psi must be positive, got {psi}")

    z_mean = float(np.sum(pi_z * z_grid))
    K = (gamma_n + eta * gamma_c) / (1.0 - eta)
    one_plus_K = 1.0 + K  # = (1 - eta + gamma_n + eta*gamma_c) / (1 - eta)
    log_w = (1.0 / one_plus_K) * (
        np.log(psi)
        + (gamma_c + gamma_n) * np.log(z_mean)
        + K * np.log(eta)
    )
    return float(np.exp(log_w))


def _resolve_eta(eta: Optional[float], theta: Optional[float]) -> float:
    """Accept either ``eta`` or the deprecated ``theta`` kwarg."""
    if eta is None and theta is None:
        raise TypeError("must pass `eta` (or the deprecated `theta`)")
    if eta is not None and theta is not None:
        raise TypeError("pass only one of `eta` or `theta`")
    if theta is not None:
        warnings.warn(
            "`theta` kwarg is deprecated in the Virgiliu closure; "
            "use `eta` to match Virgiliu's notation.",
            DeprecationWarning,
            stacklevel=3,
        )
        return theta
    return eta


def _virgiliu_employment_grid(
    z_grid: np.ndarray,
    w: float,
    eta: float,
    n_n: int,
    pad: float = 1.0,
) -> np.ndarray:
    """Log-spaced employment grid matching ``start.m`` lines 42-46.

    The MATLAB bounds in log space are::

        nmin = (1/(1-eta)) * log(eta / w) + log(z_min) - pad
        nmax = (1/(1-eta)) * log(eta / w) + log(z_max) + pad

    which places log(n_min) and log(n_max) ``pad`` units below/above the
    frictionless log-FOC at the extremes of the z grid.
    """
    log_z_min = float(np.log(z_grid[0]))
    log_z_max = float(np.log(z_grid[-1]))
    center = np.log(eta / w) / (1.0 - eta)
    log_n_min = center + log_z_min - pad
    log_n_max = center + log_z_max + pad
    return build_employment_grid(
        n_min=float(np.exp(log_n_min)),
        n_max=float(np.exp(log_n_max)),
        n_n=n_n,
        spacing="log",
    )


# ── Section B. Vectorized golden-section search ──────────────────────────


def _vectorized_golden_section(
    f,
    a: np.ndarray,
    b: np.ndarray,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> np.ndarray:
    """Maximize ``f`` componentwise on ``[a, b]`` by golden section.

    Line-by-line port of ``solve_golden.m``. ``a``, ``b`` are arrays of the
    same shape. ``f`` is a callable taking an array of candidate points with
    the same shape and returning an array of objective values (same shape).

    Returns
    -------
    x_star : ndarray
        Argmax for each component.

    Notes
    -----
    Unimodality over ``[a, b]`` is assumed componentwise. Each iteration
    performs *one* vectorized call to ``f`` by fusing the two branches with
    ``np.where``.
    """
    alpha1 = (3.0 - np.sqrt(5.0)) / 2.0   # ~0.38197  # PS8 solve_golden.m:5
    alpha2 = (np.sqrt(5.0) - 1.0) / 2.0   # ~0.61803  # PS8 solve_golden.m:6

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    d0 = b - a                             # PS8 solve_golden.m:8
    x1 = a + alpha1 * d0                   # PS8 solve_golden.m:10
    x2 = a + alpha2 * d0                   # PS8 solve_golden.m:11
    f1 = f(x1)
    f2 = f(x2)

    d = alpha1 * alpha2 * d0               # PS8 solve_golden.m:16

    for _ in range(max_iter):              # PS8 solve_golden.m:23
        if np.all(d <= tol):
            break
        d = d * alpha2                     # PS8 solve_golden.m:30
        mask = f2 >= f1
        # In one branch we need x2 + d; in the other, x1 - d.
        # PS8 solve_golden.m:31-35 (vectorized fusion of the two scalar branches).
        x_new = np.where(mask, x2 + d, x1 - d)
        f_new = f(x_new)
        # Rebraket:
        #   If f2 >= f1 (keep right half): new x1 = x2, new x2 = x_new.
        #   Else (keep left half):        new x2 = x1, new x1 = x_new.
        x1, x2 = np.where(mask, x2, x_new), np.where(mask, x_new, x1)
        f1, f2 = np.where(mask, f2, f_new), np.where(mask, f_new, f1)

    return np.where(f2 >= f1, x2, x1)     # PS8 solve_golden.m:39


# ── Section C. Bellman step ──────────────────────────────────────────────


def _build_Vbar_splines(
    V: np.ndarray, Pi_z: np.ndarray, n_grid: np.ndarray
) -> List[CubicSpline]:
    """One CubicSpline per productivity column of V-bar.

    ``V-bar[i, j] = sum_{j'} Pi_z[j, j'] * V[i, j']`` is the expected
    continuation value at ``(n_prev=n_grid[i], z=z_j)``. We build ``n_z``
    separable splines, one per column in ``n``. This mirrors MATLAB's
    ``griddedInterpolant({ngrid, zgrid}, ..., 'spline', 'linear')``, which
    evaluates linearly in the z direction at grid points anyway.
    """
    Vbar = V @ Pi_z.T
    return [
        CubicSpline(n_grid, Vbar[:, j], bc_type="not-a-knot", extrapolate=True)
        for j in range(Vbar.shape[1])
    ]


def _eval_Vbar(
    splines: List[CubicSpline],
    n_query: np.ndarray,
    z_idx: np.ndarray,
) -> np.ndarray:
    """Evaluate the per-column spline selected by ``z_idx`` at ``n_query``.

    ``n_query`` and ``z_idx`` share shape. For each productivity index j
    present in ``z_idx`` we invoke the corresponding spline on a boolean
    mask. For n_z ~ 51 this inner loop is cheap.
    """
    out = np.empty_like(n_query, dtype=float)
    n_z = len(splines)
    for j in range(n_z):
        mask = z_idx == j
        if mask.any():
            out[mask] = splines[j](n_query[mask])
    return out


def _bellman_step_golden(
    V: np.ndarray,
    Pi_z: np.ndarray,
    n_grid: np.ndarray,
    z_grid: np.ndarray,
    w: float,
    eta: float,
    tau: float,
    beta: float,
) -> Tuple[np.ndarray, np.ndarray, List[CubicSpline]]:
    """One full Bellman maximization step via golden section over ``n'``.

    Returns the new ``V``, the continuous policy ``n'(n_prev, z)`` in levels,
    and the spline list (useful for Howard policy evaluation with the same V).
    """
    n_n, n_z = V.shape
    splines = _build_Vbar_splines(V, Pi_z, n_grid)

    n_prev = np.broadcast_to(n_grid[:, None], (n_n, n_z))          # (n_n, n_z)
    z_mat = np.broadcast_to(z_grid[None, :], (n_n, n_z))            # (n_n, n_z)
    z_idx = np.broadcast_to(np.arange(n_z)[None, :], (n_n, n_z))    # (n_n, n_z)

    # Golden-section bracket from bellman.m: [n_grid[0], n_grid[-1]].
    lo = np.full((n_n, n_z), float(n_grid[0]))
    hi = np.full((n_n, n_z), float(n_grid[-1]))

    def objective(n_next: np.ndarray) -> np.ndarray:
        # Clip the query to the spline support (drift can push just outside).
        n_clip = np.clip(n_next, n_grid[0], n_grid[-1])
        cur = revenue(z_mat, n_clip, eta) - w * n_clip
        fc = tau * np.maximum(0.0, n_prev - n_clip)
        Vb = _eval_Vbar(splines, n_clip, z_idx)
        return cur - fc + beta * Vb

    n_policy_level = _vectorized_golden_section(objective, lo, hi)
    V_new = objective(n_policy_level)
    return V_new, n_policy_level, splines


def _policy_evaluation_step_golden(
    V: np.ndarray,
    Pi_z: np.ndarray,
    n_policy_level: np.ndarray,
    n_grid: np.ndarray,
    z_grid: np.ndarray,
    w: float,
    eta: float,
    tau: float,
    beta: float,
) -> np.ndarray:
    """Howard improvement: evaluate V at the fixed continuous policy.

    The stored policy is in levels (not indices) because golden section
    returns an off-grid ``n'``, so this step must also interpolate V-bar.
    """
    n_n, n_z = V.shape
    splines = _build_Vbar_splines(V, Pi_z, n_grid)
    n_prev = np.broadcast_to(n_grid[:, None], (n_n, n_z))
    z_mat = np.broadcast_to(z_grid[None, :], (n_n, n_z))
    z_idx = np.broadcast_to(np.arange(n_z)[None, :], (n_n, n_z))

    n_clip = np.clip(n_policy_level, n_grid[0], n_grid[-1])
    cur = revenue(z_mat, n_clip, eta) - w * n_clip
    fc = tau * np.maximum(0.0, n_prev - n_clip)
    Vb = _eval_Vbar(splines, n_clip, z_idx)
    return cur - fc + beta * Vb


# ── Section D. VFI driver ────────────────────────────────────────────────


def _solve_firm_problem_golden(
    Pi_z: np.ndarray,
    n_grid: np.ndarray,
    z_grid: np.ndarray,
    w: float,
    eta: float,
    tau: float,
    beta: float,
    V_init: Optional[np.ndarray] = None,
    tol: float = 1e-9,
    max_iter: int = 1000,
    howard_freq: int = 5,
    howard_steps: int = 3,
    warmup_iters: int = 5,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Value function iteration with Howard improvement (continuous n').

    Mirrors ``solve_dynamic.m``:

    1. Initial guess: frictionless perpetuity ``V0[i, j] = pi_z_frict / (1-beta)``
       broadcast across n_prev.
    2. ``warmup_iters`` full Bellman updates.
    3. Howard phase: every ``howard_freq`` iterations do a full Bellman update;
       otherwise run ``howard_steps`` of policy evaluation with the latest
       continuous policy held fixed.

    Convergence: relative Frobenius norm ``||dV||/||V|| < tol``, matching
    ``solve_dynamic.m`` line 41.

    Note: unlike the discrete-grid Howard in :mod:`hopenhayn_rogerson`, each
    policy-evaluation step here rebuilds the per-z splines (because V has
    changed), so the savings over a full Bellman are modest. Defaults reflect
    this: a small ``howard_freq`` and ``howard_steps`` perform best.
    """
    n_n = len(n_grid)
    n_z = len(z_grid)

    if V_init is None:
        # Frictionless profits at each z, ignoring firing cost.
        n_star_z = frictionless_n_star(z_grid, w, eta)          # (n_z,)
        pi_z_frict = revenue(z_grid, n_star_z, eta) - w * n_star_z  # (n_z,)
        V = np.broadcast_to(
            pi_z_frict[None, :] / (1.0 - beta), (n_n, n_z)
        ).copy()
    else:
        V = V_init.copy()

    n_policy_level = np.empty((n_n, n_z), dtype=float)

    # Warmup: full Bellman updates.  PS8 solve_dynamic.m:21-43
    for it in range(warmup_iters):
        V_new, n_policy_level, _ = _bellman_step_golden(
            V, Pi_z, n_grid, z_grid, w, eta, tau, beta
        )
        rel = np.linalg.norm(V_new - V) / max(1e-12, np.linalg.norm(V_new))  # PS8 solve_dynamic.m:37
        V = V_new
        if verbose:
            print(f"  warmup {it + 1}: rel={rel:.2e}")
        if rel < tol:
            return V, n_policy_level, it + 1

    # Howard phase.  PS8 solve_dynamic.m:47-77
    for it in range(warmup_iters, max_iter):
        V_old = V
        if (it - warmup_iters) % howard_freq == 0:          # PS8 solve_dynamic.m:60
            V, n_policy_level, _ = _bellman_step_golden(
                V, Pi_z, n_grid, z_grid, w, eta, tau, beta
            )
            rel = np.linalg.norm(V - V_old) / max(1e-12, np.linalg.norm(V))  # PS8 solve_dynamic.m:64
            if verbose:
                print(f"  VFI {it + 1}: rel={rel:.2e}")
            if rel < tol:
                return V, n_policy_level, it + 1
        else:
            # howard_steps of policy evaluation at the current n_policy_level.
            # PS8 solve_dynamic.m:73 (valfunc call)
            for _ in range(howard_steps):
                V = _policy_evaluation_step_golden(
                    V, Pi_z, n_policy_level, n_grid, z_grid, w, eta, tau, beta
                )

    # Compute rel fresh in case the last iteration was a Howard step (rel would
    # otherwise be stale from the last Bellman update several iters ago).
    rel = np.linalg.norm(V - V_old) / max(1e-12, np.linalg.norm(V))
    warnings.warn(
        f"Virgiliu VFI did not converge in {max_iter} iters "
        f"(last rel = {rel:.2e})",
        stacklevel=2,
    )
    return V, n_policy_level, max_iter


# ── Section E. Stationary distribution (lottery) ─────────────────────────


def _lottery_indices(
    n_policy_level: np.ndarray, n_grid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Linear lottery onto the n grid: return (ind, w_lo).

    ``ind[i, j]`` is the index of the largest grid point ``<= n'(i, j)``,
    clipped so that ``ind + 1`` is always a valid index. ``w_lo`` is the
    weight on ``n_grid[ind]``. Mirrors findstationary.m lines 10-11.
    """
    n_n = len(n_grid)
    ind = np.searchsorted(n_grid, n_policy_level, side="right") - 1  # PS8 findstationary.m:9
    ind = np.clip(ind, 0, n_n - 2)
    span = n_grid[ind + 1] - n_grid[ind]
    w_lo = (n_grid[ind + 1] - n_policy_level) / span                  # PS8 findstationary.m:11
    w_lo = np.clip(w_lo, 0.0, 1.0)
    return ind, w_lo


def stationary_distribution_lottery(
    n_policy_level: np.ndarray,
    n_grid: np.ndarray,
    Pi_z: np.ndarray,
    method: str = "iterate",
    tol: float = 1e-13,
    max_iter: int = 10_000,
    lam_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Joint ergodic distribution on (n_prev, z) under the policy-induced kernel.

    Parameters
    ----------
    n_policy_level : (n_n, n_z) array
        Continuous policy in levels.
    n_grid : (n_n,) array
    Pi_z : (n_z, n_z) array
    method : {'iterate', 'eigen'}
        Power iteration via ``np.add.at`` (matches findstationary.m case 2)
        or sparse-eigs on the explicit transition matrix (case 1).
    tol : float
        L1 convergence tolerance (iterate) or eigenvalue solver tol (eigen).
    max_iter : int
    lam_init : (n_n, n_z) array, optional
        Warm start. If None, initialise with the exogenous z ergodic on the
        lowest-n row.

    Returns
    -------
    lam : (n_n, n_z) array, summing to 1.
    """
    n_n, n_z = n_policy_level.shape
    ind, w_lo = _lottery_indices(n_policy_level, n_grid)
    w_hi = 1.0 - w_lo

    if method == "iterate":
        if lam_init is None:
            pi_z_erg = stationary_distribution(Pi_z)
            lam = np.zeros((n_n, n_z))
            lam[0, :] = pi_z_erg
        else:
            lam = lam_init / lam_init.sum()

        diff = np.inf
        for _ in range(max_iter):              # PS8 findstationary.m:39 (case 2 loop)
            Ftilde = np.zeros((n_n, n_z))
            for j in range(n_z):               # PS8 findstationary.m:43
                np.add.at(Ftilde[:, j], ind[:, j], lam[:, j] * w_lo[:, j])
                np.add.at(Ftilde[:, j], ind[:, j] + 1, lam[:, j] * w_hi[:, j])
            lam_new = Ftilde @ Pi_z            # PS8 findstationary.m:45
            total = lam_new.sum()
            if total > 0:
                lam_new /= total
            diff = float(np.abs(lam_new - lam).sum())
            lam = lam_new
            if diff < tol:
                return lam

        warnings.warn(
            f"stationary_distribution_lottery(method='iterate') did not "
            f"converge in {max_iter} iters (last L1 diff = {diff:.2e})",
            stacklevel=2,
        )
        return lam

    if method == "eigen":
        # Build sparse (n_s x n_s) transition matrix Q with s = i*n_z + j.
        n_s = n_n * n_z
        rows: List[np.ndarray] = []
        cols: List[np.ndarray] = []
        vals: List[np.ndarray] = []
        row_base = np.arange(n_n)[:, None] * n_z  # (n_n, 1)
        for j_src in range(n_z):
            src = (row_base + j_src).ravel()                       # (n_n,)
            ind_j = ind[:, j_src]                                  # (n_n,)
            w_lo_j = w_lo[:, j_src]
            w_hi_j = w_hi[:, j_src]
            for j_dst in range(n_z):
                p = Pi_z[j_src, j_dst]
                if p == 0.0:
                    continue
                col_lo = ind_j * n_z + j_dst
                col_hi = (ind_j + 1) * n_z + j_dst
                rows.append(src)
                cols.append(col_lo)
                vals.append(p * w_lo_j)
                rows.append(src)
                cols.append(col_hi)
                vals.append(p * w_hi_j)
        rows_all = np.concatenate(rows)
        cols_all = np.concatenate(cols)
        vals_all = np.concatenate(vals)
        Q = csr_matrix((vals_all, (rows_all, cols_all)), shape=(n_s, n_s))

        # Power-iteration warm start avoids eigs' convergence issues.
        if lam_init is None:
            pi_z_erg = stationary_distribution(Pi_z)
            lam0 = np.zeros((n_n, n_z))
            lam0[0, :] = pi_z_erg
            lam0 = lam0.ravel()
        else:
            lam0 = (lam_init / lam_init.sum()).ravel()

        try:
            _, vecs = sparse_eigs(Q.T, k=1, sigma=1.0, v0=lam0, tol=tol)
            lam = np.real(vecs[:, 0])
            lam = np.maximum(lam, 0.0)
            s = lam.sum()
            if s <= 0:
                raise RuntimeError("degenerate eigs result")
            lam /= s
            return lam.reshape(n_n, n_z)
        except Exception as exc:
            warnings.warn(
                f"sparse eigs failed ({exc}); falling back to power iteration.",
                stacklevel=2,
            )
            return stationary_distribution_lottery(
                n_policy_level, n_grid, Pi_z,
                method="iterate", tol=tol, max_iter=max_iter,
                lam_init=lam_init,
            )

    raise ValueError(f"method must be 'iterate' or 'eigen', got '{method}'")


# ── Section F. Aggregates (ergodic.m) ────────────────────────────────────


def virgiliu_aggregates(
    n_grid: np.ndarray,
    z_grid: np.ndarray,
    lam: np.ndarray,
    n_policy_level: np.ndarray,
    eta: float,
    tau: float,
    w: float,
) -> Dict[str, Any]:
    """Compute aggregates under the Virgiliu closure.

    Mirrors ergodic.m. Unit firm mass: ``lam`` sums to 1.

    Returns
    -------
    dict with keys
        N                  aggregate labor demand = E_lam[n']
        Y                  aggregate output = E_lam[z^(1-eta) * n'^eta]
        firing             firing cost flow   = E_lam[tau * max(0, n_prev - n')]
        TFP                Y / N^eta           (misallocated economy)
        TFP_frictionless   (E_lam[z])^(1-eta)  (PS8 ergodic.m line 24)
        misallocation_loss 1 - TFP / TFP_frictionless
        mrpl               firm-level marginal revenue product of labor
        labor_wedge_agg    mass-weighted wedge mu_bar
    """
    n_prev = n_grid[:, None]                                    # (n_n, 1)
    z_mat = z_grid[None, :]                                      # (1, n_z)

    N = float(np.sum(lam * n_policy_level))
    Y = float(np.sum(lam * revenue(z_mat, n_policy_level, eta)))
    firing = float(
        np.sum(lam * tau * np.maximum(0.0, n_prev - n_policy_level))
    )

    TFP = Y / (N ** eta) if N > 0 else 0.0
    # Frictionless benchmark at the ergodic z: ergodic.m uses the z-marginal
    # of the joint distribution, which converges to the exogenous z ergodic.
    z_marg = lam.sum(axis=0)
    z_mean = float(np.sum(z_marg * z_grid))
    TFP_frictionless = z_mean ** (1.0 - eta)
    misallocation_loss = (
        1.0 - TFP / TFP_frictionless if TFP_frictionless > 0 else 0.0
    )

    # Firm-level marginal revenue product of labor:
    #   mrpl = eta * z^(1-eta) * n'^(eta-1) = eta * revenue / n'
    with np.errstate(divide="ignore", invalid="ignore"):
        mrpl = np.where(
            n_policy_level > 0,
            eta * revenue(z_mat, n_policy_level, eta) / n_policy_level,
            0.0,
        )
    # Mass-weighted aggregate wedge mu_i = mrpl_i / w.
    labor_wedge_agg = float(np.sum(lam * mrpl)) / w if w > 0 else np.nan

    return dict(
        N=N,
        Y=Y,
        firing=firing,
        TFP=TFP,
        TFP_frictionless=TFP_frictionless,
        misallocation_loss=misallocation_loss,
        mrpl=mrpl,
        labor_wedge_agg=labor_wedge_agg,
        z_mean=z_mean,
    )


# ── Section G. Result container ──────────────────────────────────────────


@dataclass
class HopenhaynVirgiliuSteadyState:
    """Stationary equilibrium of Virgiliu's simplified Hopenhayn closure."""

    # Prices and taxes
    w: float
    r: float
    tau: float

    # Firm problem
    V: np.ndarray
    n_policy_level: np.ndarray  # (n_n, n_z) continuous policy in levels
    mrpl: np.ndarray            # (n_n, n_z) marginal revenue product of labor

    # Distribution and aggregates
    lambda_star: np.ndarray
    N: float
    Y: float
    C: float
    firing_total: float
    Pi_agg: float               # aggregate net firm profits (Y - w*N - firing)
    TFP: float
    TFP_frictionless: float
    misallocation_loss: float   # 1 - TFP / TFP_frictionless
    labor_wedge_agg: float      # mass-weighted aggregate labor wedge

    # Grids
    n_grid: np.ndarray
    z_grid: np.ndarray
    Pi_z: np.ndarray

    # Metadata
    params: Dict[str, Any]
    labor_supply: str
    gamma_c: Optional[float] = None
    gamma_n: Optional[float] = None
    psi: Optional[float] = None
    n_vfi_iter: int = 0
    n_outer_iter: int = 0
    wage_residual: float = 0.0
    walras_residual: float = 0.0
    elapsed: float = 0.0

    def summary(self) -> str:
        lines = [
            "Hopenhayn (Virgiliu PS8 closure) Stationary Equilibrium",
            f"  Labor supply: {self.labor_supply}",
        ]
        if self.labor_supply == "separable":
            lines.append(
                f"    gamma_c = {self.gamma_c:.3f}, "
                f"gamma_n = {self.gamma_n:.3f}, psi = {self.psi:.3f}"
            )
            # FOC residual: |w - psi * N^gamma_n * C^gamma_c| / w
            foc_rhs = self.psi * self.N ** self.gamma_n * self.C ** self.gamma_c
            foc_res = abs(self.w - foc_rhs) / max(1e-12, self.w)
            lines.append(
                f"    HH FOC residual |w - psi*N^gn*C^gc| / w = {foc_res:.2e}"
            )
        lines += [
            f"  tau (firing tax) = {self.tau:.4f}",
            f"  w = {self.w:.6f},  r = {self.r:.6f}",
            f"  N = {self.N:.6f},  Y = {self.Y:.6f},  C = {self.C:.6f}",
            f"  Net firm profits = {self.Pi_agg:.6f}",
            f"  Firing flow      = {self.firing_total:.6f}",
            f"  TFP            = {self.TFP:.6f}",
            f"  TFP (frictionless) = {self.TFP_frictionless:.6f}",
            f"  Misallocation loss = {100 * self.misallocation_loss:+.4f}%",
            f"  Aggregate labor wedge = {self.labor_wedge_agg:.6f}",
            f"  |dW|  residual = {self.wage_residual:.2e}",
            f"  Walras residual = {self.walras_residual:.2e}",
            f"  VFI iter (final) = {self.n_vfi_iter}",
            f"  Outer iter (wage)= {self.n_outer_iter}",
            f"  Elapsed = {self.elapsed:.2f}s",
        ]
        return "\n".join(lines)


# Backward-compatible alias for the prior (more verbose) class name.
HopenhaynRogersonVirgiliuSteadyState = HopenhaynVirgiliuSteadyState


# ── Section H. Top-level solver ──────────────────────────────────────────


def solve_virgiliu_ss(
    # Productivity process
    n_z: int = 51,
    rho_z: float = 0.90,
    sigma_z: float = 0.50,
    # Employment grid
    n_n: int = 251,
    log_grid_pad: float = 1.0,
    n_min: Optional[float] = None,
    n_max: Optional[float] = None,
    # Technology / preferences
    eta: Optional[float] = None,
    tau: float = 0.5,
    beta: float = 0.95,
    # Household closure
    labor_supply: str = "inelastic",
    N_supply: float = 1.0,
    gamma_c: Optional[float] = None,
    gamma_n: Optional[float] = None,
    psi: float = 1.0,
    # Outer loop
    w_init: Optional[float] = None,
    tol: float = 1e-7,
    max_outer: int = 35,
    damping: float = 0.5,
    # Inner VFI
    vfi_tol: float = 1e-9,
    vfi_max_iter: int = 1000,
    howard_freq: int = 5,
    howard_steps: int = 3,
    warmup_iters: int = 5,
    # Distribution
    dist_method: str = "iterate",
    dist_tol: float = 1e-13,
    dist_max_iter: int = 10_000,
    verbose: bool = False,
    # Deprecated alias
    theta: Optional[float] = None,
) -> HopenhaynVirgiliuSteadyState:
    """Solve the PS8 Inelastic-Labor Virgiliu closure.

    Parameters
    ----------
    n_z, rho_z, sigma_z : int, float, float
        Rouwenhorst discretization of log z ~ AR(1).
    n_n : int
        Number of employment grid points.
    log_grid_pad : float
        Buffer (in log units) below/above the frictionless FOC at the extreme
        z values when building the n grid, matching ``start.m`` line 42.
    n_min, n_max : float, optional
        Override the auto-derived grid endpoints.
    eta : float
        Revenue curvature in ``z^(1-eta) * n^eta`` (Virgiliu's η; the
        deprecated ``theta`` kwarg is also accepted).
    tau : float
        Firing cost per worker.
    beta : float
        Discount factor.
    labor_supply : {'inelastic', 'separable'}
        Household side. ``'inelastic'`` fixes ``N^s = N_supply``. ``'separable'``
        uses ``u(C, N) = C^{1-γ_c}/(1-γ_c) - ψ N^{1+γ_n}/(1+γ_n)`` and clears
        with the FOC ``w · C^{-γ_c} = ψ · N^{γ_n}`` (with ``C = Y``, since the
        firing tax is rebated lump-sum). ``'elastic'`` is accepted as an alias
        for ``'separable'``.
    N_supply : float
        Target aggregate labor supply (only meaningful for the inelastic
        closure).
    gamma_c, gamma_n : float, optional
        CRRA on consumption (``γ_c``) and Frisch inverse on labor (``γ_n``)
        for the separable closure. Required if ``labor_supply='separable'``.
        Match Virgiliu's ``Elastic Labor/start.m`` ``p.theta`` and ``p.gamma``.
    psi : float, default 1.0
        Disutility-of-labor scale ``ψ`` in the separable utility.
    w_init : float, optional
        Initial wage guess. Defaults to the frictionless LMC wage.
    tol : float
        Convergence tolerance on the wage.
    max_outer : int
        Maximum outer (wage) iterations.
    damping : float
        Weight on the old wage: ``w = damping*w_old + (1-damping)*w_implied``.
    vfi_tol, vfi_max_iter, howard_freq, howard_steps, warmup_iters : ...
        Inner VFI controls; see _solve_firm_problem_golden.
    dist_method : {'iterate', 'eigen'}
        Joint stationary distribution solver.
    verbose : bool
        Print per-iteration diagnostics.
    theta : float, optional
        Deprecated alias for ``eta``.

    Returns
    -------
    HopenhaynVirgiliuSteadyState
    """
    t0 = time.time()
    eta = _resolve_eta(eta, theta)

    # 0. Input validation.
    if not 0.0 < eta < 1.0:
        raise ValueError(f"eta must be in (0, 1), got {eta}")
    if not 0.0 < beta < 1.0:
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    if tau < 0.0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    # Accept 'elastic' as an alias for 'separable' (same closure).
    if labor_supply == "elastic":
        labor_supply = "separable"
    if labor_supply not in ("inelastic", "separable"):
        raise ValueError(
            f"labor_supply must be one of 'inelastic', 'separable' "
            f"(or 'elastic' as an alias for 'separable'); got '{labor_supply}'"
        )
    if labor_supply == "separable":
        if gamma_c is None or gamma_n is None:
            raise TypeError(
                "labor_supply='separable' requires gamma_c and gamma_n "
                "(CRRA on C and Frisch inverse on N)."
            )
        if gamma_c <= 0.0 or gamma_n <= 0.0:
            raise ValueError(
                f"gamma_c, gamma_n must be positive (got {gamma_c}, {gamma_n})"
            )
        if psi <= 0.0:
            raise ValueError(f"psi must be positive, got {psi}")

    r = 1.0 / beta - 1.0

    # 1. Rouwenhorst discretization of log z.
    log_z, Pi_z = rouwenhorst(n_z, rho_z, sigma_z)
    z_grid = np.exp(log_z)
    pi_z_erg = stationary_distribution(Pi_z)

    # 2. Initial wage + employment grid.
    if labor_supply == "inelastic":
        w_frict = _frictionless_wage(z_grid, pi_z_erg, eta=eta, N_target=N_supply)
    else:  # separable
        w_frict = _frictionless_wage_separable(
            z_grid, pi_z_erg, eta=eta,
            gamma_c=gamma_c, gamma_n=gamma_n, psi=psi,
        )
    w = float(w_init) if w_init is not None else w_frict

    if n_min is None and n_max is None:
        n_grid = _virgiliu_employment_grid(
            z_grid, w_frict, eta, n_n, pad=log_grid_pad
        )
    else:
        if n_min is None or n_max is None:
            raise ValueError("pass both n_min and n_max, or neither")
        n_grid = build_employment_grid(n_min, n_max, n_n, spacing="log")

    # 3. Outer wage fixed-point loop (matches start.m:64-80).
    V_warm: Optional[np.ndarray] = None
    lam_warm: Optional[np.ndarray] = None
    n_vfi_iter_last = 0
    n_outer = 0
    wage_hist: List[Tuple[int, float, float, float, float]] = []
    w_implied = w
    N_d = N_supply

    for outer in range(max_outer):
        n_outer = outer + 1
        w_old = w

        V, n_policy_level, n_vfi_iter_last = _solve_firm_problem_golden(
            Pi_z, n_grid, z_grid, w, eta, tau, beta,
            V_init=V_warm, tol=vfi_tol, max_iter=vfi_max_iter,
            howard_freq=howard_freq, howard_steps=howard_steps,
            warmup_iters=warmup_iters, verbose=False,
        )
        V_warm = V

        lam = stationary_distribution_lottery(
            n_policy_level, n_grid, Pi_z,
            method=dist_method, tol=dist_tol, max_iter=dist_max_iter,
            lam_init=lam_warm,
        )
        lam_warm = lam

        agg = virgiliu_aggregates(
            n_grid, z_grid, lam, n_policy_level, eta, tau, w
        )
        N_d = agg["N"]
        Y_d = agg["Y"]

        if labor_supply == "inelastic":
            # Implied inelastic LMC wage (ergodic.m TODO 7). Treat aggregate
            # labor demand as if it were frictionless (N ∝ w^{-1/(1-eta)}):
            # the wage that would yield N_supply under the SAME shape is
            #     w_new = w_old * (N_d / N_supply) ** (1 - eta).
            # Standard Newton-step for constant-elasticity demand; contractive
            # near the equilibrium for damping in [0, 1).
            if N_d > 0:
                w_implied = w_old * (N_d / N_supply) ** (1.0 - eta)
            else:
                w_implied = w_old
        else:  # separable: Newton-style update with zero slope at the FP.
            # Household FOC at the equilibrium: w = psi * N^gamma_n * Y^gamma_c.
            # Naive iteration on this has slope ~ -(gamma_n+eta*gamma_c)/(1-eta)
            # = -K, which diverges for large K (e.g. K = 38 with γ_c=γ_n=2,
            # η=0.9). Instead, treat the firm side as approximately frictionless
            # (N ∝ w^{-1/(1-eta)}, Y ∝ w^{-eta/(1-eta)}) and solve the FOC for w
            # in closed form, holding the realised TFP_eff = Y/N^η fixed:
            #     w_new = (psi * N^γ_n * Y^γ_c * w_old^K)^(1/(1+K))
            # This has zero slope at the fixed point (super-linear convergence)
            # and only the slow-moving misallocation drift remains.
            K = (gamma_n + eta * gamma_c) / (1.0 - eta)
            if N_d > 0 and Y_d > 0:
                w_implied = (
                    psi * N_d ** gamma_n * Y_d ** gamma_c * w_old ** K
                ) ** (1.0 / (1.0 + K))
            else:
                w_implied = w_old
        w_new = damping * w_old + (1.0 - damping) * w_implied

        wage_hist.append((outer, w_old, w_implied, w_new, N_d))
        if verbose:
            print(
                f"outer {outer + 1:3d}: w_old={w_old:.6f}  "
                f"w_implied={w_implied:.6f}  N_d={N_d:.6f}  "
                f"VFI_iter={n_vfi_iter_last}"
            )

        w = w_new
        if abs(w_new - w_old) < tol:
            break
    else:
        warnings.warn(
            f"Virgiliu outer wage loop did not converge in {max_outer} "
            f"iters; last |dW|={abs(w - wage_hist[-1][1]):.2e}",
            stacklevel=2,
        )

    # 4. Final aggregates at the converged w.
    wage_residual = float(abs(w - wage_hist[-1][1])) if wage_hist else 0.0

    N = agg["N"]
    Y = agg["Y"]
    firing_total = agg["firing"]
    Pi_agg = Y - w * N - firing_total  # net firm profits after wages & firing
    # Household budget (firing rebated as a transfer): C = Y.
    C = Y
    walras_residual = float(abs(C - Y))  # zero by construction; kept for symmetry

    TFP = agg["TFP"]
    TFP_frict = agg["TFP_frictionless"]
    misalloc = agg["misallocation_loss"]

    # Grid saturation check.
    ind, _ = _lottery_indices(n_policy_level, n_grid)
    mass_top = float(lam[ind == (n_n - 2)].sum())  # mass landing in top bin
    if mass_top > 0.01:
        warnings.warn(
            f"{100 * mass_top:.2f}% of firm mass lands in the top employment "
            f"bin (n_max = {n_grid[-1]:.3e}); widen `log_grid_pad` if this is "
            f"consistent across tau values.",
            stacklevel=2,
        )

    params = dict(
        n_z=n_z, rho_z=rho_z, sigma_z=sigma_z,
        n_n=n_n, log_grid_pad=log_grid_pad,
        n_min=float(n_grid[0]), n_max=float(n_grid[-1]),
        eta=eta, tau=tau, beta=beta,
        labor_supply=labor_supply, N_supply=N_supply,
        gamma_c=gamma_c, gamma_n=gamma_n, psi=psi,
        tol=tol, max_outer=max_outer, damping=damping,
    )

    return HopenhaynVirgiliuSteadyState(
        w=float(w),
        r=float(r),
        tau=float(tau),
        V=V,
        n_policy_level=n_policy_level,
        mrpl=agg["mrpl"],
        lambda_star=lam,
        N=float(N),
        Y=float(Y),
        C=float(C),
        firing_total=float(firing_total),
        Pi_agg=float(Pi_agg),
        TFP=float(TFP),
        TFP_frictionless=float(TFP_frict),
        misallocation_loss=float(misalloc),
        labor_wedge_agg=float(agg["labor_wedge_agg"]),
        n_grid=n_grid,
        z_grid=z_grid,
        Pi_z=Pi_z,
        params=params,
        labor_supply=labor_supply,
        gamma_c=gamma_c,
        gamma_n=gamma_n,
        psi=psi if labor_supply == "separable" else None,
        n_vfi_iter=int(n_vfi_iter_last),
        n_outer_iter=int(n_outer),
        wage_residual=wage_residual,
        walras_residual=walras_residual,
        elapsed=float(time.time() - t0),
    )
