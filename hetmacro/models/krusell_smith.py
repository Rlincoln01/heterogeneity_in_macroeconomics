"""Krusell-Smith model: SSJ GE closure, 4D EGM, and simulation.

Provides:
- ``ssj_ge_irfs``: Compute IRFs via SSJ with Cobb-Douglas GE closure.
- ``simulate_from_ssj``: Simulate K path from SSJ IRF coefficients.
- ``backward_egm_ks``: Single EGM step on the 4D (k, K, e, Z) state.
- ``policy_iteration_ks``: Iterate EGM to convergence for given law of motion.
- ``simulate_ks_economy``: Forward simulate distribution along a shock path.
- ``solve_krusell_smith``: Full KS outer loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ..backward import backward_egm
from ..chebyshev import fit_law_of_motion, eval_law_of_motion
from ..forward import forward_iteration
from ..interpolation import get_lottery
from ..ssj import jacobian, build_ge_matrices, solve_ge


# ── Result container ─────────────────────────────────────────────────────


@dataclass
class KSResult:
    """Container for Krusell-Smith algorithm output."""

    K_path: np.ndarray
    Z_path: np.ndarray
    ck: np.ndarray
    Rsq: float
    converged: bool
    n_iterations: int
    Rsq_history: List[float] = field(default_factory=list)
    error_history: List[float] = field(default_factory=list)


# ── SSJ with GE equilibrium closure ─────────────────────────────────────


def ssj_ge_irfs(
    ss: Dict[str, np.ndarray],
    e_grid: np.ndarray,
    alpha: float,
    delta: float,
    rho_z: float,
    T: int = 500,
    h: float = 1e-4,
) -> Dict[str, np.ndarray]:
    """Compute SSJ impulse responses with Cobb-Douglas GE closure.

    Parameters
    ----------
    ss : dict
        Steady state from ``household_steady_state``.  Must contain
        D, Va, a, c, a_i, a_pi, Pi, a_grid, y, r, beta, eis.
    e_grid : ndarray, shape (ne,)
        Normalised idiosyncratic productivity grid (E[e]=1).
    alpha : float
        Capital share.
    delta : float
        Depreciation rate.
    rho_z : float
        Persistence of aggregate TFP (AR(1) coefficient).
    T : int
        Truncation horizon for Jacobians.
    h : float
        Finite-difference step size.

    Returns
    -------
    dict with:
        'dlogK'  : ndarray (T,) — IRF of log K/Kbar to a unit TFP shock
        'Kt_ssj' : ndarray (T,) — IRF coefficients normalised by shock size
        'J_r'    : ndarray (T,T) — partial Jacobian w.r.t. r
        'J_W'    : ndarray (T,T) — partial Jacobian w.r.t. W
    """
    r_ss = ss["r"]
    K_ss = ss["A"]
    R_ss = r_ss + delta  # gross rental rate (MPK)
    W_ss = (1.0 - alpha) * K_ss**alpha  # wage at Z=1

    # --- Step 1: partial equilibrium Jacobians ---
    shocks = {
        "r_shock": {"r": 1.0},
        "W_shock": {"y": e_grid},
    }
    Js = jacobian(ss, shocks, T)

    J_r = Js["A"]["r_shock"]   # dK/dr, shape (T, T)
    J_W = Js["A"]["W_shock"]   # dK/dW, shape (T, T)

    # --- Step 2: construct GE matrices ---
    # Firm FOCs (log-linearised around Z=1 steady state):
    #   dr = R_ss * ((alpha-1)*dlogK + dlogZ)
    #   dW = W_ss * (alpha * dlogK + dlogZ)
    #
    # With lag: prices at time s depend on K_s, which was determined at s-1.
    # In the Jacobian framework, this means we multiply by the lag operator L.
    L = np.zeros((T, T))
    L[1:, :-1] = np.eye(T - 1)

    # Price → dlogK mapping (through firm FOCs, with lag)
    price_elast_K = {
        "r_shock": R_ss * (alpha - 1) * L,
        "W_shock": W_ss * alpha * L,
    }
    # Price → dlogZ mapping (direct, no lag: Z enters contemporaneously)
    price_elast_Z = {
        "r_shock": R_ss * np.eye(T),
        "W_shock": W_ss * np.eye(T),
    }

    H_U, H_Z = build_ge_matrices(
        Js_A={"r_shock": J_r, "W_shock": J_W},
        price_elasticities_K=price_elast_K,
        price_elasticities_Z=price_elast_Z,
        K_ss=K_ss,
        T=T,
    )

    # --- Step 3: construct TFP shock path ---
    dlogZ = np.zeros(T)
    dlogZ[0] = 1.0  # unit innovation
    for t in range(1, T):
        dlogZ[t] = rho_z * dlogZ[t - 1]

    # --- Step 4: solve GE ---
    dlogK = solve_ge(H_U, H_Z, dlogZ)

    # Normalised IRF coefficients (response per unit innovation)
    Kt_ssj = dlogK.copy()

    return {
        "dlogK": dlogK,
        "Kt_ssj": Kt_ssj,
        "J_r": J_r,
        "J_W": J_W,
        "dlogZ": dlogZ,
    }


def simulate_from_ssj(
    Kt_ssj: np.ndarray,
    eps_path: np.ndarray,
    rho_z: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate K and Z paths from SSJ IRF coefficients.

    Parameters
    ----------
    Kt_ssj : ndarray, shape (T_ssj,)
        IRF coefficients from ``ssj_ge_irfs``.
    eps_path : ndarray, shape (T_sim,)
        Simulated TFP innovations.
    rho_z : float
        AR(1) persistence of aggregate TFP.

    Returns
    -------
    K_path : ndarray, shape (T_sim,)
        Log deviation of K from steady state.
    Z_path : ndarray, shape (T_sim,)
        Log TFP path.
    """
    T_sim = len(eps_path)
    T_ssj = len(Kt_ssj)

    # Simulate Z
    Z_path = np.zeros(T_sim)
    for t in range(1, T_sim):
        Z_path[t] = rho_z * Z_path[t - 1] + eps_path[t]

    # K from convolution with IRF coefficients
    # K_t = sum_{j=1}^{T_ssj-1} Kt_ssj[j] * eps_{t-j+1}
    # (Kt_ssj[0] is the contemporaneous response, Kt_ssj[1] is one period later, etc.)
    K_path = np.zeros(T_sim)
    padded_eps = np.concatenate([np.zeros(T_ssj), eps_path])
    for t in range(T_sim):
        # eps at time t is padded_eps[T_ssj + t]
        # Kt_ssj[j] responds to eps at t - j + 1, i.e. j periods earlier
        idx_start = T_ssj + t
        K_path[t] = np.dot(
            Kt_ssj[1:],
            padded_eps[idx_start - 1 : idx_start - T_ssj : -1],
        )

    return K_path, Z_path


# ── Phase 4: Core Krusell-Smith Algorithm (EGM-based) ─────────────────────


def backward_egm_ks(
    Va_4d: np.ndarray,
    kgrid: np.ndarray,
    Kgrid: np.ndarray,
    egrid: np.ndarray,
    zgrid: np.ndarray,
    Qee: np.ndarray,
    Qzz: np.ndarray,
    ck_lom: np.ndarray,
    params: dict,
    ncheb_K: int = 3,
    ncheb_Z: int = 3,
    K_bounds: Tuple[float, float] = None,
    Z_bounds: Tuple[float, float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single EGM backward step on the 4D Krusell-Smith state space.

    For each aggregate state (K_j, Z_m) on the grid:

    1. Compute K' from the Chebyshev law of motion.
    2. Interpolate Va to K' (linear in the K dimension).
    3. Integrate over future (e', Z') to get expected marginal value.
    4. Run the standard 1D EGM in individual assets k.

    This is standard CRRA utility, not Epstein-Zin.

    Parameters
    ----------
    Va_4d : ndarray, shape (nk, nK, ne, nz)
        Marginal value of assets on the 4D grid.
    kgrid : ndarray, shape (nk,)
        Individual asset grid.
    Kgrid : ndarray, shape (nK,)
        Aggregate capital grid (log deviations).
    egrid : ndarray, shape (ne,)
        Individual productivity levels (E[e]=1).
    zgrid : ndarray, shape (nz,)
        Aggregate TFP grid (log deviations).
    Qee : ndarray, shape (ne, ne)
        Idiosyncratic income transition matrix.
    Qzz : ndarray, shape (nz, nz)
        Aggregate TFP transition matrix.
    ck_lom : ndarray
        Chebyshev coefficients for K' = Gamma(K, Z).
    params : dict
        Must contain: ``beta``, ``theta``, ``alpha``, ``delta``, ``Kss``.
    ncheb_K, ncheb_Z : int
        Chebyshev orders used in the law of motion.
    K_bounds, Z_bounds : tuple, optional
        Domain bounds for Chebyshev evaluation.

    Returns
    -------
    Va_new : ndarray, shape (nk, nK, ne, nz)
    a_pol  : ndarray, shape (nk, nK, ne, nz)
    c_pol  : ndarray, shape (nk, nK, ne, nz)
    """
    nk, nK_dim, ne, nz_dim = Va_4d.shape
    beta = params["beta"]
    eis = 1.0 / params["theta"]
    alpha_val = params["alpha"]
    delta_val = params["delta"]
    Kss = params["Kss"]

    if K_bounds is None:
        K_bounds = (Kgrid[0], Kgrid[-1])
    if Z_bounds is None:
        Z_bounds = (zgrid[0], zgrid[-1])

    Va_new = np.empty_like(Va_4d)
    a_pol = np.empty_like(Va_4d)
    c_pol = np.empty_like(Va_4d)

    for jK in range(nK_dim):
        for mz in range(nz_dim):
            logK = Kgrid[jK]
            logZ = zgrid[mz]
            K_level = np.exp(logK) * Kss
            Z_level = np.exp(logZ)

            # Prices
            r = alpha_val * Z_level * K_level ** (alpha_val - 1) - delta_val
            W = (1.0 - alpha_val) * Z_level * K_level ** alpha_val
            y = W * egrid  # (ne,)

            # Law of motion: K' in log deviation
            Kp = float(eval_law_of_motion(
                ck_lom, np.array([logK]), np.array([logZ]),
                nK=ncheb_K, nZ=ncheb_Z,
                K_bounds=K_bounds, Z_bounds=Z_bounds,
            ))
            Kp = np.clip(Kp, Kgrid[0], Kgrid[-1])

            # Interpolate Va in K dimension at K' (linear)
            idx = np.searchsorted(Kgrid, Kp, side="right") - 1
            idx = np.clip(idx, 0, nK_dim - 2)
            w = (Kp - Kgrid[idx]) / (Kgrid[idx + 1] - Kgrid[idx])
            # Va_at_Kp shape: (nk, ne, nz)
            Va_at_Kp = (1.0 - w) * Va_4d[:, idx, :, :] + w * Va_4d[:, idx + 1, :, :]

            # Integrate over Z': temp[k, e'] = sum_{Z'} Qzz[mz, Z'] * Va[k, e', Z']
            temp = Va_at_Kp @ Qzz[mz, :]  # (nk, ne)

            # Integrate over e': E_Va[e, k] = sum_{e'} Qee[e, e'] * temp[k, e']
            E_Va = Qee @ temp.T  # (ne, nk)

            # Standard EGM step (CRRA utility)
            Wa = beta * E_Va
            c_endog = Wa ** (-eis)
            coh = y[:, np.newaxis] + (1.0 + r) * kgrid[np.newaxis, :]

            a_pol_2d = np.empty((ne, nk))
            for ie in range(ne):
                a_pol_2d[ie, :] = np.interp(
                    coh[ie, :],
                    c_endog[ie, :] + kgrid,
                    kgrid,
                )
            a_pol_2d = np.maximum(a_pol_2d, kgrid[0])
            c_pol_2d = coh - a_pol_2d
            Va_new_2d = (1.0 + r) * c_pol_2d ** (-1.0 / eis)

            # Store — transpose (ne, nk) -> (nk, ne) for the 4D layout
            Va_new[:, jK, :, mz] = Va_new_2d.T
            a_pol[:, jK, :, mz] = a_pol_2d.T
            c_pol[:, jK, :, mz] = c_pol_2d.T

    return Va_new, a_pol, c_pol


def policy_iteration_ks(
    ss: dict,
    kgrid: np.ndarray,
    Kgrid: np.ndarray,
    egrid: np.ndarray,
    zgrid: np.ndarray,
    Qee: np.ndarray,
    Qzz: np.ndarray,
    ck_lom: np.ndarray,
    params: dict,
    ncheb_K: int = 3,
    ncheb_Z: int = 3,
    K_bounds: Tuple[float, float] = None,
    Z_bounds: Tuple[float, float] = None,
    tol: float = 1e-9,
    max_iter: int = 10_000,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterate EGM to convergence on the 4D KS state space.

    Standard CRRA utility.  For each aggregate state (K, Z) the household
    faces a 1D savings problem, so the existing EGM machinery applies
    directly.  Much faster than VFI with golden search.

    Parameters
    ----------
    ss : dict
        Steady state from ``household_steady_state``.  Used only to
        initialise Va from ``ss['Va']``.
    kgrid, Kgrid, egrid, zgrid : ndarray
        Grid points for each state dimension.
    Qee, Qzz : ndarray
        Transition matrices for idiosyncratic income and aggregate TFP.
    ck_lom : ndarray
        Chebyshev coefficients for the law of motion K' = Gamma(K, Z).
    params : dict
        Must contain: ``beta``, ``theta``, ``alpha``, ``delta``, ``Kss``.
    ncheb_K, ncheb_Z : int
        Chebyshev orders for the law of motion.
    K_bounds, Z_bounds : tuple, optional
        Domain bounds for Chebyshev evaluation.
    tol : float
        Convergence tolerance on max policy change.
    max_iter : int
        Maximum EGM iterations.
    verbose : bool
        Print convergence information.

    Returns
    -------
    Va_4d : ndarray, shape (nk, nK, ne, nz)
        Converged marginal value of assets.
    a_pol_4d : ndarray, shape (nk, nK, ne, nz)
        Asset policy a'(k, K, e, Z).
    c_pol_4d : ndarray, shape (nk, nK, ne, nz)
        Consumption policy c(k, K, e, Z).
    """
    nk = len(kgrid)
    nK_dim = len(Kgrid)
    ne = len(egrid)
    nz_dim = len(zgrid)

    # Initialise Va from steady state, replicated across (K, Z)
    Va_ss = ss["Va"]  # (ne, nk)
    Va_4d = np.empty((nk, nK_dim, ne, nz_dim))
    for jK in range(nK_dim):
        for mz in range(nz_dim):
            Va_4d[:, jK, :, mz] = Va_ss.T  # (ne, nk) -> (nk, ne)

    a_old = np.zeros_like(Va_4d)

    for it in range(1, max_iter + 1):
        Va_4d, a_pol_4d, c_pol_4d = backward_egm_ks(
            Va_4d, kgrid, Kgrid, egrid, zgrid,
            Qee, Qzz, ck_lom, params,
            ncheb_K=ncheb_K, ncheb_Z=ncheb_Z,
            K_bounds=K_bounds, Z_bounds=Z_bounds,
        )

        err = np.max(np.abs(a_pol_4d - a_old))
        if verbose and (it <= 5 or it % 100 == 0):
            print(f"  EGM iter {it:4d}  err={err:.2e}")

        if err < tol:
            if verbose:
                print(f"  Converged at iteration {it}")
            break
        a_old = a_pol_4d.copy()

    return Va_4d, a_pol_4d, c_pol_4d


def simulate_ks_economy(
    a_pol_4d: np.ndarray,
    D_ss: np.ndarray,
    kgrid: np.ndarray,
    Kgrid: np.ndarray,
    egrid: np.ndarray,
    zgrid: np.ndarray,
    Qee: np.ndarray,
    Z_path: np.ndarray,
    params: dict,
    Tsim: int = None,
) -> np.ndarray:
    """Forward simulate the distribution along an aggregate shock path.

    At each period *t*:

    1. Compute aggregate capital K_t from the cross sectional distribution.
    2. Interpolate the 4D asset policy at the current (K_t, Z_t).
    3. Update the distribution using lottery based forward iteration.

    Parameters
    ----------
    a_pol_4d : ndarray, shape (nk, nK, ne, nz)
        Asset policy a'(k, K, e, Z) from ``policy_iteration_ks``.
    D_ss : ndarray, shape (ne, nk)
        Steady state distribution over (e, k).
    kgrid : ndarray, shape (nk,)
        Individual asset grid.
    Kgrid : ndarray, shape (nK,)
        Aggregate capital grid (log deviations).
    egrid : ndarray, shape (ne,)
        Individual productivity grid (levels, E[e]=1).
    zgrid : ndarray, shape (nz,)
        Aggregate TFP grid (log deviations).
    Qee : ndarray, shape (ne, ne)
        Idiosyncratic income transition matrix.
    Z_path : ndarray, shape (Tsim,)
        Aggregate TFP path (log deviations from 1).
    params : dict
        Must contain: ``alpha``, ``delta``, ``Kss``.
    Tsim : int, optional
        Simulation length. Defaults to ``len(Z_path)``.

    Returns
    -------
    K_path_new : ndarray, shape (Tsim,)
        Simulated aggregate capital path in log deviations from Kss.
    """
    Kss = params["Kss"]

    if Tsim is None:
        Tsim = len(Z_path)

    nk = len(kgrid)
    ne = len(egrid)
    nK_dim = len(Kgrid)
    nz_dim = len(zgrid)

    D = D_ss.copy()
    K_new = np.zeros(Tsim)

    for t in range(Tsim):
        # Aggregate capital from distribution
        K_t = np.sum(D * kgrid[np.newaxis, :])
        K_new[t] = K_t

        logK_t = np.log(K_t / Kss)
        logZ_t = Z_path[t]

        # Bilinear interpolation of a_pol_4d in (K, Z) dimensions
        # to get a_pol_2d(k, e) at the current aggregate state
        iK = np.searchsorted(Kgrid, logK_t, side="right") - 1
        iK = np.clip(iK, 0, nK_dim - 2)
        wK = (logK_t - Kgrid[iK]) / (Kgrid[iK + 1] - Kgrid[iK])

        iz = np.searchsorted(zgrid, logZ_t, side="right") - 1
        iz = np.clip(iz, 0, nz_dim - 2)
        wz = (logZ_t - zgrid[iz]) / (zgrid[iz + 1] - zgrid[iz])

        # a_pol_4d has shape (nk, nK, ne, nz)
        a_interp = (
            (1 - wK) * (1 - wz) * a_pol_4d[:, iK, :, iz]
            + wK * (1 - wz) * a_pol_4d[:, iK + 1, :, iz]
            + (1 - wK) * wz * a_pol_4d[:, iK, :, iz + 1]
            + wK * wz * a_pol_4d[:, iK + 1, :, iz + 1]
        )  # (nk, ne)

        # Transpose to (ne, nk) for get_lottery / forward_iteration
        kprime_2d = np.clip(a_interp.T, kgrid[0], kgrid[-1])

        # Update distribution
        if t < Tsim - 1:
            a_i, a_pi = get_lottery(kprime_2d, kgrid)
            D = forward_iteration(D, Qee, a_i, a_pi)

    K_path_new = np.log(K_new / Kss)
    return K_path_new


def solve_krusell_smith(
    ss: dict,
    params: dict,
    Z_sim: np.ndarray,
    K_init: np.ndarray,
    Qee: np.ndarray,
    Qzz: np.ndarray,
    kgrid: np.ndarray,
    Kgrid: np.ndarray,
    egrid: np.ndarray,
    zgrid: np.ndarray,
    ncheb_K: int = 3,
    ncheb_Z: int = 3,
    step: float = 0.5,
    tol_outer: float = 1e-6,
    max_outer: int = 20,
    burn_in: int = 100,
    verbose: bool = True,
    **egm_kwargs,
) -> KSResult:
    """Run the full Krusell-Smith outer loop.

    The algorithm iterates between:

    1. **Inner loop** — Solve the household problem on the 4D state space
       (k, K, e, Z) via EGM, given a conjectured law of motion.
    2. **Simulation** — Forward simulate the economy to get a new K path.
    3. **Projection** — Re-estimate the law of motion via Chebyshev regression.
    4. **Damping** — Update K = K + step * (K_new - K).

    Uses standard CRRA utility and the endogenous grid method.

    Parameters
    ----------
    ss : dict
        Steady state from ``household_steady_state``.  Must contain
        ``D``, ``Va``, ``a``, ``c``, ``A``.
    params : dict
        Must contain: ``beta``, ``theta``, ``alpha``, ``delta``, ``Kss``.
    Z_sim : ndarray, shape (Tsim,)
        Simulated aggregate TFP path (log deviations).
    K_init : ndarray, shape (Tsim,)
        Initial guess for capital path (log deviations).
    Qee : ndarray, shape (ne, ne)
        Idiosyncratic income transition matrix.
    Qzz : ndarray, shape (nz, nz)
        Aggregate TFP transition matrix.
    kgrid, Kgrid, egrid, zgrid : ndarray
        Grids for each state dimension.
    ncheb_K, ncheb_Z : int
        Chebyshev polynomial orders for the law of motion.
    step : float
        Damping step size (weight on the new K path).
    tol_outer : float
        Convergence tolerance on the capital path error.
    max_outer : int
        Maximum outer loop iterations.
    burn_in : int
        Periods to discard when fitting the law of motion.
    verbose : bool
        Print progress information.
    **egm_kwargs
        Extra keyword arguments forwarded to ``policy_iteration_ks``
        (e.g. ``tol``, ``max_iter``).

    Returns
    -------
    KSResult
        Container with ``K_path``, ``Z_path``, ``ck`` (Chebyshev
        coefficients), ``Rsq``, convergence flag, and history.
    """
    Tsim = len(Z_sim)
    Kss = params["Kss"]

    # Chebyshev bounds
    K_bounds = (Kgrid[0], Kgrid[-1])
    Z_bounds = (zgrid[0], zgrid[-1])

    # Initial law of motion projection
    Kt = K_init.copy()
    ck, Rsq = fit_law_of_motion(
        Kt[burn_in:-1],
        Z_sim[burn_in:-1],
        Kt[burn_in + 1 :],
        nK=ncheb_K,
        nZ=ncheb_Z,
        K_bounds=K_bounds,
        Z_bounds=Z_bounds,
    )

    if verbose:
        print(f"Initial law of motion R² = {Rsq:.6f}")

    Rsq_history = [Rsq]
    error_history: List[float] = []

    for outer_it in range(1, max_outer + 1):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"KS Outer Iteration {outer_it}")
            print(f"{'=' * 60}")
            print("Solving household problem (EGM)...")

        # Solve 4D household problem via EGM
        Va_4d, a_pol_4d, c_pol_4d = policy_iteration_ks(
            ss, kgrid, Kgrid, egrid, zgrid,
            Qee, Qzz, ck, params,
            ncheb_K=ncheb_K, ncheb_Z=ncheb_Z,
            K_bounds=K_bounds, Z_bounds=Z_bounds,
            verbose=verbose,
            **egm_kwargs,
        )

        if verbose:
            print("Simulating economy...")

        # Forward simulate
        K_new = simulate_ks_economy(
            a_pol_4d, ss["D"], kgrid, Kgrid, egrid, zgrid,
            Qee, Z_sim, params, Tsim,
        )

        # Compute error (after burn-in)
        err = np.linalg.norm(K_new[burn_in:] - Kt[burn_in:])
        error_history.append(err)

        if verbose:
            print(f"Outer iter {outer_it}: K path error = {err:.2e}")

        if err < tol_outer:
            if verbose:
                print("Converged!")
            return KSResult(
                K_path=Kt,
                Z_path=Z_sim,
                ck=ck,
                Rsq=Rsq,
                converged=True,
                n_iterations=outer_it,
                Rsq_history=Rsq_history,
                error_history=error_history,
            )

        # Damped update of K path
        Kt = Kt + step * (K_new - Kt)

        # Re-project law of motion
        ck, Rsq = fit_law_of_motion(
            Kt[burn_in:-1],
            Z_sim[burn_in:-1],
            Kt[burn_in + 1 :],
            nK=ncheb_K,
            nZ=ncheb_Z,
            K_bounds=K_bounds,
            Z_bounds=Z_bounds,
        )
        Rsq_history.append(Rsq)

        if verbose:
            print(f"R² = {Rsq:.6f}")

    if verbose:
        print(f"Did not converge after {max_outer} iterations (err={err:.2e})")

    return KSResult(
        K_path=Kt,
        Z_path=Z_sim,
        ck=ck,
        Rsq=Rsq,
        converged=False,
        n_iterations=max_outer,
        Rsq_history=Rsq_history,
        error_history=error_history,
    )
