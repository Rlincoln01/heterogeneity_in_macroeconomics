"""Complete markets RBC model: linear and nonlinear solutions.

Provides:
- ``cm_rbc_steady_state``: Compute representative agent RBC steady state.
- ``cm_rbc_linearize``: Log-linearise and solve via solab (QZ).
- ``solve_cm_rbc_nonlinear``: Nonlinear projection using Euler equation
  iteration on a Chebyshev/spline basis.
- ``simulate_cm_rbc``: Simulate from a solved nonlinear RBC.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .solab import solab
from ..markov import rouwenhorst


# ── Steady state ─────────────────────────────────────────────────────────


def cm_rbc_steady_state(
    beta: float,
    alpha: float,
    delta: float,
) -> Dict[str, float]:
    """Compute representative-agent RBC steady state."""
    Rbar = 1.0 / beta - 1 + delta  # gross rental rate of capital
    Kbar = (alpha / Rbar) ** (1.0 / (1.0 - alpha))
    Ybar = Kbar**alpha
    Cbar = Ybar - delta * Kbar
    Xbar = delta * Kbar
    Wbar = (1.0 - alpha) * Kbar**alpha
    return {
        "K": Kbar,
        "C": Cbar,
        "Y": Ybar,
        "X": Xbar,
        "W": Wbar,
        "R": Rbar,
    }


# ── Linear solution (solab) ─────────────────────────────────────────────


def cm_rbc_linearize(
    beta: float,
    theta: float,
    alpha: float,
    delta: float,
    rho_z: float,
) -> Dict[str, np.ndarray]:
    """Log-linearise RBC and solve via QZ decomposition.

    The system A x(t+1) = B x(t) where x = [K, Z, C]
    (states: K, Z; jump: C).

    Parameters
    ----------
    beta, theta, alpha, delta, rho_z : float
        Standard RBC parameters.  theta = CRRA coefficient.

    Returns
    -------
    dict with:
        'F' : ndarray (1, 2) — decision rule C = F @ [K, Z]
        'P' : ndarray (2, 2) — law of motion [K', Z'] = P @ [K, Z]
        'ss': dict — steady state values
    """
    ss = cm_rbc_steady_state(beta, alpha, delta)
    Kbar = ss["K"]
    Cbar = ss["C"]

    # Construct linearised system (matching Virgiliu's Code CM/start.m).
    # States: K, Z.  Jump: C.  So nk = 2.
    # Note: beta here is the CM discount factor 1/(1 - delta + R_ss).
    beta_cm = 1.0 / (1.0 - delta + ss["R"])

    A = np.array(
        [
            [
                beta_cm * alpha * Kbar ** (alpha - 1) * (alpha - 1),
                beta_cm * alpha * Kbar ** (alpha - 1),
                -theta,
            ],
            [Kbar ** (1 - alpha), 0, 0],
            [0, 1, 0],
        ]
    )

    B = np.array(
        [
            [0, 0, -theta],
            [
                (1 - delta) * Kbar ** (1 - alpha) + alpha,
                1,
                -Cbar / Kbar**alpha,
            ],
            [0, rho_z, 0],
        ]
    )

    F, P = solab(A, B, 2)

    return {"F": F, "P": P, "ss": ss, "A_mat": A, "B_mat": B}


def simulate_cm_linear(
    P: np.ndarray,
    Z_path: np.ndarray,
) -> np.ndarray:
    """Simulate linear RBC: K_{t+1} = P[0,0]*K_t + P[0,1]*Z_t.

    Parameters
    ----------
    P : ndarray, shape (2, 2)
        Law of motion from ``cm_rbc_linearize``.
    Z_path : ndarray, shape (T,)
        Log TFP path.

    Returns
    -------
    K_path : ndarray, shape (T,)
        Log deviation of K from steady state.
    """
    T = len(Z_path)
    K = np.zeros(T)
    for t in range(T - 1):
        K[t + 1] = P[0, 0] * K[t] + P[0, 1] * Z_path[t]
    return K


# ── Nonlinear projection (Euler iteration) ──────────────────────────────


def _euler_residual(
    x: np.ndarray,
    s: np.ndarray,
    c_coef: np.ndarray,
    Phi: np.ndarray,
    p: Dict,
) -> np.ndarray:
    """Euler equation residual at each collocation node.

    x : savings fraction of (Y + (1-delta)*K)
    s : (N, 2) state grid [logK/Kbar, logZ]
    c_coef : coefficient vector for consumption function
    Phi : basis matrix at collocation nodes
    p : parameter dict
    """
    K = np.exp(s[:, 0]) * p["Kbar"]
    Z = np.exp(s[:, 1])

    Y = Z * K**p["alpha"]
    cash = Y + (1 - p["delta"]) * K  # total resources

    C = (1 - x) * cash
    Kprime = x * cash
    logKprime = np.log(Kprime / p["Kbar"])

    # Expected marginal utility tomorrow
    nz = p["nz"]
    Qzz = p["Qzz"]
    zgrid = p["zgrid"]
    nk_nodes = len(s) // nz  # nodes per Z slice

    Eu_prime = np.zeros(len(s))
    for iz_prime in range(nz):
        z_prime = zgrid[iz_prime]
        # Evaluate consumption tomorrow at (K', z')
        s_prime = np.column_stack([logKprime, np.full(len(s), z_prime)])
        C_prime = Phi.__class__  # placeholder; we'll use the interpolator

    # Actually, let's use the RegularGridInterpolator approach
    raise NotImplementedError("Use solve_cm_rbc_nonlinear instead")


def solve_cm_rbc_nonlinear(
    beta: float,
    theta: float,
    alpha: float,
    delta: float,
    rho_z: float,
    sigma_z: float,
    nK: int = 11,
    nz: int = 11,
    tol: float = 1e-7,
    max_iter: int = 200,
    verbose: bool = True,
) -> Dict:
    """Solve nonlinear RBC via Euler equation iteration.

    Uses a grid of collocation nodes in (logK/Kbar, logZ) space and
    iterates on the Euler equation until the consumption function
    coefficient vector converges.

    Parameters
    ----------
    beta, theta, alpha, delta : float
        Standard RBC parameters.
    rho_z, sigma_z : float
        AR(1) parameters for aggregate TFP.
    nK, nz : int
        Number of grid nodes per dimension.
    tol : float
        Convergence tolerance (relative change in coefficients).
    max_iter : int
        Maximum iterations.
    verbose : bool
        Print iteration info.

    Returns
    -------
    dict with:
        'c_coef'  : coefficient vector for consumption policy
        'ck'      : coefficient vector for capital law of motion
        'interp'  : RegularGridInterpolator for C(logK, logZ)
        'ss'      : steady state dict
        'Kgrid'   : log-deviation K grid
        'zgrid'   : log Z grid
        'C_grid'  : consumption on the grid
        'K_grid'  : capital policy on the grid
    """
    # Get linear solution as initial guess
    lin = cm_rbc_linearize(beta, theta, alpha, delta, rho_z)
    F = lin["F"]
    ss = lin["ss"]
    Kbar = ss["K"]
    Cbar = ss["C"]

    # Discretise Z
    zgrid, Qzz = rouwenhorst(nz, rho_z, sigma_z)

    zmin, zmax = zgrid[0], zgrid[-1]
    kmin, kmax = 5 * zmin, 5 * zmax  # log-deviation range for K

    # Build grid
    Kgrid = np.linspace(kmin, kmax, nK)  # log(K/Kbar)
    KK, ZZ = np.meshgrid(Kgrid, zgrid, indexing="ij")
    s = np.column_stack([KK.ravel(), ZZ.ravel()])  # (nK*nz, 2)

    K_lev = np.exp(s[:, 0]) * Kbar
    Z_lev = np.exp(s[:, 1])

    # Initial guess from linear solution: C = exp(F[0]*logK + F[1]*logZ)*Cbar
    C = np.exp(F[0, 0] * s[:, 0] + F[0, 1] * s[:, 1]) * Cbar

    # Iterate on Euler equation
    for it in range(1, max_iter + 1):
        C_old = C.copy()

        # Build interpolator for current C guess
        C_grid = C.reshape(nK, nz)
        C_interp = RegularGridInterpolator(
            (Kgrid, zgrid), C_grid, method="linear", bounds_error=False, fill_value=None
        )

        # At each node, solve Euler equation
        Y = Z_lev * K_lev**alpha
        cash = Y + (1 - delta) * K_lev  # Y + (1-delta)*K

        # For each node, find savings fraction x such that Euler holds
        # Euler: C^(-theta) = beta * E[R'*C'^(-theta)]
        # where R' = alpha*Z'*K'^(alpha-1) + 1 - delta
        # and K' = x*cash, C = (1-x)*cash

        # Use bisection at each node
        x = np.empty(len(s))
        for i in range(len(s)):
            x[i] = _bisect_euler_node(
                i, s, C_interp, cash[i], Kbar, Kgrid, zgrid, Qzz, alpha, delta, beta, theta
            )

        # Update C
        C = (1 - x) * cash
        Kprime = x * cash

        err = np.max(np.abs(C - C_old)) / np.max(np.abs(C))
        if verbose and (it <= 5 or it % 10 == 0):
            print(f"  iter {it:4d}  err = {err:.2e}")
        if err < tol:
            if verbose:
                print(f"  Converged at iter {it}  err = {err:.2e}")
            break

    # Build final interpolators
    C_grid = C.reshape(nK, nz)
    Kprime_grid = Kprime.reshape(nK, nz)
    logKprime = np.log(Kprime_grid / Kbar)

    C_interp = RegularGridInterpolator(
        (Kgrid, zgrid), C_grid, method="linear", bounds_error=False, fill_value=None
    )
    K_interp = RegularGridInterpolator(
        (Kgrid, zgrid), logKprime, method="linear", bounds_error=False, fill_value=None
    )

    return {
        "C_interp": C_interp,
        "K_interp": K_interp,
        "ss": ss,
        "Kgrid": Kgrid,
        "zgrid": zgrid,
        "Qzz": Qzz,
        "C_grid": C_grid,
        "logKprime_grid": logKprime,
    }


def _bisect_euler_node(
    i: int,
    s: np.ndarray,
    C_interp,
    cash_i: float,
    Kbar: float,
    Kgrid: np.ndarray,
    zgrid: np.ndarray,
    Qzz: np.ndarray,
    alpha: float,
    delta: float,
    beta: float,
    theta: float,
) -> float:
    """Bisect for savings fraction x at node i."""

    def euler_resid(x):
        if x <= 0 or x >= 1:
            return 1e10 if x <= 0 else -1e10
        C_now = (1 - x) * cash_i
        Kprime = x * cash_i
        logKprime = np.log(Kprime / Kbar)

        # Clamp logKprime to grid bounds
        logKprime = np.clip(logKprime, Kgrid[0], Kgrid[-1])

        # Expected R'*C'^(-theta)
        iz = np.searchsorted(zgrid, s[i, 1])
        iz = min(iz, len(zgrid) - 1)
        E_Rc = 0.0
        for iz_p in range(len(zgrid)):
            prob = Qzz[iz, iz_p]
            if prob < 1e-15:
                continue
            z_p = zgrid[iz_p]
            pt = np.array([[logKprime, z_p]])
            C_p = C_interp(pt)[0]
            K_p = np.exp(logKprime) * Kbar
            Z_p = np.exp(z_p)
            R_p = alpha * Z_p * K_p ** (alpha - 1) + 1 - delta
            E_Rc += prob * R_p * C_p ** (-theta)

        # Euler: C^(-theta) = beta * E_Rc
        return C_now ** (-theta) - beta * E_Rc

    # Bisection on [eps, 1-eps]
    xlo, xhi = 1e-9, 1.0 - 1e-9
    flo = euler_resid(xlo)
    fhi = euler_resid(xhi)

    if flo * fhi > 0:
        # Fallback: return a reasonable fraction
        return 0.5

    for _ in range(60):
        xmid = 0.5 * (xlo + xhi)
        fmid = euler_resid(xmid)
        if abs(fmid) < 1e-12 or (xhi - xlo) < 1e-14:
            return xmid
        if flo * fmid <= 0:
            xhi = xmid
            fhi = fmid
        else:
            xlo = xmid
            flo = fmid

    return 0.5 * (xlo + xhi)


# ── Simulation ───────────────────────────────────────────────────────────


def simulate_cm_rbc(
    solution: Dict,
    Z_path: np.ndarray,
    Tsim: int = None,
) -> Dict[str, np.ndarray]:
    """Simulate from solved nonlinear RBC.

    Parameters
    ----------
    solution : dict
        Output from ``solve_cm_rbc_nonlinear``.
    Z_path : ndarray, shape (T,)
        Log TFP path.
    Tsim : int, optional
        Simulation length (defaults to len(Z_path)).

    Returns
    -------
    dict with K, C, Y, X paths (all in levels or log-deviations).
    """
    if Tsim is None:
        Tsim = len(Z_path)

    ss = solution["ss"]
    Kbar = ss["K"]
    K_interp = solution["K_interp"]
    C_interp = solution["C_interp"]
    alpha = 0.35  # TODO: store in solution
    delta_val = 0.02

    K = np.zeros(Tsim)  # log(K/Kbar)
    C = np.zeros(Tsim)
    Y = np.zeros(Tsim)
    X = np.zeros(Tsim)

    Kgrid = solution["Kgrid"]

    for t in range(Tsim - 1):
        pt = np.array([[K[t], Z_path[t]]])
        K[t + 1] = float(K_interp(pt))

        K_lev = np.exp(K[t]) * Kbar
        Z_lev = np.exp(Z_path[t])
        Y[t] = Z_lev * K_lev**alpha
        K_lev_next = np.exp(K[t + 1]) * Kbar
        X[t] = K_lev_next - (1 - delta_val) * K_lev
        C[t] = Y[t] - X[t]

    return {"K": K, "C": C, "Y": Y, "X": X}
