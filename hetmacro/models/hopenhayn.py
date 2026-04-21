"""Hopenhayn (1992) model: entry, exit, and the stationary firm size distribution.

Provides building blocks for the vanilla Hopenhayn model (no firing costs,
no capital) and two GE closures:

- **Inelastic labor supply** (N^s = 1): wage pinned by free entry alone.
- **Elastic labor supply** (u(C) - aN): representative household with
  employment lotteries (Rogerson 1988), log or CRRA utility.

The value function solves by direct matrix inversion (no VFI), since the
only intertemporal decision is the binary stay/exit choice.

References
----------
Hopenhayn (1992). "Entry, Exit, and Firm Dynamics in Long Run Equilibrium."
    Econometrica 60(5), 1127-1150.
Hopenhayn and Rogerson (1993). "Job Turnover and Policy Evaluation: A General
    Equilibrium Analysis." JPE 101(5), 915-938.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..markov import rouwenhorst, stationary_distribution
from ..optimize import brentq


# ── Static Firm Problem ───────────────────────────────────────────────────


def hopenhayn_profit(
    z_grid: np.ndarray, w: float, theta: float, phi: float = 0.0,
) -> np.ndarray:
    """Static profit for each productivity level.

    Parameters
    ----------
    z_grid : (n_z,) array
        Productivity levels (in levels, not logs).
    w : float
        Wage rate.
    theta : float
        Span of control (DRS exponent), in (0, 1).
    phi : float
        Per-period fixed operating cost.

    Returns
    -------
    profit : (n_z,) array
        Maximized profit pi(z) = (1-theta)*(theta/w)^{theta/(1-theta)}
        * z^{1/(1-theta)} - phi.
    """
    exp = theta / (1.0 - theta)
    return (1.0 - theta) * (theta / w) ** exp * z_grid ** (1.0 / (1.0 - theta)) - phi


def hopenhayn_labor_demand(
    z_grid: np.ndarray, w: float, theta: float,
) -> np.ndarray:
    """Optimal labor demand from the static FOC.

    Returns
    -------
    n : (n_z,) array
        n(z) = (z * theta / w)^{1/(1-theta)}.
    """
    return (z_grid * theta / w) ** (1.0 / (1.0 - theta))


def hopenhayn_output(
    z_grid: np.ndarray, n: np.ndarray, theta: float,
) -> np.ndarray:
    """Per-firm output: y(z) = z * n(z)^theta."""
    return z_grid * n ** theta


# ── Value Function (Matrix Inversion) ─────────────────────────────────────


def hopenhayn_value_function(
    Pi_z: np.ndarray,
    profit: np.ndarray,
    beta: float,
    z_star_idx: int,
) -> np.ndarray:
    """Solve V = (I - beta*Q)^{-1} * pi by matrix inversion.

    Constructs the sub-Markov kernel Q by zeroing rows of Pi_z for
    firms below the exit threshold, then solves the linear system.

    Parameters
    ----------
    Pi_z : (n_z, n_z) array
        Full transition matrix for productivity.
    profit : (n_z,) array
        Static profit vector.
    beta : float
        Discount factor.
    z_star_idx : int
        Grid index of the exit threshold. Firms with i < z_star_idx exit.

    Returns
    -------
    V : (n_z,) array
        Value function. For exiting firms (i < z_star_idx), V[i] = profit[i].
    """
    n_z = len(profit)
    Q = Pi_z.copy()
    Q[:z_star_idx, :] = 0.0  # exiting firms: zero rows
    A = np.eye(n_z) - beta * Q
    V = np.linalg.solve(A, profit)
    return V


def hopenhayn_exit_threshold(
    V: np.ndarray, Pi_z: np.ndarray, beta: float,
) -> int:
    """Find the exit threshold index where the continuation value crosses zero.

    Parameters
    ----------
    V : (n_z,) array
        Value function.
    Pi_z : (n_z, n_z) array
        Full transition matrix (not the sub-Markov Q).
    beta : float
        Discount factor.

    Returns
    -------
    z_star_idx : int
        Smallest index i such that the continuation value
        beta * (Pi_z @ V)[i] >= 0.  Returns 0 if all non-negative.
    """
    cont_val = beta * Pi_z @ V
    # cont_val is increasing in z under FOSD; find first non-negative
    idx = np.searchsorted(cont_val, 0.0)
    return int(idx)


def _solve_firm_problem(
    Pi_z: np.ndarray,
    profit: np.ndarray,
    beta: float,
    max_iter: int = 50,
) -> Tuple[np.ndarray, int]:
    """Iterate on exit threshold and matrix inversion until z* converges.

    Parameters
    ----------
    Pi_z : (n_z, n_z) array
        Full productivity transition matrix.
    profit : (n_z,) array
        Static profit vector at the candidate wage.
    beta : float
        Discount factor.
    max_iter : int
        Maximum inner-loop iterations (typically converges in 2-5).

    Returns
    -------
    V : (n_z,) array
        Converged value function.
    z_star_idx : int
        Converged exit threshold index.
    """
    z_star_idx = 0  # optimistic: no exit
    for _ in range(max_iter):
        V = hopenhayn_value_function(Pi_z, profit, beta, z_star_idx)
        z_star_new = hopenhayn_exit_threshold(V, Pi_z, beta)
        if z_star_new == z_star_idx:
            break
        z_star_idx = z_star_new
    return V, z_star_idx


# ── Entry ─────────────────────────────────────────────────────────────────


def hopenhayn_entry_value(
    V: np.ndarray, G: np.ndarray, kappa: float, beta: float,
) -> float:
    """Entry value: V^e = -kappa + beta * G' @ V.

    Parameters
    ----------
    V : (n_z,) array
        Incumbent value function.
    G : (n_z,) array
        Entrant productivity distribution (probability vector).
    kappa : float
        Sunk entry cost.
    beta : float
        Discount factor.

    Returns
    -------
    V_e : float
        Expected value of entry. Free entry requires V_e = 0.
    """
    return -kappa + beta * np.dot(G, V)


# ── Stationary Distribution ───────────────────────────────────────────────


def hopenhayn_stationary_distribution(
    Q: np.ndarray, G: np.ndarray,
) -> np.ndarray:
    """Normalized stationary distribution for M = 1.

    Solves (I - Q') lambda_hat = G.  The actual equilibrium distribution
    is lambda_star = M * lambda_hat.

    Parameters
    ----------
    Q : (n_z, n_z) array
        Sub-Markov kernel (rows zeroed for exiting firms).
    G : (n_z,) array
        Entrant distribution.

    Returns
    -------
    lambda_hat : (n_z,) array
        Firm density per unit of entry mass M.
    """
    n_z = len(G)
    A = np.eye(n_z) - Q.T
    return np.linalg.solve(A, G)


# ── Aggregation ───────────────────────────────────────────────────────────


def hopenhayn_aggregates(
    z_grid: np.ndarray,
    lambda_hat: np.ndarray,
    n_vec: np.ndarray,
    y_vec: np.ndarray,
    profit_vec: np.ndarray,
    z_star_idx: int,
    phi: float,
    kappa: float,
) -> Dict[str, float]:
    """Per-unit-M aggregates from the normalized distribution.

    Parameters
    ----------
    z_grid, lambda_hat, n_vec, y_vec, profit_vec : arrays of shape (n_z,)
    z_star_idx : int
        Exit threshold index.
    phi : float
        Per-period fixed operating cost.
    kappa : float
        Entry cost.

    Returns
    -------
    dict with keys: N_hat, Y_hat, Pi_hat, Phi_hat, exit_rate,
    resource_hat (= Y_hat - phi*Phi_hat - kappa).
    """
    active = np.zeros_like(lambda_hat)
    active[z_star_idx:] = lambda_hat[z_star_idx:]

    N_hat = np.dot(active, n_vec)
    Y_hat = np.dot(active, y_vec)
    Pi_hat = np.dot(active, profit_vec)
    Phi_hat = active.sum()

    total_mass = lambda_hat.sum()
    exit_mass = lambda_hat[:z_star_idx].sum()
    exit_rate = exit_mass / total_mass if total_mass > 0 else 0.0

    resource_hat = Y_hat - phi * Phi_hat - kappa  # C per unit M

    return dict(
        N_hat=N_hat, Y_hat=Y_hat, Pi_hat=Pi_hat,
        Phi_hat=Phi_hat, exit_rate=exit_rate, resource_hat=resource_hat,
    )


# ── Result Container ──────────────────────────────────────────────────────


@dataclass
class HopenhaynSteadyState:
    """All equilibrium objects from the Hopenhayn stationary equilibrium."""

    # Prices
    w: float
    r: float

    # Firm problem
    V: np.ndarray
    z_star: float
    z_star_idx: int
    profit: np.ndarray
    labor_demand: np.ndarray
    output: np.ndarray
    cont_value: np.ndarray

    # Distribution
    lambda_hat: np.ndarray
    lambda_star: np.ndarray
    M: float

    # Aggregates
    Y: float
    N: float
    C: float
    Pi_agg: float
    exit_rate: float
    n_active: float

    # Grids and transition
    z_grid: np.ndarray
    Pi_z: np.ndarray
    Q: np.ndarray
    G: np.ndarray

    # Metadata
    params: Dict[str, Any]
    labor_supply: str
    n_outer_iter: int = 0
    walras_residual: float = 0.0
    elapsed: float = 0.0

    def summary(self) -> str:
        """Print a summary of the equilibrium."""
        lines = [
            "Hopenhayn (1992) Stationary Equilibrium",
            f"  Labor supply: {self.labor_supply}",
            f"  w = {self.w:.6f},  r = {self.r:.6f}",
            f"  z* = {self.z_star:.4f} (grid index {self.z_star_idx})",
            f"  M (entrants) = {self.M:.4f}",
            f"  Exit rate = {self.exit_rate:.4f}",
            f"  Active firms = {self.n_active:.4f}",
            f"  Y = {self.Y:.4f},  N = {self.N:.4f},  C = {self.C:.4f}",
            f"  Aggregate profits = {self.Pi_agg:.4f}",
            f"  Walras residual = {self.walras_residual:.2e}",
            f"  Outer iterations = {self.n_outer_iter}",
            f"  Elapsed = {self.elapsed:.2f}s",
        ]
        return "\n".join(lines)


# ── Top-Level Solver ──────────────────────────────────────────────────────


def solve_hopenhayn_ss(
    # Productivity process
    n_z: int = 101,
    rho_z: float = 0.90,
    sigma_z: float = 0.50,
    # Technology
    theta: float = 0.64,
    phi: float = 0.0,
    kappa: float = 1.0,
    # Discount factor
    beta: float = 0.80,
    # Entrant distribution
    G: Optional[np.ndarray] = None,
    # Household closure
    labor_supply: str = "inelastic",
    N_supply: float = 1.0,
    a_disutility: float = 1.0,
    gamma_hh: float = 1.0,
    nu_labor: float = 1.0,
    # Solver options
    w_bounds: Tuple[float, float] = (0.01, 10.0),
    tol: float = 1e-8,
    verbose: bool = False,
) -> HopenhaynSteadyState:
    """Solve the stationary equilibrium of the Hopenhayn (1992) model.

    Parameters
    ----------
    n_z : int
        Number of grid points for the productivity process.
        Values below 51 produce coarse exit threshold quantization;
        101 (default) is recommended for accurate results.
    rho_z : float
        AR(1) persistence of log productivity.
    sigma_z : float
        Standard deviation of productivity innovations.
    theta : float
        Span of control (DRS exponent), in (0, 1).
    phi : float
        Per-period fixed operating cost.
    kappa : float
        Sunk entry cost.
    beta : float
        Discount factor.
    G : (n_z,) array or None
        Entrant distribution. If None, uses the ergodic distribution of Pi_z.
    labor_supply : str
        'inelastic' (N^s fixed), 'elastic' (u(C) - aN, employment lotteries),
        or 'separable' (C^{1-sigma}/(1-sigma) - N^{1+nu}/(1+nu)).
    N_supply : float
        Fixed labor supply (only used if labor_supply='inelastic').
    a_disutility : float
        Disutility weight (elastic: coefficient a in u(C) - aN).
    gamma_hh : float
        CRRA on consumption (elastic: gamma in w = a*C^gamma;
        separable: sigma in C^{-sigma}*w = N^nu).
    nu_labor : float
        Labor supply curvature (separable only): exponent nu in N^{1+nu}/(1+nu).
        Frisch elasticity = 1/nu.
    w_bounds : (float, float)
        Bracket for Brent's method over the wage.
    tol : float
        Tolerance for free entry condition.
    verbose : bool
        Print iteration info.

    Returns
    -------
    HopenhaynSteadyState
        Dataclass with all equilibrium objects.
    """
    t0 = time.time()

    # ── 0. Input validation ───────────────────────────────────────────
    if not 0 < theta < 1:
        raise ValueError(f"theta must be in (0, 1), got {theta}")
    if not 0 < beta < 1:
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    if w_bounds[0] <= 0:
        raise ValueError(f"w_bounds[0] must be positive, got {w_bounds[0]}")

    # ── 1. Discretize productivity ────────────────────────────────────
    log_z, Pi_z = rouwenhorst(n_z, rho_z, sigma_z)
    z_grid = np.exp(log_z)

    if G is None:
        G = stationary_distribution(Pi_z)

    r = 1.0 / beta - 1.0

    # ── 2. Define free-entry residual f(w) ────────────────────────────
    n_eval = [0]

    def free_entry_residual(w: float) -> float:
        profit = hopenhayn_profit(z_grid, w, theta, phi)
        V, z_star_idx = _solve_firm_problem(Pi_z, profit, beta)
        Ve = hopenhayn_entry_value(V, G, kappa, beta)
        n_eval[0] += 1
        if verbose:
            print(f"  eval {n_eval[0]:3d}: w={w:.6f}, z*_idx={z_star_idx:3d}, V^e={Ve:+.6e}")
        return Ve

    # ── 3. Bracket check ──────────────────────────────────────────────
    Ve_low = free_entry_residual(w_bounds[0])
    Ve_high = free_entry_residual(w_bounds[1])
    if Ve_low < 0:
        raise ValueError(
            f"V^e({w_bounds[0]:.4f}) = {Ve_low:.4e} < 0. "
            "Lower wage bound too high; lower w_bounds[0]."
        )
    if Ve_high > 0:
        raise ValueError(
            f"V^e({w_bounds[1]:.4f}) = {Ve_high:.4e} > 0. "
            "Upper wage bound too low; raise w_bounds[1]."
        )

    # ── 4. Root-find: V^e(w*) = 0 ────────────────────────────────────
    w_star = brentq(free_entry_residual, w_bounds[0], w_bounds[1], tol=tol)

    # ── 5. Solve at converged wage ────────────────────────────────────
    profit = hopenhayn_profit(z_grid, w_star, theta, phi)
    n_vec = hopenhayn_labor_demand(z_grid, w_star, theta)
    y_vec = hopenhayn_output(z_grid, n_vec, theta)
    V, z_star_idx = _solve_firm_problem(Pi_z, profit, beta)
    cont_val = beta * Pi_z @ V

    # Sub-Markov kernel
    Q = Pi_z.copy()
    Q[:z_star_idx, :] = 0.0

    # ── 6. Stationary distribution ────────────────────────────────────
    if z_star_idx == 0:
        # No exit: Q = Pi_z, so (I - Q.T) is singular. Use the ergodic
        # distribution as lambda_hat. Since no firm exits, M -> 0 in
        # equilibrium and lambda_star is proportional to the ergodic.
        import warnings
        warnings.warn(
            "No firm exits (z_star_idx=0, likely phi=0). "
            "lambda_hat set to ergodic distribution.",
            stacklevel=2,
        )
        lambda_hat = stationary_distribution(Pi_z)
    else:
        lambda_hat = hopenhayn_stationary_distribution(Q, G)
    agg = hopenhayn_aggregates(z_grid, lambda_hat, n_vec, y_vec, profit, z_star_idx, phi, kappa)

    # ── 7. Determine M from household closure ─────────────────────────
    if labor_supply == "inelastic":
        M = N_supply / agg["N_hat"]
        N = N_supply
        C = M * agg["resource_hat"]
        Pi_agg = M * agg["Pi_hat"] - kappa * M

    elif labor_supply == "elastic":
        # FOC: w = a * C^gamma_hh  =>  C = (w/a)^{1/gamma_hh}
        if gamma_hh == 1.0:
            C = w_star / a_disutility
        else:
            C = (w_star / a_disutility) ** (1.0 / gamma_hh)
        # Resource constraint: C = M * resource_hat
        if agg["resource_hat"] <= 0:
            raise ValueError(
                f"Per-unit-M net output = {agg['resource_hat']:.4e} <= 0. "
                "Cannot sustain positive consumption. Check calibration."
            )
        M = C / agg["resource_hat"]
        N = M * agg["N_hat"]
        Pi_agg = M * agg["Pi_hat"] - kappa * M

    elif labor_supply == "separable":
        # u(C,N) = C^{1-sigma}/(1-sigma) - N^{1+nu}/(1+nu)
        # FOC: C^{-sigma} * w = N^nu
        # With N = M*N_hat, C = M*R:  N^{nu+sigma} = w * (N_hat/R)^sigma
        sigma_c = gamma_hh
        nu_n = nu_labor
        R = agg["resource_hat"]
        if R <= 0:
            raise ValueError(
                f"Per-unit-M net output = {R:.4e} <= 0. "
                "Cannot sustain positive consumption. Check calibration."
            )
        N = (w_star * (agg["N_hat"] / R) ** sigma_c) ** (1.0 / (nu_n + sigma_c))
        M = N / agg["N_hat"]
        C = M * R
        Pi_agg = M * agg["Pi_hat"] - kappa * M

    else:
        raise ValueError(
            f"labor_supply must be 'inelastic', 'elastic', or 'separable', "
            f"got '{labor_supply}'"
        )

    Y = M * agg["Y_hat"]
    lambda_star = M * lambda_hat
    n_active = M * agg["Phi_hat"]

    # ── 8. Walras' law check ──────────────────────────────────────────
    goods_supply = Y - phi * n_active - kappa * M
    walras_residual = abs(C - goods_supply)

    elapsed = time.time() - t0

    # ── 9. Pack results ───────────────────────────────────────────────
    params = dict(
        n_z=n_z, rho_z=rho_z, sigma_z=sigma_z, theta=theta,
        phi=phi, kappa=kappa, beta=beta,
        a_disutility=a_disutility, gamma_hh=gamma_hh, nu_labor=nu_labor,
        N_supply=N_supply,
    )

    ss = HopenhaynSteadyState(
        w=w_star, r=r,
        V=V, z_star=z_grid[z_star_idx] if z_star_idx < n_z else np.inf,
        z_star_idx=z_star_idx,
        profit=profit, labor_demand=n_vec, output=y_vec,
        cont_value=cont_val,
        lambda_hat=lambda_hat, lambda_star=lambda_star, M=M,
        Y=Y, N=N, C=C, Pi_agg=Pi_agg,
        exit_rate=agg["exit_rate"], n_active=n_active,
        z_grid=z_grid, Pi_z=Pi_z, Q=Q, G=G,
        params=params, labor_supply=labor_supply,
        n_outer_iter=n_eval[0], walras_residual=walras_residual,
        elapsed=elapsed,
    )
    return ss
