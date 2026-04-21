"""Hopenhayn-Rogerson (1993) model: firm dynamics with firing taxes.

Extends the vanilla Hopenhayn (1992) model in `hopenhayn.py` by adding a
per-worker firing cost that applies whenever employment contracts:

    g(n_prev, n) = tau * max(0, n_prev - n)

This introduces a second state variable, lagged employment `n_prev`, and the
value function can no longer be solved by matrix inversion. Instead, we use
value function iteration with Howard improvement on the 2D grid (n_prev, z).

Entrants draw z from G and enter with n_prev = 0, so they face no firing
cost in their first period. Exit entails a lump-sum cost of tau * n_prev.

The GE structure (free entry, three household closures, stationary
distribution) is preserved from `hopenhayn.py`. The firing tax revenue is
rebated to the household as a lump-sum transfer.

References
----------
Hopenhayn and Rogerson (1993). "Job Turnover and Policy Evaluation: A General
    Equilibrium Analysis." JPE 101(5), 915-938.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..markov import rouwenhorst, stationary_distribution
from ..optimize import brentq


# ── Grid construction ─────────────────────────────────────────────────────


def build_employment_grid(
    n_min: float, n_max: float, n_n: int, spacing: str = "log",
) -> np.ndarray:
    """Construct an employment grid.

    Parameters
    ----------
    n_min, n_max : float
        Grid endpoints. `n_min` should be strictly positive.
    n_n : int
        Number of grid points.
    spacing : {'log', 'linear'}
        Grid spacing. 'log' is recommended since employment is right-skewed
        in equilibrium.

    Returns
    -------
    n_grid : (n_n,) array
        Employment grid including `n_min` and `n_max` as endpoints.
    """
    if n_min <= 0:
        raise ValueError(f"n_min must be positive, got {n_min}")
    if n_max <= n_min:
        raise ValueError(f"n_max must exceed n_min, got {n_max} <= {n_min}")
    if n_n < 3:
        raise ValueError(f"n_n must be at least 3, got {n_n}")

    if spacing == "log":
        return np.exp(np.linspace(np.log(n_min), np.log(n_max), n_n))
    elif spacing == "linear":
        return np.linspace(n_min, n_max, n_n)
    else:
        raise ValueError(f"spacing must be 'log' or 'linear', got '{spacing}'")


def frictionless_labor_demand(z: float, w: float, theta: float) -> float:
    """Static FOC labor demand: n* = (z*theta/w)^(1/(1-theta))."""
    return (z * theta / w) ** (1.0 / (1.0 - theta))


# ── Flow payoff ───────────────────────────────────────────────────────────


def flow_payoff(
    n_prev: np.ndarray,
    n_next: np.ndarray,
    z: np.ndarray,
    w: float,
    theta: float,
    phi: float,
    tau: float,
) -> np.ndarray:
    """Firm flow payoff for a continuing firm.

    flow(n_prev, n, z) = z * n^theta - w*n - phi - tau * max(0, n_prev - n)

    Arguments are broadcast against each other, so pass with appropriate
    shapes to build a 3D tensor over (n_prev, n_next, z).
    """
    output = z * n_next ** theta
    wage_bill = w * n_next
    separations = tau * np.maximum(0.0, n_prev - n_next)
    return output - wage_bill - phi - separations


# ── Value function iteration ──────────────────────────────────────────────


def _build_flow_tensor(
    n_grid: np.ndarray,
    z_grid: np.ndarray,
    w: float,
    theta: float,
    phi: float,
    tau: float,
) -> np.ndarray:
    """Build the (n_n, n_n, n_z) tensor of flow payoffs.

    F[i, k, j] = flow(n_grid[i], n_grid[k], z_grid[j], w, theta, phi, tau)

    The static part (z*n^theta - w*n - phi) depends on (k, j); the firing
    cost -tau * max(0, n_prev - n) depends on (i, k). We build each and sum.
    """
    n_n = len(n_grid)
    n_z = len(z_grid)

    # Static part: shape (n_n, n_z) indexed by (k, j)
    # z_grid[j] * n_grid[k]**theta
    static = z_grid[None, :] * (n_grid ** theta)[:, None] - w * n_grid[:, None] - phi

    # Firing cost: shape (n_n, n_n) indexed by (i, k)
    # -tau * max(0, n_grid[i] - n_grid[k])
    firing = -tau * np.maximum(0.0, n_grid[:, None] - n_grid[None, :])

    # Broadcast to (n_n, n_n, n_z): static expanded along i, firing along j
    F = static[None, :, :] + firing[:, :, None]
    return F


def _vfi_step(
    V: np.ndarray,
    flow_tensor: np.ndarray,
    Pi_z: np.ndarray,
    n_grid: np.ndarray,
    beta: float,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One VFI Bellman update.

    Uses the Hopenhayn (1992) timing convention adapted to HR1993:
    the firm always operates this period, then decides at the end of
    the period (based on expected continuation) whether to exit next
    period. This makes the model reduce exactly to Hopenhayn (1992) at
    tau = 0.

        V[i, j] = max_k { F[i, k, j] + beta * max(EV[k, j], -tau * n_k) }

    where EV[k, j] = E[V(n_k, z') | z_j] is the expected continuation
    value of carrying n_k workers into next period, and -tau * n_k is
    the value of exiting at the start of next period (firing all
    workers). Since the expectation is taken before the max, exit is a
    deterministic decision based on (n, z) — i.e., the firm commits to
    continue-or-exit at the end of the current period without seeing z'.

    Parameters
    ----------
    V : (n_n, n_z) array
        Current value function guess.
    flow_tensor : (n_n, n_n, n_z) array
        F[i, k, j] = flow payoff for (n_prev=n_i, n=n_k, z=z_j).
    Pi_z : (n_z, n_z) array
        Productivity transition matrix, Pi_z[j, j'] = P(z'=z_j' | z=z_j).
    n_grid : (n_n,) array
    beta : float
    tau : float

    Returns
    -------
    V_new : (n_n, n_z) array
    n_policy : (n_n, n_z) int array
        Grid index of chosen next-period employment.
    exit_mask : (n_n, n_z) bool array
        True if the firm at state (n_{-1}, z) exits at end of period
        (i.e., if at the chosen n, expected continuation is below -tau*n).
    """
    # EV[k, j] = E[V(n_k, z') | z_j]
    EV = V @ Pi_z.T  # (n_n, n_z)

    # Next-period exit value: -tau * n_k  (independent of j)
    exit_next = -tau * n_grid[:, None]  # (n_n, 1)

    # Effective continuation: max(EV, -tau * n_k)
    cont = np.maximum(EV, exit_next)  # (n_n, n_z)

    # Bellman value candidates: W[i, k, j] = F[i, k, j] + beta * cont[k, j]
    W = flow_tensor + beta * cont[None, :, :]  # (n_n, n_n, n_z)

    # Maximize over k
    n_policy = np.argmax(W, axis=1)  # (n_n, n_z)
    V_new = np.take_along_axis(W, n_policy[:, None, :], axis=1).squeeze(1)

    # Exit mask: did the chosen k trigger next-period exit?
    # exit if cont[k_chosen, j] == exit_next[k_chosen] (i.e., EV was below -tau*n_k)
    j_idx = np.arange(V.shape[1])[None, :]
    EV_at_k = EV[n_policy, j_idx]
    exit_next_at_k = -tau * n_grid[n_policy]
    exit_mask = EV_at_k <= exit_next_at_k

    return V_new, n_policy, exit_mask


def _policy_evaluation_step(
    V: np.ndarray,
    flow_tensor: np.ndarray,
    Pi_z: np.ndarray,
    n_policy: np.ndarray,
    exit_mask: np.ndarray,
    n_grid: np.ndarray,
    beta: float,
    tau: float,
) -> np.ndarray:
    """One Bellman update with the policy held fixed (Howard improvement).

    Matches the timing of `_vfi_step`: firm operates this period, then
    the effective continuation is max(EV, -tau*n_k) at the chosen k.
    """
    EV = V @ Pi_z.T  # (n_n, n_z)

    n_n, n_z = V.shape
    i_idx = np.arange(n_n)[:, None]
    j_idx = np.arange(n_z)[None, :]
    k_idx = n_policy  # (n_n, n_z)

    F_chosen = flow_tensor[i_idx, k_idx, j_idx]  # (n_n, n_z)
    EV_chosen = EV[k_idx, j_idx]
    exit_val_chosen = -tau * n_grid[k_idx]
    cont_chosen = np.where(exit_mask, exit_val_chosen, EV_chosen)
    V_new = F_chosen + beta * cont_chosen
    return V_new


def _solve_firm_problem_hr(
    Pi_z: np.ndarray,
    n_grid: np.ndarray,
    z_grid: np.ndarray,
    w: float,
    beta: float,
    theta: float,
    phi: float,
    tau: float,
    V_init: Optional[np.ndarray] = None,
    tol: float = 1e-7,
    max_iter: int = 2000,
    howard_freq: int = 25,
    howard_steps: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """VFI with Howard improvement on the 2D (n_prev, z) grid.

    Parameters
    ----------
    V_init : (n_n, n_z) array or None
        Warm start. If None, initialize to zero.
    tol : float
        Sup-norm tolerance on V updates.
    max_iter : int
        Maximum number of Bellman updates (counting policy evaluation steps).
    howard_freq : int
        Every `howard_freq` full Bellman maximizations, perform `howard_steps`
        policy evaluation updates with the policy held fixed.

    Returns
    -------
    V : (n_n, n_z) array
    n_policy : (n_n, n_z) int array
    exit_mask : (n_n, n_z) bool array
    n_iter : int
    """
    n_n = len(n_grid)
    n_z = len(z_grid)

    flow_tensor = _build_flow_tensor(n_grid, z_grid, w, theta, phi, tau)

    if V_init is None:
        V = np.zeros((n_n, n_z))
    else:
        V = V_init.copy()

    n_policy = np.zeros((n_n, n_z), dtype=np.int64)
    exit_mask = np.zeros((n_n, n_z), dtype=bool)

    for it in range(max_iter):
        V_new, n_policy, exit_mask = _vfi_step(
            V, flow_tensor, Pi_z, n_grid, beta, tau
        )
        diff = np.max(np.abs(V_new - V))
        V = V_new

        if diff < tol:
            return V, n_policy, exit_mask, it + 1

        # Howard improvement: policy evaluation sweep
        if (it + 1) % howard_freq == 0:
            for _ in range(howard_steps):
                V = _policy_evaluation_step(
                    V, flow_tensor, Pi_z, n_policy, exit_mask, n_grid, beta, tau
                )

    warnings.warn(
        f"VFI did not converge in {max_iter} iterations (last diff = {diff:.2e})",
        stacklevel=2,
    )
    return V, n_policy, exit_mask, max_iter


# ── Entry ─────────────────────────────────────────────────────────────────


def hopenhayn_rogerson_entry_value(
    V: np.ndarray,
    G: np.ndarray,
    kappa: float,
    beta: float,
    n0_idx: int = 0,
) -> float:
    """Entry value: V^e = -kappa + beta * sum_j G_j * V[n0_idx, j].

    Entrants arrive with n_prev = 0 (no firing cost), approximated on the
    grid by the smallest employment level n_grid[n0_idx].
    """
    return -kappa + beta * np.dot(G, V[n0_idx, :])


# ── Stationary distribution ───────────────────────────────────────────────


def stationary_distribution_hr(
    n_policy: np.ndarray,
    exit_mask: np.ndarray,
    Pi_z: np.ndarray,
    G: np.ndarray,
    n0_idx: int = 0,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> np.ndarray:
    """Normalized stationary distribution lambda_hat on the 2D grid.

    Law of motion: lambda[n', z'] = sum_{n, z: not exit} 1{n_pol(n,z)=n'} *
                                       Pi_z[z, z'] * lambda[n, z] + G[z'] * 1{n'=n0}.

    Forward iteration until convergence, then return lambda_hat (per unit M).

    Parameters
    ----------
    n_policy : (n_n, n_z) int array
        Optimal next-period employment grid index.
    exit_mask : (n_n, n_z) bool array
    Pi_z : (n_z, n_z) array
    G : (n_z,) array
        Entrant productivity distribution (probability vector).
    n0_idx : int
        Grid index of the entrant employment level (default: lowest grid point).

    Returns
    -------
    lambda_hat : (n_n, n_z) array
        Per-unit-M stationary distribution. Satisfies the fixed-point
        equation with one unit of entry flow.
    """
    n_n, n_z = n_policy.shape
    lam = np.zeros((n_n, n_z))
    lam[n0_idx, :] = G  # initialize with fresh entrants

    for it in range(max_iter):
        lam_new = np.zeros_like(lam)

        # Surviving firms: for each (i, j) with mass lam[i, j] and not exiting,
        # they move to (n_policy[i, j], z') with probability Pi_z[j, z'].
        # Vectorize: build the post-decision mass by employment bin.
        survivors = lam * (~exit_mask)  # (n_n, n_z)

        # Aggregate by destination employment n' (n_policy index)
        # mass_by_new_n[k', j] = sum over (i, j) with n_policy[i,j]=k' of survivors[i,j]
        # but z is not yet updated, so: mass_at_kj[k, j] = sum_{i: n_pol[i,j]=k} survivors[i, j]
        mass_at_kj = np.zeros((n_n, n_z))
        # np.add.at for scatter-add along axis 0 (employment)
        for j in range(n_z):
            np.add.at(mass_at_kj[:, j], n_policy[:, j], survivors[:, j])

        # Now transition z: lam_new[k, j'] = sum_j mass_at_kj[k, j] * Pi_z[j, j']
        lam_new = mass_at_kj @ Pi_z

        # Add entrants
        lam_new[n0_idx, :] += G

        diff = np.sum(np.abs(lam_new - lam))
        lam = lam_new
        if diff < tol:
            return lam

    warnings.warn(
        f"Stationary distribution did not converge in {max_iter} iterations "
        f"(last L1 diff = {diff:.2e})",
        stacklevel=2,
    )
    return lam


# ── Aggregation ───────────────────────────────────────────────────────────


def hopenhayn_rogerson_aggregates(
    n_grid: np.ndarray,
    z_grid: np.ndarray,
    lambda_hat: np.ndarray,
    n_policy: np.ndarray,
    exit_mask: np.ndarray,
    theta: float,
    phi: float,
    kappa: float,
    tau: float,
) -> Dict[str, float]:
    """Per-unit-M aggregates from the normalized distribution.

    Returns
    -------
    dict with keys:
      - N_hat: aggregate labor demand (using chosen n, not n_prev)
      - Y_hat: aggregate output
      - Pi_hat: aggregate flow profit (net of firing flow and phi)
      - Firing_hat: total firing cost flow (incumbents downsizing + exiters)
      - Phi_hat: mass of active firms
      - exit_rate: fraction of mass exiting per period
      - resource_hat: Y_hat - phi * Phi_hat - kappa - Firing_hat
                    = per-unit-M consumption (since the firing transfer
                      just reshuffles within the household budget)
    """
    active = ~exit_mask  # (n_n, n_z)
    active_mass = lambda_hat * active

    # Chosen n for each active firm
    n_chosen = n_grid[n_policy]  # (n_n, n_z)

    # Labor demand: sum active_mass * n_chosen
    N_hat = np.sum(active_mass * n_chosen)

    # Output: sum active_mass * z * n_chosen^theta
    y_vec = z_grid[None, :] * n_chosen ** theta  # (n_n, n_z)
    Y_hat = np.sum(active_mass * y_vec)

    # Active mass
    Phi_hat = np.sum(active_mass)

    # Firing cost flow
    # Incumbents that downsize: tau * max(0, n_prev - n_chosen) * active_mass
    n_prev_grid = n_grid[:, None]  # (n_n, 1)
    firing_incumbent = tau * np.maximum(0.0, n_prev_grid - n_chosen) * active_mass
    # Exiters: tau * n_prev * lambda_hat[exit]
    firing_exit = tau * n_prev_grid * lambda_hat * exit_mask
    Firing_hat = float(np.sum(firing_incumbent) + np.sum(firing_exit))

    # Flow profit (gross of the firing tax, which is a transfer)
    # Each active firm earns z*n^theta - w*n - phi - firing cost
    # For aggregation purposes we track gross profit before tax:
    # gross_profit = output - w*n - phi (the wage bill cancels out in C = wN + Pi + T)
    # Pi_hat computed later needs w. Here we return the pieces.
    Pi_hat_gross = Y_hat - phi * Phi_hat  # gross of wages, firing, entry

    # Exit rate (mass exiting / total mass)
    total_mass = float(np.sum(lambda_hat))
    exit_mass = float(np.sum(lambda_hat * exit_mask))
    exit_rate = exit_mass / total_mass if total_mass > 0 else 0.0

    # Per-unit-M resource for consumption:
    #   Y - phi*Phi - kappa
    # The firing cost is paid by firms to the government and rebated to the
    # household as a lump-sum transfer, so it nets out of the goods market.
    resource_hat = Y_hat - phi * Phi_hat - kappa

    return dict(
        N_hat=float(N_hat),
        Y_hat=float(Y_hat),
        Pi_hat_gross=float(Pi_hat_gross),
        Firing_hat=Firing_hat,
        Phi_hat=float(Phi_hat),
        exit_rate=exit_rate,
        resource_hat=float(resource_hat),
    )


# ── Result container ──────────────────────────────────────────────────────


@dataclass
class HopenhaynRogersonSteadyState:
    """All equilibrium objects from the HR1993 stationary equilibrium."""

    # Prices and taxes
    w: float
    r: float
    tau: float

    # Firm problem (2D)
    V: np.ndarray
    n_policy: np.ndarray        # grid indices
    n_policy_level: np.ndarray  # employment levels
    exit_mask: np.ndarray

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
    firing_total: float  # scaled
    transfer: float      # lump-sum transfer = firing_total

    # Grids and transitions
    n_grid: np.ndarray
    z_grid: np.ndarray
    Pi_z: np.ndarray
    G: np.ndarray

    # Metadata
    params: Dict[str, Any]
    labor_supply: str
    n_vfi_iter: int = 0
    n_outer_iter: int = 0
    walras_residual: float = 0.0
    elapsed: float = 0.0

    def summary(self) -> str:
        lines = [
            "Hopenhayn-Rogerson (1993) Stationary Equilibrium",
            f"  Labor supply: {self.labor_supply}",
            f"  tau (firing tax) = {self.tau:.4f}",
            f"  w = {self.w:.6f},  r = {self.r:.6f}",
            f"  M (entrants) = {self.M:.4f}",
            f"  Exit rate = {self.exit_rate:.4f}",
            f"  Active firms = {self.n_active:.4f}",
            f"  Y = {self.Y:.4f},  N = {self.N:.4f},  C = {self.C:.4f}",
            f"  Aggregate profits = {self.Pi_agg:.4f}",
            f"  Firing flow = {self.firing_total:.4f}  (transfer = {self.transfer:.4f})",
            f"  Walras residual = {self.walras_residual:.2e}",
            f"  VFI iterations (final) = {self.n_vfi_iter}",
            f"  Outer iterations = {self.n_outer_iter}",
            f"  Elapsed = {self.elapsed:.2f}s",
        ]
        return "\n".join(lines)


# ── Top-level solver ──────────────────────────────────────────────────────


def solve_hopenhayn_rogerson_ss(
    # Productivity process
    n_z: int = 25,
    rho_z: float = 0.90,
    sigma_z: float = 0.50,
    # Employment grid
    n_n: int = 101,
    n_min: float = 1e-3,
    n_max: Optional[float] = None,
    # Technology
    theta: float = 0.64,
    phi: float = 0.5,
    kappa: float = 5.0,
    tau: float = 0.0,
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
    tol: float = 1e-7,
    vfi_tol: float = 1e-7,
    vfi_max_iter: int = 2000,
    howard_freq: int = 25,
    howard_steps: int = 50,
    verbose: bool = False,
) -> HopenhaynRogersonSteadyState:
    """Solve the stationary equilibrium of Hopenhayn-Rogerson (1993).

    Parameters
    ----------
    n_z : int
        Number of grid points for productivity (Rouwenhorst).
    rho_z, sigma_z : float
        AR(1) parameters for log productivity.
    n_n : int
        Number of grid points for employment.
    n_min, n_max : float, optional
        Employment grid endpoints. If `n_max` is None, it is set to twice
        the frictionless optimum at `z_max` using the midpoint of `w_bounds`
        as a reference wage.

        Caveat: `n_max` is a numerical parameter that can silently bias the
        equilibrium if set too low — the policy will saturate at the grid's
        upper bound for high-productivity firms. The solver emits a warning
        when more than 1% of states hit the upper bound. If the warning
        fires, increase `n_max` (manually) and re-solve.
    theta : float
        Span of control, in (0, 1).
    phi : float
        Per-period fixed operating cost.
    kappa : float
        Sunk entry cost.
    tau : float
        Firing cost per worker laid off. Must be non-negative.
    beta : float
        Discount factor.
    G : (n_z,) array, optional
        Entrant productivity distribution. Defaults to ergodic distribution.
    labor_supply : {'inelastic', 'elastic', 'separable'}
        Household closure.
    N_supply : float
        Fixed labor supply (inelastic case).
    a_disutility : float
        Labor disutility weight (elastic / separable).
    gamma_hh : float
        CRRA on consumption (separable: sigma; elastic: exponent in w = a*C^gamma).
    nu_labor : float
        Labor curvature (separable only).
    w_bounds : (float, float)
        Bracket for Brent's method over the wage.
    tol : float
        Tolerance for the free entry condition.
    vfi_tol : float
        Tolerance for the VFI inner loop.
    vfi_max_iter : int
        Cap on VFI iterations.
    howard_freq : int
        Frequency of Howard improvement sweeps.
    howard_steps : int
        Number of policy-evaluation updates per Howard sweep.
    verbose : bool
        Print iteration info.

    Returns
    -------
    HopenhaynRogersonSteadyState
    """
    t0 = time.time()

    # ── 0. Input validation ───────────────────────────────────────────
    if not 0 < theta < 1:
        raise ValueError(f"theta must be in (0, 1), got {theta}")
    if not 0 < beta < 1:
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    if tau < 0:
        raise ValueError(
            f"tau (firing tax) must be non-negative, got {tau}. "
            "Firing subsidies are not supported."
        )
    if w_bounds[0] <= 0:
        raise ValueError(f"w_bounds[0] must be positive, got {w_bounds[0]}")
    if phi < 0:
        raise ValueError(f"phi must be non-negative, got {phi}")
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")

    # ── 1. Productivity discretization ────────────────────────────────
    log_z, Pi_z = rouwenhorst(n_z, rho_z, sigma_z)
    z_grid = np.exp(log_z)
    if G is None:
        G = stationary_distribution(Pi_z)
    r = 1.0 / beta - 1.0

    # ── 2. Employment grid ────────────────────────────────────────────
    if n_max is None:
        # Upper bound: frictionless optimum at z_max using the midpoint
        # of the wage bracket as a reference wage. Padded 2x for safety.
        # Using w_bounds[0] would give an absurd n_max for small lower bounds.
        w_ref = 0.5 * (w_bounds[0] + w_bounds[1])
        n_max = 2.0 * frictionless_labor_demand(z_grid[-1], w_ref, theta)
    n_grid = build_employment_grid(n_min, n_max, n_n, spacing="log")

    # Entrants arrive with n_prev = 0, which we map to the lowest grid point
    n0_idx = 0

    # ── 3. Free-entry residual with VFI warm start ────────────────────
    V_warm: Optional[np.ndarray] = None
    vfi_iter_last = [0]
    n_eval = [0]

    def free_entry_residual(w: float) -> float:
        nonlocal V_warm
        V, n_policy, exit_mask, n_iter = _solve_firm_problem_hr(
            Pi_z, n_grid, z_grid, w, beta, theta, phi, tau,
            V_init=V_warm, tol=vfi_tol, max_iter=vfi_max_iter,
            howard_freq=howard_freq, howard_steps=howard_steps,
        )
        V_warm = V
        vfi_iter_last[0] = n_iter
        Ve = hopenhayn_rogerson_entry_value(V, G, kappa, beta, n0_idx=n0_idx)
        n_eval[0] += 1
        if verbose:
            print(f"  eval {n_eval[0]:3d}: w={w:.6f}, VFI_iter={n_iter:4d}, V^e={Ve:+.6e}")
        return Ve

    # ── 4. Bracket check ──────────────────────────────────────────────
    Ve_low = free_entry_residual(w_bounds[0])
    Ve_high = free_entry_residual(w_bounds[1])
    if Ve_low < 0:
        raise ValueError(
            f"V^e({w_bounds[0]:.4f}) = {Ve_low:.4e} < 0. "
            "Lower w_bounds[0]."
        )
    if Ve_high > 0:
        raise ValueError(
            f"V^e({w_bounds[1]:.4f}) = {Ve_high:.4e} > 0. "
            "Raise w_bounds[1]."
        )

    # ── 5. Root-find ──────────────────────────────────────────────────
    w_star = brentq(free_entry_residual, w_bounds[0], w_bounds[1], tol=tol)

    # ── 6. Solve at converged wage ────────────────────────────────────
    V, n_policy, exit_mask, n_iter = _solve_firm_problem_hr(
        Pi_z, n_grid, z_grid, w_star, beta, theta, phi, tau,
        V_init=V_warm, tol=vfi_tol, max_iter=vfi_max_iter,
        howard_freq=howard_freq, howard_steps=howard_steps,
    )
    n_policy_level = n_grid[n_policy]

    # ── 7. Stationary distribution ────────────────────────────────────
    lambda_hat = stationary_distribution_hr(
        n_policy, exit_mask, Pi_z, G, n0_idx=n0_idx
    )

    # Warn if the employment grid's upper bound is binding on a meaningful
    # mass of firms at the equilibrium. For right-skewed productivity
    # distributions, the extreme high-z states saturate at any reasonable
    # n_max, but they carry negligible mass — so we measure binding by
    # mass-weighted saturation, not by raw grid-point count.
    top_idx = n_n - 1
    binding_mask = (n_policy == top_idx) & (~exit_mask)
    total_active = float((lambda_hat * (~exit_mask)).sum())
    if total_active > 0:
        mass_binding = float((lambda_hat * binding_mask).sum()) / total_active
        if mass_binding > 0.01:
            warnings.warn(
                f"Employment grid upper bound binds for {mass_binding:.1%} "
                f"of active firm mass (n_max = {n_grid[-1]:.2e}). "
                f"The equilibrium may be biased — consider increasing n_max.",
                stacklevel=2,
            )

    # ── 8. Per-unit-M aggregates ──────────────────────────────────────
    agg = hopenhayn_rogerson_aggregates(
        n_grid, z_grid, lambda_hat, n_policy, exit_mask,
        theta, phi, kappa, tau,
    )

    # ── 9. Determine M from household closure ────────────────────────
    if labor_supply == "inelastic":
        if agg["N_hat"] <= 0:
            raise RuntimeError(f"N_hat = {agg['N_hat']:.4e} <= 0, cannot solve.")
        M = N_supply / agg["N_hat"]
        N = N_supply
        C = M * agg["resource_hat"]

    elif labor_supply == "elastic":
        # w = a * C^gamma  =>  C = (w/a)^{1/gamma}
        if gamma_hh == 1.0:
            C = w_star / a_disutility
        else:
            C = (w_star / a_disutility) ** (1.0 / gamma_hh)
        if agg["resource_hat"] <= 0:
            raise ValueError(
                f"resource_hat = {agg['resource_hat']:.4e} <= 0 at w* = {w_star:.4f}. "
                "Check calibration."
            )
        M = C / agg["resource_hat"]
        N = M * agg["N_hat"]

    elif labor_supply == "separable":
        # u = C^{1-sigma}/(1-sigma) - N^{1+nu}/(1+nu)
        # FOC: C^{-sigma} * w = N^nu
        # With C = M*R, N = M*N_hat:
        #   N^{nu+sigma} = w * (N_hat/R)^sigma
        sigma_c = gamma_hh
        nu_n = nu_labor
        R = agg["resource_hat"]
        if R <= 0:
            raise ValueError(
                f"resource_hat = {R:.4e} <= 0 at w* = {w_star:.4f}. "
                "Check calibration."
            )
        N = (w_star * (agg["N_hat"] / R) ** sigma_c) ** (1.0 / (nu_n + sigma_c))
        M = N / agg["N_hat"]
        C = M * R

    else:
        raise ValueError(
            f"labor_supply must be 'inelastic', 'elastic', or 'separable', "
            f"got '{labor_supply}'"
        )

    # ── 10. Scale aggregates ─────────────────────────────────────────
    Y = M * agg["Y_hat"]
    firing_total = M * agg["Firing_hat"]
    transfer = firing_total
    n_active = M * agg["Phi_hat"]
    # Aggregate net firm profits (after wage bill, fixed costs, entry costs,
    # and firing costs). Equivalently: what's left for the household after
    # wages, before the firing-tax rebate transfer.
    Pi_agg = Y - w_star * N - phi * n_active - kappa * M - firing_total

    lambda_star = M * lambda_hat

    # ── 11. Walras check ─────────────────────────────────────────────
    # Household budget: C = w*N + Pi_agg + T
    # With T = firing_total (transferred back), this becomes:
    #   C = w*N + (Y - w*N - phi*n_active - kappa*M - firing_total) + firing_total
    #     = Y - phi*n_active - kappa*M
    # Goods market clearing: Y = C + phi*n_active + kappa*M  (firing costs net out)
    goods_supply = Y - phi * n_active - kappa * M
    walras_residual = abs(C - goods_supply)

    elapsed = time.time() - t0

    # ── 12. Package ──────────────────────────────────────────────────
    params = dict(
        n_z=n_z, rho_z=rho_z, sigma_z=sigma_z,
        n_n=n_n, n_min=n_min, n_max=n_max,
        theta=theta, phi=phi, kappa=kappa, tau=tau, beta=beta,
        a_disutility=a_disutility, gamma_hh=gamma_hh, nu_labor=nu_labor,
        N_supply=N_supply,
    )

    return HopenhaynRogersonSteadyState(
        w=w_star, r=r, tau=tau,
        V=V, n_policy=n_policy, n_policy_level=n_policy_level, exit_mask=exit_mask,
        lambda_hat=lambda_hat, lambda_star=lambda_star, M=M,
        Y=Y, N=N, C=C, Pi_agg=Pi_agg,
        exit_rate=agg["exit_rate"], n_active=n_active,
        firing_total=firing_total, transfer=transfer,
        n_grid=n_grid, z_grid=z_grid, Pi_z=Pi_z, G=G,
        params=params, labor_supply=labor_supply,
        n_vfi_iter=vfi_iter_last[0], n_outer_iter=n_eval[0],
        walras_residual=walras_residual, elapsed=elapsed,
    )
