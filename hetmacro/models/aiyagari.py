"""Aiyagari model using the modern Household + solver framework.

Provides:
- ``firm_k_demand`` / ``firm_wage``: Cobb-Douglas firm FOCs (labor = 1).
- ``capital_supply_curve``: Evaluate K_s(r) on a grid for plotting.
- ``solve_aiyagari_ge``: Find the GE steady state via Brent or bisection.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..household import Household
from ..income_process import IncomeProcess
from ..optimize import bisect, brentq


# ── Firm FOCs (Cobb-Douglas, L = 1) ─────────────────────────────────────

def firm_k_demand(r: float, alpha: float, delta: float) -> float:
    """Capital demand from MPK = r + delta with labor normalized to 1."""
    return (alpha / (r + delta)) ** (1.0 / (1.0 - alpha))


def firm_wage(K: float, alpha: float) -> float:
    """Wage from MPL with labor normalized to 1."""
    return (1.0 - alpha) * K**alpha


# ── Helper: solve household at given r ──────────────────────────────────

def _solve_household_at_r(
    r: float,
    solver,
    income_process: IncomeProcess,
    a_grid: np.ndarray,
    beta: float,
    gamma: float,
    alpha: float,
    delta: float,
    solver_kwargs: Optional[Dict] = None,
) -> Tuple[Household, float, float]:
    """Construct a Household at interest rate *r*, solve, and return aggregates.

    Returns ``(household, K_supply, K_demand)``.
    """
    K_d = firm_k_demand(r, alpha, delta)
    w = firm_wage(K_d, alpha)
    hh = Household(
        income_process=income_process,
        a_grid=a_grid,
        beta=beta,
        gamma=gamma,
        r=r,
        w=w,
    )
    kw = solver_kwargs or {}
    hh.solve(solver, **kw)
    hh.compute_ergodic()
    agg = hh.compute_aggregates()
    return hh, agg["A"], K_d


# ── Capital supply curve ────────────────────────────────────────────────

def capital_supply_curve(
    solver,
    income_process: IncomeProcess,
    a_grid: np.ndarray,
    beta: float,
    gamma: float,
    alpha: float = 0.36,
    delta: float = 0.08,
    r_values: Optional[np.ndarray] = None,
    n_points: int = 20,
    r_bounds: Tuple[float, float] = (-0.01, 0.009),
    verbose: bool = False,
    solver_kwargs: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    """Evaluate capital supply K_s(r) on a grid of interest rates.

    Parameters
    ----------
    solver : solver object
        Any solver with ``solve(household) -> SolvedPolicy``.
    income_process : IncomeProcess
        Discretized income process (z_grid should be normalized so E[z]=1).
    a_grid : ndarray
        Asset grid.
    beta, gamma : float
        Discount factor and CRRA coefficient.
    alpha, delta : float
        Capital share and depreciation rate.
    r_values : ndarray, optional
        Explicit interest rate grid.  If *None*, a uniform grid of
        *n_points* values in *r_bounds* is created.
    n_points : int
        Number of grid points when *r_values* is None.
    r_bounds : tuple
        Bracket for the interest rate grid.
    verbose : bool
        Print progress for each r evaluation.
    solver_kwargs : dict, optional
        Extra keyword arguments forwarded to ``solver.solve()``.

    Returns
    -------
    dict
        ``'r'``: interest rate grid,
        ``'K_supply'``: aggregate asset supply at each r,
        ``'K_demand'``: firm capital demand at each r.
    """
    if r_values is None:
        r_values = np.linspace(r_bounds[0], r_bounds[1], n_points)

    K_supply = np.empty_like(r_values)
    K_demand = np.empty_like(r_values)

    for i, r in enumerate(r_values):
        t0 = time.time()
        _, Ks, Kd = _solve_household_at_r(
            r, solver, income_process, a_grid, beta, gamma, alpha, delta,
            solver_kwargs=solver_kwargs,
        )
        K_supply[i] = Ks
        K_demand[i] = Kd
        if verbose:
            elapsed = time.time() - t0
            print(
                f"  [{i+1:3d}/{len(r_values)}]  r={r:.6f}  "
                f"K_s={Ks:.4f}  K_d={Kd:.4f}  "
                f"excess={Ks - Kd:.4e}  ({elapsed:.2f}s)"
            )

    return {"r": r_values, "K_supply": K_supply, "K_demand": K_demand}


# ── GE steady state solver ──────────────────────────────────────────────

def solve_aiyagari_ge(
    solver,
    income_process: IncomeProcess,
    a_grid: np.ndarray,
    beta: float,
    gamma: float,
    alpha: float = 0.36,
    delta: float = 0.08,
    r_bounds: Tuple[float, float] = (0.005, 0.04),
    tol_ge: float = 1e-6,
    method: str = "brentq",
    max_iter: int = 100,
    verbose: bool = False,
    solver_kwargs: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Solve the Aiyagari GE steady state.

    Parameters
    ----------
    solver : solver object
        Any solver with ``solve(household) -> SolvedPolicy``.
    income_process : IncomeProcess
        Discretized income process (z_grid should be normalized so E[z]=1).
    a_grid : ndarray
        Asset grid.
    beta, gamma : float
        Discount factor and CRRA coefficient.
    alpha, delta : float
        Capital share and depreciation rate.
    r_bounds : tuple
        Bracket ``(r_low, r_high)`` for the net interest rate.
    tol_ge : float
        Convergence tolerance for the root finder.
    method : str
        ``"brentq"`` (Brent, superlinear) or ``"bisection"`` (simple, linear).
    max_iter : int
        Maximum iterations (used by bisection; brentq uses SciPy default).
    verbose : bool
        Print iteration info: candidate r, K_s, K_d, excess, timing.
    solver_kwargs : dict, optional
        Extra keyword arguments forwarded to ``solver.solve()``.

    Returns
    -------
    dict
        Keys: ``r, K, w, Y, C, household, solved_policy, distribution,
        aggregates, solver_name, method, n_iterations, elapsed``.
    """
    cache: Dict[str, Any] = {}
    n_eval = [0]
    t_start = time.time()

    def excess_demand(r: float) -> float:
        n_eval[0] += 1
        t0 = time.time()
        hh, K_s, K_d = _solve_household_at_r(
            r, solver, income_process, a_grid, beta, gamma, alpha, delta,
            solver_kwargs=solver_kwargs,
        )
        excess = K_s - K_d
        dt = time.time() - t0
        if verbose:
            print(
                f"  iter {n_eval[0]:3d}  r={r:.8f}  "
                f"K_d={K_d:.4f}  K_s={K_s:.4f}  "
                f"excess={excess:+.4e}  ({dt:.2f}s)"
            )
        cache["hh"] = hh
        cache["r"] = r
        cache["K_d"] = K_d
        cache["w"] = firm_wage(K_d, alpha)
        return excess

    # Dispatch root finder
    if method == "brentq":
        r_star = brentq(excess_demand, r_bounds[0], r_bounds[1], tol=tol_ge)
    elif method == "bisection":
        r_star = bisect(
            excess_demand, r_bounds[0], r_bounds[1], tol=tol_ge, max_iter=max_iter,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'brentq' or 'bisection'.")

    total_time = time.time() - t_start

    # Ensure cache is at r_star (brentq may end at a nearby point)
    if abs(cache["r"] - r_star) > 1e-12:
        excess_demand(r_star)

    hh = cache["hh"]
    K_d = cache["K_d"]
    w = cache["w"]
    agg = hh.compute_aggregates()

    if verbose:
        print(
            f"\n  Converged: r*={r_star:.8f}  K*={K_d:.4f}  w*={w:.4f}  "
            f"Y*={K_d**alpha:.4f}  C*={agg['C']:.4f}"
        )
        rc = K_d**alpha - delta * K_d - agg["C"]
        print(f"  Resource constraint residual: {rc:.4e}")
        print(f"  Evaluations: {n_eval[0]}  Total time: {total_time:.2f}s")

    return {
        "r": r_star,
        "K": K_d,
        "w": w,
        "Y": K_d**alpha,
        "C": agg["C"],
        "household": hh,
        "solved_policy": hh.solved_policy,
        "distribution": hh.g,
        "aggregates": agg,
        "solver_name": type(solver).__name__,
        "method": method,
        "n_iterations": n_eval[0],
        "elapsed": total_time,
    }


# ── Transition path convenience wrapper ────────────────────────────────

def aiyagari_transition_path(
    shock_path: Dict[str, np.ndarray],
    Pi: np.ndarray,
    a_grid: np.ndarray,
    y: np.ndarray,
    beta: float,
    gamma: float,
    alpha: float = 0.36,
    delta: float = 0.08,
    ss_initial: Optional[Dict] = None,
    ss_terminal: Optional[Dict] = None,
    T: int = 300,
    method: str = "shooting",
    damping: float = 0.5,
    tol: float = 1e-6,
    max_iter: int = 200,
    backward_fn=None,
    verbose: bool = False,
    plot_progress: bool = False,
) -> "TransitionResult":
    """Compute transition dynamics in the Aiyagari model after an MIT shock.

    Parameters
    ----------
    shock_path : dict
        Time paths of shocks, e.g. ``{'Z': np.ndarray}`` for TFP.
        Each array must have length *T*.
    Pi : ndarray
        Markov transition matrix for income states.
    a_grid : ndarray
        Asset grid.
    y : ndarray
        Steady-state income grid (z_grid values such that E[z]=1).
    beta, gamma : float
        Discount factor and CRRA coefficient.
    alpha, delta : float
        Capital share and depreciation rate.
    ss_initial : dict, optional
        Pre-computed initial steady state from ``household_steady_state``.
        If *None*, computed internally from the steady-state prices.
    ss_terminal : dict, optional
        Pre-computed terminal steady state.  If *None*, same as
        *ss_initial* (mean-reverting shock).
    T : int
        Number of transition periods.
    method : str
        ``"shooting"`` or ``"root_finding"``.
    damping : float
        Damping parameter for shooting.
    tol, max_iter : float, int
        Convergence tolerance and iteration limit.
    backward_fn : callable, optional
        Custom backward step (see ``make_backward_fn``).
    verbose, plot_progress : bool
        Print iteration info / show live plots.

    Returns
    -------
    TransitionResult
    """
    from ..steady_state import household_steady_state
    from ..transition import TransitionResult, compute_transition_path

    eis = 1.0 / gamma

    # Compute initial SS if not provided
    # NOTE: y is the productivity grid (E[z]=1). Income = w * y.
    z_grid = y  # productivity levels (renamed for clarity)

    if ss_initial is None:
        from ..optimize import brentq

        def _excess(r):
            K_d = firm_k_demand(r, alpha, delta)
            w = firm_wage(K_d, alpha)
            y_income = w * z_grid
            ss = household_steady_state(Pi, a_grid, y_income, r, beta, eis)
            return ss["A"] - K_d

        r_upper = 1.0 / beta - 1.0 - 1e-4
        r_star = brentq(_excess, 0.001, r_upper, tol=1e-8)
        K_d = firm_k_demand(r_star, alpha, delta)
        w_star = firm_wage(K_d, alpha)
        ss_initial = household_steady_state(
            Pi, a_grid, w_star * z_grid, r_star, beta, eis,
        )
        if verbose:
            print(f"  Computed initial SS: r*={r_star:.6f}  K={ss_initial['A']:.4f}  "
                  f"w*={w_star:.4f}")

    if ss_terminal is None:
        ss_terminal = ss_initial

    # Build TFP path (default to 1 if not in shock_path)
    Z_path = shock_path.get("Z", np.ones(T))

    def price_fn(K, t):
        Z = float(Z_path[t])
        r = Z * alpha * K ** (alpha - 1.0) - delta
        w = Z * (1.0 - alpha) * K ** alpha
        Y = Z * K ** alpha  # L=1
        return {"r": r, "y": w * z_grid, "w": w, "Y": Y}

    result = compute_transition_path(
        ss_initial=ss_initial,
        ss_terminal=ss_terminal,
        price_fn=price_fn,
        T=T,
        method=method,
        damping=damping,
        tol=tol,
        max_iter=max_iter,
        backward_fn=backward_fn,
        verbose=verbose,
        plot_progress=plot_progress,
    )

    return result
