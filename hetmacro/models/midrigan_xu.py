"""Virgiliu Midrigan's simplified Midrigan-Xu (2014) finance-and-misallocation
model (Problem Set 10, Advanced Macro I).

Ports the MATLAB teaching code (``start.m``, ``findstatic.m``, ``solve_dynamic.m``,
``ergodic.m``, ``simulate.m``) to Python on top of the ``hetmacro`` primitives.

Model
-----
Continuum of entrepreneur-households with idiosyncratic ability ``z_it``
following ``log z' = rho_z log z + sigma_z eps'``. Each period the household
runs a span-of-control business with Cobb-Douglas production::

    y = z^(1-eta) * (k^alpha * l^(1-alpha))^eta,    eta < 1

subject to a collateral constraint ``k <= lambda * a`` where ``lambda = 1/(1-xi)``.
Static factor demands solve

    max_{k, l}  y - W l - R k   s.t.  k <= lambda * a,

with Lagrange multiplier ``mu >= 0`` on the collateral constraint and
``R = r + delta`` the user cost of capital.

Profits ``pi(a, z) = y - W l - R k`` depend on wealth ``a`` through the
binding constraint. The dynamic program is

    V(a, z) = max_{a'}  c^(1-theta)/(1-theta) + beta E[V(a', z') | z]
              s.t.  c + a' = pi(a, z) + (1 + r) a.

Closures
--------
- **Closed economy** (``solve_mx_closed``): wage, interest rate and user cost
  are updated from aggregate FOCs with labor supply ``L = 1`` and capital
  supply ``K = A`` (savings-assets market clearing).
- **Small open economy** (``solve_mx_open``): ``r`` is fixed exogenously;
  only ``W`` is iterated to clear the labor market at ``L_d = 1``.

Numerics
--------
- Productivity discretized via ``rouwenhorst`` (levels on ``z_grid``).
- Asset grid: power-law curvature 3 (matches ``start.m`` line 42).
- Dynamic program: VFI with Howard acceleration; continuous ``a'`` from
  vectorized golden-section on a cubic-spline interpolation of ``E V``.
- Stationary distribution: lottery-based power iteration (reused from
  ``forward.stationary_distribution``).
- Outer loop: damped Gauss-Seidel on prices, matching ``start.m`` line 72.

References
----------
Midrigan, V. "Heterogeneity in Macroeconomics," NYU PhD course, Lecture 3
"Finance and Misallocation"; Problem Set 10 starter code. Original paper:
Midrigan and Xu (2014), "Finance and Misallocation: Evidence from Plant-Level
Data," American Economic Review.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from ..forward import stationary_distribution
from .hopenhayn_rogerson_virgiliu import _vectorized_golden_section


# ── Section A. Static firm block (``findstatic.m``) ─────────────────────


def firm_static(
    a_grid: np.ndarray,
    z_grid: np.ndarray,
    W: float,
    R: float,
    lam: float,
    alpha: float,
    eta: float,
) -> Dict[str, np.ndarray]:
    """Vectorized static firm problem on the ``(a, z)`` grid.

    Computes capital, labor, output, profit and the collateral multiplier for
    every ``(a_i, z_j)`` pair, given factor prices ``(W, R)`` and the
    collateral tightness ``lam = 1/(1 - xi)``.

    The constrained multiplier follows the closed form on pset page 2::

        mu_virtual = alpha * eta^{1/(1 - eta(1-alpha))}
                     * (z / (lam a))^{(1 - eta) / (1 - eta(1-alpha))}
                     * (W / (1-alpha))^{-eta(1-alpha) / (1 - eta(1-alpha))}
                     - R
        mu = max(mu_virtual, 0)

    ``mu_virtual`` is the multiplier that would make ``k = lam a`` optimal
    under the unconstrained FOC. It is negative precisely when the constraint
    is slack, so clipping at zero gives the true multiplier.

    With ``mu`` in hand, output and factor demands follow from the FOCs::

        y = z * [ (1/eta) ((R + mu)/alpha)^alpha (W/(1-alpha))^{1-alpha} ]^{eta/(eta-1)}
        l = (1 - alpha) eta y / W
        k = alpha eta y / (R + mu)

    When ``mu = 0`` this returns the unconstrained frictionless choice; when
    ``mu > 0`` it returns exactly ``k = lam a`` by construction of the virtual
    multiplier.

    Parameters
    ----------
    a_grid : (na,) array
        Asset (net-worth) grid. Must be strictly positive for the formula to
        be well defined; at ``a=0`` the constraint forces all inputs to zero.
    z_grid : (nz,) array
        Productivity grid in levels (not logs).
    W, R : float
        Wage and user cost of capital.
    lam : float
        Collateral tightness ``1 / (1 - xi)``.
    alpha, eta : float
        Capital share and span-of-control curvature, with ``0 < alpha < 1``
        and ``0 < eta < 1``.

    Returns
    -------
    dict
        Keys ``y, k, l, mu, pi``, each a ``(na, nz)`` array indexed by
        ``(i_a, i_z)``.

    Notes
    -----
    Numerically stable when ``z / (lam * a)`` stays roughly in
    ``[1e-10, 1e10]``. Extreme calibrations or very small ``a`` may trigger
    intermediate overflow in the ``**`` chain; the default
    ``amin = 1e-2`` with Korean-scale z values is comfortable.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 < eta < 1.0:
        raise ValueError(f"eta must be in (0, 1), got {eta}")
    if lam <= 0.0:
        raise ValueError(f"lam = 1/(1-xi) must be positive, got {lam}")
    if W <= 0.0 or R <= 0.0:
        raise ValueError(f"W and R must be positive, got W={W}, R={R}")

    a_arr = np.asarray(a_grid, dtype=float)
    if (a_arr <= 0.0).any():
        raise ValueError(
            "a_grid entries must be strictly positive; at a=0 the collateral "
            "constraint forces all inputs to zero and 1/(lam*a) is undefined."
        )
    a = a_arr[:, None]                                    # (na, 1)
    z = np.asarray(z_grid, dtype=float)[None, :]          # (1, nz)

    denom = 1.0 - eta * (1.0 - alpha)                     # constant, positive
    exp_a = (1.0 - eta) / denom
    exp_w = -eta * (1.0 - alpha) / denom
    exp_eta = 1.0 / denom

    # Virtual multiplier: the value of mu that would make k = lam*a optimal.
    # Negative in the unconstrained region, positive in the constrained one.
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_virtual = (
            alpha
            * eta ** exp_eta
            * (z / (lam * a)) ** exp_a
            * (W / (1.0 - alpha)) ** exp_w
            - R
        )
    mu = np.maximum(mu_virtual, 0.0)                      # (na, nz)
    R_plus_mu = R + mu                                    # (na, nz)

    # Output from the envelope of the two FOCs (pset page 2).
    bracket = (
        (1.0 / eta)
        * (R_plus_mu / alpha) ** alpha
        * (W / (1.0 - alpha)) ** (1.0 - alpha)
    )
    y = z * bracket ** (eta / (eta - 1.0))                # (na, nz)
    l = (1.0 - alpha) * eta * y / W
    k = alpha * eta * y / R_plus_mu
    pi = y - W * l - R * k                                # user cost R, not R+mu

    return {"y": y, "k": k, "l": l, "mu": mu, "pi": pi}


def unconstrained_firm(
    z_grid: np.ndarray,
    W: float,
    R: float,
    alpha: float,
    eta: float,
) -> Dict[str, np.ndarray]:
    """Unconstrained (``mu = 0``) static firm choice per productivity.

    Useful as a reference for testing ``firm_static`` in the region
    ``lam * a >= k*(z, W, R)``.

    Returns
    -------
    dict
        Keys ``y, k, l, pi`` as ``(nz,)`` arrays; ``mu`` omitted (zero).
    """
    z = np.asarray(z_grid, dtype=float)
    bracket = (
        (1.0 / eta)
        * (R / alpha) ** alpha
        * (W / (1.0 - alpha)) ** (1.0 - alpha)
    )
    y = z * bracket ** (eta / (eta - 1.0))
    l = (1.0 - alpha) * eta * y / W
    k = alpha * eta * y / R
    pi = y - W * l - R * k
    return {"y": y, "k": k, "l": l, "pi": pi}


# ── Section B. Dynamic entrepreneur VFI (``solve_dynamic.m``) ───────────


def _u_crra(c: np.ndarray, theta: float, c_min: float = 1e-12) -> np.ndarray:
    """CRRA utility with safe clipping at ``c_min``.

    Values of ``c < c_min`` are replaced with a very negative utility so that
    the optimizer avoids them without producing NaNs. The clip is on the input
    ``c`` side (not ``u`` side) to keep the function continuous and
    monotonically increasing near zero.
    """
    c_safe = np.maximum(c, c_min)
    if theta == 1.0:
        u = np.log(c_safe)
    else:
        u = c_safe ** (1.0 - theta) / (1.0 - theta)
    # Penalize the clipped region so golden section steers away from infeasibility.
    return np.where(c >= c_min, u, u - 1e8 * (c_min - c))


def _build_EV_splines(
    V: np.ndarray, Pi: np.ndarray, a_grid: np.ndarray
) -> List[CubicSpline]:
    """Build one cubic spline of ``Vbar(a', z_j)`` per current productivity j.

    ``Vbar[i, j] = sum_{j'} Pi[j, j'] V[i, j']`` is the expected continuation
    value at next-period assets ``a' = a_grid[i]`` conditional on today's
    productivity being ``z_j``. The list contains one spline per column.
    """
    Vbar = V @ Pi.T                                       # (na, nz)
    return [
        CubicSpline(a_grid, Vbar[:, j], bc_type="not-a-knot", extrapolate=True)
        for j in range(Vbar.shape[1])
    ]


def _eval_EV(
    splines: List[CubicSpline],
    a_next: np.ndarray,
) -> np.ndarray:
    """Evaluate the per-column spline at ``a_next[:, j]`` for each j.

    ``a_next`` has shape ``(na, nz)``; returns the same shape. Each column j
    is evaluated on its own spline so the routine is O(na * nz) with tight
    inner loops in scipy.
    """
    out = np.empty_like(a_next)
    for j in range(a_next.shape[1]):
        out[:, j] = splines[j](a_next[:, j])
    return out


def _bellman_step(
    V: np.ndarray,
    coh: np.ndarray,
    Pi: np.ndarray,
    a_grid: np.ndarray,
    beta: float,
    theta: float,
    a_min: float,
    a_max: float,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """One full Bellman maximization via vectorized golden section over ``a'``.

    Builds expected-value splines from the incoming ``V``, then maximizes
    ``u(coh - a') + beta * EV(a', z)`` pointwise on the ``(a, z)`` grid.

    Returns
    -------
    V_new, a_policy : (na, nz) arrays
        Updated value function and continuous savings policy.
    """
    splines = _build_EV_splines(V, Pi, a_grid)

    lo = np.full_like(coh, a_min)
    hi = np.minimum(coh - eps, a_max)
    hi = np.maximum(hi, lo + eps)                          # guard infeasibility

    def objective(a_next: np.ndarray) -> np.ndarray:
        a_clip = np.clip(a_next, a_grid[0], a_grid[-1])
        c = coh - a_next
        return _u_crra(c, theta) + beta * _eval_EV(splines, a_clip)

    a_policy = _vectorized_golden_section(objective, lo, hi)
    V_new = objective(a_policy)
    return V_new, a_policy


def _howard_evaluate(
    V: np.ndarray,
    a_policy: np.ndarray,
    coh: np.ndarray,
    Pi: np.ndarray,
    a_grid: np.ndarray,
    beta: float,
    theta: float,
    n_steps: int,
) -> np.ndarray:
    """Policy evaluation for ``n_steps`` iterations at fixed policy.

    Given a policy ``a_policy[i, j]`` and starting ``V``, iterate

        V_{k+1}(a, z) = u(coh(a,z) - a_policy(a,z))
                        + beta * E_{z'|z}[V_k(a_policy(a,z), z')]

    using the same cubic-spline interpolation as the Bellman step. Accelerates
    convergence of the outer VFI substantially (~10x in practice).
    """
    c = coh - a_policy
    u_flow = _u_crra(c, theta)
    for _ in range(n_steps):
        splines = _build_EV_splines(V, Pi, a_grid)
        V = u_flow + beta * _eval_EV(splines, a_policy)
    return V


def entrepreneur_vfi(
    pi_mat: np.ndarray,
    a_grid: np.ndarray,
    z_grid: np.ndarray,
    Pi: np.ndarray,
    r: float,
    beta: float,
    theta: float,
    tol: float = 1e-6,
    max_iter: int = 2500,
    howard_steps: int = 50,
    V_init: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Solve the entrepreneur dynamic program by VFI with Howard acceleration.

    Parameters
    ----------
    pi_mat : (na, nz) array
        Static profits ``pi(a, z)`` from :func:`firm_static`.
    a_grid : (na,) array
        Asset grid (strictly positive).
    z_grid : (nz,) array
        Productivity grid (unused inside the DP but kept for output clarity).
    Pi : (nz, nz) array
        Row-stochastic transition matrix ``Pi[j, j'] = P(z' = z_{j'} | z = z_j)``.
    r, beta, theta : float
        Interest rate, discount factor, CRRA coefficient.
    tol, max_iter : float, int
        Outer VFI tolerance on ``max|V_new - V_old|`` and iteration cap.
    howard_steps : int
        Number of Howard policy-evaluation steps after each Bellman step.
        Use 0 to disable Howard and run plain VFI.
    V_init : (na, nz) array, optional
        Initial guess. If None, uses the present value of flow utility at
        ``c = coh - a_grid[0]`` as a warm start.

    Returns
    -------
    dict
        ``V`` : (na, nz) value function.
        ``a_policy`` : (na, nz) continuous savings policy.
        ``c_policy`` : (na, nz) consumption policy.
        ``n_iter`` : int, outer iterations used.
        ``final_resid`` : float, last ``max|V_new - V_old|``.
    """
    na = len(a_grid)
    nz = len(z_grid)
    if pi_mat.shape != (na, nz):
        raise ValueError(
            f"pi_mat shape {pi_mat.shape} does not match (na, nz) = ({na}, {nz})"
        )
    if Pi.shape != (nz, nz):
        raise ValueError(f"Pi shape {Pi.shape} does not match (nz, nz) = ({nz}, {nz})")

    a_min = float(a_grid[0])
    a_max = float(a_grid[-1])
    coh = pi_mat + (1.0 + r) * a_grid[:, None]            # (na, nz)

    if V_init is None:
        # Warm start: PV of flow utility at stay-put consumption.
        c0 = np.maximum(coh - a_min, 1e-6)
        V = _u_crra(c0, theta) / (1.0 - beta)
    else:
        V = np.array(V_init, dtype=float, copy=True)

    resid = np.inf
    for it in range(max_iter):
        V_new, a_policy = _bellman_step(V, coh, Pi, a_grid, beta, theta, a_min, a_max)
        if howard_steps > 0:
            V_new = _howard_evaluate(
                V_new, a_policy, coh, Pi, a_grid, beta, theta, howard_steps,
            )
        resid = float(np.max(np.abs(V_new - V)))
        V = V_new
        if verbose and (it % 20 == 0 or it < 5):
            print(f"  [vfi] iter {it:4d}  resid = {resid:.3e}")
        if resid < tol:
            break

    c_policy = coh - a_policy
    return {
        "V": V,
        "a_policy": a_policy,
        "c_policy": c_policy,
        "n_iter": it + 1,
        "final_resid": resid,
    }


# ── Section C. Stationary aggregates (``ergodic.m``) ────────────────────


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, q: float
) -> float:
    """Weighted quantile ``q in [0, 1]`` via cumulative distribution lookup.

    Linear interpolation between adjacent sorted values. ``weights`` must be
    nonnegative and sum to a positive number.
    """
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    order = np.argsort(v)
    v_s = v[order]
    w_s = w[order]
    w_sum = w_s.sum()
    if w_sum <= 0:
        return float("nan")
    cum = np.cumsum(w_s) / w_sum
    return float(np.interp(q, cum, v_s))


def aggregate_mx(
    sol: Dict[str, np.ndarray],
    fs: Dict[str, np.ndarray],
    a_grid: np.ndarray,
    z_grid: np.ndarray,
    Pi: np.ndarray,
    r: float,
    delta: float,
    alpha: float,
    eta: float,
    L_supply: float = 1.0,
) -> Dict[str, Any]:
    """Compute the stationary distribution and all aggregates required to
    update prices and report slide-style tables.

    The stationary distribution is found by lottery-based power iteration on
    the joint ``(z, a)`` grid (reused from :func:`hetmacro.forward.stationary_distribution`;
    that routine expects the ``(n_z, n_a)`` convention so we transpose at the
    interface).

    Parameters
    ----------
    sol : dict
        Output of :func:`entrepreneur_vfi`; uses ``a_policy`` and ``c_policy``
        (both ``(na, nz)``).
    fs : dict
        Output of :func:`firm_static` at the same ``(W, R)`` used for ``sol``.
    a_grid, z_grid : arrays
        Asset and productivity grids.
    Pi : (nz, nz) array
        Row-stochastic productivity transition.
    r, delta, alpha, eta : float
        Net interest rate, depreciation, capital share, span-of-control.
    L_supply : float, default 1.0
        Aggregate labor supply (closed economy L = 1, open economy also L=1
        in this pset — the distinction shows up only in price-updates outside
        this function).

    Returns
    -------
    dict with keys
        ``D`` (na, nz) stationary distribution on the ``(a, z)`` grid.
        ``A, K, L_d, Y, C, Pi_agg`` scalar aggregates.
        ``tau`` scalar k-weighted aggregate wedge.
        ``TFP, TFP_star`` measured and first-best TFP.
        ``tau_i`` (na, nz) firm-level wedge ``(R + mu)/R``.
        ``W_new, R_new, r_new`` aggregate-FOC price updates
        (closed-economy form with ``K = A, L = L_supply``).
        ``wedge_quantiles`` dict with ``q -> {unweighted, k_weighted, z_weighted}``.
    """
    # --- Stationary distribution (lottery-based) -----------------------
    a_policy = sol["a_policy"]
    D_zn = stationary_distribution(Pi, a_policy.T, a_grid)        # (nz, na)
    D = D_zn.T                                                    # (na, nz)

    # --- Basic aggregates ---------------------------------------------
    a_mat = np.broadcast_to(a_grid[:, None], D.shape)
    z_mat = np.broadcast_to(z_grid[None, :], D.shape)

    A = float(np.sum(D * a_mat))
    K = float(np.sum(D * fs["k"]))
    L_d = float(np.sum(D * fs["l"]))
    Y = float(np.sum(D * fs["y"]))
    C = float(np.sum(D * sol["c_policy"]))
    Pi_agg = float(np.sum(D * fs["pi"]))

    # --- Wedges and TFP (slides 15-16) --------------------------------
    R = r + delta
    tau_i = (R + fs["mu"]) / R                                    # (na, nz)
    tau = float(np.sum(D * tau_i * fs["k"]) / K) if K > 0 else float("nan")

    # TFP_t = (∫ z τ^{-αη/(1-η)})^{1-(1-α)η}  /  (∫ z τ^{-(1-(1-α)η)/(1-η)})^{αη}
    exp_num = -alpha * eta / (1.0 - eta)
    exp_den = -(1.0 - (1.0 - alpha) * eta) / (1.0 - eta)
    I_num = float(np.sum(D * z_mat * tau_i ** exp_num))
    I_den = float(np.sum(D * z_mat * tau_i ** exp_den))
    TFP = I_num ** (1.0 - (1.0 - alpha) * eta) / I_den ** (alpha * eta)
    # First-best: TFP* = (∫ z)^{1-η}
    E_z = float(np.sum(D * z_mat))
    TFP_star = E_z ** (1.0 - eta)

    # --- Aggregate-FOC price updates (slide 17, K = A, L = L_supply) --
    # Y = TFP · K^{αη} · L^{(1-α)η}, so
    #   W = (1-α)η · Y / L_supply
    #   R = αη · Y / (τ · K)
    # Implementing directly from TFP × aggregates avoids double-counting.
    K_implied = A                                                  # closed-economy clearing
    Y_agg = TFP * K_implied ** (alpha * eta) * L_supply ** ((1.0 - alpha) * eta)
    W_new = (1.0 - alpha) * eta * Y_agg / L_supply
    R_new = alpha * eta * Y_agg / (tau * K_implied) if tau > 0 else float("nan")
    r_new = R_new - delta

    # --- Wedge percentiles (slide 20) ---------------------------------
    # Convention matches Virgiliu's ``ergodic.m`` line 46-50:
    #   - ``unweighted``  := plain percentile over flattened tau_i (one
    #     entry per (a, z) grid cell, equal weight per cell), i.e.
    #     MATLAB's ``prctile(taui, q)``. This does NOT weight by the
    #     stationary mass D: cells the distribution rarely visits count
    #     the same as cells with most of the mass.
    #   - ``k_weighted`` := weights = D * k, i.e. ``wprctile(taui, q, F.*ki)``.
    #   - ``z_weighted`` := weights = D * z, i.e. ``wprctile(taui, q, F.*zi)``.
    # We also store ``mass_weighted`` (weights = D) for completeness since
    # it's a more conventional density-based summary, but slide-20 tables
    # use the grid-cell version.
    tau_flat = tau_i.ravel()
    wedge_quantiles: Dict[float, Dict[str, float]] = {}
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        wedge_quantiles[q] = {
            "unweighted":   float(np.percentile(tau_flat, q * 100)),
            "mass_weighted": _weighted_quantile(tau_i, D, q),
            "k_weighted":    _weighted_quantile(tau_i, D * fs["k"], q),
            "z_weighted":    _weighted_quantile(tau_i, D * z_mat, q),
        }

    return {
        "D": D,
        "A": A, "K": K, "L_d": L_d, "Y": Y, "C": C, "Pi_agg": Pi_agg,
        "tau": tau, "tau_i": tau_i,
        "TFP": TFP, "TFP_star": TFP_star,
        "W_new": W_new, "R_new": R_new, "r_new": r_new,
        "wedge_quantiles": wedge_quantiles,
    }


# ── Persistence helpers (pickle-based) ─────────────────────────────────


def save_steady_state(ss: "MXSteadyState", path) -> None:
    """Pickle an :class:`MXSteadyState` to ``path``.

    Useful to cache expensive solves (e.g. the closed-economy ``xi=0`` run
    at ``damping=0.5`` is ~10 minutes) and reuse the result for simulations
    or comparative plots without resolving. Pickle is convenient but not
    portable across major Python/NumPy versions — regenerate from source
    if you upgrade.
    """
    import pickle
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(ss, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_steady_state(path) -> "MXSteadyState":
    """Load an :class:`MXSteadyState` previously saved with
    :func:`save_steady_state`.
    """
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Section D. Closed-economy general equilibrium (``start.m``) ─────────


# Outer-loop residual threshold below which we trust the warm-started V
# from the previous iteration. Above it, we restart VFI from scratch so
# that a stale V doesn't anchor the Bellman iteration at the wrong level.
_WARM_START_RESID_THRESHOLD = 5e-2


@dataclass
class MXSteadyState:
    """Midrigan-Xu stationary equilibrium bundle."""

    W: float
    r: float
    R: float
    lam: float
    xi: float
    a_grid: np.ndarray
    z_grid: np.ndarray
    Pi: np.ndarray
    fs: Dict[str, np.ndarray]
    sol: Dict[str, np.ndarray]
    agg: Dict[str, Any]
    n_outer: int
    final_resid: float
    resid_trace: np.ndarray
    mode: str = "closed"


def _build_grids(
    na: int,
    nz: int,
    amin: float,
    amax: float,
    rho_z: float,
    sigma_z: float,
    a_curvature: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build asset and productivity grids (shared by closed and open solvers)."""
    from ..grids import make_asset_grid
    from ..markov import rouwenhorst

    a_grid = make_asset_grid(amin, amax, na, curvature=a_curvature)
    z_log, Pi = rouwenhorst(nz, rho_z, sigma_z)
    z_grid = np.exp(z_log)
    return a_grid, z_grid, Pi


def solve_mx_closed(
    *,
    beta: float,
    theta: float,
    alpha: float,
    eta: float,
    delta: float,
    rho_z: float,
    sigma_z: float,
    xi: float,
    na: int = 501,
    nz: int = 11,
    amin: float = 1e-2,
    amax: float = 2500.0,
    a_curvature: float = 3.0,
    W_init: float = 1.0,
    r_init: float = 0.02,
    damping: float = 0.5,
    tol: float = 1e-7,
    max_iter: int = 2500,
    vfi_tol: float = 1e-6,
    vfi_max_iter: int = 2500,
    howard_steps: int = 50,
    method: str = "damped",
    r_bracket: Tuple[float, float] = (-0.055, 0.05),
    r_root_tol: float = 1e-5,
    verbose: bool = False,
) -> MXSteadyState:
    """Closed-economy stationary equilibrium with ``K = A`` and ``L = 1``.

    Two solver modes via ``method``:

    - ``"damped"`` (default): outer loop is damped Gauss-Seidel on ``(W, r)``
      using the slide-17 aggregate FOCs at the implied ``K = A``. Matches
      ``start.m`` lines 57-80 (step = 1/2). Stable and simple, but converges
      slowly at tight constraints (ξ near 0) because the contraction rate
      approaches 1.
    - ``"brentq"``: scalar root-finding on r with residual
      ``K_demand(r) - A(r)``. Each outer evaluation runs a small open
      economy (only W iterates) at the trial r. Converges in ~8 evaluations
      even at ξ = 0; use this when ``"damped"`` stalls.

    Parameters
    ----------
    damping : float in (0, 1)
        Weight on the OLD guess; ``W_next = damping*W + (1-damping)*W_new``.
        ``damping=0.5`` reproduces Virgiliu's ``step = 1/2``. (Used only by
        ``"damped"`` method.)
    tol : float
        Outer-loop stop when ``max(|dW|, |dr|) < tol``. (``"damped"`` only.)
    method : {"damped", "brentq"}
        Which outer solver to use.
    r_bracket : (low, high)
        Bracket for Brent root-find. The residual ``K_d - A`` must change
        sign inside the bracket (checked at bracket endpoints). Ignored by
        ``"damped"``.
    r_root_tol : float
        Absolute tolerance on r for Brent's method.
    vfi_tol, vfi_max_iter, howard_steps : passed to :func:`entrepreneur_vfi`.
    """
    if xi < 0.0 or xi >= 1.0:
        raise ValueError(f"xi must lie in [0, 1), got {xi}")
    if method not in ("damped", "brentq"):
        raise ValueError(f"method must be 'damped' or 'brentq', got {method!r}")
    if method == "brentq":
        return _solve_mx_closed_brentq(
            beta=beta, theta=theta, alpha=alpha, eta=eta, delta=delta,
            rho_z=rho_z, sigma_z=sigma_z, xi=xi,
            na=na, nz=nz, amin=amin, amax=amax, a_curvature=a_curvature,
            W_init=W_init,
            r_bracket=r_bracket, r_root_tol=r_root_tol,
            vfi_tol=vfi_tol, vfi_max_iter=vfi_max_iter, howard_steps=howard_steps,
            verbose=verbose,
        )

    lam = 1.0 / (1.0 - xi)

    a_grid, z_grid, Pi = _build_grids(
        na, nz, amin, amax, rho_z, sigma_z, a_curvature,
    )

    W, r = float(W_init), float(r_init)
    V_warm: Optional[np.ndarray] = None
    resid_trace = []
    resid = float("inf")
    prev_resid = float("inf")
    it = -1
    last_fs: Dict[str, np.ndarray] = {}
    last_sol: Dict[str, np.ndarray] = {}
    last_agg: Dict[str, Any] = {}

    for it in range(max_iter):
        R_user = r + delta
        fs = firm_static(a_grid, z_grid, W, R_user, lam, alpha, eta)
        # Only warm-start V when the outer loop is settling; if the prices
        # just moved a lot, V_warm is a stale anchor that can stall VFI.
        use_warm = V_warm is not None and prev_resid < _WARM_START_RESID_THRESHOLD
        sol = entrepreneur_vfi(
            pi_mat=fs["pi"], a_grid=a_grid, z_grid=z_grid, Pi=Pi,
            r=r, beta=beta, theta=theta,
            tol=vfi_tol, max_iter=vfi_max_iter, howard_steps=howard_steps,
            V_init=V_warm if use_warm else None, verbose=False,
        )
        V_warm = sol["V"]
        agg = aggregate_mx(
            sol=sol, fs=fs, a_grid=a_grid, z_grid=z_grid, Pi=Pi,
            r=r, delta=delta, alpha=alpha, eta=eta, L_supply=1.0,
        )
        W_new = agg["W_new"]
        r_new = agg["r_new"]

        W_next = damping * W + (1.0 - damping) * W_new
        r_next = damping * r + (1.0 - damping) * r_new
        resid = max(abs(W_next - W), abs(r_next - r))
        resid_trace.append(resid)

        last_fs, last_sol, last_agg = fs, sol, agg
        W, r = W_next, r_next
        prev_resid = resid

        if verbose and (it < 5 or it % 25 == 0):
            print(
                f"  [ge] iter {it:4d}  W={W:.5f}  r={r:+.5f}  resid={resid:.2e}"
                f"  A={agg['A']:.3f}  K_d={agg['K']:.3f}  L_d={agg['L_d']:.4f}"
                f"  TFP_loss={100*(1 - agg['TFP']/agg['TFP_star']):.3f}%"
            )

        if resid < tol:
            break

    return MXSteadyState(
        W=W, r=r, R=r + delta, lam=lam, xi=xi,
        a_grid=a_grid, z_grid=z_grid, Pi=Pi,
        fs=last_fs, sol=last_sol, agg=last_agg,
        n_outer=it + 1, final_resid=resid,
        resid_trace=np.asarray(resid_trace),
        mode="closed",
    )


# ── Section E. Small open economy (r fixed, only W clears labor) ────────


def solve_mx_open(
    *,
    r: float,
    beta: float,
    theta: float,
    alpha: float,
    eta: float,
    delta: float,
    rho_z: float,
    sigma_z: float,
    xi: float,
    na: int = 501,
    nz: int = 11,
    amin: float = 1e-2,
    amax: float = 2500.0,
    a_curvature: float = 3.0,
    W_init: float = 1.0,
    damping: float = 0.7,
    tol: float = 1e-5,
    max_iter: int = 600,
    vfi_tol: float = 1e-6,
    vfi_max_iter: int = 2500,
    howard_steps: int = 50,
    verbose: bool = False,
) -> MXSteadyState:
    """Small open economy at fixed interest rate ``r``.

    The world capital market supplies any capital at user cost ``R = r + δ``,
    so the asset-demand / asset-supply identity ``K = A`` no longer binds;
    savings flow abroad whenever ``A > K_demand``. Only the labor market
    clears at the fixed aggregate supply ``L = 1``, giving the wage update

        W_new = (1-α) η · TFP · K_d^{αη}

    where ``K_d = ∫ k_i dD`` is aggregate firm capital demand at the current
    prices. This reuses :func:`aggregate_mx` for the firm-side distribution
    and overrides only the wage update relative to :func:`solve_mx_closed`.
    """
    if xi < 0.0 or xi >= 1.0:
        raise ValueError(f"xi must lie in [0, 1), got {xi}")
    lam = 1.0 / (1.0 - xi)
    a_grid, z_grid, Pi = _build_grids(
        na, nz, amin, amax, rho_z, sigma_z, a_curvature,
    )

    W = float(W_init)
    V_warm: Optional[np.ndarray] = None
    prev_resid = float("inf")
    resid_trace: list = []
    resid = float("inf")
    it = -1
    last_fs: Dict[str, np.ndarray] = {}
    last_sol: Dict[str, np.ndarray] = {}
    last_agg: Dict[str, Any] = {}

    R_user = r + delta
    for it in range(max_iter):
        fs = firm_static(a_grid, z_grid, W, R_user, lam, alpha, eta)
        use_warm = V_warm is not None and prev_resid < _WARM_START_RESID_THRESHOLD
        sol = entrepreneur_vfi(
            pi_mat=fs["pi"], a_grid=a_grid, z_grid=z_grid, Pi=Pi,
            r=r, beta=beta, theta=theta,
            tol=vfi_tol, max_iter=vfi_max_iter, howard_steps=howard_steps,
            V_init=V_warm if use_warm else None, verbose=False,
        )
        V_warm = sol["V"]
        agg = aggregate_mx(
            sol=sol, fs=fs, a_grid=a_grid, z_grid=z_grid, Pi=Pi,
            r=r, delta=delta, alpha=alpha, eta=eta, L_supply=1.0,
        )
        # Open economy: wage clears labor market at K = K_demand (not A).
        K_d = agg["K"]
        Y_open = agg["TFP"] * K_d ** (alpha * eta)     # L = 1
        W_new = (1.0 - alpha) * eta * Y_open

        W_next = damping * W + (1.0 - damping) * W_new
        resid = abs(W_next - W)
        resid_trace.append(resid)

        last_fs, last_sol, last_agg = fs, sol, agg
        W = W_next
        prev_resid = resid

        if verbose and (it < 5 or it % 25 == 0):
            tfp_loss = 100.0 * (1.0 - agg["TFP"] / agg["TFP_star"])
            print(
                f"  [soe] iter {it:4d}  W={W:.5f}  resid={resid:.2e}"
                f"  A={agg['A']:.3f}  K_d={agg['K']:.3f}  L_d={agg['L_d']:.4f}"
                f"  TFP_loss={tfp_loss:.3f}%"
            )

        if resid < tol:
            break

    return MXSteadyState(
        W=W, r=r, R=R_user, lam=lam, xi=xi,
        a_grid=a_grid, z_grid=z_grid, Pi=Pi,
        fs=last_fs, sol=last_sol, agg=last_agg,
        n_outer=it + 1, final_resid=resid,
        resid_trace=np.asarray(resid_trace),
        mode="open",
    )


def _solve_mx_closed_brentq(
    *,
    beta: float,
    theta: float,
    alpha: float,
    eta: float,
    delta: float,
    rho_z: float,
    sigma_z: float,
    xi: float,
    na: int,
    nz: int,
    amin: float,
    amax: float,
    a_curvature: float,
    W_init: float,
    r_bracket: Tuple[float, float],
    r_root_tol: float,
    vfi_tol: float,
    vfi_max_iter: int,
    howard_steps: int,
    verbose: bool,
) -> MXSteadyState:
    """Closed-economy GE via Brent root-finding on ``r`` with
    ``K_demand(r) − A(r)`` as the residual.

    Each evaluation runs :func:`solve_mx_open` at the trial ``r``; labor
    clears internally. At the root, capital clears too, so the resulting
    steady state is the closed-economy GE.
    """
    from ..optimize import brentq

    r_low, r_high = r_bracket
    if r_low >= r_high:
        raise ValueError(f"r_bracket must satisfy low < high, got {r_bracket}")
    if r_low <= -delta:
        raise ValueError(
            f"r_bracket low = {r_low} implies R = r+delta <= 0; "
            f"must have r > -delta (= {-delta})."
        )

    # Cache the last SOE result so we don't resolve at r_star.
    last = {"r": None, "ss": None}
    W_state = {"W": float(W_init)}  # warm start shared across evaluations

    def excess_capital(r: float) -> float:
        ss = solve_mx_open(
            xi=xi, r=float(r), W_init=W_state["W"],
            beta=beta, theta=theta, alpha=alpha, eta=eta, delta=delta,
            rho_z=rho_z, sigma_z=sigma_z,
            na=na, nz=nz, amin=amin, amax=amax, a_curvature=a_curvature,
            damping=0.7, tol=1e-6, max_iter=400,
            vfi_tol=vfi_tol, vfi_max_iter=vfi_max_iter, howard_steps=howard_steps,
            verbose=False,
        )
        W_state["W"] = ss.W            # warm start next trial
        last["r"] = float(r)
        last["ss"] = ss
        resid = ss.agg["K"] - ss.agg["A"]
        if verbose:
            print(
                f"  [brentq] r = {r:+.5f}  K_d = {ss.agg['K']:.4f}  "
                f"A = {ss.agg['A']:.4f}  K_d - A = {resid:+.4e}"
            )
        return resid

    # Bracket sanity check.
    f_low = excess_capital(r_low)
    f_high = excess_capital(r_high)
    if f_low * f_high > 0:
        raise RuntimeError(
            f"K_d - A does not change sign inside r_bracket = {r_bracket}: "
            f"f({r_low})={f_low:.3e}, f({r_high})={f_high:.3e}. "
            f"Widen the bracket."
        )

    r_star = brentq(excess_capital, r_low, r_high, tol=r_root_tol)

    if abs(last["r"] - r_star) > r_root_tol:
        # Final evaluation at r_star to populate caches.
        excess_capital(r_star)

    ss = last["ss"]
    ss.mode = "closed"
    ss.r = float(r_star)
    ss.R = r_star + delta
    # Reuse ss.resid_trace from the SOE run; final_resid is the |K_d - A|.
    ss.final_resid = abs(ss.agg["K"] - ss.agg["A"])
    return ss


# ── Section F. Monte-Carlo panel simulation (``simulate.m``) ────────────


def _sample_stationary_z(
    pi_z: np.ndarray, N: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``N`` initial z indices from the stationary distribution."""
    cum = np.cumsum(pi_z)
    u = rng.random(N)
    return np.searchsorted(cum, u).clip(0, len(pi_z) - 1)


def _markov_chain_step(
    z_prev: np.ndarray, Pi: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One-step Markov transition ``z_prev -> z_next`` using per-row CDFs."""
    cum = np.cumsum(Pi, axis=1)                   # (nz, nz)
    u = rng.random(z_prev.shape[0])
    # For each agent, inverse-CDF sample from Pi[z_prev, :].
    rows = cum[z_prev]                            # (N, nz)
    return (rows < u[:, None]).sum(axis=1).clip(0, Pi.shape[0] - 1)


def simulate_firms(
    ss: MXSteadyState,
    N: int = 100_000,
    T: int = 250,
    seed: Optional[int] = 0,
    burn: int = 50,
) -> Dict[str, np.ndarray]:
    """Simulate ``N`` entrepreneurs for ``T`` periods from the steady state.

    The productivity process is drawn from the Rouwenhorst-discretized Markov
    chain (indices, not continuous z), so static decisions are looked up
    column-by-column without z-interpolation. Asset dynamics use linear
    interpolation of ``a_policy`` in ``a``.

    Parameters
    ----------
    ss : MXSteadyState
        Converged steady state from :func:`solve_mx_closed` or
        :func:`solve_mx_open`.
    N, T : int
        Cross-section and time dimensions of the panel.
    seed : int, optional
        Numpy RNG seed.
    burn : int
        Number of initial periods discarded when returning moments; the
        full panel is returned separately.

    Returns
    -------
    dict with panels of shape ``(N, T)`` (after burn-in index truncation):
        ``a``, ``z`` (level, ``z_grid`` lookup), ``z_idx`` (grid index),
        ``k``, ``l``, ``y``, ``mu``, ``pi``, ``c``.
        Plus scalars ``std_log_y``, ``std_dlog_y``, ``corr_log_y[1..5]``.
    """
    rng = np.random.default_rng(seed)
    from ..forward import stationary_markov

    a_grid = ss.a_grid
    z_grid = ss.z_grid
    Pi = ss.Pi
    a_policy = ss.sol["a_policy"]               # (na, nz)
    c_policy = ss.sol["c_policy"]
    fs = ss.fs

    nz = len(z_grid)

    # Productivity panel via discrete Markov chain
    z_idx = np.empty((N, T), dtype=np.int32)
    pi_z = stationary_markov(Pi)
    z_idx[:, 0] = _sample_stationary_z(pi_z, N, rng)
    for t in range(1, T):
        z_idx[:, t] = _markov_chain_step(z_idx[:, t - 1], Pi, rng)

    # Asset / policy panels
    a_sim = np.empty((N, T))
    k_sim = np.empty((N, T))
    l_sim = np.empty((N, T))
    y_sim = np.empty((N, T))
    mu_sim = np.empty((N, T))
    pi_sim = np.empty((N, T))
    c_sim = np.empty((N, T))

    # Initialize at stationary distribution mean (weighted) for a
    a_mean = float(np.sum(
        stationary_markov(Pi)[None, :] * (a_grid[:, None] / len(a_grid)) * 0.0
    ))  # placeholder; actually start everyone at the within-z mean asset
    # Simpler: start at median of a_grid for all firms; burn-in removes transient.
    a_sim[:, 0] = a_grid[len(a_grid) // 2]

    for t in range(T):
        # Look up decisions at (a_sim[:, t], z_idx[:, t]) column by column.
        for j in range(nz):
            mask = z_idx[:, t] == j
            if not mask.any():
                continue
            a_vals = a_sim[mask, t]
            k_sim[mask, t] = np.interp(a_vals, a_grid, fs["k"][:, j])
            l_sim[mask, t] = np.interp(a_vals, a_grid, fs["l"][:, j])
            y_sim[mask, t] = np.interp(a_vals, a_grid, fs["y"][:, j])
            mu_sim[mask, t] = np.interp(a_vals, a_grid, fs["mu"][:, j])
            pi_sim[mask, t] = np.interp(a_vals, a_grid, fs["pi"][:, j])
            c_sim[mask, t] = np.interp(a_vals, a_grid, c_policy[:, j])
            if t + 1 < T:
                a_sim[mask, t + 1] = np.interp(
                    a_vals, a_grid, a_policy[:, j]
                )

    z_sim = z_grid[z_idx]

    # Moments on (post-burn, y > 0) slice
    y_post = y_sim[:, burn:]
    y_post = np.where(y_post > 0, y_post, np.nan)
    log_y = np.log(y_post)

    std_log_y = float(np.nanstd(log_y))
    d_log_y = np.diff(log_y, axis=1)
    std_d_log_y = float(np.nanstd(d_log_y))

    def _corr_at_lag(k: int) -> float:
        if k >= log_y.shape[1]:
            return float("nan")
        x = log_y[:, k:].ravel()
        y = log_y[:, :log_y.shape[1] - k].ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        return float(np.corrcoef(x, y)[0, 1])

    corrs = {k: _corr_at_lag(k) for k in (1, 3, 5)}

    return {
        "a": a_sim, "z": z_sim, "z_idx": z_idx,
        "k": k_sim, "l": l_sim, "y": y_sim,
        "mu": mu_sim, "pi": pi_sim, "c": c_sim,
        "std_log_y": std_log_y,
        "std_dlog_y": std_d_log_y,
        "corr_log_y": corrs,
        "burn": burn,
    }
