"""Transition path computation for heterogeneous agent models.

Implements the backward-forward shooting algorithm and a root-finding
alternative for computing perfect-foresight transition dynamics after
an MIT shock.  The core routines are model-agnostic: the user supplies
a ``price_fn(K, t)`` that maps aggregate capital and time to household
inputs (r, w, y).

Typical usage (Aiyagari model)::

    from hetmacro.transition import compute_transition_path
    from hetmacro.steady_state import household_steady_state

    ss = household_steady_state(Pi, a_grid, y, r_ss, beta, eis)

    def price_fn(K, t):
        Z = Z_path[t]
        r = Z * alpha * K**(alpha - 1) - delta
        w = Z * (1 - alpha) * K**alpha
        return {'r': r, 'y': w * z_grid, 'w': w}

    result = compute_transition_path(ss, ss, price_fn, T=500,
                                     method='shooting', damping=1/3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .backward import backward_egm
from .forward import forward_iteration
from .interpolation import get_lottery


# ── Result container ───────────────────────────────────────────────────


@dataclass
class TransitionResult:
    """Container for transition path output."""

    T: int
    K_path: np.ndarray
    r_path: np.ndarray
    w_path: np.ndarray
    C_path: np.ndarray
    policies: List[Tuple[np.ndarray, np.ndarray]]
    distributions: List[np.ndarray]
    converged: bool
    n_iterations: int
    final_error: float
    error_history: List[float] = field(default_factory=list)


# ── Backward / forward primitives ─────────────────────────────────────


def _backward_path(
    Va_terminal: np.ndarray,
    prices_path: List[Dict],
    Pi: np.ndarray,
    a_grid: np.ndarray,
    beta: float,
    eis: float,
    backward_fn: Optional[Callable] = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Backward iterate value/policy from *t = T-1* to *t = 0*.

    Returns
    -------
    policies : list of (a_pol, c_pol)
        One tuple per period, in forward-time order (index 0 = t=0).
    Va_0 : ndarray
        Marginal value of assets at t=0 (useful for diagnostics).
    """
    T = len(prices_path)
    policies: list = [None] * T
    Va = Va_terminal

    for t in range(T - 1, -1, -1):
        p = prices_path[t]
        if backward_fn is not None:
            Va, a_pol, c_pol = backward_fn(Va, p["r"], p["y"])
        else:
            Va, a_pol, c_pol = backward_egm(
                Va, Pi, a_grid, p["y"], p["r"], beta, eis,
            )
        policies[t] = (a_pol, c_pol)

    return policies, Va


def _forward_path(
    D_initial: np.ndarray,
    policies: List[Tuple[np.ndarray, np.ndarray]],
    Pi: np.ndarray,
    a_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Forward iterate distribution from *t = 0* to *t = T-1*.

    Returns
    -------
    K_path : ndarray, shape (T,)
        Aggregate capital at each date (mean assets in D_t).
    C_path : ndarray, shape (T,)
        Aggregate consumption at each date.
    distributions : list of ndarray
        Distribution at each date (length T).
    """
    T = len(policies)
    K_path = np.empty(T)
    C_path = np.empty(T)
    a_grid_2d = a_grid[np.newaxis, :]  # broadcast shape for D @ a_grid
    distributions: list = []

    D = D_initial.copy()
    for t in range(T):
        a_pol, c_pol = policies[t]
        # Capital at time t = mean asset holdings agents bring into period t
        K_path[t] = float(np.sum(D * a_grid_2d))
        C_path[t] = float(np.vdot(D, c_pol))
        distributions.append(D.copy())

        if t < T - 1:
            a_i, a_pi = get_lottery(a_pol, a_grid)
            D = forward_iteration(D, Pi, a_i, a_pi)

    return K_path, C_path, distributions


# ── Smoke tests / guardrails ──────────────────────────────────────────


def diagnose_transition(
    result: TransitionResult,
    ss_initial: Dict,
    ss_terminal: Dict,
    price_fn: Callable,
    alpha: float = 0.35,
    delta: float = 0.02,
) -> Dict[str, float]:
    """Run post-convergence smoke tests and print diagnostics.

    Returns dict of diagnostic values.
    """
    T = result.T
    K = result.K_path
    C = result.C_path

    diag: Dict[str, float] = {}

    # 1. Resource constraint residual: Y_t - C_t - A_t + (1-delta)*K_t = 0
    #    where A_t = aggregate savings at t (from policies, not distribution).
    Y_path = np.empty(T)
    savings = np.empty(T)
    a_grid = result.distributions[0].shape[1]  # infer from distribution shape
    for t in range(T):
        p = price_fn(K[t], t)
        if "Y" in p:
            Y_path[t] = p["Y"]
        elif "w" in p and "r" in p:
            Y_path[t] = p["w"] + (p["r"] + delta) * K[t]
        else:
            Y_path[t] = np.nan
        a_pol, _ = result.policies[t]
        savings[t] = float(np.vdot(result.distributions[t], a_pol))

    # Flow RC uses actual savings (avoids terminal boundary contamination)
    rc = Y_path - C - savings + (1.0 - delta) * K
    diag["max_resource_residual"] = float(np.max(np.abs(rc)))

    # 2. Distribution mass
    mass_errors = [abs(float(np.sum(D)) - 1.0) for D in result.distributions]
    diag["max_mass_error"] = max(mass_errors)

    # 3. Boundary check
    K_init_ss = ss_initial["A"]
    K_term_ss = ss_terminal["A"]
    diag["K_start_deviation"] = abs(K[0] - K_init_ss) / max(abs(K_init_ss), 1e-12)
    diag["K_end_deviation"] = abs(K[-1] - K_term_ss) / max(abs(K_term_ss), 1e-12)

    # 4. Policy feasibility
    min_c = min(float(np.min(c)) for _, c in result.policies)
    min_a = min(float(np.min(a)) for a, _ in result.policies)
    diag["min_consumption"] = min_c
    diag["min_assets"] = min_a

    # Print report
    print("\n  ── Transition diagnostics ──")
    print(f"  Resource constraint:  max|RC_t| = {diag['max_resource_residual']:.4e}")
    if diag["max_resource_residual"] > 1e-4:
        print("    WARNING: Resource constraint violated. Consider increasing T or decreasing tol.")
    print(f"  Distribution mass:    max|sum-1| = {diag['max_mass_error']:.4e}")
    if diag["max_mass_error"] > 1e-8:
        print("    WARNING: Distribution mass deviates from 1.")
    print(f"  K boundary:           start dev = {diag['K_start_deviation']:.4e}, "
          f"end dev = {diag['K_end_deviation']:.4e}")
    if diag["K_end_deviation"] > 0.01:
        print("    WARNING: K_path has not returned to terminal SS. Consider increasing T.")
    if min_c <= 0:
        print(f"    WARNING: Negative consumption detected (min c = {min_c:.4e}).")
    print()

    return diag


# ── Shooting method ───────────────────────────────────────────────────


def _shooting(
    ss_initial: Dict,
    ss_terminal: Dict,
    price_fn: Callable,
    T: int,
    damping: float,
    tol: float,
    max_iter: int,
    verbose: bool,
    backward_fn: Optional[Callable],
    plot_progress: bool,
    plot_iterations: bool = False,
) -> TransitionResult:
    """Transition path via shooting with damped iteration."""
    Pi = ss_initial["Pi"]
    a_grid = ss_initial["a_grid"]
    beta = ss_initial["beta"]
    eis = ss_initial["eis"]
    D_initial = ss_initial["D"]
    Va_terminal = ss_terminal["Va"]
    K_init = ss_initial["A"]
    K_term = ss_terminal["A"]

    # Buffer verbose output so it survives Jupyter clear_output
    _verbose_buf: list = []
    if verbose:
        _verbose_buf.append(
            f"  Initial SS: K={K_init:.4f}  r={ss_initial['r']:.6f}  "
            f"C={ss_initial['C']:.4f}")
        if ss_terminal is not ss_initial:
            _verbose_buf.append(
                f"  Terminal SS: K={K_term:.4f}  r={ss_terminal['r']:.6f}  "
                f"C={ss_terminal['C']:.4f}")
        else:
            _verbose_buf.append(
                f"  Terminal SS: K={K_term:.4f}  (same as initial, mean-reverting shock)")
        for _line in _verbose_buf:
            print(_line)

    # Initial guess: linear interpolation
    K_guess = np.linspace(K_init, K_term, T)

    # Set up live plotting if requested
    fig_live = None
    if plot_progress:
        try:
            import matplotlib.pyplot as plt
            plt.ion()
            fig_live, axs_live = plt.subplots(2, 2, figsize=(12, 8))
            fig_live.suptitle("Transition path convergence", fontsize=13)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show(block=False)
        except Exception:
            plot_progress = False

    # Set up live iteration overlay plot
    iter_plotter = None
    if plot_iterations:
        try:
            iter_plotter = _IterationPlotter(K_init, T)
        except Exception:
            plot_iterations = False

    error_history: list = []
    verbose_lines: list = list(_verbose_buf)  # buffer for verbose output
    policies = None
    distributions = None
    K_new = K_guess.copy()

    for it in range(max_iter):
        # Compute prices from capital path guess
        prices_path = [price_fn(K_guess[t], t) for t in range(T)]

        # Backward
        policies, _ = _backward_path(
            Va_terminal, prices_path, Pi, a_grid, beta, eis, backward_fn,
        )

        # Forward
        K_new, C_new, distributions = _forward_path(
            D_initial, policies, Pi, a_grid,
        )

        # Convergence check
        err = float(np.max(np.abs(K_new - K_guess)))
        mean_err = float(np.mean(np.abs(K_new - K_guess)))
        error_history.append(err)

        if verbose:
            line = (f"  Transition iter {it + 1:3d}: "
                    f"max|K_new - K_old| = {err:.4e}  "
                    f"mean|K_new - K_old| = {mean_err:.4e}")
            verbose_lines.append(line)
            if iter_plotter is None:
                print(line)

        # Live iteration overlay plot (after verbose line so it's included)
        if iter_plotter is not None:
            iter_plotter.add_iteration(K_guess, price_fn, it, max_iter,
                                       verbose_lines=verbose_lines)

        # Live convergence plot
        if plot_progress and fig_live is not None:
            _update_convergence_plot(
                fig_live, axs_live, K_guess, K_new, prices_path,
                error_history, K_init, it,
            )

        if err < tol:
            if verbose:
                conv_line = f"  Shooting converged at iteration {it + 1}."
                verbose_lines.append(conv_line)
                if iter_plotter is None:
                    print(conv_line)
            if iter_plotter is not None:
                iter_plotter.add_iteration(K_new, price_fn, it, max_iter,
                                           final=True,
                                           verbose_lines=verbose_lines)
                iter_plotter.finalize(verbose_lines=verbose_lines)
            # Final prices for result
            r_path = np.array([prices_path[t]["r"] for t in range(T)])
            w_path = np.array([prices_path[t]["w"] for t in range(T)])
            if plot_progress:
                try:
                    import matplotlib.pyplot as plt
                    plt.ioff()
                except Exception:
                    pass
            return TransitionResult(
                T=T, K_path=K_new, r_path=r_path, w_path=w_path,
                C_path=C_new, policies=policies, distributions=distributions,
                converged=True, n_iterations=it + 1, final_error=err,
                error_history=error_history,
            )

        # Damped update
        K_guess = damping * K_guess + (1.0 - damping) * K_new

        # Monotonicity warning
        if verbose and len(error_history) >= 4:
            last3 = error_history[-3:]
            if all(last3[i] > last3[i - 1] for i in range(1, 3)):
                warn = ("    WARNING: Error increasing for 3 consecutive "
                        "iterations. Consider reducing damping.")
                verbose_lines.append(warn)
                if iter_plotter is None:
                    print(warn)

    # Did not converge
    if verbose:
        stop_line = (f"  Shooting stopped at max_iter={max_iter}; "
                     f"last err = {err:.4e} (did not converge).")
        verbose_lines.append(stop_line)
        if iter_plotter is None:
            print(stop_line)

    if iter_plotter is not None:
        iter_plotter.finalize(verbose_lines=verbose_lines)

    prices_path = [price_fn(K_guess[t], t) for t in range(T)]
    r_path = np.array([prices_path[t]["r"] for t in range(T)])
    w_path = np.array([prices_path[t]["w"] for t in range(T)])
    if plot_progress:
        try:
            import matplotlib.pyplot as plt
            plt.ioff()
        except Exception:
            pass

    return TransitionResult(
        T=T, K_path=K_guess, r_path=r_path, w_path=w_path,
        C_path=C_new, policies=policies, distributions=distributions,
        converged=False, n_iterations=max_iter, final_error=err,
        error_history=error_history,
    )


def _update_convergence_plot(fig, axs, K_old, K_new, prices_path, error_history, K_ss, it):
    """Update the live convergence figure."""
    try:
        import matplotlib.pyplot as plt
        for ax in axs.flat:
            ax.clear()

        T = len(K_old)
        r_path = [prices_path[t]["r"] for t in range(T)]
        w_path = [prices_path[t]["w"] for t in range(T)]

        # Top-left: K path
        axs[0, 0].plot(K_old, "b-", alpha=0.5, label="guess")
        axs[0, 0].plot(K_new, "r-", alpha=0.8, label="implied")
        axs[0, 0].axhline(K_ss, color="k", linestyle=":", linewidth=0.8)
        axs[0, 0].set_title("Capital path")
        axs[0, 0].legend(fontsize=7)
        axs[0, 0].grid(alpha=0.25)

        # Top-right: prices
        axs[0, 1].plot(r_path, label="r")
        ax_tw = axs[0, 1].twinx()
        ax_tw.plot(w_path, color="C1", label="w")
        axs[0, 1].set_title("Prices (r left, w right)")
        axs[0, 1].grid(alpha=0.25)

        # Bottom-left: error profile
        axs[1, 0].plot(np.abs(np.array(K_new) - np.array(K_old)))
        axs[1, 0].set_title("|K_new - K_old| by period")
        axs[1, 0].set_yscale("log")
        axs[1, 0].grid(alpha=0.25)

        # Bottom-right: convergence trace
        axs[1, 1].plot(range(1, len(error_history) + 1), error_history, "o-", markersize=3)
        axs[1, 1].set_title("max error vs iteration")
        axs[1, 1].set_yscale("log")
        axs[1, 1].set_xlabel("iteration")
        axs[1, 1].grid(alpha=0.25)

        fig.suptitle(f"Transition path convergence (iter {it + 1})", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    except Exception:
        pass


class _IterationPlotter:
    """Live overlay plot of K/r/w paths across shooting iterations.

    Uses ``IPython.display`` for Jupyter notebooks (clear_output + display)
    and falls back to ``draw_idle`` for terminal/desktop backends.
    """

    def __init__(self, K_ss, T):
        import matplotlib.pyplot as plt
        self._plt = plt
        self._t_plot = T
        self._cmap = plt.get_cmap("viridis")
        self._n_lines = 0

        plt.ioff()  # we manage display ourselves
        self._fig, self._axs = plt.subplots(1, 3, figsize=(16, 5))
        self._axs[0].axhline(K_ss, color="k", linestyle=":", linewidth=0.8)
        self._axs[0].set_xlabel("t"); self._axs[0].set_ylabel("K")
        self._axs[0].set_title("Capital path by iteration")
        self._axs[0].grid(alpha=0.25)
        self._axs[1].set_xlabel("t"); self._axs[1].set_ylabel("r")
        self._axs[1].set_title("Interest rate by iteration")
        self._axs[1].grid(alpha=0.25)
        self._axs[2].set_xlabel("t"); self._axs[2].set_ylabel("w")
        self._axs[2].set_title("Wage by iteration")
        self._axs[2].grid(alpha=0.25)
        plt.tight_layout()

        # Detect Jupyter
        self._jupyter = False
        self._display_fn = None
        self._clear_fn = None
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                from IPython.display import display, clear_output
                self._display_fn = display
                self._clear_fn = clear_output
                self._jupyter = True
        except Exception:
            pass

        if not self._jupyter:
            plt.ion()
            plt.show(block=False)

    def add_iteration(self, K_path, price_fn, it, max_iter, final=False,
                       verbose_lines=None):
        """Add one iteration's paths and refresh the display.

        Parameters
        ----------
        verbose_lines : list of str, optional
            Buffered verbose output to re-print after clearing Jupyter output.
        """
        try:
            frac = 1.0 if final else min(self._n_lines / max(max_iter - 1, 1), 0.9)
            color = self._cmap(frac)
            lw = 2.0 if final else 0.8
            alpha = 1.0 if final else 0.6
            label = f"iter {it + 1} (final)" if final else f"iter {it + 1}"
            tp = self._t_plot

            # Compute prices once per t
            r_vals = np.empty(tp)
            w_vals = np.empty(tp)
            for t in range(tp):
                p = price_fn(K_path[t], t)
                r_vals[t] = p["r"]
                w_vals[t] = p["w"]

            self._axs[0].plot(K_path[:tp], color=color, linewidth=lw,
                              alpha=alpha, label=label)
            self._axs[1].plot(r_vals, color=color, linewidth=lw, alpha=alpha)
            self._axs[2].plot(w_vals, color=color, linewidth=lw, alpha=alpha)

            self._fig.suptitle(
                f"Shooting iteration convergence (iter {it + 1})",
                fontsize=13, y=1.02,
            )
            self._n_lines += 1

            # Refresh display
            if self._jupyter:
                self._clear_fn(wait=True)
                # Re-print buffered verbose lines so they aren't lost
                if verbose_lines:
                    print("\n".join(verbose_lines))
                self._display_fn(self._fig)
            else:
                self._fig.canvas.draw_idle()
                self._fig.canvas.flush_events()
        except Exception:
            pass

    def finalize(self, verbose_lines=None):
        """Render the final static figure."""
        try:
            self._axs[0].legend(fontsize=7, loc="best")
            self._fig.suptitle("Shooting iteration convergence", fontsize=13,
                               y=1.02)
            self._plt.tight_layout()
            if self._jupyter:
                self._clear_fn(wait=True)
                if verbose_lines:
                    print("\n".join(verbose_lines))
                self._display_fn(self._fig)
            else:
                self._plt.ioff()
                self._plt.show()
        except Exception:
            pass


# ── Root-finding method ───────────────────────────────────────────────


def _root_finding(
    ss_initial: Dict,
    ss_terminal: Dict,
    price_fn: Callable,
    T: int,
    tol: float,
    max_iter: int,
    verbose: bool,
    backward_fn: Optional[Callable],
) -> TransitionResult:
    """Transition path via root-finding on market clearing residuals."""
    from scipy.optimize import root

    Pi = ss_initial["Pi"]
    a_grid = ss_initial["a_grid"]
    beta = ss_initial["beta"]
    eis = ss_initial["eis"]
    D_initial = ss_initial["D"]
    Va_terminal = ss_terminal["Va"]
    K_init = ss_initial["A"]
    K_term = ss_terminal["A"]

    if verbose:
        print(f"  Initial SS: K={K_init:.4f}  r={ss_initial['r']:.6f}  "
              f"C={ss_initial['C']:.4f}")
        print(f"  Terminal SS: K={K_term:.4f}")
        print(f"  Root-finding with T={T} unknowns...")

    error_history: list = []
    eval_count = [0]

    def residual(K_vec):
        eval_count[0] += 1
        prices_path = [price_fn(K_vec[t], t) for t in range(T)]
        policies, _ = _backward_path(
            Va_terminal, prices_path, Pi, a_grid, beta, eis, backward_fn,
        )
        K_implied, _, _ = _forward_path(D_initial, policies, Pi, a_grid)
        res = K_implied - K_vec
        err = float(np.max(np.abs(res)))
        error_history.append(err)
        if verbose:
            print(f"  Root eval {eval_count[0]:3d}: max|residual| = {err:.4e}")
        return res

    K0 = np.linspace(K_init, K_term, T)
    sol = root(residual, K0, method="broyden1",
               options={"maxiter": max_iter, "fatol": tol, "line_search": "armijo"})

    K_path = sol.x
    prices_path = [price_fn(K_path[t], t) for t in range(T)]
    policies, _ = _backward_path(
        Va_terminal, prices_path, Pi, a_grid, beta, eis, backward_fn,
    )
    K_final, C_final, distributions = _forward_path(D_initial, policies, Pi, a_grid)

    final_err = float(np.max(np.abs(K_final - K_path)))
    r_path = np.array([prices_path[t]["r"] for t in range(T)])
    w_path = np.array([prices_path[t]["w"] for t in range(T)])

    if verbose:
        if sol.success:
            print(f"  Root-finding converged in {eval_count[0]} evaluations.")
        else:
            print(f"  Root-finding stopped: {sol.message}")

    return TransitionResult(
        T=T, K_path=K_final, r_path=r_path, w_path=w_path,
        C_path=C_final, policies=policies, distributions=distributions,
        converged=sol.success, n_iterations=eval_count[0],
        final_error=final_err, error_history=error_history,
    )


# ── Household+Solver backward function wrapper ────────────────────────


def make_backward_fn(solver, income_process, a_grid, beta, gamma):
    """Wrap a Household+Solver pair into a backward_fn callable.

    The returned callable has signature::

        backward_fn(Va, r, y) -> (Va_new, a_pol, c_pol)

    It creates a temporary Household at the given (r, w), solves to
    convergence using the provided solver, and extracts Va from the
    envelope condition.

    Parameters
    ----------
    solver : solver object
        Any solver with ``solve(household) -> SolvedPolicy``.
    income_process : IncomeProcess
        Income process object.
    a_grid : ndarray
        Asset grid.
    beta, gamma : float
        Discount factor and CRRA coefficient.
    """
    from .household import Household
    from .utils import crra_marginal

    def backward_fn(Va, r, y):
        # Infer wage from y and z_grid
        z_grid = income_process.z_grid
        w = float(y[0] / z_grid[0]) if z_grid[0] != 0 else 1.0

        hh = Household(
            income_process=income_process,
            a_grid=a_grid,
            beta=beta,
            gamma=gamma,
            r=r,
            w=w,
        )
        sol = hh.solve(solver)
        Va_new = (1.0 + r) * crra_marginal(sol.policy_c, gamma)
        return Va_new, sol.policy_a, sol.policy_c

    return backward_fn


# ── Main entry point ──────────────────────────────────────────────────


def compute_transition_path(
    ss_initial: Dict,
    ss_terminal: Dict,
    price_fn: Callable,
    T: int = 300,
    method: str = "shooting",
    damping: float = 0.5,
    tol: float = 1e-6,
    max_iter: int = 200,
    backward_fn: Optional[Callable] = None,
    verbose: bool = False,
    plot_progress: bool = False,
    plot_iterations: bool = False,
) -> TransitionResult:
    """Compute the perfect-foresight transition path after an MIT shock.

    Parameters
    ----------
    ss_initial : dict
        Initial steady state from ``household_steady_state()``.
        Must contain: D, Va, a, c, a_i, a_pi, A, C, Pi, a_grid, y,
        r, beta, eis.
    ss_terminal : dict
        Terminal steady state.  For a mean-reverting shock, pass the
        same dict as *ss_initial*.
    price_fn : callable
        ``price_fn(K, t) -> dict`` mapping aggregate capital and time
        to household inputs.  Must return at least
        ``{'r': float, 'y': ndarray, 'w': float}``.
    T : int
        Number of transition periods.
    method : str
        ``"shooting"`` (damped iteration) or ``"root_finding"``
        (Broyden).
    damping : float
        Weight on old guess in shooting update:
        ``K = damping * K_old + (1 - damping) * K_new``.
    tol : float
        Convergence tolerance on max|K_new - K_old|.
    max_iter : int
        Maximum outer iterations.
    backward_fn : callable, optional
        Custom backward step: ``backward_fn(Va, r, y) -> (Va_new,
        a_pol, c_pol)``.  If *None*, uses ``backward_egm``.
        Use ``make_backward_fn`` to wrap a Household+Solver pair.
    verbose : bool
        Print per-iteration convergence info.
    plot_progress : bool
        Show live convergence plots (shooting only).
    plot_iterations : bool
        Show overlaid K/r/w paths from each shooting iteration to
        visualize convergence. Rendered after the solver finishes
        (shooting only).

    Returns
    -------
    TransitionResult
    """
    if method == "shooting":
        return _shooting(
            ss_initial, ss_terminal, price_fn, T,
            damping, tol, max_iter, verbose, backward_fn, plot_progress,
            plot_iterations,
        )
    elif method == "root_finding":
        return _root_finding(
            ss_initial, ss_terminal, price_fn, T,
            tol, max_iter, verbose, backward_fn,
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'shooting' or 'root_finding'.")
