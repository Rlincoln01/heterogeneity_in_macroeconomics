"""Boar-Midrigan (2022) progressive taxation model.

Heterogeneous-agent model with endogenous labor supply, Cobb-Douglas
firms, and HSV progressive taxation. The government balances its budget
via a lump-sum transfer.

References
----------
Boar, Midrigan (2022). "Efficient Redistribution." JME.
"""

import numpy as np

from ..backward_labor import backward_step_labor
from ..forward import forward_iteration, stationary_distribution
from ..grids import make_asset_grid
from ..income_process import RouwenhorstIncome
from ..interpolation import get_lottery
from ..kronm import kronm
from ..solvers.howard_labor import solve_household_labor
from ..taxes import hsv_iota_balanced_budget, hsv_tax


# ── Firm block ───────────────────────────────────────────────────────


def firm_prices(K, L, alpha, delta):
    """Cobb-Douglas firm FOCs.

    Returns
    -------
    r : float — net interest rate = alpha * K^{alpha-1} * L^{1-alpha} - delta
    W : float — wage = (1-alpha) * K^alpha * L^{-alpha}
    """
    KL = K / L
    r = alpha * KL ** (alpha - 1.0) - delta
    W = (1.0 - alpha) * KL ** alpha
    return r, W


def firm_prices_from_r(r, alpha, delta):
    """Compute W from r using firm FOCs (no L needed).

    R = r + delta = alpha * (K/L)^{alpha-1}
    W = (1-alpha) * (alpha/R)^{alpha/(1-alpha)}
    """
    R = r + delta
    W = (1.0 - alpha) * (alpha / R) ** (alpha / (1.0 - alpha))
    return W


# ── Steady state ─────────────────────────────────────────────────────


def solve_boar_midrigan_ss(
    n_e=9, n_a=251, a_max=500.0, curvature=3.0,
    rho_e=0.98, sigma_e=0.20,
    beta=0.97, theta=1.5, gamma=2.0,
    alpha=1.0/3.0, delta=0.06,
    tau=0.25, xi=0.05, G=0.5,
    r_init=0.02, iota_init=0.20,
    K_init=None, L_init=None,
    damping=2.0/3.0, tol=1e-7, max_iter=50,
    method='shooting',
    solver_tol=1e-7, verbose=True,
):
    """Solve the Boar-Midrigan GE steady state.

    Parameters
    ----------
    n_e, n_a : int
        Grid sizes for productivity and assets.
    a_max : float
        Upper bound on asset grid.
    curvature : float
        Asset grid curvature parameter.
    rho_e, sigma_e : float
        AR(1) parameters for log productivity.
    beta, theta, gamma : float
        Discount factor, CRRA, inverse Frisch.
    alpha, delta : float
        Capital share, depreciation.
    tau, xi, G : float
        Tax level, progressivity, government spending.
    r_init, iota_init : float
        Initial guesses for interest rate and transfer.
    K_init, L_init : float or None
        Initial guesses for capital and labor (used by KLiota methods).
        If None, derived from r_init via firm FOCs with L=1.
    damping : float
        Weight on old guess (shooting only).
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum outer loop iterations.
    method : str
        'shooting' — damped fixed-point on (r, iota). Unstable for progressive taxes.
        'root_finding' — scipy.optimize.root on (r, iota). More robust.
        'shooting_KLiota' — damped fixed-point on (K, L, iota). Stable for
            progressive taxes because the K,L mapping is contractive.
        'root_finding_KLiota' — scipy.optimize.root on (K, L, iota).
    solver_tol : float
        Inner solver convergence tolerance.
    verbose : bool
        Print convergence info.

    Returns
    -------
    dict with steady state values.
    """
    from scipy.optimize import root as scipy_root

    # Setup grids
    a_grid = make_asset_grid(0.0, a_max, n_a, curvature=curvature)
    income = RouwenhorstIncome.from_ar1(n_e, rho=rho_e, sigma=sigma_e)
    e_grid = income.z_grid
    Pi = income.Pi

    # Persistent solver state (warm start)
    _state = {'h': None, 'g': None, 'v': None, 'call_count': 0}

    def _eval_ss(r_val, iota_val):
        """Evaluate one SS iteration: solve HH, compute distribution, aggregate."""
        W = firm_prices_from_r(r_val, alpha, delta)

        result = solve_household_labor(
            Pi, a_grid, e_grid, W, r_val, tau, xi, iota_val,
            beta, theta, gamma,
            h_init=_state['h'], g_init=_state['g'], v_init=_state['v'],
            tol=solver_tol, verbose=False,
        )

        _state['h'] = result.policy_h
        _state['g'] = result.marginal_value
        _state['v'] = result.value
        _state['call_count'] += 1

        D = stationary_distribution(Pi, result.policy_a, a_grid)

        K = float(np.sum(D * result.policy_a))
        L = float(np.sum(D * result.policy_h * np.tile(e_grid[:, None], (1, n_a))))
        C = float(np.sum(D * result.policy_c))

        iota_new = hsv_iota_balanced_budget(result.policy_y, D, tau, xi, G)
        r_new = alpha * K ** (alpha - 1.0) * L ** (1.0 - alpha) - delta

        return r_new, iota_new, result, D, K, L, C

    def _eval_ss_KLiota(K_val, L_val, iota_val, use_warm_start=True):
        """Evaluate one SS iteration using (K, L, iota) as unknowns."""
        r_val, W = firm_prices(K_val, L_val, alpha, delta)

        h_init = _state['h'] if use_warm_start else None
        g_init = _state['g'] if use_warm_start else None
        v_init = _state['v'] if use_warm_start else None

        result = solve_household_labor(
            Pi, a_grid, e_grid, W, r_val, tau, xi, iota_val,
            beta, theta, gamma,
            h_init=h_init, g_init=g_init, v_init=v_init,
            tol=solver_tol, verbose=False,
        )

        _state['h'] = result.policy_h
        _state['g'] = result.marginal_value
        _state['v'] = result.value
        _state['call_count'] += 1

        D = stationary_distribution(Pi, result.policy_a, a_grid)

        K_s = float(np.sum(D * result.policy_a))
        L_s = float(np.sum(D * result.policy_h * np.tile(e_grid[:, None], (1, n_a))))
        C = float(np.sum(D * result.policy_c))

        iota_new = hsv_iota_balanced_budget(result.policy_y, D, tau, xi, G)

        return K_s, L_s, iota_new, result, D, r_val, W, C

    # Derive initial (K, L) from r_init if not provided
    def _init_KL():
        K0, L0 = K_init, L_init
        if L0 is None:
            L0 = 1.0
        if K0 is None:
            # K/L = ((r+delta)/alpha)^{1/(alpha-1)}
            KL = ((r_init + delta) / alpha) ** (1.0 / (alpha - 1.0))
            K0 = L0 * KL
        return K0, L0

    if method == 'root_finding':
        # Use scipy.optimize.root for robustness
        def _residual(x):
            r_val, iota_val = x
            # No warm-start: ensures deterministic mapping for Jacobian estimation
            _state['h'] = None
            _state['g'] = None
            _state['v'] = None
            r_new, iota_new, _, _, _, _, _ = _eval_ss(r_val, iota_val)
            res = np.array([r_new - r_val, iota_new - iota_val])
            if verbose:
                n = _state['call_count']
                print(f"  eval {n:3d}: r={r_val:.6f}, iota={iota_val:.6f}, "
                      f"||res||={np.linalg.norm(res):.2e}")
            return res

        if verbose:
            print("Boar-Midrigan Steady State (root-finding, r-iota)")

        sol = scipy_root(
            _residual,
            x0=np.array([r_init, iota_init]),
            method='hybr',
            tol=tol,
            options={'maxfev': max_iter * 2, 'epsfcn': 1e-4},
        )

        r, iota = sol.x
        if verbose:
            print(f"  scipy.root: success={sol.success}, nfev={sol.nfev}")

    elif method == 'root_finding_KLiota':
        def _residual_KLi(x):
            K_val, L_val, iota_val = x
            # No warm-start: ensures deterministic mapping for Jacobian estimation
            K_s, L_s, iota_new, _, _, r_val, _, _ = _eval_ss_KLiota(
                K_val, L_val, iota_val, use_warm_start=False
            )
            res = np.array([K_s - K_val, L_s - L_val, iota_new - iota_val])
            if verbose:
                n = _state['call_count']
                print(f"  eval {n:3d}: K={K_val:.4f}, L={L_val:.4f}, "
                      f"iota={iota_val:.4f}, r={r_val:.6f}, "
                      f"||res||={np.linalg.norm(res):.2e}")
            return res

        K0, L0 = _init_KL()
        if verbose:
            print("Boar-Midrigan Steady State (root-finding, K-L-iota)")

        sol = scipy_root(
            _residual_KLi,
            x0=np.array([K0, L0, iota_init]),
            method='hybr',
            tol=tol,
            options={'maxfev': max_iter * 3, 'epsfcn': 1e-4},
        )

        K_sol, L_sol, iota = sol.x
        r, _ = firm_prices(K_sol, L_sol, alpha, delta)
        if verbose:
            print(f"  scipy.root: success={sol.success}, nfev={sol.nfev}")

    elif method == 'shooting_KLiota':
        K, L = _init_KL()
        iota = iota_init
        step = 1.0 - damping

        if verbose:
            print("Boar-Midrigan Steady State (shooting, K-L-iota)")
            print(f"{'iter':>4s} {'||change||':>12s} {'K':>10s} {'L':>10s} "
                  f"{'iota':>10s} {'r':>10s}")

        for it in range(max_iter):
            K_s, L_s, iota_new, result, D, r_val, W, C = _eval_ss_KLiota(
                K, L, iota
            )

            par_old = np.array([K, L, iota])
            par_new = np.array([K_s, L_s, iota_new])

            # Reject if inner solver produced garbage (K_s far from K)
            if K_s > 5.0 * K or K_s < 0.2 * K or np.isnan(K_s):
                if verbose:
                    print(f"{it+1:4d} {'REJECTED':>12s} {K:10.4f} {L:10.4f} "
                          f"{iota:10.4f} {r_val:10.6f}  K_s={K_s:.1f}")
                # Tighten damping and retry
                step = max(step * 0.5, 0.05)
                continue

            change = np.linalg.norm(par_old - par_new)

            if verbose:
                print(f"{it+1:4d} {change:12.2e} {K:10.4f} {L:10.4f} "
                      f"{iota:10.4f} {r_val:10.6f}")

            if change < tol:
                if verbose:
                    print(f"\nConverged in {it+1} iterations.")
                break

            # Clamped update: limit relative change per step
            dx = step * (par_new - par_old)
            max_rel_change = 0.15 * np.abs(par_old) + 0.01
            dx = np.clip(dx, -max_rel_change, max_rel_change)

            K = K + dx[0]
            L = L + dx[1]
            iota = iota + dx[2]

            # Ensure positivity
            K = max(K, 1.0)
            L = max(L, 0.3)
            iota = max(iota, 0.001)

        r = r_val

    else:
        # Damped fixed-point (shooting) on (r, iota)
        r = r_init
        iota = iota_init
        step = 1.0 - damping

        if verbose:
            print("Boar-Midrigan Steady State (shooting, r-iota)")
            print(f"{'iter':>4s} {'||change||':>12s} {'r':>10s} {'iota':>10s}")

        for it in range(max_iter):
            r_new, iota_new, result, D, K, L, C = _eval_ss(r, iota)

            par_old = np.array([r, iota])
            par_new = np.array([r_new, iota_new])
            change = np.linalg.norm(par_old - par_new)

            if verbose:
                print(f"{it+1:4d} {change:12.2e} {r:10.4f} {iota:10.4f}")

            if change < tol:
                if verbose:
                    print(f"\nConverged in {it+1} iterations.")
                break

            r = r + step * (r_new - r)
            iota = iota + step * (iota_new - iota)

    # Final evaluation at converged prices to get all outputs
    if method in ('shooting_KLiota', 'root_finding_KLiota'):
        K_s, L_s, iota_final, result, D, r_val, W, C = _eval_ss_KLiota(
            K if method == 'shooting_KLiota' else K_sol,
            L if method == 'shooting_KLiota' else L_sol,
            iota,
        )
        K, L = K_s, L_s
    else:
        r_final, iota_final, result, D, K, L, C = _eval_ss(r, iota)

    # Final values
    W = firm_prices_from_r(r, alpha, delta)
    Y = K ** alpha * L ** (1.0 - alpha)
    Tax = float(np.sum(D * hsv_tax(result.policy_y, tau, xi, iota)))

    # Wealth/income moments
    a_flat = result.policy_a.ravel()
    y_flat = result.policy_y.ravel()
    D_flat = D.ravel()
    moments = _compute_moments(a_flat, y_flat, D_flat)

    ss = {
        # Prices
        'r': r, 'W': W, 'R': r + delta,
        # Aggregates
        'K': K, 'L': L, 'Y': Y, 'C': C, 'Tax': Tax, 'iota': iota,
        # Policy parameters
        'tau': tau, 'xi': xi, 'G': G,
        'alpha': alpha, 'delta': delta, 'beta': beta,
        'theta': theta, 'gamma': gamma,
        # Grids
        'a_grid': a_grid, 'e_grid': e_grid, 'Pi': Pi,
        'n_e': n_e, 'n_a': n_a,
        # Policies (flat)
        'h': result.policy_h.ravel(),
        'c': result.policy_c.ravel(),
        'kprime': result.policy_a.ravel(),
        'y': result.policy_y.ravel(),
        'yhat': result.policy_yhat.ravel(),
        'mtax': result.policy_mtax.ravel(),
        # Value/marginal value
        'v': result.value.ravel(),
        'g': result.marginal_value.ravel(),
        # Distribution
        'D': D, 'D_flat': D.ravel(),
        # Expectations (for transition initial condition)
        'gbar': kronm([n_a, Pi], result.marginal_value.ravel()),
        'vbar': kronm([n_a, Pi], result.value.ravel()),
        # Moments
        'moments': moments,
        # Solver result
        'result': result,
    }

    if verbose:
        _print_ss(ss)

    return ss


# ── Transition dynamics (global method) ──────────────────────────────


class _TransitionPlotter:
    """Live overlay plot of r_t, iota_t, K_t paths across shooting iterations."""

    def __init__(self, ss_old, ss_new, T):
        import matplotlib.pyplot as plt
        self._plt = plt
        self._T = T
        self._cmap = plt.get_cmap("viridis")
        self._n_lines = 0

        plt.ioff()
        self._fig, self._axs = plt.subplots(1, 3, figsize=(16, 5))

        # r panel
        self._axs[0].axhline(ss_old['r'], color='gray', ls=':', lw=0.8, label='old SS')
        self._axs[0].axhline(ss_new['r'], color='k', ls='--', lw=0.8, label='new SS')
        self._axs[0].set_xlabel('t'); self._axs[0].set_ylabel('r')
        self._axs[0].set_title('Interest rate path')
        self._axs[0].legend(fontsize=7); self._axs[0].grid(alpha=0.25)

        # iota panel
        self._axs[1].axhline(ss_old['iota'], color='gray', ls=':', lw=0.8)
        self._axs[1].axhline(ss_new['iota'], color='k', ls='--', lw=0.8)
        self._axs[1].set_xlabel('t'); self._axs[1].set_ylabel(r'$\iota$')
        self._axs[1].set_title('Transfer path')
        self._axs[1].grid(alpha=0.25)

        # K panel
        self._axs[2].axhline(ss_old['K'], color='gray', ls=':', lw=0.8)
        self._axs[2].axhline(ss_new['K'], color='k', ls='--', lw=0.8)
        self._axs[2].set_xlabel('t'); self._axs[2].set_ylabel('K')
        self._axs[2].set_title('Capital path')
        self._axs[2].grid(alpha=0.25)

        plt.tight_layout()

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

    def add_iteration(self, rt, iotat, Kt, it, max_iter, verbose_lines=None,
                      final=False):
        try:
            frac = 1.0 if final else min(self._n_lines / max(max_iter - 1, 1), 0.9)
            color = self._cmap(frac)
            lw = 2.0 if final else 0.7
            alpha = 1.0 if final else 0.5
            label = f"iter {it+1}" if (it < 3 or final) else None

            self._axs[0].plot(rt, color=color, lw=lw, alpha=alpha, label=label)
            self._axs[1].plot(iotat, color=color, lw=lw, alpha=alpha)
            self._axs[2].plot(Kt, color=color, lw=lw, alpha=alpha)
            self._n_lines += 1

            self._fig.suptitle(
                f"Transition path convergence (iter {it+1})",
                fontsize=13, y=1.02,
            )

            if self._jupyter:
                self._clear_fn(wait=True)
                if verbose_lines:
                    print("\n".join(verbose_lines))
                self._display_fn(self._fig)
            else:
                self._fig.canvas.draw_idle()
                self._fig.canvas.flush_events()
        except Exception:
            pass

    def finalize(self, verbose_lines=None):
        try:
            self._axs[0].legend(fontsize=7, loc="best")
            self._fig.suptitle("Transition path convergence", fontsize=13, y=1.02)
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


def boar_midrigan_transition(
    ss_old, ss_new,
    T=200, damping=0.9, tol=1e-7, max_iter=150,
    method='KL_iota', plot_iterations=False, verbose=True,
):
    """Compute transition dynamics after a tax reform (global method).

    Backward-forward shooting with damped fixed-point iteration.

    Parameters
    ----------
    ss_old : dict
        Initial steady state (baseline tax).
    ss_new : dict
        Terminal steady state (reform tax).
    T : int
        Number of transition periods.
    damping : float
        Weight on old guess in path updates.
    tol : float
        Convergence tolerance on path changes.
    max_iter : int
        Maximum iterations.
    method : str
        'r_iota' — guess/update (r_t, iota_t) paths.
        'KL_iota' — guess/update (K_t, L_t, iota_t) paths (more stable
        for progressive taxes).
    plot_iterations : bool
        If True, show live overlay plot of r_t, iota_t, K_t paths.
    verbose : bool
        Print convergence info.

    Returns
    -------
    dict with transition paths.
    """
    step = 1.0 - damping
    a_grid = ss_new['a_grid']
    e_grid = ss_new['e_grid']
    Pi = ss_new['Pi']
    n_e, n_a = ss_new['n_e'], ss_new['n_a']
    n_s = n_e * n_a
    alpha, delta = ss_new['alpha'], ss_new['delta']
    beta, theta, gamma = ss_new['beta'], ss_new['theta'], ss_new['gamma']
    tau, xi, G = ss_new['tau'], ss_new['xi'], ss_new['G']
    e_rep = np.repeat(e_grid, n_a)

    use_KL = (method == 'KL_iota')

    # ── Initial guess: geometric mean-reversion toward new SS ──
    rho_init = 0.9
    rt = np.zeros(T)
    Wt = np.zeros(T)
    iotat = np.zeros(T)
    Kguess = np.zeros(T)
    Lguess = np.zeros(T)

    rt[0] = ss_old['r'];       rt[-1] = ss_new['r']
    iotat[0] = ss_old['iota']; iotat[-1] = ss_new['iota']
    Kguess[0] = ss_old['K'];   Kguess[-1] = ss_new['K']
    Lguess[0] = ss_old['L'];   Lguess[-1] = ss_new['L']
    for t in range(1, T - 1):
        rt[t] = rho_init * rt[t - 1] + (1.0 - rho_init) * ss_new['r']
        iotat[t] = rho_init * iotat[t - 1] + (1.0 - rho_init) * ss_new['iota']
        Kguess[t] = rho_init * Kguess[t - 1] + (1.0 - rho_init) * ss_new['K']
        Lguess[t] = rho_init * Lguess[t - 1] + (1.0 - rho_init) * ss_new['L']

    if use_KL:
        for t in range(T):
            rt[t], Wt[t] = firm_prices(Kguess[t], Lguess[t], alpha, delta)
    else:
        for t in range(T):
            Wt[t] = firm_prices_from_r(rt[t], alpha, delta)

    # ── Plotter ──
    plotter = None
    if plot_iterations:
        plotter = _TransitionPlotter(ss_old, ss_new, T)

    verbose_lines = []
    if verbose:
        hdr = f"Boar-Midrigan Transition Dynamics (method={method})"
        if use_KL:
            col = (f"{'iter':>4s} {'err':>12s}  "
                   f"{'|dK|':>10s} {'|dL|':>10s} {'|di|':>10s}")
        else:
            col = (f"{'iter':>4s} {'err':>12s}  "
                   f"{'|dr|':>10s} {'|di|':>10s}")
        verbose_lines = [hdr, col]
        print(hdr)
        print(col)

    a_grid_2d = np.tile(a_grid[None, :], (n_e, 1))

    for iteration in range(max_iter):
        # ── Prices from current guess ──
        if use_KL:
            for t in range(1, T - 1):
                rt[t], Wt[t] = firm_prices(Kguess[t], Lguess[t], alpha, delta)
        else:
            for t in range(1, T - 1):
                Wt[t] = firm_prices_from_r(rt[t], alpha, delta)

        # ── Backward: solve decision rules from new SS ──
        vt = np.zeros((n_s, T))
        gt = np.zeros((n_s, T))
        ht = np.zeros((n_s, T))
        ct = np.zeros((n_s, T))
        kprimet = np.zeros((n_s, T))
        yt = np.zeros((n_s, T))
        yhatt = np.zeros((n_s, T))
        mtaxt = np.zeros((n_s, T))

        # Terminal condition: new SS
        for arr, key in [(vt,'v'),(gt,'g'),(ht,'h'),(ct,'c'),
                         (kprimet,'kprime'),(yt,'y'),(yhatt,'yhat'),(mtaxt,'mtax')]:
            arr[:, -1] = ss_new[key]

        # Initial condition: old SS
        for arr, key in [(vt,'v'),(gt,'g'),(ht,'h'),(ct,'c'),
                         (kprimet,'kprime'),(yt,'y'),(yhatt,'yhat'),(mtaxt,'mtax')]:
            arr[:, 0] = ss_old[key]

        gbar_next = ss_new['gbar'].copy()
        vbar_next = ss_new['vbar'].copy()

        for t in range(T - 2, 0, -1):
            bstep = backward_step_labor(
                gbar_next, vbar_next, np.full(n_s, 1e-5), Pi,
                a_grid, e_grid, Wt[t], rt[t], tau, xi, iotat[t],
                beta, theta, gamma,
            )
            vt[:, t] = bstep['v']
            gt[:, t] = bstep['g']
            ht[:, t] = bstep['h']
            ct[:, t] = bstep['c']
            kprimet[:, t] = bstep['kprime']
            yt[:, t] = bstep['y']
            yhatt[:, t] = bstep['yhat']
            mtaxt[:, t] = bstep['mtax']
            gbar_next = bstep['gbar_out']
            vbar_next = bstep['vbar_out']

        # ── Forward: iterate distribution from old SS ──
        # Convention: D_t = distribution at start of period t.
        # D_t is determined by PREVIOUS period's savings: kprimet[:, t-1].
        # K_t = sum(D_t * a_grid) = asset holdings = capital stock at t.
        # L_t, C_t use CURRENT period policies (agents choose at t).
        Dt = np.zeros((n_e, n_a, T))
        Dt[:, :, 0] = ss_old['D']

        Kt = np.zeros(T)
        Lt = np.zeros(T)
        Ct = np.zeros(T)
        Taxt = np.zeros(T)

        Kt[0] = ss_old['K']
        Lt[0] = ss_old['L']
        Ct[0] = ss_old['C']
        Taxt[0] = ss_old['Tax']

        iotatnew = np.zeros(T)
        iotatnew[0] = ss_old['iota']
        iotatnew[-1] = ss_new['iota']

        for t in range(1, T):
            # Evolve distribution using PREVIOUS period's savings policy
            kp_prev = kprimet[:, t - 1].reshape(n_e, n_a)
            a_i, a_pi = get_lottery(kp_prev, a_grid)
            Dt[:, :, t] = forward_iteration(Dt[:, :, t - 1], Pi, a_i, a_pi)

            if t < T - 1:
                D_t = Dt[:, :, t]
                D_t_flat = D_t.ravel()

                # Capital = asset holdings at start of period t
                Kt[t] = float(np.sum(D_t * a_grid_2d))
                Lt[t] = float(np.sum(D_t_flat * ht[:, t] * e_rep))
                Ct[t] = float(np.sum(D_t_flat * ct[:, t]))
                Taxt[t] = float(np.sum(
                    D_t_flat * hsv_tax(yt[:, t], tau, xi, iotat[t])))

                iotatnew[t] = hsv_iota_balanced_budget(
                    yt[:, t].reshape(n_e, n_a), D_t, tau, xi, G
                )

        # Terminal aggregates from forward-simulated distribution
        D_T = Dt[:, :, -1]
        D_T_flat = D_T.ravel()
        Kt[-1] = float(np.sum(D_T * a_grid_2d))
        Lt[-1] = float(np.sum(D_T_flat * ht[:, -1] * e_rep))
        Ct[-1] = float(np.sum(D_T_flat * ct[:, -1]))
        Taxt[-1] = float(np.sum(
            D_T_flat * hsv_tax(yt[:, -1], tau, xi, iotat[-1])))

        Yt = Kt ** alpha * Lt ** (1.0 - alpha)

        # ── Convergence check (max-norm) ──
        if use_KL:
            resid_K = Kt[1:-1] - Kguess[1:-1]
            resid_L = Lt[1:-1] - Lguess[1:-1]
            resid_iota = iotatnew[1:-1] - iotat[1:-1]

            maxK = float(np.max(np.abs(resid_K)))
            maxL = float(np.max(np.abs(resid_L)))
            maxiota = float(np.max(np.abs(resid_iota)))
            err = max(maxK, maxL, maxiota)

            iK = int(np.argmax(np.abs(resid_K))) + 1
            iL = int(np.argmax(np.abs(resid_L))) + 1
            iiota = int(np.argmax(np.abs(resid_iota))) + 1
        else:
            rtnew = np.zeros(T)
            rtnew[0] = ss_old['r']; rtnew[-1] = ss_new['r']
            for t in range(1, T - 1):
                rtnew[t] = (alpha * Kt[t] ** (alpha - 1.0)
                            * Lt[t] ** (1.0 - alpha) - delta)
            resid_r = rtnew[1:-1] - rt[1:-1]
            resid_iota = iotatnew[1:-1] - iotat[1:-1]
            maxK = float(np.max(np.abs(resid_r)))
            maxiota = float(np.max(np.abs(resid_iota)))
            maxL = 0.0
            err = max(maxK, maxiota)
            iK = int(np.argmax(np.abs(resid_r))) + 1
            iiota = int(np.argmax(np.abs(resid_iota))) + 1
            iL = 0

        if use_KL:
            line = (f"{iteration+1:4d} {err:12.2e}  "
                    f"{maxK:10.2e} {maxL:10.2e} {maxiota:10.2e}")
        else:
            line = (f"{iteration+1:4d} {err:12.2e}  "
                    f"{maxK:10.2e} {maxiota:10.2e}")
        if verbose:
            verbose_lines.append(line)
            print(line)

        if plotter is not None:
            r_show = np.array([firm_prices(Kt[t], Lt[t], alpha, delta)[0]
                               if Kt[t] > 0 and Lt[t] > 0 else rt[t]
                               for t in range(T)]) if use_KL else rt.copy()
            plotter.add_iteration(r_show, iotat, Kt, iteration, max_iter,
                                  verbose_lines=verbose_lines)

        if err < tol:
            if verbose:
                msg = f"\nConverged in {iteration+1} iterations."
                verbose_lines.append(msg)
                print(msg)
            break

        # ── Damped update ──
        if use_KL:
            Kguess[1:-1] += step * (Kt[1:-1] - Kguess[1:-1])
            Lguess[1:-1] += step * (Lt[1:-1] - Lguess[1:-1])
        else:
            rt[1:-1] += step * (rtnew[1:-1] - rt[1:-1])
        iotat[1:-1] += step * (iotatnew[1:-1] - iotat[1:-1])

    if plotter is not None:
        r_final = np.array([firm_prices(Kt[t], Lt[t], alpha, delta)[0]
                            if Kt[t] > 0 and Lt[t] > 0 else rt[t]
                            for t in range(T)]) if use_KL else rt.copy()
        plotter.add_iteration(r_final, iotat, Kt, iteration, max_iter,
                              verbose_lines=verbose_lines, final=True)
        plotter.finalize(verbose_lines=verbose_lines)

    # Final r, W paths
    rt_final = np.zeros(T)
    Wt_final = np.zeros(T)
    for t in range(T):
        if use_KL:
            rt_final[t], Wt_final[t] = firm_prices(
                Kguess[t], Lguess[t], alpha, delta)
        else:
            rt_final[t] = rt[t]
            Wt_final[t] = firm_prices_from_r(rt[t], alpha, delta)

    return {
        'rt': rt_final, 'Wt': Wt_final, 'iotat': iotat,
        'Kt': Kt, 'Lt': Lt, 'Yt': Yt, 'Ct': Ct, 'Taxt': Taxt,
        'vt': vt, 'gt': gt, 'ht': ht, 'ct': ct,
        'kprimet': kprimet, 'yt': yt, 'yhatt': yhatt, 'mtaxt': mtaxt,
        'Dt': Dt,
        'T': T,
        'ss_old': ss_old, 'ss_new': ss_new,
    }


# ── Welfare ──────────────────────────────────────────────────────────


def compute_welfare_cev(v, D_flat, beta, theta):
    """Compute consumption-equivalent welfare.

    omega = [(1-beta)*(1-theta)*V]^{1/(1-theta)}

    Parameters
    ----------
    v : ndarray
        Value function (flat).
    D_flat : ndarray
        Distribution (flat, sums to 1).
    beta, theta : float
        Preference parameters.

    Returns
    -------
    omega_agg : float
        Aggregate CEV welfare.
    """
    if abs(theta - 1.0) < 1e-12:
        omega = np.exp((1.0 - beta) * v)
    else:
        omega = ((1.0 - beta) * (1.0 - theta) * v) ** (1.0 / (1.0 - theta))

    omega_agg = (float(np.sum(D_flat * omega ** (1.0 - theta)))) ** (1.0 / (1.0 - theta))
    return omega_agg


def welfare_by_tercile(v_old, v_new, D_old_flat, beta, theta):
    """CEV decomposed by value-function terciles.

    Parameters
    ----------
    v_old : ndarray
        Old SS value function (flat).
    v_new : ndarray
        New value function at t=1 (flat, after reform announcement).
    D_old_flat : ndarray
        Old SS distribution (flat).
    beta, theta : float
        Preference parameters.

    Returns
    -------
    dict with keys: 'all', 'bottom', 'middle', 'top'
        Each contains (omega_old, omega_new, cev_gain).
    """
    # Compute CEV for full population
    omega_old_all = compute_welfare_cev(v_old, D_old_flat, beta, theta)
    omega_new_all = compute_welfare_cev(v_new, D_old_flat, beta, theta)

    # Tercile cutoffs based on old value function
    p33 = _weighted_percentile(v_old, 100.0/3.0, D_old_flat)
    p67 = _weighted_percentile(v_old, 200.0/3.0, D_old_flat)

    bot = v_old <= p33
    top = v_old > p67
    mid = ~bot & ~top

    results = {'all': (omega_old_all, omega_new_all, omega_new_all / omega_old_all - 1.0)}

    for name, mask in [('bottom', bot), ('middle', mid), ('top', top)]:
        if np.sum(D_old_flat[mask]) > 0:
            D_group = D_old_flat.copy()
            D_group[~mask] = 0.0
            D_group = D_group / np.sum(D_group)  # normalize

            omega_old = compute_welfare_cev(v_old, D_group, beta, theta)
            omega_new = compute_welfare_cev(v_new, D_group, beta, theta)
            results[name] = (omega_old, omega_new, omega_new / omega_old - 1.0)
        else:
            results[name] = (0.0, 0.0, 0.0)

    return results


# ── Helpers ──────────────────────────────────────────────────────────


def _weighted_percentile(values, percentile, weights):
    """Compute weighted percentile."""
    sorted_idx = np.argsort(values)
    sorted_v = values[sorted_idx]
    sorted_w = weights[sorted_idx]
    cum_w = np.cumsum(sorted_w)
    target = percentile / 100.0
    idx = np.searchsorted(cum_w, target)
    idx = min(idx, len(sorted_v) - 1)
    return sorted_v[idx]


def _compute_moments(a_flat, y_flat, D_flat):
    """Compute wealth and income distribution moments."""
    k_mean = float(np.sum(D_flat * a_flat))
    y_mean = float(np.sum(D_flat * y_flat))

    # Wealth moments
    sorted_idx_k = np.argsort(a_flat)
    dk = D_flat[sorted_idx_k]
    ak = a_flat[sorted_idx_k]
    cum_dk = np.cumsum(dk)

    top01_k = cum_dk >= 0.999
    top1_k = cum_dk >= 0.99
    top5_k = cum_dk >= 0.95

    share_top01_k = float(np.sum(dk[top01_k] * ak[top01_k])) / max(k_mean, 1e-15)
    share_top1_k = float(np.sum(dk[top1_k] * ak[top1_k])) / max(k_mean, 1e-15)
    share_top5_k = float(np.sum(dk[top5_k] * ak[top5_k])) / max(k_mean, 1e-15)

    gini_k = 1.0 - float(np.sum(dk * ak / max(k_mean, 1e-15) * (dk + 2.0 * (1.0 - cum_dk))))

    # Income moments
    sorted_idx_y = np.argsort(y_flat)
    dy = D_flat[sorted_idx_y]
    ay = y_flat[sorted_idx_y]
    cum_dy = np.cumsum(dy)

    top01_y = cum_dy >= 0.999
    top1_y = cum_dy >= 0.99
    top5_y = cum_dy >= 0.95

    share_top01_y = float(np.sum(dy[top01_y] * ay[top01_y])) / max(y_mean, 1e-15)
    share_top1_y = float(np.sum(dy[top1_y] * ay[top1_y])) / max(y_mean, 1e-15)
    share_top5_y = float(np.sum(dy[top5_y] * ay[top5_y])) / max(y_mean, 1e-15)

    gini_y = 1.0 - float(np.sum(dy * ay / max(y_mean, 1e-15) * (dy + 2.0 * (1.0 - cum_dy))))

    return {
        'wealth_to_income': k_mean / max(y_mean, 1e-15),
        'gini_wealth': gini_k,
        'gini_income': gini_y,
        'share_wealth_top01': share_top01_k,
        'share_wealth_top1': share_top1_k,
        'share_wealth_top5': share_top5_k,
        'share_income_top01': share_top01_y,
        'share_income_top1': share_top1_y,
        'share_income_top5': share_top5_y,
    }


def _print_ss(ss):
    """Print steady state summary."""
    print()
    print("Macro Variables:")
    print(f"  Y = {ss['Y']:9.2f}    C + dK + G = {ss['C'] + ss['delta'] * ss['K'] + ss['G']:9.2f}")
    print(f"  Tax = {ss['Tax']:7.2f}    G = {ss['G']:7.2f}")
    print(f"  r = {ss['r']:9.4f}")
    print(f"  W = {ss['W']:9.4f}")
    print(f"  iota = {ss['iota']:7.4f}")
    print(f"  K = {ss['K']:9.2f}")
    print(f"  L = {ss['L']:9.4f}")
    print()
    m = ss['moments']
    print("Moments:")
    print(f"  Wealth/Income = {m['wealth_to_income']:7.2f}")
    print(f"  Gini wealth   = {m['gini_wealth']:7.2f}")
    print(f"  Gini income   = {m['gini_income']:7.2f}")
    print(f"  Wealth top 0.1% = {m['share_wealth_top01']:7.2f}")
    print(f"  Wealth top 1%   = {m['share_wealth_top1']:7.2f}")
    print(f"  Wealth top 5%   = {m['share_wealth_top5']:7.2f}")
    print(f"  Income top 0.1% = {m['share_income_top01']:7.2f}")
    print(f"  Income top 1%   = {m['share_income_top1']:7.2f}")
    print(f"  Income top 5%   = {m['share_income_top5']:7.2f}")
    print()
