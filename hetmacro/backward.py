"""Backward iteration routines for dynamic programming."""

from typing import Tuple

import numpy as np


def backward_egm(
    Va: np.ndarray,
    Pi: np.ndarray,
    a_grid: np.ndarray,
    y: np.ndarray,
    r: float,
    beta: float,
    eis: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single EGM backward step.

    Parameters
    ----------
    Va : ndarray, shape (n_y, n_a)
        Marginal value of assets.
    Pi : ndarray, shape (n_y, n_y)
        Transition matrix for income states.
    a_grid : ndarray, shape (n_a,)
        Asset grid.
    y : ndarray, shape (n_y,)
        Income grid.
    r, beta, eis : float
        Interest rate, discount factor, elasticity of intertemporal substitution.
    """
    Wa = beta * (Pi @ Va)
    c_endog = Wa ** (-eis)
    coh = y[:, np.newaxis] + (1.0 + r) * a_grid[np.newaxis, :]

    a_pol = np.empty_like(coh)
    for e in range(len(y)):
        a_pol[e, :] = np.interp(
            coh[e, :],
            c_endog[e, :] + a_grid,
            a_grid,
        )

    a_pol = np.maximum(a_pol, a_grid[0])
    c_pol = coh - a_pol
    Va_new = (1.0 + r) * c_pol ** (-1.0 / eis)
    return Va_new, a_pol, c_pol


def backward_vfi(
    V: np.ndarray,
    Pi: np.ndarray,
    a_grid: np.ndarray,
    y: np.ndarray,
    r: float,
    beta: float,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single VFI step with discrete choice over a' in the grid."""
    n_y, n_a = V.shape
    EV = Pi @ V

    V_new = np.empty_like(V)
    a_pol = np.empty_like(V)
    c_pol = np.empty_like(V)

    for e in range(n_y):
        for i, a in enumerate(a_grid):
            c = y[e] + (1.0 + r) * a - a_grid
            util = np.where(c > 0, _crra_utility(c, gamma), -1e18)
            val = util + beta * EV[e, :]
            j = np.argmax(val)
            V_new[e, i] = val[j]
            a_pol[e, i] = a_grid[j]
            c_pol[e, i] = c[j]

    return V_new, a_pol, c_pol


def policy_iteration(
    Pi: np.ndarray,
    a_grid: np.ndarray,
    y: np.ndarray,
    r: float,
    beta: float,
    eis: float,
    method: str = "egm",
    tol: float = 1e-9,
    max_iter: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iterate policy until convergence using EGM or VFI."""
    method = method.lower()
    coh = y[:, np.newaxis] + (1.0 + r) * a_grid[np.newaxis, :]
    c_init = 0.05 * coh
    Va = (1.0 + r) * c_init ** (-1.0 / eis)

    a_old = np.zeros_like(Va)
    for _ in range(max_iter):
        if method == "egm":
            Va, a_pol, c_pol = backward_egm(Va, Pi, a_grid, y, r, beta, eis)
        elif method == "vfi":
            V = _value_from_marginal(Va, a_grid, eis)
            V, a_pol, c_pol = backward_vfi(V, Pi, a_grid, y, r, beta, 1.0 / eis)
            Va = (1.0 + r) * c_pol ** (-1.0 / eis)
        else:
            raise ValueError("method must be 'egm' or 'vfi'")

        if np.max(np.abs(a_pol - a_old)) < tol:
            return Va, a_pol, c_pol
        a_old = a_pol

    return Va, a_pol, c_pol


def _crra_utility(c: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return np.log(c)
    return c ** (1.0 - gamma) / (1.0 - gamma)


def _value_from_marginal(Va: np.ndarray, a_grid: np.ndarray, eis: float) -> np.ndarray:
    """Approximate value function from its derivative using trapezoid rule."""
    c = ((1.0 + 1e-12) / Va) ** eis
    V = np.zeros_like(Va)
    for i in range(1, a_grid.size):
        da = a_grid[i] - a_grid[i - 1]
        V[:, i] = V[:, i - 1] + 0.5 * da * (c[:, i - 1] + c[:, i])
    return V


def policy_evaluation_discrete(
    a_choice_idx: np.ndarray, Pi: np.ndarray, utility: np.ndarray, beta: float
) -> np.ndarray:
    """Evaluate value function for a fixed discrete policy via linear system solve.

    Parameters
    ----------
    a_choice_idx : ndarray, shape (n_y, n_a)
        Chosen next-asset index at each state.
    Pi : ndarray, shape (n_y, n_y)
        Income transition matrix.
    utility : ndarray, shape (n_y, n_a)
        One-period utility under the fixed policy.
    beta : float
        Discount factor.
    """
    n_y, n_a = a_choice_idx.shape
    n = n_y * n_a
    T = np.zeros((n, n))
    for e in range(n_y):
        for i in range(n_a):
            row = e * n_a + i
            j = int(a_choice_idx[e, i])
            for ep in range(n_y):
                col = ep * n_a + j
                T[row, col] += Pi[e, ep]

    u = utility.reshape(-1)
    V = np.linalg.solve(np.eye(n) - beta * T, u)
    return V.reshape(n_y, n_a)


def howard_policy_evaluation(
    a_choice_idx: np.ndarray,
    Pi: np.ndarray,
    a_grid: np.ndarray,
    y: np.ndarray,
    r: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Howard policy evaluation for fixed discrete policy indices."""
    n_y, n_a = a_choice_idx.shape
    utility = np.empty((n_y, n_a))
    for e in range(n_y):
        for i, a in enumerate(a_grid):
            j = int(a_choice_idx[e, i])
            c = y[e] + (1.0 + r) * a - a_grid[j]
            utility[e, i] = _crra_utility(np.maximum(c, 1e-14), gamma)
    return policy_evaluation_discrete(a_choice_idx, Pi, utility, beta)


def coefficient_vfi_step(theta: np.ndarray, bellman_operator) -> Tuple[np.ndarray, np.ndarray]:
    """Generic coefficient-space Bellman step.

    Parameters
    ----------
    theta : ndarray
        Current coefficient vector/matrix.
    bellman_operator : callable
        Function returning ``(theta_new, policy)``.
    """
    theta_new, policy = bellman_operator(theta)
    return theta_new, policy

