"""Sequence-space Jacobian tools (fake news algorithm)."""

from typing import Dict, Tuple

import numpy as np

from .backward import backward_egm
from .forward import expectation_functions, forward_iteration
from .interpolation import get_lottery


def jacobian(ss: Dict[str, np.ndarray], shocks: Dict[str, Dict[str, float]], T: int) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute Jacobians of aggregate variables using fake news algorithm.

    Parameters
    ----------
    ss : dict
        Steady state dict with keys like 'D', 'Pi', 'a_i', 'a_pi', 'a', 'c'.
    shocks : dict
        Mapping shock_name -> dict of input perturbations.
    T : int
        Horizon length.
    """
    curlyY = {"A": {}, "C": {}}
    curlyD = {}

    for shock_name, shock in shocks.items():
        curlyYi, curlyD[shock_name] = step1_backward(ss, shock, T, h=1e-4)
        curlyY["A"][shock_name] = curlyYi["A"]
        curlyY["C"][shock_name] = curlyYi["C"]

    curlyE = {}
    for out in ("A", "C"):
        curlyE[out] = expectation_functions(
            ss[out.lower()],
            ss["Pi"],
            ss["a_i"],
            ss["a_pi"],
            T - 1,
        )

    Js = {"A": {}, "C": {}}
    for out in Js:
        for shock_name in shocks:
            F = np.empty((T, T))
            F[0, :] = curlyY[out][shock_name]
            F[1:, :] = (
                curlyE[out].reshape(T - 1, -1) @ curlyD[shock_name].reshape(T, -1).T
            )
            Js[out][shock_name] = J_from_F(F)

    return Js


def step1_backward(
    ss: Dict[str, np.ndarray],
    shock: Dict[str, float],
    T: int,
    h: float = 1e-4,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Step 1 of fake news algorithm: compute curlyY and curlyD."""
    D1_noshock = forward_iteration(ss["D"], ss["Pi"], ss["a_i"], ss["a_pi"])
    ss_inputs = {k: ss[k] for k in ("Va", "Pi", "a_grid", "y", "r", "beta", "eis")}

    curlyY = {"A": np.empty(T), "C": np.empty(T)}
    curlyD = np.empty((T,) + ss["D"].shape)

    Va = ss_inputs["Va"]
    for s in range(T):
        if s == 0:
            shocked_inputs = {k: ss[k] + h * shock[k] for k in shock}
            Va, a_pol, c_pol = backward_egm(**{**ss_inputs, **shocked_inputs})
        else:
            Va, a_pol, c_pol = backward_egm(**{**ss_inputs, "Va": Va})

        curlyY["A"][s] = np.vdot(ss["D"], a_pol - ss["a"]) / h
        curlyY["C"][s] = np.vdot(ss["D"], c_pol - ss["c"]) / h

        a_i_shocked, a_pi_shocked = get_lottery(a_pol, ss["a_grid"])
        curlyD[s] = (
            forward_iteration(ss["D"], ss["Pi"], a_i_shocked, a_pi_shocked) - D1_noshock
        ) / h

    return curlyY, curlyD


def J_from_F(F: np.ndarray) -> np.ndarray:
    """Convert fake news matrix F to Jacobian J."""
    J = F.copy()
    for t in range(1, F.shape[0]):
        J[1:, t] += J[:-1, t - 1]
    return J


# ── General equilibrium closure ──────────────────────────────────────────


def solve_ge(
    H_U: np.ndarray,
    H_Z: np.ndarray,
    dZ: np.ndarray,
) -> np.ndarray:
    """Solve linearised GE equilibrium for endogenous unknowns.

    Given an equilibrium condition  H(U, Z) = 0  linearised as

        H_U @ dU + H_Z @ dZ = 0

    solves for dU = -H_U^{-1} H_Z dZ.

    Parameters
    ----------
    H_U : ndarray, shape (T, T)
        Jacobian of equilibrium target w.r.t. unknowns (e.g. dK).
    H_Z : ndarray, shape (T, T)
        Jacobian of equilibrium target w.r.t. exogenous shock (e.g. dZ).
    dZ : ndarray, shape (T,)
        Exogenous shock path.

    Returns
    -------
    dU : ndarray, shape (T,)
        Endogenous response.

    Notes
    -----
    For a single-unknown SHADE model, the equilibrium condition is
    asset market clearing:

        H(K, Z) = K_supply(K, Z) - K = 0

    H_U and H_Z are constructed from partial-equilibrium Jacobians
    (from ``jacobian``) and the firm-side price mappings.
    """
    return np.linalg.solve(H_U, -H_Z @ dZ)


def build_ge_matrices(
    Js_A: Dict[str, np.ndarray],
    price_elasticities_K: Dict[str, np.ndarray],
    price_elasticities_Z: Dict[str, np.ndarray],
    K_ss: float,
    T: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build H_U, H_Z matrices for a single-unknown GE model.

    Parameters
    ----------
    Js_A : dict
        Partial-equilibrium Jacobians of aggregate assets w.r.t. each
        price/input.  ``Js_A[name]`` is a (T, T) matrix giving
        dK_t / d(price_name)_s.
    price_elasticities_K : dict
        How each price responds to dK (log-deviation).
        ``price_elasticities_K[name]`` is a (T, T) matrix such that
        d(price) = price_elast_K @ dlogK.
    price_elasticities_Z : dict
        How each price responds to dlogZ.
        ``price_elast_Z[name]`` is a (T, T) matrix or (T,) vector.
    K_ss : float
        Steady-state capital.
    T : int
        Truncation horizon.

    Returns
    -------
    H_U : ndarray, shape (T, T)
        Jacobian of excess demand w.r.t. dlogK.
    H_Z : ndarray, shape (T, T)
        Jacobian of excess demand w.r.t. dlogZ.
    """
    H_U = K_ss * np.eye(T)
    H_Z = np.zeros((T, T))

    for name in Js_A:
        J = Js_A[name]
        GK = price_elasticities_K[name]
        GZ = price_elasticities_Z[name]

        H_U -= J @ GK
        if GZ.ndim == 1:
            H_Z -= J * GZ[np.newaxis, :]  # broadcast: each column scaled
        else:
            H_Z -= J @ GZ

    return H_U, H_Z

