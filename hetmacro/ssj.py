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

