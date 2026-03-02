"""PSet 2 - Problem 4: Life-cycle consumption-savings with unemployment risk."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


@dataclass
class Params:
    beta: float = 0.95
    sigma: float = 0.10  # E->U
    lamb: float = 0.25  # U->E
    b: float = 0.25
    r: float = 0.04
    T: int = 50
    na: int = 501
    amin: float = 0.0
    amax: float = 10.0


def _argmax_bounded(fun, lo: float, hi: float) -> float:
    res = minimize_scalar(lambda x: -fun(x), bounds=(lo, hi), method="bounded")
    return float(res.x)


def solve_lifecycle(p: Params):
    a = p.amin + (p.amax - p.amin) * np.linspace(0.0, 1.0, p.na) ** 2

    Ve = np.zeros((p.T, p.na))
    Vu = np.zeros((p.T, p.na))
    Ce = np.zeros((p.T, p.na))
    Cu = np.zeros((p.T, p.na))
    Aprime_e = np.zeros((p.T, p.na))
    Aprime_u = np.zeros((p.T, p.na))

    # Terminal values
    Ce[-1, :] = 1.0 + (1.0 + p.r) * a
    Cu[-1, :] = p.b + (1.0 + p.r) * a
    Ve[-1, :] = np.log(Ce[-1, :])
    Vu[-1, :] = np.log(Cu[-1, :])

    for t in range(p.T - 2, -1, -1):
        Ve_next_spline = CubicSpline(a, Ve[t + 1, :], bc_type="natural", extrapolate=True)
        Vu_next_spline = CubicSpline(a, Vu[t + 1, :], bc_type="natural", extrapolate=True)

        for i, ai in enumerate(a):
            m_e = 1.0 + (1.0 + p.r) * ai
            m_u = p.b + (1.0 + p.r) * ai

            # Employed choice
            ce = _argmax_bounded(
                lambda c: np.log(max(c, 1e-14))
                + p.beta
                * (
                    (1.0 - p.sigma) * Ve_next_spline(max((1.0 + p.r) * ai + 1.0 - c, p.amin))
                    + p.sigma * Vu_next_spline(max((1.0 + p.r) * ai + 1.0 - c, p.amin))
                ),
                1e-12,
                m_e - 1e-12,
            )
            ap_e = max((1.0 + p.r) * ai + 1.0 - ce, p.amin)
            ve = np.log(max(ce, 1e-14)) + p.beta * ((1.0 - p.sigma) * Ve_next_spline(ap_e) + p.sigma * Vu_next_spline(ap_e))

            # Unemployed choice
            cu = _argmax_bounded(
                lambda c: np.log(max(c, 1e-14))
                + p.beta
                * (
                    p.lamb * Ve_next_spline(max((1.0 + p.r) * ai + p.b - c, p.amin))
                    + (1.0 - p.lamb) * Vu_next_spline(max((1.0 + p.r) * ai + p.b - c, p.amin))
                ),
                1e-12,
                m_u - 1e-12,
            )
            ap_u = max((1.0 + p.r) * ai + p.b - cu, p.amin)
            vu = np.log(max(cu, 1e-14)) + p.beta * (p.lamb * Ve_next_spline(ap_u) + (1.0 - p.lamb) * Vu_next_spline(ap_u))

            Ce[t, i] = ce
            Cu[t, i] = cu
            Ve[t, i] = ve
            Vu[t, i] = vu
            Aprime_e[t, i] = ap_e
            Aprime_u[t, i] = ap_u

    # t=1 policies on fine grid
    aa = np.linspace(p.amin, p.amax, 1000)
    ce_t1 = CubicSpline(a, Ce[0, :], bc_type="natural")(aa)
    cu_t1 = CubicSpline(a, Cu[0, :], bc_type="natural")(aa)
    ap_e_t1 = (1.0 + p.r) * aa + 1.0 - ce_t1
    ap_u_t1 = (1.0 + p.r) * aa + p.b - cu_t1

    # Euler residuals at t=1
    ce_next = CubicSpline(a, Ce[1, :], bc_type="natural")
    cu_next = CubicSpline(a, Cu[1, :], bc_type="natural")
    euler_e = 1.0 / ce_t1 - p.beta * (1.0 + p.r) * ((1.0 - p.sigma) * (1.0 / ce_next(ap_e_t1)) + p.sigma * (1.0 / cu_next(ap_e_t1)))
    euler_u = 1.0 / cu_t1 - p.beta * (1.0 + p.r) * (p.lamb * (1.0 / ce_next(ap_u_t1)) + (1.0 - p.lamb) * (1.0 / cu_next(ap_u_t1)))

    # Simulation
    at = np.zeros(p.T)
    ct = np.zeros(p.T)
    yt = np.zeros(p.T)
    et = np.ones(p.T)
    et[9:15] = 0  # periods 10-15 unemployment

    for t in range(p.T):
        ce_pol = CubicSpline(a, Ce[t, :], bc_type="natural")
        cu_pol = CubicSpline(a, Cu[t, :], bc_type="natural")
        c_now = ce_pol(at[t]) if et[t] == 1 else cu_pol(at[t])
        ct[t] = float(c_now)
        yt[t] = 1.0 * et[t] + p.b * (1.0 - et[t])
        if t < p.T - 1:
            at[t + 1] = (1.0 + p.r) * at[t] + yt[t] - ct[t]

    return {
        "a_grid": a,
        "Ve": Ve,
        "Vu": Vu,
        "Ce": Ce,
        "Cu": Cu,
        "Aprime_e": Aprime_e,
        "Aprime_u": Aprime_u,
        "aa": aa,
        "ce_t1": ce_t1,
        "cu_t1": cu_t1,
        "ap_e_t1": ap_e_t1,
        "ap_u_t1": ap_u_t1,
        "euler_e": euler_e,
        "euler_u": euler_u,
        "sim_assets": at,
        "sim_consumption": ct,
        "sim_income": yt,
        "sim_employment": et,
    }


if __name__ == "__main__":
    out = solve_lifecycle(Params())
    print("Solved lifecycle model.")
    print(f"Max |Euler employed|   : {np.max(np.abs(out['euler_e'])):.2e}")
    print(f"Max |Euler unemployed| : {np.max(np.abs(out['euler_u'])):.2e}")
