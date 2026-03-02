"""Euler-equation-based household solvers (continuous income branch).

This module implements two PSet-4 style methods:
- Naive Euler equation iteration on consumption-policy coefficients.
- Howard-improved Euler iteration using a policy-evaluation step for V_a.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse.linalg import factorized, spsolve

from ..household import SolvedPolicy, resolve_expectation_mode
from ..income_process import ContinuousQuadratureIncome
from ..interpolation import cubic_basis_matrix, greville_abscissae, tensor_basis_matrix
from ..optimize import solve_broyden
from ..utils import crra_marginal


def _cubic_basis_derivative_matrix(breakpoints: np.ndarray, x: np.ndarray):
    """Evaluate derivative of cubic B-spline basis at x."""
    bp = np.asarray(breakpoints, dtype=float)
    x_eval = np.asarray(x, dtype=float).reshape(-1)
    if bp.ndim != 1 or bp.size < 2:
        raise ValueError("breakpoints must be one-dimensional with at least two points.")
    if not np.all(np.diff(bp) > 0):
        raise ValueError("breakpoints must be strictly increasing.")

    degree = 3
    knots = np.concatenate(
        [np.repeat(bp[0], degree + 1), bp[1:-1], np.repeat(bp[-1], degree + 1)]
    )
    n_basis = knots.size - degree - 1

    dense = np.empty((x_eval.size, n_basis), dtype=float)
    eye = np.eye(n_basis)
    for j in range(n_basis):
        dense[:, j] = BSpline(knots, eye[j], degree, extrapolate=True).derivative()(x_eval)
    return sparse.csr_matrix(dense)


class _EulerEngine:
    """Shared computational engine for naive/Howard Euler iteration."""

    def __init__(
        self,
        household,
        n_a: int = 51,
        n_z: int = 15,
        a_power: float = 2.0,
        c_floor: float = 1e-10,
    ):
        income = household.income_process
        if not isinstance(income, ContinuousQuadratureIncome):
            raise NotImplementedError(
                "EulerIteration solvers v1 support ContinuousQuadratureIncome only."
            )

        self.household = household
        self.beta = float(household.beta)
        self.gamma = float(household.gamma)
        self.r = float(household.r)
        self.amin = float(household.a_grid[0])
        self.amax = float(household.a_grid[-1])
        self.c_floor = float(c_floor)

        self.rho = float(income.rho)
        self.mu = float(income.mu)
        self.eps_nodes = np.asarray(income.eps_nodes, dtype=float)
        self.eps_weights = np.asarray(income.eps_weights, dtype=float)

        z_std = income.sigma_eps / np.sqrt(1.0 - income.rho**2)
        self.zmin = self.mu - 4.0 * z_std
        self.zmax = self.mu + 4.0 * z_std

        ua = np.linspace(0.0, 1.0, int(n_a))
        self.a_breaks = self.amin + (self.amax - self.amin) * ua**float(a_power)
        self.z_breaks = np.linspace(self.zmin, self.zmax, int(n_z))

        self.a_nodes = greville_abscissae(self.a_breaks, degree=3)
        self.z_nodes = greville_abscissae(self.z_breaks, degree=3)
        self.n_basis_a = self.a_nodes.size
        self.n_basis_z = self.z_nodes.size
        self.n_states = self.n_basis_a * self.n_basis_z

        self._A_nodes = np.repeat(self.a_nodes, self.n_basis_z)
        self._Z_nodes = np.tile(self.z_nodes, self.n_basis_a)
        self._m_nodes = self._resources(self._A_nodes, self._Z_nodes)

        Phi_a = cubic_basis_matrix(self.a_breaks, self._A_nodes)
        Phi_z = cubic_basis_matrix(self.z_breaks, self._Z_nodes)
        self.Phi = tensor_basis_matrix(Phi_a, Phi_z).tocsr()
        self._phi_solver = factorized(self.Phi.tocsc())
        self._eye_csc = sparse.eye(self.n_states, format="csc")

        dPhi_a = _cubic_basis_derivative_matrix(self.a_breaks, self._A_nodes)
        self.DPhi_da = tensor_basis_matrix(dPhi_a, Phi_z).tocsr()

        self._z_basis_sparse = []
        self._z_basis_dense = []
        for eps in self.eps_nodes:
            z_next = np.clip(
                self.mu + self.rho * (self._Z_nodes - self.mu) + eps, self.zmin, self.zmax
            )
            Bz = cubic_basis_matrix(self.z_breaks, z_next).tocsr()
            self._z_basis_sparse.append(Bz)
            self._z_basis_dense.append(Bz.toarray())

    def _resources(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.exp(z) + (1.0 + self.r) * a

    def fit_theta(self, c_nodes: np.ndarray) -> np.ndarray:
        return self._phi_solver(np.asarray(c_nodes, dtype=float))

    def initial_theta(self) -> np.ndarray:
        c0 = np.maximum(self._m_nodes - self.amin, self.c_floor)
        return self.fit_theta(c0)

    def _theta_to_mat(self, theta: np.ndarray) -> np.ndarray:
        return np.asarray(theta, dtype=float).reshape(self.n_basis_a, self.n_basis_z, order="F")

    def _expected_from_theta(self, a_prime: np.ndarray, theta: np.ndarray) -> np.ndarray:
        a_prime = np.clip(np.asarray(a_prime, dtype=float), self.amin, self.amax)
        B_a = cubic_basis_matrix(self.a_breaks, a_prime).toarray()
        BATheta = B_a @ self._theta_to_mat(theta)
        out = np.zeros_like(a_prime)
        for w, Bz in zip(self.eps_weights, self._z_basis_dense):
            out += w * np.sum(BATheta * Bz, axis=1)
        return out

    def _expected_marginal_from_c_theta(self, a_prime: np.ndarray, theta: np.ndarray) -> np.ndarray:
        a_prime = np.clip(np.asarray(a_prime, dtype=float), self.amin, self.amax)
        B_a = cubic_basis_matrix(self.a_breaks, a_prime).toarray()
        BATheta = B_a @ self._theta_to_mat(theta)
        out = np.zeros_like(a_prime)
        for w, Bz in zip(self.eps_weights, self._z_basis_dense):
            c_next = np.maximum(np.sum(BATheta * Bz, axis=1), self.c_floor)
            out += w * c_next ** (-self.gamma)
        return out

    def naive_euler_residual(self, a_prime: np.ndarray, theta: np.ndarray) -> np.ndarray:
        a_prime = np.clip(np.asarray(a_prime, dtype=float), self.amin, self.amax)
        c = np.maximum(self._m_nodes - a_prime, self.c_floor)
        emuc = self._expected_marginal_from_c_theta(a_prime, theta)
        return crra_marginal(c, self.gamma) - self.beta * (1.0 + self.r) * emuc

    def howard_euler_residual(self, a_prime: np.ndarray, thetaa: np.ndarray) -> np.ndarray:
        a_prime = np.clip(np.asarray(a_prime, dtype=float), self.amin, self.amax)
        c = np.maximum(self._m_nodes - a_prime, self.c_floor)
        eva = self._expected_from_theta(a_prime, thetaa)
        return crra_marginal(c, self.gamma) - self.beta * eva

    @staticmethod
    def vectorized_bisect(f, lo, hi, tol: float = 1e-12, max_iter: int = 80):
        lo = np.asarray(lo, dtype=float).copy()
        hi = np.asarray(hi, dtype=float).copy()
        f_lo = np.asarray(f(lo), dtype=float)
        f_hi = np.asarray(f(hi), dtype=float)
        good = (f_lo <= 0.0) & (f_hi >= 0.0)
        x = np.where(np.abs(f_lo) <= np.abs(f_hi), lo, hi)
        if not np.any(good):
            return x

        l = lo[good].copy()
        h = hi[good].copy()
        fl = f_lo[good].copy()
        for _ in range(max_iter):
            m = 0.5 * (l + h)
            fm = np.asarray(f(m), dtype=float)
            left = fm >= 0.0
            h[left] = m[left]
            l[~left] = m[~left]
            fl[~left] = fm[~left]
            if np.max(h - l) < tol:
                break
        x[good] = 0.5 * (l + h)
        return x

    def _solve_step(self, residual_fn, method: str, x_guess: np.ndarray):
        lo = np.full(self.n_states, self.amin)
        hi = np.clip(self._m_nodes - self.c_floor, self.amin, self.amax)
        hi = np.maximum(hi, lo + 1e-12)

        r_lo = residual_fn(lo)
        constrained = r_lo > 0.0
        lam = np.zeros(self.n_states)
        lam[constrained] = r_lo[constrained]

        a_prime = lo.copy()
        active = (~constrained) & (hi > lo + 1e-12)
        if np.any(active):
            lo_a = lo[active]
            hi_a = hi[active]

            def f_active(xa):
                af = a_prime.copy()
                af[active] = xa
                return residual_fn(af)[active]

            if method == "broyden":
                x0 = np.asarray(x_guess, dtype=float)
                if x0.shape != (self.n_states,):
                    x0 = 0.5 * (lo + hi)
                x0 = np.clip(x0[active], lo_a, hi_a)
                xa = solve_broyden(
                    f_active,
                    x0,
                    a=lo_a,
                    b=hi_a,
                    tol=1e-13,
                    max_iter=20,
                )
            elif method == "bisect":
                r_hi = f_active(hi_a)
                bracket = r_hi >= 0.0
                xa = hi_a.copy()
                if np.any(bracket):
                    lo_b = lo_a[bracket]
                    hi_b = hi_a[bracket]

                    def f_bis(xb):
                        tmp = xa.copy()
                        tmp[bracket] = xb
                        return f_active(tmp)[bracket]

                    xa[bracket] = self.vectorized_bisect(f_bis, lo_b, hi_b, tol=1e-12, max_iter=80)
            else:
                raise ValueError("root_method must be 'broyden' or 'bisect'.")

            a_prime[active] = np.clip(xa, lo_a, hi_a)

        c_new = np.maximum(self._m_nodes - a_prime, self.c_floor)
        return c_new, a_prime, lam

    def solve_euler_step(self, theta: np.ndarray, method: str, x_guess: np.ndarray):
        return self._solve_step(lambda ap: self.naive_euler_residual(ap, theta), method, x_guess)

    def solve_howard_step(self, thetaa: np.ndarray, method: str, x_guess: np.ndarray):
        return self._solve_step(
            lambda ap: self.howard_euler_residual(ap, thetaa), method, x_guess
        )

    def compute_thetaa(self, theta: np.ndarray, a_prime: np.ndarray, lam: np.ndarray) -> np.ndarray:
        c = np.maximum(self.Phi @ theta, self.c_floor)
        c_a = self.DPhi_da @ theta
        a_prime = np.clip(np.asarray(a_prime, dtype=float), self.amin, self.amax)

        B_a_next = cubic_basis_matrix(self.a_breaks, a_prime).tocsr()
        d = np.asarray(1.0 + self.r - c_a, dtype=float)
        lhs = self.Phi.copy().tocsr()
        for w, B_z in zip(self.eps_weights, self._z_basis_sparse):
            phi_next = tensor_basis_matrix(B_a_next, B_z).tocsr()
            lhs = lhs - self.beta * float(w) * phi_next.multiply(d[:, None])

        rhs = crra_marginal(c, self.gamma) * c_a + np.asarray(lam, dtype=float) * d
        lhs_csc = lhs.tocsc()
        try:
            thetaa = spsolve(lhs_csc, rhs)
            if not np.all(np.isfinite(thetaa)):
                raise np.linalg.LinAlgError("non-finite solution")
        except Exception:
            reg = 1e-10
            thetaa = spsolve(lhs_csc + reg * self._eye_csc, rhs)
        return np.asarray(thetaa, dtype=float)

    def evaluate_on_grid(self, theta: np.ndarray, a_eval: np.ndarray, z_eval: np.ndarray) -> SolvedPolicy:
        a_eval = np.asarray(a_eval, dtype=float)
        z_eval = np.asarray(z_eval, dtype=float)
        na, nz = a_eval.size, z_eval.size

        A_flat = np.repeat(a_eval, nz)
        Z_flat = np.tile(z_eval, na)
        m = self._resources(A_flat, Z_flat)

        Phi_a = cubic_basis_matrix(self.a_breaks, np.clip(A_flat, self.amin, self.amax))
        Phi_z = cubic_basis_matrix(self.z_breaks, np.clip(Z_flat, self.zmin, self.zmax))
        Phi_eval = tensor_basis_matrix(Phi_a, Phi_z)
        c_interp = np.asarray(Phi_eval @ theta, dtype=float)
        c_upper = np.maximum(m - self.amin, self.c_floor)
        c_flat = np.clip(c_interp, self.c_floor, c_upper)
        a_flat = np.clip(m - c_flat, self.amin, self.amax)

        return SolvedPolicy(
            policy_a=a_flat.reshape(na, nz).T,
            policy_c=c_flat.reshape(na, nz).T,
            a_grid=a_eval,
            z_grid=z_eval,
            value=None,
        )


@dataclass
class NaiveEulerIteration:
    """Naive Euler-equation iteration with spline-collocation consumption policy."""

    n_a_breaks: int = 51
    n_z_breaks: int = 15
    a_power: float = 2.0
    tol: float = 1e-8
    max_iter: int = 2_000
    root_method: str = "broyden"
    expectation: str = "auto"
    verbose: bool = False

    n_iter_: int = 0
    last_error_: float = np.inf
    constrained_share_: float = np.nan
    timings_: dict = field(default_factory=dict)

    def solve(self, household, **kwargs) -> SolvedPolicy:
        mode = resolve_expectation_mode(household, kwargs.get("expectation", self.expectation))
        if mode != "quadrature":
            raise NotImplementedError(
                "NaiveEulerIteration currently supports ContinuousQuadratureIncome only."
            )

        tol = kwargs.get("tol", self.tol)
        max_iter = kwargs.get("max_iter", self.max_iter)
        method = kwargs.get("root_method", self.root_method)
        verbose = kwargs.get("verbose", self.verbose)

        engine = _EulerEngine(
            household,
            n_a=self.n_a_breaks,
            n_z=self.n_z_breaks,
            a_power=self.a_power,
        )
        theta = engine.initial_theta()
        x = np.full(engine.n_states, engine.amin, dtype=float)
        lam = np.zeros(engine.n_states, dtype=float)
        t_total0 = perf_counter()
        t_solve = 0.0
        t_fit = 0.0

        err = np.inf
        for it in range(1, max_iter + 1):
            t0 = perf_counter()
            theta_old = theta.copy()
            ts0 = perf_counter()
            c_new, x, lam = engine.solve_euler_step(theta, method, x)
            t_solve += perf_counter() - ts0
            tf0 = perf_counter()
            theta = engine.fit_theta(c_new)
            t_fit += perf_counter() - tf0
            err = float(np.max(np.abs(theta - theta_old)))
            if verbose and (it <= 10 or it % 25 == 0):
                dt = perf_counter() - t0
                cshare = float(np.mean(lam > 0.0))
                print(
                    f"NaiveEuler iter {it:4d} | err={err:.3e} | constrained={cshare:.3f} | {dt:.3f}s"
                )
            if err < tol:
                break

        self.n_iter_ = it
        self.last_error_ = err
        self.constrained_share_ = float(np.mean(lam > 0.0))
        self.timings_ = {
            "total_s": perf_counter() - t_total0,
            "solve_euler_step_s": t_solve,
            "fit_theta_s": t_fit,
        }
        return engine.evaluate_on_grid(theta, household.a_grid, household.z_grid)


@dataclass
class HowardEulerIteration:
    """Howard-improved Euler iteration with V_a policy evaluation."""

    n_a_breaks: int = 51
    n_z_breaks: int = 15
    a_power: float = 2.0
    tol: float = 1e-8
    max_iter: int = 500
    n_warmup: int = 10
    root_method: str = "broyden"
    expectation: str = "auto"
    verbose: bool = False

    n_iter_: int = 0
    last_error_: float = np.inf
    constrained_share_: float = np.nan
    timings_: dict = field(default_factory=dict)

    def solve(self, household, **kwargs) -> SolvedPolicy:
        mode = resolve_expectation_mode(household, kwargs.get("expectation", self.expectation))
        if mode != "quadrature":
            raise NotImplementedError(
                "HowardEulerIteration currently supports ContinuousQuadratureIncome only."
            )

        tol = kwargs.get("tol", self.tol)
        max_iter = kwargs.get("max_iter", self.max_iter)
        n_warmup = kwargs.get("n_warmup", self.n_warmup)
        method = kwargs.get("root_method", self.root_method)
        verbose = kwargs.get("verbose", self.verbose)

        engine = _EulerEngine(
            household,
            n_a=self.n_a_breaks,
            n_z=self.n_z_breaks,
            a_power=self.a_power,
        )
        theta = engine.initial_theta()
        x = np.full(engine.n_states, engine.amin, dtype=float)
        lam = np.zeros(engine.n_states, dtype=float)
        t_total0 = perf_counter()
        t_warm = 0.0
        t_main = 0.0
        t_compute = 0.0
        t_step = 0.0
        t_fit = 0.0

        for w in range(1, n_warmup + 1):
            t0 = perf_counter()
            ts0 = perf_counter()
            c_new, x, lam = engine.solve_euler_step(theta, method, x)
            t_step += perf_counter() - ts0
            tf0 = perf_counter()
            theta = engine.fit_theta(c_new)
            t_fit += perf_counter() - tf0
            t_warm += perf_counter() - t0
            if verbose and (w <= 5 or w % 5 == 0):
                dt = perf_counter() - t0
                print(f"HowardEuler warmup {w:3d} | {dt:.3f}s")

        err = np.inf
        for it in range(1, max_iter + 1):
            t0 = perf_counter()
            theta_old = theta.copy()
            tc0 = perf_counter()
            thetaa = engine.compute_thetaa(theta, x, lam)
            t_compute += perf_counter() - tc0
            ts0 = perf_counter()
            c_new, x, lam = engine.solve_howard_step(thetaa, method, x)
            t_step += perf_counter() - ts0
            tf0 = perf_counter()
            theta = engine.fit_theta(c_new)
            t_fit += perf_counter() - tf0
            err = float(np.max(np.abs(theta - theta_old)))
            t_main += perf_counter() - t0
            if verbose and (it <= 10 or it % 10 == 0):
                dt = perf_counter() - t0
                cshare = float(np.mean(lam > 0.0))
                print(
                    f"HowardEuler iter {it:4d} | err={err:.3e} | constrained={cshare:.3f} | {dt:.3f}s"
                )
            if err < tol:
                break

        self.n_iter_ = it
        self.last_error_ = err
        self.constrained_share_ = float(np.mean(lam > 0.0))
        self.timings_ = {
            "total_s": perf_counter() - t_total0,
            "warmup_total_s": t_warm,
            "main_total_s": t_main,
            "compute_thetaa_s": t_compute,
            "solve_step_s": t_step,
            "fit_theta_s": t_fit,
        }
        return engine.evaluate_on_grid(theta, household.a_grid, household.z_grid)
