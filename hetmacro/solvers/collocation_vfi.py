"""Coefficient-space collocation VFI solvers (spline / Chebyshev)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline as _BSpline
from scipy.sparse.linalg import factorized, spsolve

from ..grids import make_grid_1d
from ..household import SolvedPolicy
from ..income_process import ContinuousQuadratureIncome, DiscreteIncome
from ..interpolation import (
    FunctionSpace,
    cheb_basis,
    cheb_nodes,
    cubic_basis_matrix,
    greville_abscissae,
    tensor_basis_matrix,
)
from ..optimize import golden_max_vec
from ..quadrature import qnwnorm
from ..utils import crra_utility


class _CollocationEngine:
    """Shared coefficient-space collocation infrastructure.

    Supports two branches:
      - DiscreteIncome: 1D spline/Cheb basis in *a* only; theta shape (n_basis_a, n_z).
        Expectations use the Pi transition matrix.
      - ContinuousQuadratureIncome: 2D tensor-product basis over (a, z);
        theta is a flat vector of length n_basis_a * n_basis_z.
        Expectations use Gauss-Hermite quadrature.
    """

    def __init__(
        self,
        household,
        basis: str = "spline",
        n_a: int = 51,
        n_z: int = 11,
        a_power: float = 3.0,
        c_floor: float = 1e-8,
        policy_tol: float = 1e-8,
    ):
        self.household = household
        self.basis_type = basis
        self.c_floor = c_floor
        self.policy_tol = policy_tol

        self.beta = household.beta
        self.gamma = household.gamma
        self.r = household.r
        self.w = household.w
        self.amin = float(household.a_grid[0])
        self.amax = float(household.a_grid[-1])

        income = household.income_process
        self.discrete = isinstance(income, DiscreteIncome)

        if self.discrete:
            self._init_discrete(income, n_a, a_power)
        else:
            self._init_continuous(income, n_a, n_z, a_power)

    # ------------------------------------------------------------------
    # Fast B-spline evaluation (avoids rebuilding sparse matrices)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_knots(breaks):
        """Open knot vector for degree-3 B-splines (same as _bspline_knots)."""
        bp = np.asarray(breaks)
        return np.concatenate([np.repeat(bp[0], 4), bp[1:-1], np.repeat(bp[-1], 4)])

    def _eval_bspline_a(self, x, coeffs):
        """Evaluate B-spline(s) with given coefficients at points *x*.

        Equivalent to ``cubic_basis_matrix(self.a_breaks, x).toarray() @ coeffs``
        but avoids building or densifying the sparse matrix.

        Parameters
        ----------
        x : (n,) array
            Evaluation points (will be clipped to [amin, amax]).
        coeffs : (n_basis_a,) or (n_basis_a, m) array
            B-spline coefficients.  If 2-D, returns (n, m) array.
        """
        x_clip = np.clip(np.asarray(x), self.amin, self.amax)
        return _BSpline(self._knots_a, coeffs, 3, extrapolate=False)(x_clip)

    # ------------------------------------------------------------------
    # Discrete income branch: 1D basis in *a*, theta shape (n_basis_a, n_z)
    # ------------------------------------------------------------------

    def _init_discrete(self, income: DiscreteIncome, n_a: int, a_power: float):
        self.Pi = income.Pi
        self.z_vals = np.asarray(income.z_grid)
        self.n_z = self.z_vals.size

        if self.basis_type == "spline":
            ua = np.linspace(0.0, 1.0, n_a)
            self.a_breaks = self.amin + (self.amax - self.amin) * ua ** a_power
            self._knots_a = self._make_knots(self.a_breaks)
            self.a_nodes = greville_abscissae(self.a_breaks, degree=3)
            self.n_basis_a = self.a_nodes.size
            Phi_a = cubic_basis_matrix(self.a_breaks, self.a_nodes)
            self._Phi_a_nodes = Phi_a.toarray()
            self._phi_a_solver = factorized(Phi_a.tocsc())
        else:
            self.n_basis_a = n_a
            self.a_nodes = cheb_nodes(n_a, self.amin, self.amax)
            self._Phi_a_nodes = cheb_basis(self.a_nodes, n_a, self.amin, self.amax)
            self._phi_a_solver = None

        self.n_states = self.n_basis_a * self.n_z

        A_flat = np.repeat(self.a_nodes, self.n_z)
        Z_flat = np.tile(self.z_vals, self.n_basis_a)
        self._A_nodes = A_flat
        self._Z_nodes = Z_flat
        self._y_nodes = self.w * Z_flat

    def _solve_theta_col_discrete(self, v_col: np.ndarray) -> np.ndarray:
        """Invert basis system for one z-column: Phi_a @ theta_col = v_col."""
        if self._phi_a_solver is not None:
            return self._phi_a_solver(v_col)
        return np.linalg.solve(self._Phi_a_nodes, v_col)

    def _expected_continuation_discrete(self, a_prime_mat, theta):
        """E[V(a', z') | z_j] = sum_k Pi[j,k] * Phi_a(a') @ theta[:,k].

        Parameters
        ----------
        a_prime_mat : (n_basis_a, n_z)
        theta       : (n_basis_a, n_z)

        Returns
        -------
        (n_basis_a, n_z) expected continuation values
        """
        n_a, n_z = a_prime_mat.shape
        out = np.zeros((n_a, n_z))
        for j in range(n_z):
            a_p = np.clip(a_prime_mat[:, j], self.amin, self.amax)
            if self.basis_type == "spline":
                theta_Ev = theta @ self.Pi[j, :]
                out[:, j] = self._eval_bspline_a(a_p, theta_Ev)
            else:
                Ba = cheb_basis(a_p, self.n_basis_a, self.amin, self.amax)
                out[:, j] = Ba @ theta @ self.Pi[j, :]
        return out

    # ------------------------------------------------------------------
    # Continuous income branch: 2D tensor-product basis, flat theta
    # ------------------------------------------------------------------

    def _init_continuous(self, income: ContinuousQuadratureIncome, n_a, n_z, a_power):
        self.rho = income.rho
        self.sigma_eps = income.sigma_eps
        self.mu_z = income.mu

        self.z_std = income.sigma_eps / np.sqrt(1.0 - income.rho ** 2)
        self.zmin = self.mu_z - 4.0 * self.z_std
        self.zmax = self.mu_z + 4.0 * self.z_std

        self.eps_nodes, self.eps_w = qnwnorm(
            income.n_quad, mu=0.0, sigma=income.sigma_eps
        )
        self.n_quad = income.n_quad

        if self.basis_type == "spline":
            ua = np.linspace(0.0, 1.0, n_a)
            self.a_breaks = self.amin + (self.amax - self.amin) * ua ** a_power
            self.z_breaks = np.linspace(self.zmin, self.zmax, n_z)
            self._knots_a = self._make_knots(self.a_breaks)
            self._knots_z = self._make_knots(self.z_breaks)

            self.a_nodes = greville_abscissae(self.a_breaks, degree=3)
            self.z_nodes = greville_abscissae(self.z_breaks, degree=3)
            self.n_basis_a = self.a_nodes.size
            self.n_basis_z = self.z_nodes.size
        else:
            self.n_basis_a = n_a
            self.n_basis_z = n_z
            self.a_nodes = cheb_nodes(n_a, self.amin, self.amax)
            self.z_nodes = cheb_nodes(n_z, self.zmin, self.zmax)

        self.n_states = self.n_basis_a * self.n_basis_z

        self._A_nodes = np.repeat(self.a_nodes, self.n_basis_z)
        self._Z_nodes = np.tile(self.z_nodes, self.n_basis_a)

        # z-fast to a-fast permutation (for tensor_basis_matrix column ordering)
        self._perm_z_to_a = (
            np.arange(self.n_states)
            .reshape(self.n_basis_a, self.n_basis_z, order="C")
            .ravel(order="F")
        )
        self._perm_a_to_z = np.argsort(self._perm_z_to_a)

        if self.basis_type == "spline":
            Phi_a_nodes = cubic_basis_matrix(self.a_breaks, self._A_nodes)
            Phi_z_nodes = cubic_basis_matrix(self.z_breaks, self._Z_nodes)
            self.Phi = tensor_basis_matrix(Phi_a_nodes, Phi_z_nodes).tocsr()
            self._phi_solver = factorized(self.Phi.tocsc())

            self._z_basis_cache = [
                cubic_basis_matrix(
                    self.z_breaks,
                    np.clip(self.rho * self._Z_nodes + eps, self.zmin, self.zmax),
                ).toarray()
                for eps in self.eps_nodes
            ]
            self._Phi_a_nodes_2d = Phi_a_nodes
        else:
            Ba = cheb_basis(self._A_nodes, self.n_basis_a, self.amin, self.amax)
            Bz = cheb_basis(self._Z_nodes, self.n_basis_z, self.zmin, self.zmax)
            self.Phi = np.column_stack(
                [Ba[:, ia] * Bz[:, iz] for iz in range(self.n_basis_z) for ia in range(self.n_basis_a)]
            )
            self._phi_solver = None

            self._z_basis_cache = [
                cheb_basis(
                    np.clip(self.rho * self._Z_nodes + eps, self.zmin, self.zmax),
                    self.n_basis_z,
                    self.zmin,
                    self.zmax,
                )
                for eps in self.eps_nodes
            ]

    def _theta_to_mat(self, theta):
        return np.asarray(theta).reshape(self.n_basis_a, self.n_basis_z, order="F")

    def _solve_theta_continuous(self, v_nodes):
        if self._phi_solver is not None:
            return self._phi_solver(v_nodes)
        return np.linalg.solve(self.Phi, v_nodes)

    def _expected_continuation_continuous(self, a_prime, theta_mat):
        a_prime = np.clip(a_prime, self.amin, self.amax)
        if self.basis_type == "spline":
            BATheta = self._eval_bspline_a(a_prime, theta_mat)
        else:
            B_a = cheb_basis(a_prime, self.n_basis_a, self.amin, self.amax)
            BATheta = B_a @ theta_mat
        out = np.zeros_like(a_prime)
        for w, B_z in zip(self.eps_w, self._z_basis_cache):
            out += w * np.sum(BATheta * B_z, axis=1)
        return out

    # ------------------------------------------------------------------
    # Unified interface
    # ------------------------------------------------------------------

    def _resources(self, a, z):
        if self.discrete:
            return self.w * z + (1.0 + self.r) * a
        return np.exp(z) + (1.0 + self.r) * a

    def _u(self, c):
        c = np.maximum(np.asarray(c), self.c_floor)
        return crra_utility(c, self.gamma)

    def policy_improvement_nodes(self, theta):
        """Golden-section maximisation at collocation nodes.

        Returns (a_pol, c_pol, v_nodes) all as flat arrays of length n_states.
        For discrete branch, a_pol / c_pol are also available reshaped as
        (n_basis_a, n_z) via the caller.
        """
        if self.discrete:
            return self._policy_improvement_discrete(theta)
        return self._policy_improvement_continuous(theta)

    def _policy_improvement_discrete(self, theta):
        """theta shape: (n_basis_a, n_z)."""
        n_a = self.n_basis_a
        n_z = self.n_z

        a_pol = np.empty((n_a, n_z))
        c_pol = np.empty((n_a, n_z))
        v_nodes = np.empty((n_a, n_z))

        for j in range(n_z):
            m = self._resources(self.a_nodes, self.z_vals[j])
            lo = np.full(n_a, self.amin)
            hi = np.clip(m - self.c_floor, self.amin, self.amax)
            hi = np.maximum(hi, lo + 1e-12)

            if self.basis_type == "spline":
                theta_Ev_j = theta @ self.Pi[j, :]

                def obj(ap, _m=m, _tEv=theta_Ev_j):
                    ap = np.clip(ap, self.amin, self.amax)
                    c = np.maximum(_m - ap, self.c_floor)
                    Ev = self._eval_bspline_a(ap, _tEv)
                    return self._u(c) + self.beta * Ev
            else:
                def obj(ap, _j=j, _m=m, _theta=theta):
                    ap = np.clip(ap, self.amin, self.amax)
                    c = np.maximum(_m - ap, self.c_floor)
                    Ba = cheb_basis(ap, self.n_basis_a, self.amin, self.amax)
                    Ev = Ba @ _theta @ self.Pi[_j, :]
                    return self._u(c) + self.beta * Ev

            ap_j, vn_j = golden_max_vec(obj, lo, hi, tol=self.policy_tol)
            a_pol[:, j] = ap_j
            c_pol[:, j] = np.maximum(m - ap_j, self.c_floor)
            v_nodes[:, j] = vn_j

        return a_pol, c_pol, v_nodes

    def _policy_improvement_continuous(self, theta):
        """theta is a flat vector of length n_states."""
        theta_mat = self._theta_to_mat(theta)
        m = self._resources(self._A_nodes, self._Z_nodes)
        lo = np.full(self.n_states, self.amin)
        hi = np.clip(m - self.c_floor, self.amin, self.amax)
        hi = np.maximum(hi, lo + 1e-12)

        def obj(ap):
            ap = np.clip(ap, self.amin, self.amax)
            c = np.maximum(m - ap, self.c_floor)
            cont = self._expected_continuation_continuous(ap, theta_mat)
            return self._u(c) + self.beta * cont

        a_pol, v_nodes = golden_max_vec(obj, lo, hi, tol=self.policy_tol)
        c_pol = np.maximum(m - a_pol, self.c_floor)
        return a_pol, c_pol, v_nodes

    def vfi_step(self, theta):
        """One coefficient-space VFI step.

        Returns (theta_new, err, a_pol, c_pol).
        """
        if self.discrete:
            a_pol, c_pol, v_nodes = self._policy_improvement_discrete(theta)
            theta_new = np.empty_like(theta)
            for j in range(self.n_z):
                theta_new[:, j] = self._solve_theta_col_discrete(v_nodes[:, j])
            err = float(np.max(np.abs(theta_new - theta)))
            return theta_new, err, a_pol, c_pol
        else:
            a_pol, c_pol, v_nodes = self._policy_improvement_continuous(theta)
            theta_new = self._solve_theta_continuous(v_nodes)
            err = float(np.max(np.abs(theta_new - theta)))
            return theta_new, err, a_pol, c_pol

    def initial_theta(self):
        if self.discrete:
            return np.zeros((self.n_basis_a, self.n_z))
        return np.zeros(self.n_states)

    def evaluate_on_grid(self, theta, a_eval, z_eval) -> SolvedPolicy:
        """Re-optimise using converged theta on the household's evaluation grids.

        Returns SolvedPolicy with arrays shaped (n_z_eval, n_a_eval).
        """
        a_eval = np.asarray(a_eval)
        z_eval = np.asarray(z_eval)
        na, nz = a_eval.size, z_eval.size

        policy_a = np.empty((nz, na))
        policy_c = np.empty((nz, na))
        value = np.empty((nz, na))

        if self.discrete:
            for j in range(nz):
                m = self._resources(a_eval, z_eval[j])
                lo = np.full(na, self.amin)
                hi = np.clip(m - self.c_floor, self.amin, self.amax)
                hi = np.maximum(hi, lo + 1e-12)

                if self.basis_type == "spline":
                    theta_Ev_j = theta @ self.Pi[j, :]

                    def obj(ap, _m=m, _tEv=theta_Ev_j):
                        ap = np.clip(ap, self.amin, self.amax)
                        c = np.maximum(_m - ap, self.c_floor)
                        Ev = self._eval_bspline_a(ap, _tEv)
                        return self._u(c) + self.beta * Ev
                else:
                    def obj(ap, _j=j, _m=m, _theta=theta):
                        ap = np.clip(ap, self.amin, self.amax)
                        c = np.maximum(_m - ap, self.c_floor)
                        Ba = cheb_basis(ap, self.n_basis_a, self.amin, self.amax)
                        Ev = Ba @ _theta @ self.Pi[_j, :]
                        return self._u(c) + self.beta * Ev

                ap, vv = golden_max_vec(obj, lo, hi, tol=self.policy_tol)
                policy_a[j] = ap
                policy_c[j] = np.maximum(m - ap, self.c_floor)
                value[j] = vv
        else:
            theta_mat = self._theta_to_mat(theta)
            A_flat = np.repeat(a_eval, nz)
            Z_flat = np.tile(z_eval, na)

            m = self._resources(A_flat, Z_flat)
            lo = np.full(A_flat.size, self.amin)
            hi = np.clip(m - self.c_floor, self.amin, self.amax)
            hi = np.maximum(hi, lo + 1e-12)

            # Precompute z-basis outside golden search (does not depend on ap)
            z_basis_eval = []
            for eps, wq in zip(self.eps_nodes, self.eps_w):
                z_next = np.clip(self.rho * Z_flat + eps, self.zmin, self.zmax)
                if self.basis_type == "spline":
                    Bz = _BSpline(self._knots_z, np.eye(self.n_basis_z), 3,
                                  extrapolate=False)(z_next)
                else:
                    Bz = cheb_basis(z_next, self.n_basis_z, self.zmin, self.zmax)
                z_basis_eval.append((wq, Bz))

            if self.basis_type == "spline":
                def obj(ap):
                    ap = np.clip(ap, self.amin, self.amax)
                    c = np.maximum(m - ap, self.c_floor)
                    BATheta = self._eval_bspline_a(ap, theta_mat)
                    cont = np.zeros_like(ap)
                    for wq, Bz in z_basis_eval:
                        cont += wq * np.sum(BATheta * Bz, axis=1)
                    return self._u(c) + self.beta * cont
            else:
                def obj(ap):
                    ap = np.clip(ap, self.amin, self.amax)
                    c = np.maximum(m - ap, self.c_floor)
                    Ba = cheb_basis(ap, self.n_basis_a, self.amin, self.amax)
                    BATheta = Ba @ theta_mat
                    cont = np.zeros_like(ap)
                    for wq, Bz in z_basis_eval:
                        cont += wq * np.sum(BATheta * Bz, axis=1)
                    return self._u(c) + self.beta * cont

            ap, vv = golden_max_vec(obj, lo, hi, tol=self.policy_tol)
            policy_a = ap.reshape(na, nz).T
            policy_c = np.maximum(m - ap, self.c_floor).reshape(na, nz).T
            value = vv.reshape(na, nz).T

        return SolvedPolicy(
            policy_a=policy_a,
            policy_c=policy_c,
            a_grid=a_eval,
            z_grid=z_eval,
            value=value,
        )

    # ------------------------------------------------------------------
    # Howard policy evaluation (used by HowardImprovement)
    # ------------------------------------------------------------------

    def howard_evaluate(self, a_pol, theta):
        """Solve (Phi - beta * V_theta) @ theta = u_pol for new theta.

        Returns (theta_new, c_pol).
        """
        if self.discrete:
            return self._howard_evaluate_discrete(a_pol, theta)
        return self._howard_evaluate_continuous(a_pol)

    def _howard_evaluate_discrete(self, a_pol, theta):
        """a_pol shape (n_basis_a, n_z), theta shape (n_basis_a, n_z).

        Build block-linear system and solve for theta.
        """
        n_a = self.n_basis_a
        n_z = self.n_z
        N = n_a * n_z

        u_vec = np.empty(N)
        Phi_block = np.zeros((N, N))
        V_theta = np.zeros((N, N))

        for j in range(n_z):
            row_s = slice(j * n_a, (j + 1) * n_a)

            # Flow utility
            m = self._resources(self.a_nodes, self.z_vals[j])
            c = np.maximum(m - a_pol[:, j], self.c_floor)
            u_vec[row_s] = self._u(c)

            # Phi block diagonal
            Phi_block[row_s, row_s] = self._Phi_a_nodes

            # Basis at next-period assets
            ap_j = np.clip(a_pol[:, j], self.amin, self.amax)
            if self.basis_type == "spline":
                Ba_next = _BSpline.design_matrix(
                    ap_j, self._knots_a, 3).toarray()
            else:
                Ba_next = cheb_basis(ap_j, self.n_basis_a, self.amin, self.amax)

            for k in range(n_z):
                col_s = slice(k * n_a, (k + 1) * n_a)
                V_theta[row_s, col_s] = self.Pi[j, k] * Ba_next

        lhs = Phi_block - self.beta * V_theta
        theta_flat = np.linalg.solve(lhs, u_vec)
        theta_new = theta_flat.reshape(n_z, n_a).T

        c_pol = np.empty((n_a, n_z))
        for j in range(n_z):
            m = self._resources(self.a_nodes, self.z_vals[j])
            c_pol[:, j] = np.maximum(m - a_pol[:, j], self.c_floor)

        return theta_new, c_pol

    def _howard_evaluate_continuous(self, a_pol):
        """a_pol is flat of length n_states."""
        a_pol = np.clip(a_pol, self.amin, self.amax)
        m = self._resources(self._A_nodes, self._Z_nodes)
        c_pol = np.maximum(m - a_pol, self.c_floor)
        u_pol = self._u(c_pol)

        if self.basis_type == "spline":
            Phi_a_next = cubic_basis_matrix(self.a_breaks, a_pol)
            vtheta = sparse.csr_matrix((self.n_states, self.n_states))
            for idx, (eps, wq) in enumerate(zip(self.eps_nodes, self.eps_w)):
                # Reuse cached z-basis (evaluated at self._Z_nodes)
                Bz_dense = self._z_basis_cache[idx]
                Phi_z_next = sparse.csr_matrix(Bz_dense)
                Phi_next = tensor_basis_matrix(Phi_a_next, Phi_z_next).tocsr()
                vtheta = vtheta + self.beta * wq * Phi_next
            lhs = self.Phi - vtheta
            theta_new = spsolve(lhs.tocsc(), u_pol)
        else:
            Ba_next = cheb_basis(a_pol, self.n_basis_a, self.amin, self.amax)
            vtheta = np.zeros((self.n_states, self.n_states))
            for eps, wq in zip(self.eps_nodes, self.eps_w):
                z_next = np.clip(self.rho * self._Z_nodes + eps, self.zmin, self.zmax)
                Bz_next = cheb_basis(z_next, self.n_basis_z, self.zmin, self.zmax)
                Phi_next = np.column_stack(
                    [Ba_next[:, ia] * Bz_next[:, iz]
                     for iz in range(self.n_basis_z)
                     for ia in range(self.n_basis_a)]
                )
                vtheta += self.beta * wq * Phi_next
            lhs = self.Phi - vtheta
            theta_new = np.linalg.solve(lhs, u_pol)

        return theta_new, c_pol


@dataclass
class CollocationVFI_Spline:
    """True coefficient-space VFI with cubic B-spline basis.

    For DiscreteIncome: 1D basis in assets, theta shape (n_basis_a, n_z).
    For ContinuousQuadratureIncome: 2D tensor-product basis.

    Parameters
    ----------
    n_a_breaks : int
        Number of breakpoints for the asset spline.
    n_z_breaks : int
        Number of breakpoints for z (continuous branch only).
    a_power : float
        Power spacing for asset breakpoints.
    tol : float
        Convergence tolerance on max|theta_new - theta_old|.
    max_iter : int
        Maximum VFI iterations.
    policy_tol : float
        Tolerance for golden-section policy improvement.
    verbose : bool
        Print iteration diagnostics.
    """

    n_a_breaks: int = 51
    n_z_breaks: int = 11
    a_power: float = 3.0
    tol: float = 1e-8
    max_iter: int = 2_000
    policy_tol: float = 1e-8
    verbose: bool = True

    def solve(self, household, **kwargs) -> SolvedPolicy:
        tol = kwargs.get("tol", self.tol)
        max_iter = kwargs.get("max_iter", self.max_iter)

        engine = _CollocationEngine(
            household,
            basis="spline",
            n_a=self.n_a_breaks,
            n_z=self.n_z_breaks,
            a_power=self.a_power,
            policy_tol=self.policy_tol,
        )
        theta = engine.initial_theta()

        for it in range(1, max_iter + 1):
            theta_new, err, a_pol, c_pol = engine.vfi_step(theta)
            if self.verbose and (it <= 10 or it % 50 == 0):
                print(f"CollocationVFI_Spline iter {it:4d} | err = {err:.3e}")
            if err < tol:
                if self.verbose:
                    print(f"CollocationVFI_Spline converged at iter {it} (err={err:.3e})")
                break
            theta = theta_new
        else:
            if self.verbose:
                print(f"CollocationVFI_Spline reached max_iter={max_iter} (err={err:.3e})")
            theta = theta_new

        return engine.evaluate_on_grid(theta, household.a_grid, household.z_grid)


@dataclass
class CollocationVFI_Chebyshev:
    """True coefficient-space VFI with Chebyshev polynomial basis.

    Parameters
    ----------
    n_basis_a : int
        Number of Chebyshev basis functions / nodes in assets.
    n_basis_z : int
        Number of Chebyshev basis functions / nodes in z (continuous branch only).
    a_power : float
        Not used by Chebyshev (nodes are Chebyshev points), kept for API parity.
    tol : float
        Convergence tolerance on max|theta_new - theta_old|.
    max_iter : int
        Maximum VFI iterations.
    policy_tol : float
        Tolerance for golden-section policy improvement.
    verbose : bool
        Print iteration diagnostics.
    """

    n_basis_a: int = 25
    n_basis_z: int = 9
    a_power: float = 3.0
    tol: float = 1e-8
    max_iter: int = 2_000
    policy_tol: float = 1e-8
    verbose: bool = True

    def solve(self, household, **kwargs) -> SolvedPolicy:
        tol = kwargs.get("tol", self.tol)
        max_iter = kwargs.get("max_iter", self.max_iter)

        engine = _CollocationEngine(
            household,
            basis="chebyshev",
            n_a=self.n_basis_a,
            n_z=self.n_basis_z,
            a_power=self.a_power,
            policy_tol=self.policy_tol,
        )
        theta = engine.initial_theta()

        for it in range(1, max_iter + 1):
            theta_new, err, a_pol, c_pol = engine.vfi_step(theta)
            if self.verbose and (it <= 10 or it % 50 == 0):
                print(f"CollocationVFI_Chebyshev iter {it:4d} | err = {err:.3e}")
            if err < tol:
                if self.verbose:
                    print(f"CollocationVFI_Chebyshev converged at iter {it} (err={err:.3e})")
                break
            theta = theta_new
        else:
            if self.verbose:
                print(f"CollocationVFI_Chebyshev reached max_iter={max_iter} (err={err:.3e})")
            theta = theta_new

        return engine.evaluate_on_grid(theta, household.a_grid, household.z_grid)
