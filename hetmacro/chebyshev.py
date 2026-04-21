"""Chebyshev 2D tensor-product projection for law of motion estimation.

Extends the 1D Chebyshev tools in ``interpolation.py`` to 2D tensor products
used by the Krusell-Smith algorithm to approximate the capital law of motion
Gamma(K, Z) -> K'.

Typical usage::

    from hetmacro.chebyshev import fit_law_of_motion, eval_law_of_motion

    ck, Rsq = fit_law_of_motion(K_path[:-1], Z_path[:-1], K_path[1:],
                                 nK=3, nZ=3, K_bounds=(Kmin, Kmax),
                                 Z_bounds=(zmin, zmax))
    K_next = eval_law_of_motion(ck, K_new, Z_new, nK=3, nZ=3,
                                 K_bounds=(Kmin, Kmax), Z_bounds=(zmin, zmax))
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .interpolation import cheb_basis


# ── 2D tensor-product basis ──────────────────────────────────────────────


def cheb_basis_2d(
    K: np.ndarray,
    Z: np.ndarray,
    nK: int,
    nZ: int,
    K_bounds: Tuple[float, float],
    Z_bounds: Tuple[float, float],
) -> np.ndarray:
    """Evaluate 2D tensor-product Chebyshev basis at points (K, Z).

    Parameters
    ----------
    K, Z : ndarray, shape (N,)
        Evaluation points.
    nK, nZ : int
        Number of Chebyshev basis functions per dimension.
    K_bounds, Z_bounds : (float, float)
        Domain bounds for each dimension.

    Returns
    -------
    Phi : ndarray, shape (N, nK * nZ)
        Basis matrix.  Column ordering: K varies fastest (matches
        Matlab CompEcon ``fundef`` convention).
    """
    K = np.asarray(K, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()
    assert K.size == Z.size

    BK = cheb_basis(K, nK, K_bounds[0], K_bounds[1])  # (N, nK)
    BZ = cheb_basis(Z, nZ, Z_bounds[0], Z_bounds[1])  # (N, nZ)

    # Tensor product: for each (iz, ik) pair, column = BK[:, ik] * BZ[:, iz]
    # K varies fastest (Fortran order).
    Phi = np.empty((K.size, nK * nZ))
    col = 0
    for iz in range(nZ):
        for ik in range(nK):
            Phi[:, col] = BK[:, ik] * BZ[:, iz]
            col += 1
    return Phi


# ── Law of motion fitting ────────────────────────────────────────────────


def fit_law_of_motion(
    K: np.ndarray,
    Z: np.ndarray,
    K_next: np.ndarray,
    nK: int = 3,
    nZ: int = 3,
    K_bounds: Tuple[float, float] = None,
    Z_bounds: Tuple[float, float] = None,
) -> Tuple[np.ndarray, float]:
    """Fit K' = Gamma(K, Z) by OLS on a 2D Chebyshev basis.

    Parameters
    ----------
    K, Z : ndarray, shape (T,)
        Current period aggregate capital (log deviation) and TFP.
    K_next : ndarray, shape (T,)
        Next period aggregate capital (log deviation).
    nK, nZ : int
        Number of Chebyshev basis functions per dimension.
    K_bounds, Z_bounds : (float, float) or None
        Domain bounds.  If None, uses (min, max) of the data.

    Returns
    -------
    ck : ndarray, shape (nK * nZ,)
        Chebyshev coefficients for the law of motion.
    Rsq : float
        R-squared of the fit.
    """
    K = np.asarray(K, dtype=float).ravel()
    Z = np.asarray(Z, dtype=float).ravel()
    K_next = np.asarray(K_next, dtype=float).ravel()

    if K_bounds is None:
        K_bounds = (K.min(), K.max())
    if Z_bounds is None:
        Z_bounds = (Z.min(), Z.max())

    Phi = cheb_basis_2d(K, Z, nK, nZ, K_bounds, Z_bounds)
    ck, *_ = np.linalg.lstsq(Phi, K_next, rcond=None)

    K_hat = Phi @ ck
    ss_res = np.mean((K_next - K_hat) ** 2)
    ss_tot = np.var(K_next)
    Rsq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return ck, Rsq


def eval_law_of_motion(
    ck: np.ndarray,
    K: np.ndarray,
    Z: np.ndarray,
    nK: int = 3,
    nZ: int = 3,
    K_bounds: Tuple[float, float] = (None, None),
    Z_bounds: Tuple[float, float] = (None, None),
) -> np.ndarray:
    """Evaluate K' = Gamma(K, Z) given Chebyshev coefficients.

    Parameters
    ----------
    ck : ndarray, shape (nK * nZ,)
        Chebyshev coefficients from ``fit_law_of_motion``.
    K, Z : ndarray
        Evaluation points.
    nK, nZ : int
        Number of Chebyshev basis functions per dimension.
    K_bounds, Z_bounds : (float, float)
        Domain bounds (must match those used in fitting).

    Returns
    -------
    K_next : ndarray
        Predicted next period capital.
    """
    Phi = cheb_basis_2d(K, Z, nK, nZ, K_bounds, Z_bounds)
    return Phi @ ck
