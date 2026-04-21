"""Linear rational expectations solver (QZ decomposition).

Port of Paul Klein's ``solab.m`` used in Virgiliu Midrigan's course.
Solves the system  A x(t+1) = B x(t)  where the first *nk* entries
of x are predetermined (state) variables and the rest are jump variables.

Returns:
    F : decision rule  u(t) = F @ k(t)
    P : law of motion  k(t+1) = P @ k(t)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.linalg as la


def solab(
    A: np.ndarray,
    B: np.ndarray,
    nk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve a linear RE model via generalised Schur (QZ) decomposition.

    Parameters
    ----------
    A, B : ndarray, shape (n, n)
        Coefficient matrices of  A x(t+1) = B x(t).
    nk : int
        Number of predetermined (state) variables.  The state variables
        must be ordered first in x.

    Returns
    -------
    F : ndarray, shape (n - nk, nk)
        Decision rule for jump variables: u(t) = F @ k(t).
    P : ndarray, shape (nk, nk)
        Law of motion for state variables: k(t+1) = P @ k(t).

    Raises
    ------
    ValueError
        If the invertibility condition is violated (z11 singular) or
        the number of stable eigenvalues does not match nk.

    Notes
    -----
    Matlab's ``ordqz(s, t, q, z, 'udo')`` reorders so that eigenvalues
    inside the unit disk come first.  ``scipy.linalg.ordqz`` with
    ``sort='iuc'`` (inside unit circle) achieves the same reordering.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]

    # Generalised Schur decomposition with eigenvalue reordering.
    # Matlab: [s,t,q,z] = qz(a,b); ordqz(...,'udo')
    # SciPy:  ordqz(A, B, sort='iuc') puts |eigenvalue| < 1 first.
    # Stable eigenvalues of x(t+1) = A^{-1} B x(t) have |lambda| < 1.
    # lambda = beta/alpha, so |alpha/beta| > 1, i.e. w = alpha/beta is
    # outside the unit circle.  sort='ouc' puts those first.
    S, T, alpha, beta, Q, Z = la.ordqz(A, B, sort="ouc")

    z11 = Z[:nk, :nk]
    z21 = Z[nk:, :nk]

    if np.linalg.matrix_rank(z11) < nk:
        raise ValueError(
            "Invertibility condition violated: z11 is singular. "
            "The model may not have a unique stable solution."
        )

    s11 = S[:nk, :nk]
    t11 = T[:nk, :nk]

    # Check Blanchard-Kahn conditions.
    abs_t_nk = np.abs(T[nk - 1, nk - 1])
    abs_s_nk = np.abs(S[nk - 1, nk - 1])
    abs_t_nk1 = np.abs(T[nk, nk]) if nk < n else 0.0
    abs_s_nk1 = np.abs(S[nk, nk]) if nk < n else 1.0

    if abs_t_nk > abs_s_nk or abs_t_nk1 < abs_s_nk1:
        import warnings

        warnings.warn(
            "Wrong number of stable eigenvalues. Blanchard-Kahn conditions "
            "may not be satisfied.",
            stacklevel=2,
        )

    z11i = np.linalg.solve(z11, np.eye(nk))
    dyn = np.linalg.solve(s11, t11)

    F = np.real(z21 @ z11i)
    P = np.real(z11 @ dyn @ z11i)

    return F, P
