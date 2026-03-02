"""Distribution iteration tools for HA models."""

from typing import Tuple

import numpy as np
from scipy import sparse

from .forward import forward_iteration, stationary_distribution, stationary_eigenvector
from .interpolation import get_lottery


def forward_policy_step(D: np.ndarray, a_policy: np.ndarray, a_grid: np.ndarray) -> np.ndarray:
    """Forward step using policy and grid."""
    a_i, a_pi = get_lottery(a_policy, a_grid)
    return forward_iteration(D, np.eye(a_policy.shape[0]), a_i, a_pi)


def stationary_distribution_ha(
    Pi: np.ndarray,
    a_policy: np.ndarray,
    a_grid: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """Stationary distribution for HA models."""
    return stationary_distribution(Pi, a_policy, a_grid, tol=tol)


def lottery_indices(a_policy: np.ndarray, a_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return lottery indices and weights."""
    return get_lottery(a_policy, a_grid)


def compute_joint_transition_matrix(
    a_ind: np.ndarray, a_ind_w: np.ndarray, Pi: np.ndarray
):
    """Build the joint transition matrix for income-asset states.

    Returns a sparse CSR matrix of shape (nz*na, nz*na).

    Parameters
    ----------
    a_ind, a_ind_w : ndarray, shape (n_z, n_a)
        Lower-grid index and weight for next-period asset lottery.
    Pi : ndarray, shape (n_z, n_z)
        Income transition matrix.
    """
    nz, na = a_ind.shape
    N = nz * na

    # Flatten lottery data
    a_low = a_ind.ravel().astype(int)          # (N,)
    a_high = np.minimum(a_low + 1, na - 1)     # (N,)
    w_low = a_ind_w.ravel()                     # (N,)
    w_high = 1.0 - w_low                        # (N,)

    # Row indices: each (iz, ia) state, repeated for each next-period iz'
    # row[i] = iz * na + ia for the i-th flattened state
    rows_base = np.arange(N)  # (N,)

    # Build COO entries: for each row, we have 2*nz entries
    # (low target and high target for each next-period income state)
    row_idx = np.tile(rows_base, 2 * nz)  # (2*nz*N,)

    col_idx_parts = []
    data_parts = []

    # iz index for each flattened state
    iz_flat = np.repeat(np.arange(nz), na)  # (N,)

    for izn in range(nz):
        pz = Pi[iz_flat, izn]  # (N,) transition prob from current iz to izn
        col_low = izn * na + a_low    # (N,)
        col_high = izn * na + a_high  # (N,)
        col_idx_parts.append(col_low)
        col_idx_parts.append(col_high)
        data_parts.append(pz * w_low)
        data_parts.append(pz * w_high)

    col_idx = np.concatenate(col_idx_parts)
    data = np.concatenate(data_parts)

    T = sparse.coo_matrix((data, (row_idx, col_idx)), shape=(N, N))
    return T.tocsr()


def stationary_dist(P) -> np.ndarray:
    """Stationary distribution for a row-stochastic matrix.

    Accepts both dense and sparse matrices. Uses eigenvector method for
    sparse input and least-squares for dense input.
    """
    if sparse.issparse(P):
        return stationary_eigenvector(P)

    P = np.asarray(P)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be square.")
    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("Rows of P must sum to 1.")
    n = P.shape[0]
    A = np.vstack((np.eye(n) - P.T, np.ones((1, n))))
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.maximum(pi, 0.0)
    s = pi.sum()
    if s <= 0:
        raise RuntimeError("Failed to compute stationary distribution.")
    return pi / s
