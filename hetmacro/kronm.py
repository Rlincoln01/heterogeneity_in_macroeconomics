"""Fast Kronecker-matrix multiplication without forming the Kronecker product.

Used primarily by the VFI fallback path in the Krusell-Smith solver,
where the certainty equivalent must be computed across a 4D state space
(k, K, e, Z) with joint exogenous transitions Q_ez = kron(Q_zz, Q_ee).
"""

from __future__ import annotations

import numpy as np


def kronm(Qs: list, x: np.ndarray) -> np.ndarray:
    """Compute (Q_k ⊗ ... ⊗ Q_1) @ x without forming the Kronecker product.

    Parameters
    ----------
    Qs : list
        Each element is either an ndarray (the transition matrix for that
        dimension) or an int (representing an identity of that size, i.e.
        "skip this dimension").  Dimensions are ordered from fastest-varying
        (index 0) to slowest-varying (index -1), matching Matlab ``gridmake``
        / Fortran ordering.
    x : ndarray, shape (n,) or (n, m)
        Input vector or matrix, where n = prod(col_sizes).

    Returns
    -------
    ndarray, same shape as *x*
        Result of the Kronecker-matrix product.

    Notes
    -----
    Core identity for two matrices:
        (Q2 ⊗ Q1) @ vec_F(X) = vec_F(Q1 @ X @ Q2.T)
    where X is reshaped to (n1, n2) in Fortran (column-major) order.

    Generalises via tensordot: apply each non-identity Q_i along its
    axis of the Fortran-order tensor, leaving identity dimensions untouched.
    """
    x = np.asarray(x, dtype=float)
    was_1d = x.ndim == 1
    if was_1d:
        x = x[:, np.newaxis]

    n, m = x.shape
    k = len(Qs)

    def _col_sizes():
        return [int(Q) if np.isscalar(Q) else np.asarray(Q).shape[1] for Q in Qs]

    def _row_sizes():
        return [int(Q) if np.isscalar(Q) else np.asarray(Q).shape[0] for Q in Qs]

    in_sizes = _col_sizes()
    out_n = int(np.prod(_row_sizes()))

    result = np.empty((out_n, m))
    for j in range(m):
        sizes = _col_sizes()
        y = x[:, j].reshape(sizes, order="F").copy()

        for i in range(k):
            if np.isscalar(Qs[i]):
                continue
            Q = np.asarray(Qs[i])
            y = np.tensordot(Q, y, axes=([1], [i]))
            y = np.moveaxis(y, 0, i)
            sizes[i] = Q.shape[0]

        result[:, j] = y.ravel(order="F")

    if was_1d:
        result = result.ravel()
    return result
