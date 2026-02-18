"""Interpolation and lottery methods."""

from typing import Any, Tuple

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


def interp_linear(x_grid: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Vectorized 1D linear interpolation.

    Parameters
    ----------
    x_grid : ndarray, shape (n,)
        Grid points (must be sorted).
    y : ndarray, shape (n,) or (m, n)
        Values on grid.
    x_new : ndarray
        Points to interpolate.
    """
    x_grid = np.asarray(x_grid)
    y = np.asarray(y)
    x_new = np.asarray(x_new)

    if y.ndim == 1:
        return np.interp(x_new, x_grid, y)

    out = np.empty((y.shape[0],) + x_new.shape)
    for i in range(y.shape[0]):
        out[i] = np.interp(x_new, x_grid, y[i])
    return out


def interp_bilinear(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation on a rectilinear grid."""
    x_grid = np.asarray(x_grid)
    y_grid = np.asarray(y_grid)
    z = np.asarray(z)
    x_new = np.asarray(x_new)
    y_new = np.asarray(y_new)

    ix = np.searchsorted(x_grid, x_new, side="right") - 1
    iy = np.searchsorted(y_grid, y_new, side="right") - 1
    ix = np.clip(ix, 0, len(x_grid) - 2)
    iy = np.clip(iy, 0, len(y_grid) - 2)

    x0 = x_grid[ix]
    x1 = x_grid[ix + 1]
    y0 = y_grid[iy]
    y1 = y_grid[iy + 1]

    wx = (x_new - x0) / (x1 - x0)
    wy = (y_new - y0) / (y1 - y0)

    z00 = z[iy, ix]
    z10 = z[iy, ix + 1]
    z01 = z[iy + 1, ix]
    z11 = z[iy + 1, ix + 1]

    return (1 - wx) * (1 - wy) * z00 + wx * (1 - wy) * z10 + (1 - wx) * wy * z01 + wx * wy * z11


def get_lottery(a_policy: np.ndarray, a_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return indices and probabilities for lottery-based interpolation."""
    a_grid = np.asarray(a_grid)
    a_policy = np.asarray(a_policy)

    a_i = np.searchsorted(a_grid, a_policy, side="right") - 1
    a_i = np.clip(a_i, 0, len(a_grid) - 2)
    a_pi = (a_grid[a_i + 1] - a_policy) / (a_grid[a_i + 1] - a_grid[a_i])
    return a_i, a_pi


def _validate_breakpoints(breakpoints: np.ndarray) -> np.ndarray:
    """Validate and normalize spline breakpoints."""
    bp = np.asarray(breakpoints, dtype=float)
    if bp.ndim != 1:
        raise ValueError("`breakpoints` must be one-dimensional.")
    if bp.size < 2:
        raise ValueError("`breakpoints` must contain at least two points.")
    if not np.all(np.diff(bp) > 0):
        raise ValueError("`breakpoints` must be strictly increasing.")
    return bp


def _bspline_knots(breakpoints: np.ndarray, degree: int) -> np.ndarray:
    """Open knot vector for B-spline basis with interpolation breakpoints."""
    if degree < 1:
        raise ValueError("`degree` must be >= 1.")
    bp = _validate_breakpoints(breakpoints)
    k = degree
    return np.concatenate([np.repeat(bp[0], k + 1), bp[1:-1], np.repeat(bp[-1], k + 1)])


def greville_abscissae(breakpoints: np.ndarray, degree: int = 3) -> np.ndarray:
    """Return Greville abscissae for a clamped B-spline basis.

    These are the standard spline collocation nodes used by CompEcon
    (`splinode` / `funnode`) and obtained by knot averaging:
    ``x_j = (t_{j+1} + ... + t_{j+degree}) / degree``.

    Parameters
    ----------
    breakpoints : ndarray, shape (n_breakpoints,)
        Strictly increasing spline breakpoints.
    degree : int, optional
        B-spline degree (defaults to 3 for cubic splines).

    Returns
    -------
    ndarray, shape (n_basis,)
        Greville nodes where ``n_basis = len(knots) - degree - 1``.
    """
    if degree < 1:
        raise ValueError("`degree` must be >= 1.")

    knots = _bspline_knots(breakpoints, degree)
    n_basis = knots.size - degree - 1
    x = np.empty(n_basis, dtype=float)
    for j in range(n_basis):
        x[j] = np.mean(knots[j + 1 : j + degree + 1])
    return x


def spline_basis_matrix(breakpoints: np.ndarray, x: np.ndarray, degree: int = 3):
    """Evaluate 1D B-spline basis matrix at points ``x``.

    Parameters
    ----------
    breakpoints : ndarray, shape (n_breakpoints,)
        Strictly increasing spline breakpoints.
    x : ndarray
        Evaluation points.
    degree : int, optional
        B-spline degree (1 for linear tent basis, 3 for cubic basis).

    Returns
    -------
    scipy.sparse.csr_matrix
        Basis matrix of shape ``(len(x), n_basis)`` where
        ``n_basis = len(knots) - degree - 1``.
    """
    from scipy.interpolate import BSpline
    from scipy.sparse import csr_matrix

    bp = _validate_breakpoints(breakpoints)
    x_eval = np.asarray(x, dtype=float).reshape(-1)
    knots = _bspline_knots(bp, degree)
    n_basis = knots.size - degree - 1

    if hasattr(BSpline, "design_matrix"):
        try:
            mat = BSpline.design_matrix(x_eval, knots, degree, extrapolate=True)
            return mat.tocsr() if hasattr(mat, "tocsr") else csr_matrix(mat)
        except TypeError:  # pragma: no cover
            # Compatibility with older SciPy signatures.
            mat = BSpline.design_matrix(x_eval, knots, degree)
            return mat.tocsr() if hasattr(mat, "tocsr") else csr_matrix(mat)

    # Fallback for older SciPy versions without design_matrix.
    dense = np.empty((x_eval.size, n_basis), dtype=float)
    eye = np.eye(n_basis)
    for j in range(n_basis):
        dense[:, j] = BSpline(knots, eye[j], degree, extrapolate=True)(x_eval)
    return csr_matrix(dense)


def cubic_basis_matrix(breakpoints: np.ndarray, x: np.ndarray):
    """Cubic B-spline basis matrix (degree 3)."""
    return spline_basis_matrix(breakpoints, x, degree=3)


def linear_basis_matrix(breakpoints: np.ndarray, x: np.ndarray):
    """Linear (tent) B-spline basis matrix (degree 1)."""
    return spline_basis_matrix(breakpoints, x, degree=1)


def tensor_basis_matrix(Phi_a, Phi_z):
    """Row-wise Kronecker product for tensor-product basis.

    Parameters
    ----------
    Phi_a, Phi_z : array_like or sparse matrix
        Basis matrices with identical number of rows.

    Returns
    -------
    scipy.sparse.csr_matrix
        Matrix with row ``i`` equal to ``kron(Phi_z[i,:], Phi_a[i,:])``.
    """
    from scipy import sparse

    A = sparse.csr_matrix(Phi_a)
    Z = sparse.csr_matrix(Phi_z)
    if A.shape[0] != Z.shape[0]:
        raise ValueError("Phi_a and Phi_z must have the same number of rows.")

    rows = []
    for i in range(A.shape[0]):
        rows.append(sparse.kron(Z.getrow(i), A.getrow(i), format="csr"))
    if not rows:
        return sparse.csr_matrix((0, A.shape[1] * Z.shape[1]))
    return sparse.vstack(rows, format="csr")


def _validate_spline_input(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and normalize spline inputs."""
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1:
        raise ValueError("`x` must be a one-dimensional array.")
    if x.size < 2:
        raise ValueError("`x` must contain at least two points.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("`x` must be strictly increasing.")

    return x, y


def spline_fit(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "cubic",
    *,
    bc_type: Any = "not-a-knot",
    extrapolate: bool = True,
    axis: int = 0,
):
    """Fit a 1D interpolator with a consistent API.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Grid points (must be strictly increasing).
    y : ndarray
        Values at grid points.
    method : {"linear", "cubic", "pchip", "akima", "bspline"}
        Interpolation method.
    bc_type : Any, optional
        Boundary condition argument used by cubic/bspline methods.
    extrapolate : bool, optional
        Whether to extrapolate outside the x-domain.
    axis : int, optional
        Axis of `y` corresponding to `x`.
    """
    from scipy.interpolate import (
        Akima1DInterpolator,
        CubicSpline,
        PchipInterpolator,
        make_interp_spline,
    )

    x, y = _validate_spline_input(x, y)
    method_key = method.lower()

    if method_key == "linear":
        spline = make_interp_spline(x, y, k=1, axis=axis)
        if hasattr(spline, "extrapolate"):
            spline.extrapolate = extrapolate
        return spline

    if method_key == "cubic":
        return CubicSpline(
            x,
            y,
            axis=axis,
            bc_type=bc_type,
            extrapolate=extrapolate,
        )

    if method_key == "pchip":
        return PchipInterpolator(x, y, axis=axis, extrapolate=extrapolate)

    if method_key == "akima":
        # Support both old/new SciPy signatures for Akima extrapolation.
        try:
            return Akima1DInterpolator(x, y, axis=axis, extrapolate=extrapolate)
        except TypeError:  # pragma: no cover
            akima = Akima1DInterpolator(x, y, axis=axis)
            if hasattr(akima, "extrapolate"):
                akima.extrapolate = extrapolate
            return akima

    if method_key in {"bspline", "b-spline"}:
        spline = make_interp_spline(x, y, k=3, axis=axis, bc_type=bc_type)
        if hasattr(spline, "extrapolate"):
            spline.extrapolate = extrapolate
        return spline

    raise ValueError(
        "Unknown spline method. Use one of: "
        "'linear', 'cubic', 'pchip', 'akima', 'bspline'."
    )


def spline_coef(x: np.ndarray, y: np.ndarray, kind: str = "cubic", **kwargs):
    """Backward-compatible alias to `spline_fit`.

    Parameters
    ----------
    x, y : ndarray
        Data to fit.
    kind : str, optional
        Alias for `method` in `spline_fit`.
    **kwargs
        Additional keyword arguments forwarded to `spline_fit`.
    """
    return spline_fit(x, y, method=kind, **kwargs)


def spline_eval(model, x_new: np.ndarray) -> np.ndarray:
    """Evaluate a spline/interpolator object returned by `spline_fit`."""
    return model(x_new)


def cheb_nodes(n: int, a: float, b: float) -> np.ndarray:
    """Chebyshev nodes on [a, b]."""
    i = np.arange(1, n + 1)
    x = np.cos((2 * i - 1) * np.pi / (2 * n))
    return 0.5 * (a + b) + 0.5 * (b - a) * x


def cheb_basis(x: np.ndarray, n: int, a: float, b: float) -> np.ndarray:
    """Chebyshev basis matrix evaluated at x."""
    x = np.asarray(x)
    xt = (2 * x - (a + b)) / (b - a)
    xt = np.clip(xt, -1.0, 1.0)
    theta = np.arccos(xt)
    return np.column_stack([np.cos(k * theta) for k in range(n)])


def cheb_coef(f, n: int, a: float, b: float) -> np.ndarray:
    """Fit Chebyshev coefficients for f on [a, b]."""
    nodes = cheb_nodes(n, a, b)
    values = f(nodes)
    basis = cheb_basis(nodes, n, a, b)
    coef, *_ = np.linalg.lstsq(basis, values, rcond=None)
    return coef


def cheb_eval(coef: np.ndarray, x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Evaluate Chebyshev approximation given coefficients."""
    basis = cheb_basis(np.asarray(x), len(coef), a, b)
    return basis @ coef


if njit:
    @njit
    def _interp1d_numba(x_grid, y, x_new):  # pragma: no cover
        out = np.empty_like(x_new)
        for i in range(x_new.size):
            x = x_new[i]
            if x <= x_grid[0]:
                out[i] = y[0]
                continue
            if x >= x_grid[-1]:
                out[i] = y[-1]
                continue
            k = np.searchsorted(x_grid, x) - 1
            x0 = x_grid[k]
            x1 = x_grid[k + 1]
            w = (x - x0) / (x1 - x0)
            out[i] = (1 - w) * y[k] + w * y[k + 1]
        return out

