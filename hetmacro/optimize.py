"""Root-finding and optimization routines."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy import optimize


def bisect(f: Callable, a: float, b: float, args=(), tol: float = 1e-12, max_iter: int = 100) -> float:
    """Bisection root-finding."""
    fa = f(a, *args)
    fb = f(b, *args)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m, *args)
        if abs(fm) < tol:
            return m
        if fa * fm <= 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
    return m


def brentq(f: Callable, a: float, b: float, args=(), tol: float = 1e-12) -> float:
    """Brent's method root-finding."""
    return optimize.brentq(f, a, b, args=args, xtol=tol, rtol=tol)


def newton(
    f: Callable,
    x0: float,
    fprime: Callable,
    args=(),
    tol: float = 1e-8,
    max_iter: int = 50,
) -> float:
    """Newton-Raphson root-finding."""
    return optimize.newton(f, x0, fprime=fprime, args=args, tol=tol, maxiter=max_iter)


def secant(f: Callable, x0: float, args=(), tol: float = 1e-8, max_iter: int = 50) -> float:
    """Secant method."""
    return optimize.newton(f, x0, fprime=None, args=args, tol=tol, maxiter=max_iter)


def _broyden_scipy(
    f: Callable,
    x0: np.ndarray,
    args=(),
    tol: float = 1e-8,
    max_iter: int | None = None,
    return_info: bool = False,
) -> np.ndarray:
    """Broyden's method for systems of equations (SciPy wrapper).

    Notes
    -----
    This is a thin wrapper around ``scipy.optimize.root(..., method="broyden1")``.
    For highly nonlinear / ill-scaled problems, this method may diverge.
    """
    options: dict = {}
    if max_iter is not None:
        options["maxiter"] = int(max_iter)
    if return_info:
        hist = {"f_norm": [], "f_maxabs": []}

        def _cb(xk, fk=None):
            # SciPy's callback signature can be (x) or (x, f).
            if fk is None:
                fk = f(xk, *args)
            fk = np.asarray(fk, dtype=float)
            hist["f_norm"].append(float(np.linalg.norm(fk)))
            hist["f_maxabs"].append(float(np.max(np.abs(fk))))

        res = optimize.root(
            lambda x: f(x, *args),
            x0,
            method="broyden1",
            tol=tol,
            callback=_cb,
            options=options,
        )
        return res.x, hist, bool(res.success), str(res.message)

    res = optimize.root(lambda x: f(x, *args), x0, method="broyden1", tol=tol, options=options)
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def _fdjac(f: Callable, x: np.ndarray, args=()) -> np.ndarray:
    """Forward-difference Jacobian (similar to CompEcon fdjac)."""
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x, *args), dtype=float)
    n = x.size
    jac = np.zeros((n, n), dtype=float)
    step = 1e-6 * (1.0 + np.abs(x))
    for i in range(n):
        xh = x.copy()
        xh[i] += step[i]
        fh = np.asarray(f(xh, *args), dtype=float)
        jac[:, i] = (fh - f0) / step[i]
    return jac


def _linesearch_compecon(
    f: Callable,
    x: np.ndarray,
    dx: np.ndarray,
    fnorm: float,
    maxsteps: int,
    args=(),
) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
    """CompEcon-style backstepping line search."""
    fnormold = np.inf
    fvalold = None
    for it in range(1, maxsteps + 1):
        fvalnew = np.asarray(f(x + dx, *args), dtype=float)
        fnormnew = float(np.linalg.norm(fvalnew, ord=np.inf))
        if fnormnew < fnorm:
            return dx, fvalnew, fnormnew, 0, it
        if fnormold < fnormnew:
            # step was too small, backtrack one step and expand
            if fvalold is not None:
                fvalnew = fvalold
                fnormnew = fnormold
            dx = dx * 2.0
            return dx, fvalnew, fnormnew, 2, it
        fvalold = fvalnew
        fnormold = fnormnew
        dx = dx / 2.0
    return dx, fvalnew, fnormnew, 1, maxsteps


def _broyden_compecon(
    f: Callable,
    x0: np.ndarray,
    args=(),
    tol: float | None = None,
    max_iter: int = 100,
    max_steps: int = 25,
    show_iters: bool = False,
    init_inv: np.ndarray | None = None,
    init_identity: bool = False,
    use_jac: bool = False,
    restarts: bool = True,
    return_info: bool = False,
) -> np.ndarray:
    """Broyden's inverse method (CompEcon-style) with backstepping.

    Mirrors CEtools/broyden.m + linesearch.m:
    - inverse Jacobian approximation
    - backstepping line search
    - optional restarts
    """
    x = np.asarray(x0, dtype=float).copy()
    if tol is None:
        tol = float(np.sqrt(np.finfo(float).eps))

    info = None
    if return_info:
        info = {"fnorm": [], "backstep": [], "lineflag": []}

    # initialize inverse Jacobian
    if init_inv is not None:
        fjacinv = np.asarray(init_inv, dtype=float)
    else:
        if use_jac:
            fval, fjac = f(x, *args)
            fjacinv = np.linalg.inv(np.asarray(fjac, dtype=float))
            fval = np.asarray(fval, dtype=float)
        elif init_identity:
            fjacinv = np.eye(x.size, dtype=float)
            fval = np.asarray(f(x, *args), dtype=float)
        else:
            fjac = _fdjac(f, x, args=args)
            fjacinv = np.linalg.inv(fjac)
            fval = np.asarray(f(x, *args), dtype=float)

    fnorm = float(np.linalg.norm(fval, ord=np.inf))
    flag = 0
    for it in range(1, int(max_iter) + 1):
        if fnorm < tol:
            break

        dx = -(fjacinv @ fval)
        if not np.all(np.isfinite(dx)):
            flag = 2
            break

        dx, fvalnew, fnormnew, lineflag, backstep = _linesearch_compecon(
            f, x, dx, fnorm, int(max_steps), args=args
        )
        x = x + dx

        if lineflag > 0 and restarts:
            if use_jac:
                fval, fjac = f(x, *args)
                fjacinv = np.linalg.inv(np.asarray(fjac, dtype=float))
                fval = np.asarray(fval, dtype=float)
            elif init_identity:
                fjacinv = np.eye(x.size, dtype=float)
                fval = np.asarray(f(x, *args), dtype=float)
            else:
                fjac = _fdjac(f, x, args=args)
                fjacinv = np.linalg.inv(fjac)
                fval = np.asarray(f(x, *args), dtype=float)
        else:
            temp = fjacinv @ (fvalnew - fval)
            denom = float(dx @ temp)
            if abs(denom) < 1e-14:
                # fallback: restart if update is unstable
                if restarts:
                    fjac = _fdjac(f, x, args=args)
                    fjacinv = np.linalg.inv(fjac)
                else:
                    flag = 2
                    break
            else:
                fjacinv = fjacinv + np.outer(dx - temp, (dx @ fjacinv) / denom)
            fval = fvalnew
            fnorm = fnormnew

        if info is not None:
            info["fnorm"].append(float(fnorm))
            info["backstep"].append(int(backstep))
            info["lineflag"].append(int(lineflag))

        if show_iters:
            print(f"{it:4d} {backstep:4d} {fnorm:6.2e}")

    else:
        flag = 1

    if return_info:
        return x, fval, flag, it, fjacinv, info
    return x, fval, flag, it, fjacinv


def _solve_broyden_componentwise(
    func: Callable,
    x: np.ndarray,
    a: float | np.ndarray,
    b: float | np.ndarray,
    args=(),
    tol: float = 1e-13,
    max_iter: int = 20,
    return_info: bool = False,
) -> np.ndarray:
    """Safeguarded (componentwise) Broyden, mirroring the course MATLAB code.

    This implements the logic in the assignment's `solve_broyden.m`:
    - Finite-difference estimate of a *diagonal* Jacobian and its inverse
    - Componentwise update with bounds `a <= x <= b`
    - Accept/reject per component and step-halving when no improvement

    This is intended for *vectorized, largely componentwise* problems like
    solving many independent 1D FOCs in parallel. It is not a general-purpose
    dense Broyden method for strongly coupled systems.
    """
    x = np.asarray(x, dtype=float).copy()
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # broadcast bounds
    if a.shape != () and a.shape != x.shape:
        raise ValueError("Lower bound a must be scalar or same shape as x.")
    if b.shape != () and b.shape != x.shape:
        raise ValueError("Upper bound b must be scalar or same shape as x.")

    # initial derivative estimate (diagonal Jacobian)
    dx0 = (1.0 + x) * 1e-5
    f = np.asarray(func(x, *args), dtype=float)
    fx = (np.asarray(func(x + dx0, *args), dtype=float) - f) / dx0

    # avoid divide-by-zero / tiny slopes
    fx = np.where(np.abs(fx) < 1e-14, np.sign(fx) * 1e-14 + (fx == 0) * 1e-14, fx)
    fxi = 1.0 / fx

    step = np.ones_like(x)  # cut in 1/2 if no improvement
    term = np.abs(f) < tol

    info = None
    if return_info:
        info = {
            "f_norm": [],
            "f_maxabs": [],
            "frac_updated": [],
            "step_mean": [],
            "step_min": [],
            "step_max": [],
        }

    for _ in range(int(max_iter)):
        if bool(np.all(term)):
            break

        fold = f
        xold = x

        # quasi-newton step using diagonal inverse Jacobian estimate
        dx = -f * fxi * step
        x = np.minimum(np.maximum(x + dx, a), b)

        f = np.asarray(func(x, *args), dtype=float)
        dx = x - xold

        # Broyden update for diagonal inverse Jacobian
        u = (f - fold) * fxi
        denom = dx * u
        denom = np.where(np.abs(denom) < 1e-14, np.sign(denom) * 1e-14 + (denom == 0) * 1e-14, denom)
        fxi = fxi + ((dx - u) * dx * fxi) / denom

        # accept/reject per component
        update = np.abs(f) < np.abs(fold)
        x = x * update + xold * (~update)
        f = f * update + fold * (~update)

        step[~update] *= 0.5
        step[update] = 1.0

        term = np.abs(f) < tol

        if info is not None:
            info["f_norm"].append(float(np.linalg.norm(f)))
            info["f_maxabs"].append(float(np.max(np.abs(f))))
            info["frac_updated"].append(float(np.mean(update)))
            info["step_mean"].append(float(np.mean(step)))
            info["step_min"].append(float(np.min(step)))
            info["step_max"].append(float(np.max(step)))

    if return_info:
        return x, info
    return x


def broyden(
    f: Callable,
    x0: np.ndarray,
    method: str = "compecon",
    *,
    args=(),
    tol=None,
    max_iter=None,
    a=None,
    b=None,
    return_info: bool = False,
    **kwargs,
):
    """Unified Broyden solver: one entry point, multiple methods.

    Parameters
    ----------
    f : callable
        Residual function f(x, *args) returning array of same shape as x.
    x0 : array_like
        Initial guess.
    method : {'scipy', 'compecon', 'componentwise'}, optional
        Solver style:
        - ``'scipy'``: SciPy wrapper (``scipy.optimize.root(..., method='broyden1')``).
        - ``'compecon'``: CompEcon-style (inverse Jacobian, backstepping, restarts).
        - ``'componentwise'``: MATLAB-style safeguarded componentwise Broyden
          with bounds; requires ``a`` and ``b``.
    args : tuple, optional
        Extra arguments passed to ``f``.
    tol : float, optional
        Convergence tolerance (method-dependent default if None).
    max_iter : int, optional
        Maximum iterations (method-dependent default if None).
    a, b : array_like or float, optional
        Lower and upper bounds. **Required** for ``method='componentwise'``.
    return_info : bool, optional
        If True, return extra diagnostics (shape depends on method).
    **kwargs
        Passed through to the underlying solver (e.g. ``max_steps``, ``show_iters``
        for ``'compecon'``).

    Returns
    -------
    For ``method='scipy'``: x or (x, hist, success, message) if return_info.
    For ``method='compecon'``: (x, fval, flag, it, fjacinv) or + info if return_info.
    For ``method='componentwise'``: x or (x, info) if return_info.
    """
    method = method.lower()
    if method not in ("scipy", "compecon", "componentwise"):
        raise ValueError(f"method must be 'scipy', 'compecon', or 'componentwise'; got {method!r}")

    if method == "scipy":
        return _broyden_scipy(
            f, x0, args=args, tol=tol or 1e-8, max_iter=max_iter, return_info=return_info
        )
    if method == "compecon":
        return _broyden_compecon(
            f,
            x0,
            args=args,
            tol=tol,
            max_iter=max_iter or 100,
            return_info=return_info,
            **kwargs,
        )
    # componentwise
    if a is None or b is None:
        raise ValueError("method='componentwise' requires arguments a and b (bounds)")
    return _solve_broyden_componentwise(
        f,
        x0,
        a,
        b,
        args=args,
        tol=tol if tol is not None else 1e-13,
        max_iter=max_iter if max_iter is not None else 20,
        return_info=return_info,
        **kwargs,
    )


def broyden_compecon(
    f: Callable,
    x0: np.ndarray,
    args=(),
    tol=None,
    max_iter: int = 100,
    max_steps: int = 25,
    show_iters: bool = False,
    init_inv: np.ndarray | None = None,
    init_identity: bool = False,
    use_jac: bool = False,
    restarts: bool = True,
    return_info: bool = False,
):
    """Broyden's inverse method (CompEcon-style). Thin wrapper around broyden(..., method='compecon')."""
    return broyden(
        f,
        x0,
        method="compecon",
        args=args,
        tol=tol,
        max_iter=max_iter,
        max_steps=max_steps,
        show_iters=show_iters,
        init_inv=init_inv,
        init_identity=init_identity,
        use_jac=use_jac,
        restarts=restarts,
        return_info=return_info,
    )


def solve_broyden(
    func: Callable,
    x: np.ndarray,
    a: float | np.ndarray,
    b: float | np.ndarray,
    args=(),
    tol: float = 1e-13,
    max_iter: int = 20,
    return_info: bool = False,
):
    """Safeguarded (componentwise) Broyden. Thin wrapper around broyden(..., method='componentwise')."""
    return broyden(
        func,
        x,
        method="componentwise",
        a=a,
        b=b,
        args=args,
        tol=tol,
        max_iter=max_iter,
        return_info=return_info,
    )


def golden_search(f: Callable, a: float, b: float, args=(), tol: float = 1e-8) -> float:
    """Golden section search for scalar minimization."""
    res = optimize.minimize_scalar(f, bracket=(a, b), args=args, tol=tol, method="golden")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def golden_max_vec(
    f: Callable,
    a: np.ndarray | float,
    b: np.ndarray | float,
    args=(),
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized golden-section maximization on per-element intervals.

    Parameters
    ----------
    f : callable
        Objective function. Must accept an array ``x`` and return an array
        of the same shape (evaluated elementwise).
    a, b : array_like or float
        Lower and upper bounds for each element. Must be broadcastable to a
        common shape.
    args : tuple, optional
        Extra arguments passed to ``f``.
    tol : float, optional
        Absolute tolerance on interval width.

    Returns
    -------
    x : ndarray
        Approximate maximizers on each element's interval.
    fx : ndarray
        Objective evaluated at ``x``.
    """
    alpha1 = 0.5 * (3.0 - np.sqrt(5.0))
    alpha2 = 0.5 * (np.sqrt(5.0) - 1.0)

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = np.broadcast_arrays(a, b)
    if np.any(b < a):
        raise ValueError("All elements must satisfy b >= a.")

    d = b - a
    x1 = a + alpha1 * d
    x2 = a + alpha2 * d

    f1 = np.asarray(f(x1, *args), dtype=float)
    f2 = np.asarray(f(x2, *args), dtype=float)

    d = alpha1 * alpha2 * d
    while np.any(d > tol):
        x1old = x1.copy()
        x2old = x2.copy()
        f1old = f1.copy()
        f2old = f2.copy()

        d = d * alpha2
        goR = f2 >= f1
        goL = ~goR

        xnew = (x1old - d) * goL + (x2old + d) * goR
        fnew = np.asarray(f(xnew, *args), dtype=float)

        x1[goR] = x2old[goR]
        f1[goR] = f2old[goR]
        x2[goR] = xnew[goR]
        f2[goR] = fnew[goR]

        x1[goL] = xnew[goL]
        f1[goL] = fnew[goL]
        x2[goL] = x1old[goL]
        f2[goL] = f1old[goL]

    goR = f2 >= f1
    x = x2.copy()
    x[~goR] = x1[~goR]
    fx = np.maximum(f1, f2)
    return x, fx


def brent_min(f: Callable, a: float, b: float, args=(), tol: float = 1e-8) -> float:
    """Brent's method for scalar minimization."""
    res = optimize.minimize_scalar(f, bracket=(a, b), args=args, tol=tol, method="brent")
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


def nelder_mead(
    f: Callable,
    x0: np.ndarray,
    args=(),
    tol: float = 1e-8,
    callback=None,
    options: dict | None = None,
    return_info: bool = False,
):
    """Nelder-Mead simplex optimization (derivative-free).

    Parameters
    ----------
    f : callable
        Objective to minimize, signature f(x, *args) -> float.
    x0 : array_like
        Initial guess.
    args : tuple, optional
        Extra arguments passed to f.
    tol : float, optional
        Convergence tolerance (xatol/fatol).
    callback : callable, optional
        Called after each iteration as callback(xk); xk is the current best vertex.
    options : dict, optional
        Passed to scipy.optimize.minimize (e.g. maxiter, maxfev, disp).
    return_info : bool, optional
        If True, return (x, f_min, hist) where hist has keys "x" and "f"
        (parameter and objective value at each iteration).

    Returns
    -------
    x : ndarray
        Best point found.
    f_min : float
        Objective at x.
    hist : dict, optional
        Only if return_info=True: hist["x"] list of ndarray, hist["f"] list of float.
    """
    opts = dict(options) if options else {}
    if return_info:
        hist = {"x": [], "f": []}

        def _cb(xk):
            hist["x"].append(np.asarray(xk, dtype=float).copy())
            fk = float(f(xk, *args))
            hist["f"].append(fk)
            if callback is not None:
                callback(xk)

        res = optimize.minimize(
            lambda x: f(x, *args),
            x0,
            method="Nelder-Mead",
            tol=tol,
            callback=_cb,
            options=opts,
        )
    else:
        res = optimize.minimize(
            lambda x: f(x, *args),
            x0,
            method="Nelder-Mead",
            tol=tol,
            callback=callback,
            options=opts,
        )
    if not res.success and not return_info:
        raise RuntimeError(res.message)
    if return_info:
        return res.x, res.fun, hist
    return res.x, res.fun


def minimize_lbfgsb(
    f: Callable,
    x0: np.ndarray,
    bounds: list[tuple[float, float]],
    args=(),
    tol: float = 1e-8,
    options: dict | None = None,
) -> np.ndarray:
    """Minimize a scalar function over a box (L-BFGS-B quasi-Newton).

    Wrapper around ``scipy.optimize.minimize(..., method='L-BFGS-B', bounds=...)``.
    Use this for bounded multivariate minimization (e.g. policy choices in [0, 1]).

    Parameters
    ----------
    f : callable
        Objective to minimize, signature f(x, *args) -> float.
    x0 : ndarray
        Initial guess, shape (n,).
    bounds : list of (low, high)
        Per-dimension bounds; length n. Use None in a tuple for no bound on that side.
    args : tuple, optional
        Extra arguments passed to f.
    tol : float, optional
        Convergence tolerance (passed to L-BFGS-B).
    options : dict, optional
        Passed to scipy (e.g. maxiter, maxfun, disp).

    Returns
    -------
    x : ndarray
        Approximate minimizer.
    """
    opts = dict(options) if options else {}
    res = optimize.minimize(
        lambda x: f(x, *args),
        np.asarray(x0, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        tol=tol,
        options=opts,
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.x
