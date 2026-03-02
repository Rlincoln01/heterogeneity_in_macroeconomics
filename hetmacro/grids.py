"""Grid construction tools for computational economics."""

from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class GridSpec:
    """Specification for one grid dimension."""

    lower: float
    upper: float
    n_points: int
    spacing: str = "linear"
    power: float = 1.0
    concentration_points: Optional[Sequence[float]] = None
    concentration_weight: float = 0.0
    concentration_radius: Optional[float] = None
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None


def make_asset_grid(
    amin: float,
    amax: float,
    n: int,
    curvature: float = 2.0,
) -> np.ndarray:
    """Power-spaced grid commonly used for assets."""
    return make_grid_1d(amin, amax, n, spacing="power", power=curvature)


def power_grid(amin: float, amax: float, n: int, phi: float = 25.0, nu: float = 5.0) -> np.ndarray:
    """Nonlinear power grid with extra density near lower bound."""
    x = np.linspace(0.0, 1.0, n)
    xx = x + phi * x**nu
    return amin + ((amax - amin) / (xx.max() - xx.min())) * xx


def make_grid(vmin: float, vmax: float, n: int, smallest_step: float = 0.01) -> np.ndarray:
    """Curvature-controlled 1D grid used in HA models."""
    curv = np.log((vmax - vmin) / smallest_step) / np.log(n - 1)
    curv = np.maximum(curv, 1.0)
    return vmin + (vmax - vmin) * (np.linspace(0.0, 1.0, n) ** curv)


def double_exponential_grid(amin: float, amax: float, n: int) -> np.ndarray:
    """Rognlie-style double exponential grid."""
    if amax < amin:
        raise ValueError("amax must be >= amin")
    ubar = np.log(1 + np.log(1 + amax - amin))
    u_grid = np.linspace(0.0, ubar, n)
    return amin + np.exp(np.exp(u_grid) - 1.0) - 1.0


def log_spaced_grid(amin: float, amax: float, n: int) -> np.ndarray:
    """Log-spaced grid on (amin, amax], amin must be > 0."""
    if amin <= 0:
        raise ValueError("amin must be > 0 for log spacing")
    if amax <= amin:
        raise ValueError("amax must be > amin for log spacing")
    return np.exp(np.linspace(np.log(amin), np.log(amax), n))


def make_grid_1d(
    lower: float,
    upper: float,
    n: int,
    spacing: str = "linear",
    power: float = 1.0,
    concentration_points: Optional[Sequence[float]] = None,
    concentration_weight: float = 0.0,
    concentration_radius: Optional[float] = None,
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Create a 1D grid with flexible spacing and concentration.

    Parameters
    ----------
    lower, upper : float
        Bounds of the grid (inclusive).
    n : int
        Number of grid points.
    spacing : str
        'linear', 'power', 'log', 'double_exp', 'chebyshev', or 'custom'.
    power : float
        Power exponent for spacing='power'.
    concentration_points : sequence of float, optional
        Points around which to concentrate grid points.
    concentration_weight : float
        Strength of concentration in [0, 1].
    concentration_radius : float, optional
        Scale of concentration around each point. Defaults to 10% of range.
    custom_func : callable, optional
        Custom mapping from [0, 1] to [lower, upper] for spacing='custom'.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if upper < lower:
        raise ValueError("upper must be >= lower")

    spacing = spacing.lower()
    if spacing == "linear":
        grid = np.linspace(lower, upper, n)
    elif spacing == "power":
        t = np.linspace(0.0, 1.0, n)
        grid = lower + (upper - lower) * (t**power)
    elif spacing == "log":
        grid = log_spaced_grid(lower, upper, n)
    elif spacing == "double_exp":
        grid = double_exponential_grid(lower, upper, n)
    elif spacing == "chebyshev":
        i = np.arange(1, n + 1)
        x = np.cos((2 * i - 1) * np.pi / (2 * n))
        grid = 0.5 * (lower + upper) + 0.5 * (upper - lower) * x
        grid = np.sort(grid)
    elif spacing == "custom":
        if custom_func is None:
            raise ValueError("custom_func must be provided for spacing='custom'")
        t = np.linspace(0.0, 1.0, n)
        grid = custom_func(t)
    else:
        raise ValueError(f"Unknown spacing: {spacing}")

    if concentration_points and concentration_weight > 0:
        grid = _apply_concentration(
            grid,
            lower,
            upper,
            concentration_points,
            concentration_weight,
            concentration_radius,
        )

    grid[0] = lower
    grid[-1] = upper
    return grid


def make_grid_nd(
    specs: Sequence[GridSpec],
    cartesian: bool = False,
    tensor: bool = False,
    order: str = "C",
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """Construct a multi-dimensional grid from a list of GridSpec.

    Parameters
    ----------
    specs : sequence of GridSpec
        One GridSpec per dimension.
    cartesian : bool, optional
        If True, return the Cartesian product as a 2D array (n_prod, d).
    tensor : bool, optional
        If True, return a tensor grid with shape (n1, n2, ..., nd, d).
    order : str, optional
        Memory order for Cartesian output ('C' or 'F').
    """
    grids = [
        make_grid_1d(
            spec.lower,
            spec.upper,
            spec.n_points,
            spacing=spec.spacing,
            power=spec.power,
            concentration_points=spec.concentration_points,
            concentration_weight=spec.concentration_weight,
            concentration_radius=spec.concentration_radius,
            custom_func=spec.custom_func,
        )
        for spec in specs
    ]
    if cartesian and tensor:
        raise ValueError("cartesian and tensor cannot both be True")
    if cartesian:
        return grids, cartesian_product(*grids, order=order)
    if tensor:
        return grids, tensor_product(*grids)
    return grids, None


def cartesian_product(*grids: Iterable[float], order: str = "C") -> np.ndarray:
    """Cartesian product of 1D grids into a 2D array."""
    grids = [np.asarray(g) for g in grids]
    if not grids:
        return np.empty((0, 0))
    meshes = np.meshgrid(*grids, indexing="ij")
    stacked = np.stack([m.reshape(-1) for m in meshes], axis=1)
    if order.upper() == "F":
        return stacked.reshape([g.size for g in grids], len(grids), order="F").reshape(
            -1, len(grids), order="F"
        )
    return stacked


def gridmake(*grids: Iterable[float], order: str = "C") -> np.ndarray:
    """Compatibility alias for cartesian_product."""
    return cartesian_product(*grids, order=order)


def tensor_product(*grids: Iterable[float]) -> np.ndarray:
    """Tensor product grid with shape (n1, n2, ..., nd, d)."""
    grids = [np.asarray(g) for g in grids]
    if not grids:
        return np.empty((0, 0))
    meshes = np.meshgrid(*grids, indexing="ij")
    return np.stack(meshes, axis=-1)


def _apply_concentration(
    grid: np.ndarray,
    lower: float,
    upper: float,
    points: Sequence[float],
    weight: float,
    radius: Optional[float],
) -> np.ndarray:
    """Pull grid points toward specified locations using smooth kernels."""
    points = np.asarray(points, dtype=float)
    points = points[(points >= lower) & (points <= upper)]
    if points.size == 0:
        return grid

    span = upper - lower
    sigma = radius if radius is not None else 0.1 * span
    if sigma <= 0:
        return grid

    x = grid.copy()
    influence = np.zeros_like(x)
    total_weight = np.zeros_like(x)
    for p in points:
        dist = (x - p) / sigma
        w = np.exp(-0.5 * dist * dist)
        influence += w * (p - x)
        total_weight += w

    total_weight = np.maximum(total_weight, 1e-12)
    x = x + weight * (influence / total_weight)
    x = np.clip(x, lower, upper)
    x = np.sort(x)
    return x


def smolyak_sparse_grid(
    specs: Sequence[GridSpec],
    level: int,
    rule: str = "clenshaw_curtis",
    unique_decimals: int = 14,
) -> np.ndarray:
    """Construct a Smolyak sparse grid on hyper-rectangles.

    Parameters
    ----------
    specs : sequence of GridSpec
        One GridSpec per dimension. Only lower/upper bounds are used.
    level : int
        Smolyak level (>= 1). Higher levels add more points.
    rule : str, optional
        1D nested rule. Currently supports 'clenshaw_curtis'.
    unique_decimals : int, optional
        Decimal rounding for deduplicating points.

    Returns
    -------
    ndarray
        Sparse grid points of shape (n_points, d) in the original bounds.
    """
    if level < 1:
        raise ValueError("level must be >= 1")
    d = len(specs)
    if d == 0:
        return np.empty((0, 0))

    rule = rule.lower()
    if rule != "clenshaw_curtis":
        raise ValueError("Only 'clenshaw_curtis' rule is supported")

    # Precompute 1D nodes per level in [-1, 1]
    nodes_1d = {l: _clenshaw_curtis_nodes(l) for l in range(1, level + 1)}

    points = []
    for index in _smolyak_indices(d, level):
        grids = [nodes_1d[i] for i in index]
        mesh = np.meshgrid(*grids, indexing="ij")
        pts = np.stack([m.reshape(-1) for m in mesh], axis=1)
        points.append(pts)

    all_points = np.vstack(points) if points else np.empty((0, d))
    if all_points.size == 0:
        return all_points

    rounded = np.round(all_points, unique_decimals)
    unique_points = np.unique(rounded, axis=0)

    # Scale from [-1, 1] to [lower, upper] for each dimension
    lower = np.array([spec.lower for spec in specs], dtype=float)
    upper = np.array([spec.upper for spec in specs], dtype=float)
    scaled = 0.5 * (unique_points + 1.0) * (upper - lower) + lower
    return scaled


def _clenshaw_curtis_nodes(level: int) -> np.ndarray:
    """Nested Clenshaw-Curtis nodes on [-1, 1] for a given level."""
    if level < 1:
        raise ValueError("level must be >= 1")
    if level == 1:
        return np.array([0.0])
    n = 2 ** (level - 1) + 1
    k = np.arange(n)
    return np.cos(np.pi * k / (n - 1))


def _smolyak_indices(dim: int, level: int) -> List[Tuple[int, ...]]:
    """Multi-index set for Smolyak construction."""
    max_level = level
    indices = []
    for idx in product(range(1, max_level + 1), repeat=dim):
        if sum(idx) <= level + dim - 1:
            indices.append(idx)
    return indices

