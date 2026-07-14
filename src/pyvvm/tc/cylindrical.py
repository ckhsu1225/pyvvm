"""
Low-level Cartesian-to-cylindrical horizontal remapping kernels.

This module contains the NumPy/SciPy implementation used by the future
xarray and Dask wrappers.  The last two axes of every input array are
interpreted as ``(y, x)``; all leading axes are treated as independent
batch dimensions.  Consequently, the same kernel handles three-dimensional
fields, surface fields without a vertical coordinate, and arbitrary derived
arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_array


__all__ = [
    "CylindricalGridSpec",
    "HorizontalRemapStencil",
    "apply_cylindrical_stencil",
    "build_cylindrical_stencil",
    "cylindrical_target_coordinates",
]


InterpolationMethod = Literal["linear", "nearest"]
BoundaryMode = Literal["periodic", "nan"]
NaNPolicy = Literal["propagate", "omit"]


def _validate_target_coordinate(
    values: ArrayLike,
    name: str,
) -> NDArray[np.float64]:
    """Return an immutable-ready copy of a target coordinate."""
    coord = np.array(values, dtype=np.float64, copy=True)
    if coord.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {coord.shape}.")
    if coord.size < 1:
        raise ValueError(f"{name} must contain at least one point.")
    if not np.isfinite(coord).all():
        raise ValueError(f"{name} must contain only finite values.")
    if coord.size > 1 and not (np.diff(coord) > 0.0).all():
        raise ValueError(f"{name} must be strictly increasing.")
    return coord


@dataclass(frozen=True, eq=False)
class CylindricalGridSpec:
    """Definition of a cylindrical target grid.

    Parameters
    ----------
    r : array-like
        One-dimensional target radii in metres.  Values must be finite,
        non-negative, and strictly increasing; non-uniform spacing is allowed.
    theta : array-like
        One-dimensional target azimuths in radians, counter-clockwise from
        ``+x``.  Values must be finite and strictly increasing, and may span
        less than one complete revolution so the same direction is not
        included twice.  Non-uniform spacing is allowed.
    method : {"linear", "nearest"}
        Horizontal interpolation method.
    boundary : {"periodic", "nan"}
        Horizontal boundary handling.  ``"nan"`` marks target points outside
        the source coordinate extent as invalid.
    nan_policy : {"propagate", "omit"}
        Missing-value handling.  ``"omit"`` renormalizes the remaining
        interpolation weights.
    """

    r: ArrayLike
    theta: ArrayLike
    method: InterpolationMethod = "linear"
    boundary: BoundaryMode = "periodic"
    nan_policy: NaNPolicy = "propagate"

    def __post_init__(self) -> None:
        radius = _validate_target_coordinate(self.r, "r")
        theta = _validate_target_coordinate(self.theta, "theta")

        if (radius < 0.0).any():
            raise ValueError("r must contain only non-negative radii.")
        if theta.size > 1 and theta[-1] - theta[0] >= 2.0 * np.pi:
            raise ValueError(
                "theta must span less than 2*pi so the same azimuth is not "
                "included twice."
            )

        if self.method not in ("linear", "nearest"):
            raise ValueError(
                f"method must be 'linear' or 'nearest', got {self.method!r}."
            )
        if self.boundary not in ("periodic", "nan"):
            raise ValueError(
                f"boundary must be 'periodic' or 'nan', got {self.boundary!r}."
            )
        if self.nan_policy not in ("propagate", "omit"):
            raise ValueError(
                "nan_policy must be 'propagate' or 'omit', "
                f"got {self.nan_policy!r}."
            )

        # A frozen dataclass does not make ndarray contents immutable.  Store
        # private copies and mark them read-only so callers cannot silently
        # change a stencil's target definition after construction.
        radius.setflags(write=False)
        theta.setflags(write=False)
        object.__setattr__(self, "r", radius)
        object.__setattr__(self, "theta", theta)

    @classmethod
    def from_spacing(
        cls,
        *,
        r_max: float,
        dr: float,
        ntheta: int,
        method: InterpolationMethod = "linear",
        boundary: BoundaryMode = "periodic",
        nan_policy: NaNPolicy = "propagate",
    ) -> CylindricalGridSpec:
        """Create a uniform, cell-centred cylindrical target grid.

        Target radii are ``dr / 2, 3 * dr / 2, ...`` with cell boundaries
        inside or equal to *r_max*.  Azimuths are uniform over one revolution
        with the endpoint at ``2*pi`` excluded.
        """
        r_max = float(r_max)
        dr = float(dr)
        if not np.isfinite(r_max) or r_max <= 0.0:
            raise ValueError(f"r_max must be finite and positive, got {r_max!r}.")
        if not np.isfinite(dr) or dr <= 0.0:
            raise ValueError(f"dr must be finite and positive, got {dr!r}.")

        if isinstance(ntheta, (bool, np.bool_)):
            raise TypeError("ntheta must be an integer, not bool.")
        try:
            ntheta_int = operator.index(ntheta)
        except TypeError as exc:
            raise TypeError(f"ntheta must be an integer, got {ntheta!r}.") from exc
        if ntheta_int < 1:
            raise ValueError(f"ntheta must be positive, got {ntheta_int}.")

        nr = int(np.floor(r_max / dr + 1.0e-12))
        if nr < 1:
            raise ValueError(
                f"r_max/dr must define at least one radial cell, got "
                f"r_max={r_max}, dr={dr}."
            )

        radius = (np.arange(nr, dtype=np.float64) + 0.5) * dr
        theta = (
            np.arange(ntheta_int, dtype=np.float64)
            * (2.0 * np.pi / ntheta_int)
        )
        return cls(
            r=radius,
            theta=theta,
            method=method,
            boundary=boundary,
            nan_policy=nan_policy,
        )

    @property
    def nr(self) -> int:
        """Number of target radii."""
        return self.r.size

    @property
    def ntheta(self) -> int:
        """Number of target azimuths."""
        return self.theta.size

    @property
    def shape(self) -> tuple[int, int]:
        """Target horizontal shape ``(ntheta, nr)``."""
        return self.ntheta, self.nr


@dataclass(frozen=True)
class HorizontalRemapStencil:
    """Sparse horizontal interpolation operator for one target centre."""

    operator: csr_array
    source_shape: tuple[int, int]
    output_shape: tuple[int, int]
    valid: NDArray[np.bool_]
    method: InterpolationMethod
    nan_policy: NaNPolicy


def cylindrical_target_coordinates(
    spec: CylindricalGridSpec,
    center_x: float,
    center_y: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return Cartesian target coordinates with shape ``(theta, r)``."""
    center_x = float(center_x)
    center_y = float(center_y)
    if not np.isfinite(center_x) or not np.isfinite(center_y):
        raise ValueError(
            "center_x and center_y must both be finite, got "
            f"({center_x!r}, {center_y!r})."
        )

    theta = spec.theta[:, None]
    radius = spec.r[None, :]
    x_target = center_x + np.cos(theta) * radius
    y_target = center_y + np.sin(theta) * radius
    return x_target, y_target


def _uniform_coordinate(
    values: ArrayLike,
    name: str,
) -> tuple[NDArray[np.float64], float]:
    """Validate a strictly increasing, uniformly spaced 1-D coordinate."""
    coord = np.asarray(values, dtype=np.float64)
    if coord.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {coord.shape}.")
    if coord.size < 2:
        raise ValueError(f"{name} must contain at least two points.")
    if not np.isfinite(coord).all():
        raise ValueError(f"{name} must contain only finite values.")

    delta = np.diff(coord)
    spacing = float(delta[0])
    if spacing <= 0.0:
        raise ValueError(f"{name} must be strictly increasing.")

    atol = max(abs(spacing) * 1.0e-12, np.finfo(np.float64).eps)
    if not np.allclose(delta, spacing, rtol=1.0e-10, atol=atol):
        raise ValueError(f"{name} must be uniformly spaced.")
    return coord, spacing


def _linear_entries(
    sx: NDArray[np.float64],
    sy: NDArray[np.float64],
    nx: int,
    ny: int,
    boundary: BoundaryMode,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.bool_]]:
    """Return flattened bilinear source indices, weights, and valid rows."""
    if boundary == "periodic":
        sx_work = np.mod(sx, nx)
        sy_work = np.mod(sy, ny)
        i0 = np.floor(sx_work).astype(np.int64)
        j0 = np.floor(sy_work).astype(np.int64)
        fx = sx_work - i0
        fy = sy_work - j0
        i1 = (i0 + 1) % nx
        j1 = (j0 + 1) % ny
        valid = np.ones(sx.shape, dtype=bool)
    else:
        valid = (
            (sx >= 0.0)
            & (sx <= nx - 1)
            & (sy >= 0.0)
            & (sy <= ny - 1)
        )
        i0 = np.clip(np.floor(sx).astype(np.int64), 0, nx - 2)
        j0 = np.clip(np.floor(sy).astype(np.int64), 0, ny - 2)
        fx = sx - i0
        fy = sy - j0
        i1 = i0 + 1
        j1 = j0 + 1

    indices = np.stack(
        (
            j0 * nx + i0,
            j0 * nx + i1,
            j1 * nx + i0,
            j1 * nx + i1,
        ),
        axis=-1,
    )
    weights = np.stack(
        (
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        ),
        axis=-1,
    )
    return indices, weights, valid


def _nearest_entries(
    sx: NDArray[np.float64],
    sy: NDArray[np.float64],
    nx: int,
    ny: int,
    boundary: BoundaryMode,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.bool_]]:
    """Return flattened nearest-neighbour source indices and valid rows."""
    if boundary == "periodic":
        i = np.floor(np.mod(sx, nx) + 0.5).astype(np.int64) % nx
        j = np.floor(np.mod(sy, ny) + 0.5).astype(np.int64) % ny
        valid = np.ones(sx.shape, dtype=bool)
    else:
        valid = (
            (sx >= 0.0)
            & (sx <= nx - 1)
            & (sy >= 0.0)
            & (sy <= ny - 1)
        )
        i = np.clip(np.floor(sx + 0.5).astype(np.int64), 0, nx - 1)
        j = np.clip(np.floor(sy + 0.5).astype(np.int64), 0, ny - 1)

    indices = (j * nx + i)[..., None]
    weights = np.ones(indices.shape, dtype=np.float64)
    return indices, weights, valid


def build_cylindrical_stencil(
    x: ArrayLike,
    y: ArrayLike,
    *,
    center_x: float,
    center_y: float,
    spec: CylindricalGridSpec,
) -> HorizontalRemapStencil:
    """Build a sparse Cartesian-to-cylindrical horizontal remap stencil.

    Parameters
    ----------
    x, y : array-like
        Strictly increasing, uniformly spaced source coordinates.
    center_x, center_y : float
        Cylindrical-grid centre in source-coordinate units.
    spec : CylindricalGridSpec
        Target-grid and interpolation settings.
    """
    x_coord, dx = _uniform_coordinate(x, "x")
    y_coord, dy = _uniform_coordinate(y, "y")
    nx = x_coord.size
    ny = y_coord.size

    x_target, y_target = cylindrical_target_coordinates(
        spec,
        center_x,
        center_y,
    )
    sx = (x_target - x_coord[0]) / dx
    sy = (y_target - y_coord[0]) / dy

    if spec.method == "linear":
        indices, weights, valid = _linear_entries(
            sx,
            sy,
            nx,
            ny,
            spec.boundary,
        )
    else:
        indices, weights, valid = _nearest_entries(
            sx,
            sy,
            nx,
            ny,
            spec.boundary,
        )

    n_output = spec.ntheta * spec.nr
    n_neighbours = indices.shape[-1]
    rows = np.repeat(np.arange(n_output, dtype=np.int64), n_neighbours)
    cols = indices.reshape(-1)
    data = weights.astype(np.float32, copy=False).reshape(-1)

    # Invalid rows and zero-weight neighbours must not be stored.  In
    # particular, omitting explicit zero weights prevents ``0 * NaN`` from
    # contaminating a target that lies exactly on a source grid point.
    keep = np.repeat(valid.reshape(-1), n_neighbours) & (data != 0.0)
    operator_csr = csr_array(
        (data[keep], (rows[keep], cols[keep])),
        shape=(n_output, ny * nx),
        dtype=np.float32,
    )
    operator_csr.sum_duplicates()
    operator_csr.eliminate_zeros()
    operator_csr.sort_indices()

    return HorizontalRemapStencil(
        operator=operator_csr,
        source_shape=(ny, nx),
        output_shape=spec.shape,
        valid=valid.reshape(-1),
        method=spec.method,
        nan_policy=spec.nan_policy,
    )


def apply_cylindrical_stencil(
    values: ArrayLike,
    stencil: HorizontalRemapStencil,
    *,
    nan_policy: NaNPolicy | None = None,
) -> NDArray:
    """Apply a remap stencil to an array ending in ``(y, x)``.

    All leading dimensions are flattened into one batch dimension during the
    sparse multiplication and restored in the result.  For example:

    - ``(z, y, x) -> (z, theta, r)``
    - ``(time, z, y, x) -> (time, z, theta, r)``
    - ``(time, y, x) -> (time, theta, r)``
    - ``(y, x) -> (theta, r)``
    """
    if nan_policy is None:
        nan_policy = stencil.nan_policy
    if nan_policy not in ("propagate", "omit"):
        raise ValueError(
            "nan_policy must be 'propagate' or 'omit', "
            f"got {nan_policy!r}."
        )

    array = np.asarray(values)
    if array.ndim < 2:
        raise ValueError(
            f"values must have at least two dimensions (..., y, x), got {array.shape}."
        )
    if array.shape[-2:] != stencil.source_shape:
        raise ValueError(
            "values horizontal shape does not match stencil source shape: "
            f"got {array.shape[-2:]}, expected {stencil.source_shape}."
        )
    if not (
        np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError(f"values must have a numeric dtype, got {array.dtype}.")

    leading_shape = array.shape[:-2]
    n_source = stencil.source_shape[0] * stencil.source_shape[1]
    work_dtype = np.result_type(array.dtype, np.float32)
    flat = array.reshape(-1, n_source).astype(work_dtype, copy=False)

    if nan_policy == "propagate":
        result = (stencil.operator @ flat.T).T
    else:
        finite = np.isfinite(flat)
        safe = np.where(finite, flat, 0)
        numerator = (stencil.operator @ safe.T).T
        denominator = (
            stencil.operator @ finite.astype(stencil.operator.dtype).T
        ).T
        result = np.full(numerator.shape, np.nan, dtype=numerator.dtype)
        np.divide(
            numerator,
            denominator,
            out=result,
            where=denominator > 0.0,
        )

    if not stencil.valid.all():
        result[:, ~stencil.valid] = np.nan

    return result.reshape(leading_shape + stencil.output_shape)
