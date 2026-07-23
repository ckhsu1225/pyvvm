"""Cartesian-to-cylindrical horizontal remapping.

The low-level NumPy/SciPy kernel interprets the last two axes of an input
array as ``(y, x)`` and treats every leading axis as an independent batch
dimension.  The high-level xarray wrappers retain those leading dimensions,
support moving centres, and construct a lazy Dask graph when the source data
are Dask-backed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import operator
from typing import Literal

import dask
import dask.array as dask_array
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_array
import xarray as xr

from ..utils import assign_compatible_coords


__all__ = [
    "CylindricalGridSpec",
    "HorizontalRemapStencil",
    "apply_cylindrical_stencil",
    "build_cylindrical_stencil",
    "cylindrical_target_coordinates",
    "remap_dataarray",
    "remap_dataset",
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


# =============================================================================
# xarray / Dask wrappers
# =============================================================================


_X_DIM_CANDIDATES = ("xc", "xb")
_Y_DIM_CANDIDATES = ("yc", "yb")


@dataclass(frozen=True)
class _CenterPlan:
    """Centre coordinates aligned with an input DataArray."""

    x: NDArray[np.float64]
    y: NDArray[np.float64]
    varies_with_time: bool


def _infer_horizontal_dim(
    da_in: xr.DataArray,
    explicit: str | None,
    candidates: tuple[str, ...],
    axis_name: str,
) -> str:
    """Resolve one horizontal dimension and validate its coordinate."""
    if explicit is not None:
        if explicit not in da_in.dims:
            raise ValueError(
                f"{axis_name}_dim={explicit!r} is not an input dimension. "
                f"Available dimensions: {da_in.dims}."
            )
        dim = explicit
    else:
        matches = [name for name in candidates if name in da_in.dims]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot infer a unique {axis_name} dimension from {candidates}. "
                f"Found {matches or 'none'} in dimensions {da_in.dims}; pass "
                f"{axis_name}_dim explicitly."
            )
        dim = matches[0]

    if dim not in da_in.coords:
        raise ValueError(f"Input must provide a one-dimensional coordinate {dim!r}.")
    coord = da_in.coords[dim]
    if coord.dims != (dim,):
        raise ValueError(
            f"Coordinate {dim!r} must have dimensions ({dim!r},), got {coord.dims}."
        )
    return dim


def _track_component_for_input(
    component: xr.DataArray,
    da_in: xr.DataArray,
    name: str,
) -> NDArray[np.float64]:
    """Align one scalar/time-dependent track component with *da_in*."""
    extra_dims = [
        dim
        for dim in component.dims
        if dim != "time" and component.sizes[dim] != 1
    ]
    if extra_dims:
        raise ValueError(
            f"Center variable {name!r} may only vary along 'time'; "
            f"non-singleton extra dimensions are {extra_dims}."
        )
    if any(dim != "time" for dim in component.dims):
        component = component.squeeze(
            [dim for dim in component.dims if dim != "time"],
            drop=True,
        )

    if "time" in da_in.dims:
        nt = da_in.sizes["time"]
        if "time" in component.dims:
            try:
                component = component.sel(time=da_in.coords["time"])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Center variable {name!r} cannot be aligned exactly with "
                    "the input time coordinate."
                ) from exc
            values = np.asarray(component.values, dtype=np.float64)
            if values.shape != (nt,):
                raise ValueError(
                    f"Aligned center variable {name!r} has shape {values.shape}; "
                    f"expected ({nt},)."
                )
        else:
            values_scalar = np.asarray(component.values, dtype=np.float64).squeeze()
            if values_scalar.ndim != 0:
                raise ValueError(
                    f"Center variable {name!r} must be scalar or time-dependent."
                )
            values = np.full(nt, float(values_scalar), dtype=np.float64)
    else:
        if "time" in component.dims:
            time_coord = da_in.coords.get("time")
            if time_coord is not None and time_coord.ndim == 0:
                try:
                    component = component.sel(time=time_coord)
                except (KeyError, ValueError) as exc:
                    raise ValueError(
                        f"Center variable {name!r} has no value matching the "
                        "input's scalar time coordinate."
                    ) from exc
            elif component.sizes["time"] == 1:
                component = component.isel(time=0)
            else:
                raise ValueError(
                    "A time-dependent center requires an input 'time' dimension "
                    "or a scalar 'time' coordinate."
                )

        values_scalar = np.asarray(component.values, dtype=np.float64).squeeze()
        if values_scalar.ndim != 0:
            raise ValueError(f"Center variable {name!r} must resolve to one value.")
        values = np.asarray([float(values_scalar)], dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(f"Center variable {name!r} contains NaN or Inf.")
    return values


def _resolve_center_plan(
    center: xr.Dataset | Sequence[float],
    da_in: xr.DataArray,
) -> _CenterPlan:
    """Normalize a fixed center or moving-center Dataset for *da_in*."""
    nt = da_in.sizes.get("time", 1)

    if isinstance(center, xr.Dataset):
        if "x" not in center or "y" not in center:
            raise ValueError("center Dataset must contain variables 'x' and 'y'.")
        center_x = _track_component_for_input(center["x"], da_in, "x")
        center_y = _track_component_for_input(center["y"], da_in, "y")
    else:
        if isinstance(center, (str, bytes)):
            raise TypeError(
                "center must be an xr.Dataset or a two-item "
                "(center_x, center_y) sequence."
            )
        try:
            center_values = np.asarray(center, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "center must be an xr.Dataset or a two-item "
                "(center_x, center_y) sequence."
            ) from exc
        if center_values.shape != (2,):
            raise ValueError(
                "A fixed center must contain exactly two scalar values "
                f"(center_x, center_y), got shape {center_values.shape}."
            )
        if not np.isfinite(center_values).all():
            raise ValueError("Fixed center coordinates must both be finite.")
        center_x = np.full(nt, center_values[0], dtype=np.float64)
        center_y = np.full(nt, center_values[1], dtype=np.float64)

    if center_x.shape != center_y.shape:
        raise ValueError(
            "Center x and y coordinates must resolve to the same shape, got "
            f"{center_x.shape} and {center_y.shape}."
        )

    varies = bool(
        center_x.size > 1
        and (
            not np.equal(center_x, center_x[0]).all()
            or not np.equal(center_y, center_y[0]).all()
        )
    )
    return _CenterPlan(x=center_x, y=center_y, varies_with_time=varies)


def _build_stencil_collection(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    center_plan: _CenterPlan,
    spec: CylindricalGridSpec,
    *,
    lazy: bool,
) -> tuple[object, ...]:
    """Build eager stencils or shared lazy stencil tasks."""
    indices = (
        range(center_plan.x.size)
        if center_plan.varies_with_time
        else range(1)
    )

    if lazy:
        delayed_builder = dask.delayed(build_cylindrical_stencil, pure=True)
        return tuple(
            delayed_builder(
                x,
                y,
                center_x=float(center_plan.x[index]),
                center_y=float(center_plan.y[index]),
                spec=spec,
            )
            for index in indices
        )

    return tuple(
        build_cylindrical_stencil(
            x,
            y,
            center_x=float(center_plan.x[index]),
            center_y=float(center_plan.y[index]),
            spec=spec,
        )
        for index in indices
    )


def _apply_dask_stencils(
    source: dask_array.Array,
    stencils: tuple[object, ...],
    center_plan: _CenterPlan,
    spec: CylindricalGridSpec,
    time_axis: int | None,
) -> dask_array.Array:
    """Construct a lazy block graph while sharing each time's stencil."""
    rechunk: dict[int, int] = {
        source.ndim - 2: -1,
        source.ndim - 1: -1,
    }
    if center_plan.varies_with_time:
        if time_axis is None:
            raise ValueError("A varying center requires an input 'time' dimension.")
        rechunk[time_axis] = 1
    source = source.rechunk(rechunk)

    source_blocks = source.to_delayed(optimize_graph=False)
    output_blocks = np.empty(source_blocks.shape, dtype=object)
    output_dtype = np.result_type(source.dtype, np.float32)

    for block_index in np.ndindex(source_blocks.shape[:-2]):
        stencil_index = block_index[time_axis] if center_plan.varies_with_time else 0
        task = dask.delayed(apply_cylindrical_stencil)(
            source_blocks[block_index + (0, 0)],
            stencils[stencil_index],
        )
        block_shape = tuple(
            source.chunks[axis][block_index[axis]]
            for axis in range(source.ndim - 2)
        ) + spec.shape
        output_blocks[block_index + (0, 0)] = dask_array.from_delayed(
            task,
            shape=block_shape,
            dtype=output_dtype,
        )

    return dask_array.block(output_blocks.tolist())


def _apply_eager_stencils(
    source: NDArray,
    stencils: tuple[object, ...],
    center_plan: _CenterPlan,
    time_axis: int | None,
) -> NDArray:
    """Apply one fixed stencil or one stencil per input time."""
    if not center_plan.varies_with_time:
        return apply_cylindrical_stencil(source, stencils[0])

    if time_axis is None:
        raise ValueError("A varying center requires an input 'time' dimension.")
    source_by_time = np.moveaxis(source, time_axis, 0)
    remapped = np.stack(
        [
            apply_cylindrical_stencil(source_by_time[index], stencils[index])
            for index in range(source_by_time.shape[0])
        ],
        axis=0,
    )
    return np.moveaxis(remapped, 0, time_axis)


def _remap_dataarray_impl(
    da_in: xr.DataArray,
    center: xr.Dataset | Sequence[float],
    *,
    spec: CylindricalGridSpec,
    x_dim: str | None,
    y_dim: str | None,
    stencil_cache: dict[tuple[str, str, bool, bool], tuple[object, ...]],
) -> xr.DataArray:
    """Implementation shared by the DataArray and Dataset entry points."""
    if not isinstance(da_in, xr.DataArray):
        raise TypeError(f"da_in must be an xr.DataArray, got {type(da_in).__name__}.")
    if not isinstance(spec, CylindricalGridSpec):
        raise TypeError(
            f"spec must be a CylindricalGridSpec, got {type(spec).__name__}."
        )
    if not (
        np.issubdtype(da_in.dtype, np.number)
        or np.issubdtype(da_in.dtype, np.bool_)
    ):
        raise TypeError(f"da_in must have a numeric dtype, got {da_in.dtype}.")

    x_name = _infer_horizontal_dim(da_in, x_dim, _X_DIM_CANDIDATES, "x")
    y_name = _infer_horizontal_dim(da_in, y_dim, _Y_DIM_CANDIDATES, "y")
    if x_name == y_name:
        raise ValueError("x_dim and y_dim must be different dimensions.")

    leading_dims = tuple(
        dim for dim in da_in.dims if dim not in (y_name, x_name)
    )
    collisions = {"theta", "r"}.intersection(leading_dims)
    if collisions:
        raise ValueError(
            "Input leading dimensions collide with cylindrical output dimensions: "
            f"{sorted(collisions)}. Rename them before remapping."
        )

    source_da = da_in.transpose(*leading_dims, y_name, x_name)
    source_data = source_da.data
    lazy = isinstance(source_data, dask_array.Array)
    center_plan = _resolve_center_plan(center, da_in)
    time_axis = leading_dims.index("time") if "time" in leading_dims else None

    x_values = np.asarray(da_in.coords[x_name].values, dtype=np.float64)
    y_values = np.asarray(da_in.coords[y_name].values, dtype=np.float64)
    cache_key = (x_name, y_name, lazy, center_plan.varies_with_time)
    stencils = stencil_cache.get(cache_key)
    if stencils is None:
        stencils = _build_stencil_collection(
            x_values,
            y_values,
            center_plan,
            spec,
            lazy=lazy,
        )
        stencil_cache[cache_key] = stencils

    if lazy:
        remapped_data = _apply_dask_stencils(
            source_data,
            stencils,
            center_plan,
            spec,
            time_axis,
        )
    else:
        remapped_data = _apply_eager_stencils(
            np.asarray(source_data),
            stencils,
            center_plan,
            time_axis,
        )

    out = xr.DataArray(
        remapped_data,
        dims=leading_dims + ("theta", "r"),
        coords={
            "theta": ("theta", spec.theta),
            "r": ("r", spec.r),
        },
        name=da_in.name,
        attrs=dict(da_in.attrs),
    )
    out = assign_compatible_coords(out, da_in)

    if "time" in out.dims:
        out = out.assign_coords(
            center_x=("time", center_plan.x),
            center_y=("time", center_plan.y),
        )
    else:
        out = out.assign_coords(
            center_x=float(center_plan.x[0]),
            center_y=float(center_plan.y[0]),
        )

    out.coords["theta"].attrs = {
        "long_name": "azimuth",
        "units": "rad",
        "comment": "Counter-clockwise from the positive x-axis.",
    }
    out.coords["r"].attrs = {"long_name": "radius", "units": "m"}
    out.coords["center_x"].attrs = {
        "long_name": "cylindrical grid center x-coordinate",
        "units": "m",
    }
    out.coords["center_y"].attrs = {
        "long_name": "cylindrical grid center y-coordinate",
        "units": "m",
    }
    out.attrs.update(
        {
            "cylindrical_interpolation": spec.method,
            "cylindrical_boundary": spec.boundary,
            "cylindrical_nan_policy": spec.nan_policy,
        }
    )
    return out


def remap_dataarray(
    da_in: xr.DataArray,
    center: xr.Dataset | Sequence[float],
    *,
    spec: CylindricalGridSpec,
    x_dim: str | None = None,
    y_dim: str | None = None,
) -> xr.DataArray:
    """Remap one DataArray from a Cartesian to a cylindrical grid.

    Parameters
    ----------
    da_in : xr.DataArray
        Numeric input with one x and one y dimension.  Any other dimensions,
        including ``time``, ``zc`` or ``zb``, are preserved.  Surface fields
        therefore need no artificial vertical dimension.
    center : xr.Dataset or two-item sequence
        Either a moving-center Dataset containing ``x`` and ``y`` variables,
        optionally on ``time``, or a fixed ``(center_x, center_y)`` pair.
    spec : CylindricalGridSpec
        Target radii, azimuths and interpolation settings.
    x_dim, y_dim : str, optional
        Source horizontal dimensions.  By default VVM dimensions are inferred
        independently from ``xc``/``xb`` and ``yc``/``yb``.

    Returns
    -------
    xr.DataArray
        A DataArray whose source horizontal dimensions are replaced by
        ``(theta, r)``.  Dask-backed inputs remain lazy.

    Notes
    -----
    A varying center is resolved once per time.  Its stencil is shared by all
    vertical and other leading-dimension chunks for that time.  Horizontal
    source dimensions are rechunked to one core block when necessary.
    """
    return _remap_dataarray_impl(
        da_in,
        center,
        spec=spec,
        x_dim=x_dim,
        y_dim=y_dim,
        stencil_cache={},
    )


def remap_dataset(
    ds_in: xr.Dataset,
    center: xr.Dataset | Sequence[float],
    *,
    spec: CylindricalGridSpec,
    variables: str | Sequence[str] | None = None,
) -> xr.Dataset:
    """Remap selected or all horizontally gridded Dataset variables.

    When *variables* is ``None``, every data variable containing exactly one
    VVM x dimension (``xc`` or ``xb``) and one VVM y dimension (``yc`` or
    ``yb``) is remapped.  Variables without horizontal dimensions are omitted.
    Different C-grid locations and vertical dimensions are handled
    independently without broadcasting them against one another.
    """
    if not isinstance(ds_in, xr.Dataset):
        raise TypeError(f"ds_in must be an xr.Dataset, got {type(ds_in).__name__}.")
    if not isinstance(spec, CylindricalGridSpec):
        raise TypeError(
            f"spec must be a CylindricalGridSpec, got {type(spec).__name__}."
        )

    if variables is None:
        selected = [
            name
            for name, variable in ds_in.data_vars.items()
            if len([dim for dim in _X_DIM_CANDIDATES if dim in variable.dims]) == 1
            and len([dim for dim in _Y_DIM_CANDIDATES if dim in variable.dims]) == 1
        ]
    else:
        requested = [variables] if isinstance(variables, str) else list(variables)
        selected = list(dict.fromkeys(requested))
        missing = [name for name in selected if name not in ds_in.data_vars]
        if missing:
            raise ValueError(f"Dataset data variables not found: {missing}.")

    if not selected:
        raise ValueError("No horizontally gridded data variables were selected.")

    stencil_cache: dict[
        tuple[str, str, bool, bool], tuple[object, ...]
    ] = {}
    remapped = {
        name: _remap_dataarray_impl(
            ds_in[name],
            center,
            spec=spec,
            x_dim=None,
            y_dim=None,
            stencil_cache=stencil_cache,
        )
        for name in selected
    }
    out = xr.Dataset(remapped, attrs=dict(ds_in.attrs))
    out.attrs.update(
        {
            "cylindrical_interpolation": spec.method,
            "cylindrical_boundary": spec.boundary,
            "cylindrical_nan_policy": spec.nan_policy,
        }
    )
    return out
