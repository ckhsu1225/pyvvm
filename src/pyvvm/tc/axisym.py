"""
Axisymmetric (azimuthal) mean for TC-centered analysis.

This module provides blockwise radial binning with periodic boundary
support.  No roll or concat is needed; the algorithm slices at most
4 contiguous sub-blocks and reduces sum/count in the Dask graph.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import dask
import dask.array as da
from typing import Sequence
from dataclasses import dataclass

from ._utils import wrap_min, resolve_track
from .._utils import assign_compatible_coords


__all__ = [
    "axisym_mean",
]


# =============================================================================
# General helpers
# =============================================================================


def _periodic_slices(c: int, half: int, n: int) -> list[slice]:
    """
    Return 1 or 2 slices that cover [c-half, c+half] with periodic wrapping,
    WITHOUT concatenation. Each slice is in-bounds.
    """
    left = c - half
    right = c + half

    if 0 <= left and right < n:
        return [slice(left, right + 1)]

    # Wrap on left side: [-k .. right] -> [0..right] and [n-k..n-1]
    if left < 0:
        return [slice(0, right + 1), slice(n + left, n)]

    # Wrap on right side: [left .. n-1] and [0 .. (right-n)]
    return [slice(left, n), slice(0, (right % n) + 1)]


def _infer_dim(
    da_in: xr.DataArray,
    prefer: Sequence[str],
    *,
    optional: bool = False,
) -> str | None:
    """
    Find the first dimension name from *prefer* that exists in *da_in*.

    Raises ``ValueError`` unless *optional* is True, in which case
    returns ``None`` when no match is found.
    """
    for name in prefer:
        if name in da_in.dims:
            return name
    if optional:
        return None
    raise ValueError(
        f"Cannot infer dimension from {prefer}. "
        f"Available dims: {da_in.dims}"
    )


def _unique_dim_name(da_in: xr.DataArray, base: str) -> str:
    """Return a dimension name that does not collide with existing dims."""
    name = base
    i = 1
    while name in da_in.dims:
        name = f"{base}_{i}"
        i += 1
    return name


# =============================================================================
# Core bincount kernel (numpy)
# =============================================================================

def _bincount_sum_count(
    block_data: np.ndarray,   # (nz, ny, nx) numpy
    x_coords: np.ndarray,     # (nx,)
    y_coords: np.ndarray,     # (ny,)
    cx: float,
    cy: float,
    Lx: float,
    Ly: float,
    dr: float,
    nbins: int,
) -> np.ndarray:
    """
    Compute radial sum and count for one 3D block.

    For each vertical level, bins data points by their radial distance
    from ``(cx, cy)`` using periodic shortest-distance, and accumulates
    weighted sums and counts per bin via ``np.bincount``.

    Parameters
    ----------
    block_data : np.ndarray, shape (nz, ny, nx)
        Data values for one spatial sub-block.
    x_coords : np.ndarray, shape (nx,)
        Physical x-coordinates of the block (m).
    y_coords : np.ndarray, shape (ny,)
        Physical y-coordinates of the block (m).
    cx, cy : float
        TC center coordinates (m).
    Lx, Ly : float
        Full domain lengths in x and y (m), for periodic wrapping.
    dr : float
        Radial bin width (m).
    nbins : int
        Number of radial bins.

    Returns
    -------
    out : np.ndarray, shape (nz, 2, nbins)
        ``out[:, 0, :]`` = weighted sum, ``out[:, 1, :]`` = count.
        NaN values in *block_data* are excluded from both.
    """
    if block_data.ndim != 3:
        raise ValueError(f"block_data must be 3D (nz, ny, nx), got {block_data.shape}")

    nz, ny, nx = block_data.shape
    out = np.zeros((nz, 2, nbins), dtype=np.float64)

    # Compute 2D rbin once (ny, nx) using periodic shortest distance
    ddx = wrap_min(x_coords - cx, Lx)  # (nx,)
    ddy = wrap_min(y_coords - cy, Ly)  # (ny,)

    # Broadcasting grid without meshgrid copies: r2 (ny, nx)
    r2 = ddy[:, None] ** 2 + ddx[None, :] ** 2
    rbin2d = np.floor(np.sqrt(r2) / dr).astype(np.int32)

    valid_flat = ((rbin2d >= 0) & (rbin2d < nbins)).ravel()
    valid_idx = np.flatnonzero(valid_flat)
    if valid_idx.size == 0:
        return out

    bins_valid = rbin2d.ravel()[valid_idx]  # (npts_valid,)
    count_all = np.bincount(bins_valid, minlength=nbins)

    # Gather one vertical level at a time to avoid materializing a full
    # (nz, npts_valid) temporary array and its finite mask.
    flat_block = block_data.reshape(nz, -1)

    for k in range(nz):
        level = flat_block[k, valid_idx]
        finite_mask = np.isfinite(level)
        if not finite_mask.any():
            continue
        if finite_mask.all():
            # Fast path: no NaN at this level, so the count is reusable.
            out[k, 0] = np.bincount(bins_valid, weights=level, minlength=nbins)
            out[k, 1] = count_all
        else:
            b = bins_valid[finite_mask]
            out[k, 0] = np.bincount(b, weights=level[finite_mask], minlength=nbins)
            out[k, 1] = np.bincount(b, minlength=nbins)

    return out


def _reduce_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Small helper for delayed tree-reduce."""
    return a + b


def _chunk_bincount(
    block: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    cx: float,
    cy: float,
    Lx: float,
    Ly: float,
    dr: float,
    nbins: int,
) -> np.ndarray:
    """Partial bincount on a single dask chunk (squeezes leading time dim)."""
    if block.ndim == 4:
        block = block[0]  # (1, nz, ny_chunk, nx_chunk) → (nz, ny_chunk, nx_chunk)
    return _bincount_sum_count(block, bx, by, cx, cy, Lx, Ly, dr, nbins)


# =============================================================================
# Axisym planning helpers
# =============================================================================


@dataclass(frozen=True)
class _AxisymLayout:
    da_work: xr.DataArray
    has_time: bool
    added_z: bool
    z_dim: str
    y_dim: str
    x_dim: str
    nt: int
    nz: int
    nx: int
    ny: int
    Lx: float
    Ly: float
    dr: float
    nbins: int
    r_centers: np.ndarray
    half_x: int
    half_y: int
    x_coords: np.ndarray
    y_coords: np.ndarray
    x_offsets: np.ndarray
    y_offsets: np.ndarray
    cx_all: np.ndarray
    cy_all: np.ndarray
    cx_idx_all: np.ndarray
    cy_idx_all: np.ndarray
    delayed_chunks: np.ndarray


def _normalize_axisym_input(
    da_in: xr.DataArray,
) -> tuple[xr.DataArray, bool, str, bool, str, str]:
    """Normalize *da_in* so the working array always has time and z dims."""
    has_time = "time" in da_in.dims
    da_work = da_in if has_time else da_in.expand_dims("time")

    z_dim = _infer_dim(da_work, ("zc", "zb"), optional=True)
    added_z = z_dim is None
    if added_z:
        z_dim = _unique_dim_name(da_work, "__z_dummy__")
        da_work = da_work.expand_dims(z_dim)

    y_dim = _infer_dim(da_work, ("yc", "yb"))
    x_dim = _infer_dim(da_work, ("xc", "xb"))

    return da_work, has_time, z_dim, added_z, y_dim, x_dim


def _prepare_axisym_layout(
    da_in: xr.DataArray,
    track: xr.Dataset,
    *,
    r_max: float,
    dr: float,
) -> _AxisymLayout:
    """Pre-compute metadata, chunk layout, and track indices for axisym averaging."""
    da_work, has_time, z_dim, added_z, y_dim, x_dim = _normalize_axisym_input(da_in)

    if "dx" not in da_work.coords or "dy" not in da_work.coords:
        raise ValueError("da_in must have scalar coords 'dx' and 'dy' (meters).")
    dx = float(da_work.coords["dx"].values)
    dy = float(da_work.coords["dy"].values)

    nx = da_work.sizes[x_dim]
    ny = da_work.sizes[y_dim]
    Lx = nx * dx
    Ly = ny * dy

    nbins = int(np.floor(r_max / dr))
    if nbins <= 0:
        raise ValueError(f"nbins must be > 0, got nbins={nbins}. Check r_max/dr.")
    r_centers = (np.arange(nbins) + 0.5) * dr

    half_x = min(int(np.ceil(r_max / dx)), nx // 2)
    half_y = min(int(np.ceil(r_max / dy)), ny // 2)

    nt = da_work.sizes["time"]
    cx_all, cy_all = resolve_track(track, da_work)
    if not (np.isfinite(cx_all).all() and np.isfinite(cy_all).all()):
        raise ValueError(
            "Track contains NaN or Inf center coordinates. "
            "Ensure all center positions are finite."
        )

    # Convert center coords to integer indices for slicing.
    x0 = float(da_work.coords[x_dim].values[0])
    y0 = float(da_work.coords[y_dim].values[0])
    cx_idx_all = (np.rint((cx_all - x0) / dx).astype(np.int64) % nx)
    cy_idx_all = (np.rint((cy_all - y0) / dy).astype(np.int64) % ny)

    # Pre-extract dask array and numpy coords to avoid xarray overhead in loop.
    darr = da_work.transpose("time", z_dim, y_dim, x_dim).data
    x_coords = np.asarray(da_work.coords[x_dim].values, dtype=np.float64)
    y_coords = np.asarray(da_work.coords[y_dim].values, dtype=np.float64)
    nz = da_work.sizes[z_dim]

    # Ensure input is a dask array for to_delayed().
    if not isinstance(darr, da.Array):
        darr = da.from_array(darr, chunks=(1, nz, ny, nx))

    if darr.chunks[1] != (nz,):
        raise ValueError(
            f"axisym_mean requires the vertical dimension as a single chunk "
            f"(lev: -1), but got z chunks = {darr.chunks[1]}."
        )

    # Pre-compute chunk boundaries.
    chunks_y = darr.chunks[2]
    chunks_x = darr.chunks[3]
    y_offsets = np.concatenate([[0], np.cumsum(chunks_y)])
    x_offsets = np.concatenate([[0], np.cumsum(chunks_x)])

    # Extract pre-existing Delayed objects — one per chunk, no serialization.
    delayed_chunks = darr.to_delayed(optimize_graph=False)

    return _AxisymLayout(
        da_work=da_work,
        has_time=has_time,
        added_z=added_z,
        z_dim=z_dim,
        y_dim=y_dim,
        x_dim=x_dim,
        nt=nt,
        nz=nz,
        nx=nx,
        ny=ny,
        Lx=Lx,
        Ly=Ly,
        dr=float(dr),
        nbins=int(nbins),
        r_centers=r_centers,
        half_x=half_x,
        half_y=half_y,
        x_coords=x_coords,
        y_coords=y_coords,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        cx_all=cx_all,
        cy_all=cy_all,
        cx_idx_all=cx_idx_all,
        cy_idx_all=cy_idx_all,
        delayed_chunks=delayed_chunks,
    )


def _chunk_set_for_time(layout: _AxisymLayout, t: int) -> list[tuple[int, int]]:
    """Return the chunk indices needed for time step *t*."""
    xs = _periodic_slices(int(layout.cx_idx_all[t]), layout.half_x, layout.nx)
    ys = _periodic_slices(int(layout.cy_idx_all[t]), layout.half_y, layout.ny)

    # Map array-index slices -> set of (j, i) chunk indices.
    chunk_set: set[tuple[int, int]] = set()
    for ysl in ys:
        j0 = int(np.searchsorted(layout.y_offsets, ysl.start, side="right") - 1)
        j1 = int(np.searchsorted(layout.y_offsets, ysl.stop - 1, side="right") - 1)
        for xsl in xs:
            i0 = int(np.searchsorted(layout.x_offsets, xsl.start, side="right") - 1)
            i1 = int(np.searchsorted(layout.x_offsets, xsl.stop - 1, side="right") - 1)
            for j in range(j0, j1 + 1):
                for i in range(i0, i1 + 1):
                    chunk_set.add((j, i))

    return sorted(chunk_set)


def _build_time_task(layout: _AxisymLayout, t: int):
    """Build the delayed sum/count reduction for one time step."""
    cx = float(layout.cx_all[t])
    cy = float(layout.cy_all[t])

    # Create partial bincount tasks from pre-existing Delayed objects.
    block_tasks: list[dask.delayed] = []
    for j, i in _chunk_set_for_time(layout, t):
        bx = layout.x_coords[int(layout.x_offsets[i]):int(layout.x_offsets[i + 1])]
        by = layout.y_coords[int(layout.y_offsets[j]):int(layout.y_offsets[j + 1])]

        task = dask.delayed(_chunk_bincount)(
            layout.delayed_chunks[t, 0, j, i],
            bx,
            by,
            cx,
            cy,
            layout.Lx,
            layout.Ly,
            layout.dr,
            layout.nbins,
        )
        block_tasks.append(task)

    # Tree-reduce sum/count across blocks inside dask graph.
    total = block_tasks[0]
    for bt in block_tasks[1:]:
        total = dask.delayed(_reduce_add)(total, bt)

    return total


def _safe_divide_block(sum_block: np.ndarray, cnt_block: np.ndarray) -> np.ndarray:
    """Elementwise sum/count divide that preserves NaN for empty bins."""
    out = np.full(sum_block.shape, np.nan, dtype=np.float64)
    np.divide(sum_block, cnt_block, out=out, where=cnt_block > 0)
    return out


def _safe_divide(sum_: da.Array, cnt_: da.Array) -> da.Array:
    """Apply safe blockwise division to dask arrays with matching chunks."""
    return da.map_blocks(_safe_divide_block, sum_, cnt_, dtype=np.float64)

# =============================================================================
# Public API
# =============================================================================


def axisym_mean(
    da_in: xr.DataArray,
    track: xr.Dataset,
    *,
    r_max: float = 300e3,
    dr: float = 2e3,
) -> xr.DataArray:
    """
    Compute the axisymmetric (azimuthal) mean around a moving TC center.

    Data is binned into uniform radial bins ``[0, dr), [dr, 2·dr), …``
    up to *r_max*, and the mean is computed per bin.  Periodic boundary
    conditions are handled transparently.

    Parameters
    ----------
    da_in : xr.DataArray
        Input field with at least ``(y, x)`` horizontal dimensions.
        May also include ``time`` and/or vertical (``zc``/``zb``) dims.
        NaN values are excluded from the mean.
    track : xr.Dataset
        TC track with variables ``x`` and ``y`` (m) on the ``time`` dim.
    r_max : float
        Maximum radius (m).  Default 300 km.
    dr : float
        Radial bin width (m).  Default 2 km.

    Returns
    -------
    xr.DataArray
        Azimuthal mean with a ``r`` dimension (bin centers, in meters).
        Other dimensions are preserved:

        - (time, z, y, x) → (time, z, r)
        - (time, y, x) → (time, r)
        - (z, y, x) → (z, r)
        - (y, x) → (r)

    Notes
    -----
    The algorithm avoids ``np.roll`` and ``xr.concat`` entirely.  For
    each time step it slices at most 4 contiguous sub-blocks around the
    TC center (to handle periodic wrapping), computes ``np.bincount``
    on each block, and tree-reduces the partial sums within the Dask
    graph.  This ensures that only the spatially relevant chunks are
    read from disk.

    Examples
    --------
    >>> from pyvvm.tc import find_tc_center
    >>> from pyvvm.tc.axisym import axisym_mean
    >>> track = find_tc_center(ds, field='psi', level=1000.0)
    >>> th_az = axisym_mean(ds['th'], track, r_max=300e3, dr=2e3)
    """
    if "x" not in track or "y" not in track:
        raise ValueError("track must contain variables 'x' and 'y'.")

    layout = _prepare_axisym_layout(da_in, track, r_max=r_max, dr=dr)
    time_tasks = [_build_time_task(layout, t) for t in range(layout.nt)]

    # Convert delayed stats into dask.array then stack over time: (time, z, 2, r)
    darr_stats = da.stack(
        [
            da.from_delayed(tk, shape=(layout.nz, 2, layout.nbins), dtype=np.float64)
            for tk in time_tasks
        ],
        axis=0,
    )

    # mean = sum / count  -> (time, z, r)
    sum_ = darr_stats[:, :, 0, :]
    cnt_ = darr_stats[:, :, 1, :]
    mean = _safe_divide(sum_, cnt_)

    out = xr.DataArray(
        mean,
        dims=("time", layout.z_dim, "r"),
        coords={
            "time": layout.da_work["time"],
            layout.z_dim: layout.da_work[layout.z_dim],
            "r": ("r", layout.r_centers),
        },
        name=da_in.name,
        attrs={**da_in.attrs, "r_max": float(r_max), "dr": float(dr)},
    )
    out["r"].attrs = {"long_name": "radius", "units": "m"}
    out = assign_compatible_coords(out, da_in)

    # Restore original dimensionality.
    if layout.added_z:
        out = out.isel({layout.z_dim: 0}, drop=True)
    if not layout.has_time:
        out = out.isel(time=0)

    return out
