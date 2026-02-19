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

from ._utils import wrap_min, resolve_track
from .._utils import assign_compatible_coords


__all__ = [
    "axisym_mean",
]


# =============================================================================
# Helpers
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
    if not valid_flat.any():
        return out

    bins_valid = rbin2d.ravel()[valid_flat]  # (npts_valid,)

    # Vectorized: extract all z-levels at valid points → (nz, npts_valid)
    data_valid = block_data.reshape(nz, -1)[:, valid_flat]
    finite_mask = np.isfinite(data_valid)  # (nz, npts_valid)

    for k in range(nz):
        fm = finite_mask[k]
        if not fm.any():
            continue
        if fm.all():
            # Fast path: no NaN at this level
            out[k, 0] = np.bincount(bins_valid, weights=data_valid[k], minlength=nbins)
            out[k, 1] = np.bincount(bins_valid, minlength=nbins)
        else:
            b = bins_valid[fm]
            out[k, 0] = np.bincount(b, weights=data_valid[k, fm], minlength=nbins)
            out[k, 1] = np.bincount(b, minlength=nbins)

    return out


def _reduce_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Small helper for delayed tree-reduce."""
    return a + b


# =============================================================================
# Azimuthal averaging
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

    # Normalize to a working shape that always has both time and z.
    has_time = "time" in da_in.dims
    da_work = da_in if has_time else da_in.expand_dims("time")

    z_dim = _infer_dim(da_work, ("zc", "zb"), optional=True)
    added_z = z_dim is None
    if added_z:
        z_dim = _unique_dim_name(da_work, "__z_dummy__")
        da_work = da_work.expand_dims(z_dim)

    y_dim = _infer_dim(da_work, ("yc", "yb"))
    x_dim = _infer_dim(da_work, ("xc", "xb"))

    # Scalars
    if "dx" not in da_work.coords or "dy" not in da_work.coords:
        raise ValueError("da_in must have scalar coords 'dx' and 'dy' (meters).")
    dx = float(da_work.coords["dx"].values)
    dy = float(da_work.coords["dy"].values)

    nx = da_work.sizes[x_dim]
    ny = da_work.sizes[y_dim]
    Lx = nx * dx
    Ly = ny * dy

    # Bin setup
    nbins = int(np.floor(r_max / dr))
    if nbins <= 0:
        raise ValueError(f"nbins must be > 0, got nbins={nbins}. Check r_max/dr.")
    r_centers = (np.arange(nbins) + 0.5) * dr  # (nbins,)

    # Half-width in grid points for the box window that encloses circle
    half_x = min(int(np.ceil(r_max / dx)), nx // 2)
    half_y = min(int(np.ceil(r_max / dy)), ny // 2)

    nt = da_work.sizes["time"]
    cx_all, cy_all = resolve_track(track, da_work)

    # Validate that all center coordinates are finite.
    if not (np.isfinite(cx_all).all() and np.isfinite(cy_all).all()):
        raise ValueError(
            "Track contains NaN or Inf center coordinates. "
            "Ensure all center positions are finite."
        )

    # Convert center coords to integer indices for slicing
    x0 = float(da_work.coords[x_dim].values[0])
    y0 = float(da_work.coords[y_dim].values[0])
    cx_idx_all = (np.rint((cx_all - x0) / dx).astype(np.int64) % nx)
    cy_idx_all = (np.rint((cy_all - y0) / dy).astype(np.int64) % ny)

    # Pre-extract dask array and numpy coords to avoid xarray overhead in loop.
    darr = da_work.transpose("time", z_dim, y_dim, x_dim).data  # (nt, nz, ny, nx)
    x_coords = np.asarray(da_work.coords[x_dim].values, dtype=np.float64)
    y_coords = np.asarray(da_work.coords[y_dim].values, dtype=np.float64)
    nz = da_work.sizes[z_dim]

    # Build per-time delayed stats (nz,2,nbins), reduced inside dask graph
    time_tasks: list[dask.delayed] = []
    dr_f = float(dr)
    nbins_i = int(nbins)

    for t in range(nt):
        cx = float(cx_all[t])
        cy = float(cy_all[t])
        cx_idx = int(cx_idx_all[t])
        cy_idx = int(cy_idx_all[t])

        xs = _periodic_slices(cx_idx, half_x, nx)
        ys = _periodic_slices(cy_idx, half_y, ny)

        block_tasks: list[dask.delayed] = []

        for ysl in ys:
            for xsl in xs:
                block = darr[t, :, ysl, xsl]  # dask array (nz, sub_ny, sub_nx)
                bx = x_coords[xsl]
                by = y_coords[ysl]

                task = dask.delayed(_bincount_sum_count)(
                    block, bx, by,
                    cx, cy, Lx, Ly,
                    dr_f, nbins_i,
                )
                block_tasks.append(task)

        # Tree-reduce sum/count across blocks inside dask graph
        total = block_tasks[0]
        for bt in block_tasks[1:]:
            total = dask.delayed(_reduce_add)(total, bt)

        time_tasks.append(total)

    # Convert delayed stats into dask.array then stack over time: (time, z, 2, r)
    darr_stats = da.stack(
        [da.from_delayed(tk, shape=(nz, 2, nbins), dtype=np.float64) for tk in time_tasks],
        axis=0
    )

    # mean = sum / count  -> (time, z, r)
    sum_ = darr_stats[:, :, 0, :]
    cnt_ = darr_stats[:, :, 1, :]
    mean = sum_ / cnt_

    out = xr.DataArray(
        mean,
        dims=("time", z_dim, "r"),
        coords={
            "time": da_work["time"],
            z_dim: da_work[z_dim],
            "r": ("r", r_centers),
        },
        name=da_in.name,
        attrs={**da_in.attrs, "r_max": float(r_max), "dr": float(dr)},
    )
    out["r"].attrs = {"long_name": "radius", "units": "m"}
    out = assign_compatible_coords(out, da_in)

    # Restore original dimensionality.
    if added_z:
        out = out.isel({z_dim: 0}, drop=True)
    if not has_time:
        out = out.isel(time=0)

    return out
