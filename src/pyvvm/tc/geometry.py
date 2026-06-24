"""
TC-centered polar geometry utilities.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import dask.array as dask_array

from .utils import resolve_track, wrap_min

__all__ = [
    'polar_geometry',
]


def polar_geometry(
    da: xr.DataArray,
    track: xr.Dataset,
) -> xr.Dataset:
    """
    Compute TC-centered polar geometry on the horizontal center grid.

    Parameters
    ----------
    da : xr.DataArray
        Reference field on the center grid (xc, yc).  Must have scalar coords
        ``dx`` and ``dy``.  If a ``time`` dimension is present, *track* is
        aligned to that time coordinate.
    track : xr.Dataset
        TC track with variables ``x`` and ``y`` in metres.

    Returns
    -------
    xr.Dataset
        Dataset with ``r`` (m), ``cos_theta`` and ``sin_theta``.  At the TC
        center where ``r == 0``, both unit-vector components are set to zero.
    """
    if "xc" not in da.dims or "yc" not in da.dims:
        raise ValueError("da must be on center grid (xc, yc).")

    dx = float(da.coords["dx"].values)
    dy = float(da.coords["dy"].values)
    nx = da.sizes["xc"]
    ny = da.sizes["yc"]
    Lx = nx * dx
    Ly = ny * dy

    has_time = "time" in da.dims

    if has_time:
        cx_all, cy_all = resolve_track(track, da)
        x_np = np.asarray(da["xc"].values, dtype=np.float64)
        y_np = np.asarray(da["yc"].values, dtype=np.float64)

        t_chunks = da.chunksizes.get("time") or (da.sizes["time"],)
        x_chunks = da.chunksizes.get("xc") or (nx,)
        y_chunks = da.chunksizes.get("yc") or (ny,)

        x_coord = xr.DataArray(
            dask_array.from_array(x_np, chunks=x_chunks),
            dims=("xc",),
            coords={"xc": da["xc"]},
        )
        y_coord = xr.DataArray(
            dask_array.from_array(y_np, chunks=y_chunks),
            dims=("yc",),
            coords={"yc": da["yc"]},
        )
        cx = xr.DataArray(
            dask_array.from_array(cx_all, chunks=t_chunks),
            dims=("time",),
            coords={"time": da["time"]},
        )
        cy = xr.DataArray(
            dask_array.from_array(cy_all, chunks=t_chunks),
            dims=("time",),
            coords={"time": da["time"]},
        )

        ddx = wrap_min(x_coord - cx, Lx).transpose("time", "xc")
        ddy = wrap_min(y_coord - cy, Ly).transpose("time", "yc")
    else:
        if "time" in track.dims:
            if track.sizes["time"] != 1:
                raise ValueError(
                    "track has multiple times but da has no 'time' dim."
                )
            cx = float(track["x"].isel(time=0).values)
            cy = float(track["y"].isel(time=0).values)
        else:
            cx = float(track["x"].squeeze().values)
            cy = float(track["y"].squeeze().values)

        x_np = np.asarray(da["xc"].values, dtype=np.float64)
        y_np = np.asarray(da["yc"].values, dtype=np.float64)
        ddx_np = wrap_min(x_np - cx, Lx)
        ddy_np = wrap_min(y_np - cy, Ly)

        x_chunks = da.chunksizes.get("xc") or (nx,)
        y_chunks = da.chunksizes.get("yc") or (ny,)

        ddx = xr.DataArray(
            dask_array.from_array(ddx_np, chunks=x_chunks),
            dims=("xc",),
            coords={"xc": da["xc"]},
        )
        ddy = xr.DataArray(
            dask_array.from_array(ddy_np, chunks=y_chunks),
            dims=("yc",),
            coords={"yc": da["yc"]},
        )

    r = np.hypot(ddy, ddx)
    safe_r = xr.where(r > 0.0, r, 1.0)
    inv_r = xr.where(r > 0.0, 1.0 / safe_r, 0.0)

    cos_theta = (ddx * inv_r).transpose(*r.dims)
    sin_theta = (ddy * inv_r).transpose(*r.dims)

    r.attrs.update({"long_name": "TC-centered radius", "units": "m"})
    cos_theta.attrs.update({
        "long_name": "cosine of TC-centered azimuth angle",
        "units": "1",
    })
    sin_theta.attrs.update({
        "long_name": "sine of TC-centered azimuth angle",
        "units": "1",
    })

    return xr.Dataset({
        "r": r,
        "cos_theta": cos_theta,
        "sin_theta": sin_theta,
    })
