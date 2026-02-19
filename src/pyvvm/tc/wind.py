"""
Wind decomposition into radial/tangential components relative to TC center.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import dask.array as da

from ._utils import resolve_track, wrap_min


__all__ = [
    'compute_vr_vt',
]


def compute_vr_vt(
    u: xr.DataArray,
    v: xr.DataArray,
    track: xr.Dataset,
) -> xr.Dataset:
    """
    Decompose Cartesian (u, v) into radial and tangential wind.

    Parameters
    ----------
    u, v : xr.DataArray
        Wind components on center grid (xc, yc).  Must have scalar coords
        ``dx``, ``dy``.
    track : xr.Dataset
        TC track with variables ``x`` and ``y`` (m) on ``time`` dim.

    Returns
    -------
    xr.Dataset
        Dataset with variables ``vr`` (radial, positive outward) and
        ``vt`` (tangential, positive cyclonic / counter-clockwise).
    """
    if "xc" not in u.dims or "yc" not in u.dims:
        raise ValueError("u must be on center grid (xc, yc).")
    if "xc" not in v.dims or "yc" not in v.dims:
        raise ValueError("v must be on center grid (xc, yc).")

    has_time = "time" in u.dims
    if has_time != ("time" in v.dims):
        raise ValueError("u and v must both have 'time' dim or both be time-independent.")

    dx = float(u.coords["dx"].values)
    dy = float(u.coords["dy"].values)
    nx = u.sizes["xc"]
    ny = u.sizes["yc"]
    Lx = nx * dx
    Ly = ny * dy

    if has_time:
        cx_all, cy_all = resolve_track(track, u)
        x_np = np.asarray(u["xc"].values, dtype=np.float64)
        y_np = np.asarray(u["yc"].values, dtype=np.float64)
        ddx_np = wrap_min(x_np[None, :] - cx_all[:, None], Lx)
        ddy_np = wrap_min(y_np[None, :] - cy_all[:, None], Ly)

        t_chunks = u.chunksizes.get("time") or (u.sizes["time"],)
        x_chunks = u.chunksizes.get("xc") or (nx,)
        y_chunks = u.chunksizes.get("yc") or (ny,)

        ddx = xr.DataArray(
            da.from_array(ddx_np, chunks=(t_chunks, x_chunks)),
            dims=("time", "xc"),
            coords={"time": u["time"], "xc": u["xc"]},
        )
        ddy = xr.DataArray(
            da.from_array(ddy_np, chunks=(t_chunks, y_chunks)),
            dims=("time", "yc"),
            coords={"time": u["time"], "yc": u["yc"]},
        )
    else:
        if "time" in track.dims:
            if track.sizes["time"] != 1:
                raise ValueError(
                    "track has multiple times but u/v have no 'time' dim."
                )
            cx = float(track["x"].isel(time=0).values)
            cy = float(track["y"].isel(time=0).values)
        else:
            cx = float(track["x"].squeeze().values)
            cy = float(track["y"].squeeze().values)

        x_np = np.asarray(u["xc"].values, dtype=np.float64)
        y_np = np.asarray(u["yc"].values, dtype=np.float64)
        ddx_np = wrap_min(x_np - cx, Lx)
        ddy_np = wrap_min(y_np - cy, Ly)

        x_chunks = u.chunksizes.get("xc") or (nx,)
        y_chunks = u.chunksizes.get("yc") or (ny,)

        ddx = xr.DataArray(
            da.from_array(ddx_np, chunks=x_chunks),
            dims=("xc",),
            coords={"xc": u["xc"]},
        )
        ddy = xr.DataArray(
            da.from_array(ddy_np, chunks=y_chunks),
            dims=("yc",),
            coords={"yc": u["yc"]},
        )

    # Build radial unit vectors lazily to avoid embedding large constants in graph.
    r = np.hypot(ddy, ddx)
    safe_r = xr.where(r > 0.0, r, 1.0)
    inv_r = xr.where(r > 0.0, 1.0 / safe_r, 0.0)

    out_dtype = np.result_type(u.dtype, v.dtype)
    erx = (ddx * inv_r).astype(out_dtype)
    ery = (ddy * inv_r).astype(out_dtype)

    vr = u * erx + v * ery
    vt = -u * ery + v * erx

    vr.attrs.update({"long_name": "radial wind", "units": "m s-1"})
    vt.attrs.update({"long_name": "tangential wind", "units": "m s-1"})

    return xr.Dataset({"vr": vr, "vt": vt})
