"""
Scalar derivatives in TC-centered polar coordinates.
"""

from __future__ import annotations

import xgcm
import numpy as np
import xarray as xr

from .geometry import polar_geometry
from ..utils import assign_compatible_coords

__all__ = [
    'polar_derivatives',
]


def polar_derivatives(
    da: xr.DataArray,
    track: xr.Dataset,
    grid: xgcm.Grid,
) -> xr.Dataset:
    """
    Compute scalar derivatives in TC-centered polar coordinates.

    The input field must be on the horizontal center grid (xc, yc).  Cartesian
    derivatives are computed on the model grid and then projected onto the
    radial and azimuthal directions relative to *track*.

    Parameters
    ----------
    da : xr.DataArray
        Scalar field on the center grid (xc, yc).  Must have scalar coords
        ``dx`` and ``dy``.
    track : xr.Dataset
        TC track with variables ``x`` and ``y`` in metres.
    grid
        xgcm grid associated with *da*.

    Returns
    -------
    xr.Dataset
        Dataset with:

        - ``d_dr``: radial derivative.
        - ``d_ds``: azimuthal arc-length derivative, equal to
          ``(1 / r) * d_dtheta``.
        - ``d_dtheta``: angular derivative with respect to theta.

        At ``r == 0``, all three derivatives are set to NaN because the polar
        directions are undefined at the TC center.
    """
    if "xc" not in da.dims or "yc" not in da.dims:
        raise ValueError("da must be on center grid (xc, yc).")

    geom = polar_geometry(da, track)

    d_dx = grid.derivative(grid.interp(da, "X"), "X")
    d_dy = grid.derivative(grid.interp(da, "Y"), "Y")
    d_dx = assign_compatible_coords(d_dx, da)
    d_dy = assign_compatible_coords(d_dy, da)

    cos_theta = geom["cos_theta"]
    sin_theta = geom["sin_theta"]
    r = geom["r"]

    d_dr = d_dx * cos_theta + d_dy * sin_theta
    d_ds = -d_dx * sin_theta + d_dy * cos_theta
    d_dtheta = r * d_ds

    valid = r > 0.0
    d_dr = xr.where(valid, d_dr, np.nan)
    d_ds = xr.where(valid, d_ds, np.nan)
    d_dtheta = xr.where(valid, d_dtheta, np.nan)

    target_dims = tuple(dim for dim in da.dims if dim in d_dr.dims)
    d_dr = d_dr.transpose(*target_dims)
    d_ds = d_ds.transpose(*target_dims)
    d_dtheta = d_dtheta.transpose(*target_dims)

    units = da.attrs.get("units")
    if units:
        spatial_units = f"{units} m-1"
        angular_units = units
    else:
        spatial_units = "m-1"
        angular_units = "1"

    name = da.name or "scalar"
    d_dr.attrs.update({
        "long_name": f"radial derivative of {name}",
        "units": spatial_units,
    })
    d_ds.attrs.update({
        "long_name": f"azimuthal arc-length derivative of {name}",
        "units": spatial_units,
    })
    d_dtheta.attrs.update({
        "long_name": f"angular derivative of {name}",
        "units": angular_units,
    })

    return xr.Dataset({
        "d_dr": d_dr,
        "d_ds": d_ds,
        "d_dtheta": d_dtheta,
    })
