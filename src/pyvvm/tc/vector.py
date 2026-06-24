"""
Vector transformations in TC-centered polar coordinates.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .geometry import polar_geometry

__all__ = [
    'decompose_vector',
]


def decompose_vector(
    x: xr.DataArray,
    y: xr.DataArray,
    track: xr.Dataset,
    *,
    radial_name: str = 'radial',
    tangential_name: str = 'tangential',
) -> xr.Dataset:
    """
    Decompose horizontal vector components into radial and tangential parts.

    Parameters
    ----------
    x, y : xr.DataArray
        Vector components on the center grid (xc, yc).  ``x`` is the component
        in the positive x direction and ``y`` is the component in the positive
        y direction.  ``x`` must have scalar coords ``dx`` and ``dy``.
    track : xr.Dataset
        TC track with variables ``x`` and ``y`` in metres.
    radial_name, tangential_name : str, optional
        Variable names used in the returned dataset.

    Returns
    -------
    xr.Dataset
        Dataset with the radial component (positive outward) and tangential
        component (positive counter-clockwise).
    """
    if radial_name == tangential_name:
        raise ValueError("radial_name and tangential_name must be different.")

    if "xc" not in x.dims or "yc" not in x.dims:
        raise ValueError("x must be on center grid (xc, yc).")
    if "xc" not in y.dims or "yc" not in y.dims:
        raise ValueError("y must be on center grid (xc, yc).")

    has_time = "time" in x.dims
    if has_time != ("time" in y.dims):
        raise ValueError(
            "x and y must both have 'time' dim or both be time-independent."
        )

    geom = polar_geometry(x, track)
    out_dtype = np.result_type(x.dtype, y.dtype)
    cos_theta = geom["cos_theta"].astype(out_dtype)
    sin_theta = geom["sin_theta"].astype(out_dtype)

    radial = x * cos_theta + y * sin_theta
    tangential = -x * sin_theta + y * cos_theta

    return xr.Dataset({
        radial_name: radial,
        tangential_name: tangential,
    })
