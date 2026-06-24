"""
Wind decomposition into radial/tangential components relative to TC center.
"""

from __future__ import annotations

import xarray as xr

from .vector import decompose_vector


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
    out = decompose_vector(
        u,
        v,
        track,
        radial_name="vr",
        tangential_name="vt",
    )

    out["vr"].attrs.update({"long_name": "radial wind", "units": "m s-1"})
    out["vt"].attrs.update({"long_name": "tangential wind", "units": "m s-1"})

    return out
