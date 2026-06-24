"""
Horizontal vorticity decomposition relative to a TC center.
"""

from __future__ import annotations

import xarray as xr

from .vector import decompose_vector

__all__ = [
    'compute_vort_rt',
]


def compute_vort_rt(
    xi: xr.DataArray,
    eta: xr.DataArray,
    track: xr.Dataset,
    *,
    radial_name: str = 'vort_r',
    tangential_name: str = 'vort_t',
) -> xr.Dataset:
    """
    Decompose horizontal vorticity into radial and tangential components.

    Parameters
    ----------
    xi, eta : xr.DataArray
        Physical x- and y-components of horizontal vorticity on the center
        grid (xc, yc).  For raw VVM output, ``xi`` is the x-component, while
        the physical y-component is typically ``-eta`` after interpolation to
        the center grid.
    track : xr.Dataset
        TC track with variables ``x`` and ``y`` in metres.
    radial_name, tangential_name : str, optional
        Variable names used in the returned dataset.

    Returns
    -------
    xr.Dataset
        Dataset with radial and tangential horizontal vorticity components.
    """
    out = decompose_vector(
        xi,
        eta,
        track,
        radial_name=radial_name,
        tangential_name=tangential_name,
    )

    units = xi.attrs.get('units') or eta.attrs.get('units') or 's-1'
    out[radial_name].attrs.update({
        'long_name': 'radial horizontal vorticity',
        'units': units,
    })
    out[tangential_name].attrs.update({
        'long_name': 'tangential horizontal vorticity',
        'units': units,
    })

    return out
