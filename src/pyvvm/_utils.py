"""
Utility functions for xarray operations.

This module provides helper functions for common data manipulation tasks,
particularly for working with VVM datasets.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def assign_compatible_coords(
    out: xr.DataArray,
    src: xr.Dataset | xr.DataArray,
) -> xr.DataArray:
    """
    Carry over coords from *src* whose dims are a subset of *out*'s dims.

    Scalar coords and 1-D profile coords (e.g. ``rho(zc)``, ``dx``) are
    preserved; spatial coords replaced by the output grid are skipped.
    """
    out_dims = set(out.dims)
    for name, coord in src.coords.items():
        if name not in out.coords and set(coord.dims) <= out_dims:
            out = out.assign_coords({name: coord})
    return out


def take_along_vertical(da: xr.DataArray, indices: xr.DataArray, dim: str = 'zc') -> xr.DataArray:
    """
    Extract values at specified vertical indices for each horizontal point.

    This is useful for extracting surface values when terrain varies
    horizontally, or for extracting values at specific levels like LCL.

    Parameters
    ----------
    da : xr.DataArray
        Data array with a vertical dimension.
    indices : xr.DataArray
        Integer indices specifying which vertical level to extract at each
        horizontal point. Must be broadcastable to da's horizontal dimensions.
    dim : str, optional
        Name of the vertical dimension, default 'zc'.

    Returns
    -------
    xr.DataArray
        Data array with the vertical dimension removed, containing values
        extracted at the specified indices.

    Examples
    --------
    >>> # Extract surface values with varying terrain
    >>> sfc_idx = np.maximum(ds['topo'].astype(int) - 1, 0)
    >>> th_sfc = take_along_vertical(ds['th'], sfc_idx, dim='zc')
    """
    return xr.apply_ufunc(
        lambda x, idx: x[idx],
        da,
        indices,
        input_core_dims=[[dim], []],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[da.dtype],
    )
