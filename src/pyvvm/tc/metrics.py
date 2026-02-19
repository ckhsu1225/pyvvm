"""
Axisymmetric TC wind size/intensity diagnostics on radial profiles.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from collections.abc import Sequence

__all__ = [
    'wind_metrics_from_profile',
]


def _threshold_name(threshold: float) -> str:
    """Build variable name for threshold radius."""
    value = float(threshold)
    if np.isclose(value, np.round(value)):
        return f"r{int(np.round(value))}"
    token = f"{value:g}".replace('-', 'm').replace('.', 'p')
    return f"r{token}"


def _validate_thresholds(thresholds: Sequence[float]) -> tuple[np.ndarray, list[str]]:
    """Validate threshold values and generate variable names."""
    arr = np.asarray(list(thresholds), dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("thresholds must be a non-empty 1-D sequence.")
    if not np.isfinite(arr).all():
        raise ValueError("thresholds must contain only finite values.")
    if (arr < 0.0).any():
        raise ValueError("thresholds must be >= 0 m s-1.")

    names = [_threshold_name(v) for v in arr]
    if len(set(names)) != len(names):
        raise ValueError(
            f"thresholds generate duplicate names: {names}. "
            "Use distinct threshold values."
        )

    return arr, names


def _profile_metrics_1d(
    profile: np.ndarray,
    radius: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """
    Compute vmax/rmw/threshold radii from one 1-D radial profile.

    Threshold radius is the outermost radius where wind meets threshold,
    with outward linear interpolation to the next finite radial point
    when available.
    """
    n_out = 2 + thresholds.size
    out = np.full(n_out, np.nan, dtype=np.float64)

    if profile.ndim != 1 or radius.ndim != 1 or profile.size != radius.size:
        return out

    valid = np.isfinite(profile) & np.isfinite(radius)
    if not valid.any():
        return out

    prof = np.where(valid, profile, np.nan)
    vmax = np.nanmax(prof)
    if not np.isfinite(vmax):
        return out

    out[0] = float(vmax)
    i_max = int(np.nanargmax(prof))
    out[1] = float(radius[i_max])

    n = profile.size
    for i_thr, thr in enumerate(thresholds):
        idx = np.where(np.isfinite(prof) & (prof >= thr))[0]
        if idx.size == 0:
            continue

        i0 = int(idx[-1])
        r_thr = float(radius[i0])

        j = i0 + 1
        while j < n and not np.isfinite(prof[j]):
            j += 1

        if j < n and np.isfinite(radius[j]):
            v0 = float(prof[i0])
            v1 = float(prof[j])
            r0 = float(radius[i0])
            r1 = float(radius[j])

            # Interpolate the outward crossing when [v0, v1] brackets thr.
            if (v0 - thr) * (v1 - thr) <= 0.0 and v1 != v0:
                w = (thr - v0) / (v1 - v0)
                r_thr = r0 + w * (r1 - r0)

        out[2 + i_thr] = r_thr

    return out


def wind_metrics_from_profile(
    wind_r: xr.DataArray,
    thresholds: Sequence[float] = (17.0, 25.0, 33.0),
) -> xr.Dataset:
    """
    Compute wind size/intensity metrics from axisymmetric wind profile(s).

    Parameters
    ----------
    wind_r : xr.DataArray
        Axisymmetric wind profile with dimension ``r`` and optional
        leading dimensions (e.g., ``time``, ``zc``).
    thresholds : sequence of float, optional
        Wind thresholds (m s-1) for threshold radii.

    Returns
    -------
    xr.Dataset
        Dataset containing:
        - ``vmax`` : maximum wind speed [m s-1]
        - ``rmw`` : radius of maximum wind [m]
        - ``r*`` : threshold radii [m], e.g. ``r17``, ``r25``, ``r33``
    """
    if 'r' not in wind_r.dims:
        raise ValueError("wind_r must include 'r' dimension.")

    threshold_values, threshold_names = _validate_thresholds(thresholds)
    metric_names = ['vmax', 'rmw', *threshold_names]

    metrics = xr.apply_ufunc(
        _profile_metrics_1d,
        wind_r,
        wind_r['r'],
        kwargs={'thresholds': threshold_values},
        input_core_dims=[['r'], ['r']],
        output_core_dims=[['metric']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={'output_sizes': {'metric': len(metric_names)}},
    )
    metrics = metrics.assign_coords(metric=metric_names)

    out = xr.Dataset(
        {name: metrics.sel(metric=name, drop=True) for name in metric_names}
    )

    out['vmax'].attrs.update({
        'long_name': 'maximum azimuthal-mean wind speed',
        'units': 'm s-1',
    })
    out['rmw'].attrs.update({
        'long_name': 'radius of maximum azimuthal-mean wind',
        'units': 'm',
    })

    for value, name in zip(threshold_values, threshold_names):
        out[name].attrs.update({
            'long_name': f'radius of {value:g} m s-1 azimuthal-mean wind',
            'units': 'm',
            'threshold_wind_speed': float(value),
        })

    return out

