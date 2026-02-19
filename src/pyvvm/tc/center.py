"""
TC center finding algorithms for VVM simulations.

This module provides functions to locate tropical cyclone centers
using vorticity or streamfunction fields.
"""

from __future__ import annotations

import warnings
import numpy as np
import xarray as xr

from .._utils import assign_compatible_coords
from ..numerics import solve_poisson_spectral, periodic_gaussian_smooth

__all__ = [
    'find_tc_center',
    'smooth_zeta',
    'compute_psi',
]

_RECHUNK_WARNED = False


# =============================================================================
# Tracking
# =============================================================================

def find_tc_center(
    ds: xr.Dataset,
    field: str = 'psi',
    method: str = 'extremum',
    level: float | tuple[float, float] = 1000.0,
    sigma: float = 50e3,
    radius: float = 100e3,
) -> xr.Dataset:
    """
    Find TC center using specified field and method.
    
    Parameters
    ----------
    ds : xr.Dataset
        VVM dataset with vvm accessor.
    field : {'zeta', 'psi'}
        Field to use for center finding.
        - 'zeta': Smoothed vorticity (cyclonic maximum)
        - 'psi': Streamfunction (minimum for cyclone)
    method : {'centroid', 'extremum'}
        - 'centroid': Weighted centroid with iterative refinement
        - 'extremum': Simple argmax/argmin
    level : float or tuple[float, float]
        Vertical level (m). A single float selects the nearest level.
        A tuple (low, high) averages over the range before processing.
    sigma : float
        Gaussian smoothing sigma for 'zeta' field (m). Ignored for 'psi'.
    radius : float
        Search radius for centroid method (m). Ignored for 'extremum'.
        
    Returns
    -------
    xr.Dataset
        TC center coordinates with variables ``x`` and ``y`` on ``time`` dimension.
        
    Examples
    --------
    >>> center = find_tc_center(ds, field='zeta', method='centroid')
    >>> center = find_tc_center(ds, field='psi', method='extremum')
    """
    # Validate inputs
    if field not in ('zeta', 'psi'):
        raise ValueError(f"field must be 'zeta' or 'psi', got '{field}'")
    if method not in ('centroid', 'extremum'):
        raise ValueError(f"method must be 'centroid' or 'extremum', got '{method}'")
    if method == 'centroid' and radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    
    # Prepare field
    if field == 'zeta':
        da = smooth_zeta(ds, level, sigma)
    else:
        da = compute_psi(ds, level)
        # For cyclone, psi is negative at center, so negate for finding maximum
        da = -da
    
    # Find center
    return _get_track(da, method=method, radius=radius)


def _get_track(
    da: xr.DataArray,
    method: str = 'centroid',
    radius: float = 100e3,
) -> xr.Dataset:
    """
    Get TC track from time series of 2D field.
    
    This is a lower-level function that operates on a prepared field.
    For most use cases, prefer `find_tc_center` which handles field preparation.
    
    Parameters
    ----------
    da : xr.DataArray
        2D field with dims (time, yc, xc). Should be positive at center.
    method : {'centroid', 'extremum'}
        Center finding method.
    radius : float
        Search radius for centroid method (m).
        
    Returns
    -------
    xr.Dataset
        TC center coordinates with variables ``x`` and ``y`` on ``time`` dimension.
    """
    dx = float(da.coords['dx'].values)
    dy = float(da.coords['dy'].values)
    x0 = float(da.coords['xc'].values[0])
    y0 = float(da.coords['yc'].values[0])
    
    if method == 'centroid':
        func = _find_centroid_iterative
        kwargs = {'dx': dx, 'dy': dy, 'radius': radius}
    else:
        func = _find_extremum
        kwargs = {}
    
    track = xr.apply_ufunc(
        func,
        da,
        kwargs=kwargs,
        input_core_dims=[['yc', 'xc']],
        output_core_dims=[['coor']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={'output_sizes': {'coor': 2}},
    )
    track = track.assign_coords(coor=['x_idx', 'y_idx'])

    x = x0 + track.sel(coor='x_idx', drop=True) * dx
    y = y0 + track.sel(coor='y_idx', drop=True) * dy

    track_ds = xr.Dataset(
        data_vars={
            'x': x,
            'y': y,
        },
        attrs={
            'long_name': 'TC center coordinates',
            'units': 'm',
            'method': method,
        },
    )
    track_ds['x'].attrs = {
        'long_name': 'TC center x coordinate',
        'units': 'm',
    }
    track_ds['y'].attrs = {
        'long_name': 'TC center y coordinate',
        'units': 'm',
    }

    return track_ds


# =============================================================================
# Field preparation
# =============================================================================

def smooth_zeta(
    ds: xr.Dataset,
    level: float | tuple[float, float],
    sigma: float = 50e3,
) -> xr.DataArray:
    """
    Compute smoothed vorticity field at specified level.
    
    Parameters
    ----------
    ds : xr.Dataset
        VVM dataset with vvm accessor.
    level : float or tuple[float, float]
        Vertical level (m). A single float selects the nearest level.
        A tuple (low, high) averages over the range.
    sigma : float
        Gaussian smoothing sigma (m).
        
    Returns
    -------
    xr.DataArray
        Smoothed vorticity at specified level.
    """
    grid = ds.vvm.grid
    
    # Select and interpolate vorticity to cell centers
    zeta = _select_zeta(ds, grid, level)

    dx = float(ds.coords['dx'].values)
    dy = float(ds.coords['dy'].values)
    
    # Apply Gaussian smoothing
    zeta_smooth = xr.apply_ufunc(
        periodic_gaussian_smooth,
        zeta,
        kwargs={'dx': dx, 'dy': dy, 'sigma': sigma},
        input_core_dims=[['yc', 'xc']],
        output_core_dims=[['yc', 'xc']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[zeta.dtype],
    ).assign_coords(zeta.coords)

    return zeta_smooth


def compute_psi(
    ds: xr.Dataset,
    level: float | tuple[float, float],
) -> xr.DataArray:
    """
    Compute streamfunction field at specified level.
    
    Parameters
    ----------
    ds : xr.Dataset
        VVM dataset with vvm accessor.
    level : float or tuple[float, float]
        Vertical level (m). A single float selects the nearest level.
        A tuple (low, high) averages over the range.
        
    Returns
    -------
    xr.DataArray
        Streamfunction at specified level.
    """
    grid = ds.vvm.grid

    # Select and interpolate vorticity to cell centers
    zeta = _select_zeta(ds, grid, level)

    dx = float(ds.coords['dx'].values)
    dy = float(ds.coords['dy'].values)
    
    # Solve Poisson equation for streamfunction
    psi = xr.apply_ufunc(
        solve_poisson_spectral,
        zeta,
        kwargs={'dx': dx, 'dy': dy},
        input_core_dims=[['yc', 'xc']],
        output_core_dims=[['yc', 'xc']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[zeta.dtype],
    ).assign_coords(zeta.coords)

    return psi


def _select_zeta(
    ds: xr.Dataset,
    grid,
    level: float | tuple[float, float],
) -> xr.DataArray:
    """
    Select vorticity at a single level or averaged over a range.
    
    Parameters
    ----------
    ds : xr.Dataset
        VVM dataset with vvm accessor.
    grid : xgcm.Grid
        xgcm Grid object.
    level : float or tuple[float, float]
        Single level (nearest) or (low, high) range (mean).
        
    Returns
    -------
    xr.DataArray
        Vorticity interpolated to cell centers with terrain masked.
    """
    if isinstance(level, tuple):
        low, high = level
        zeta = ds['zeta'].sel(zc=slice(low, high))
        zeta = ds.vvm.mask(zeta).mean(dim='zc')
    else:
        zeta = ds['zeta'].sel(zc=level, method='nearest')
        zeta = ds.vvm.mask(zeta)
        zc = zeta.coords['zc'].values

    zeta = grid.interp(zeta, ['X', 'Y']).fillna(0.0)
    zeta = zeta.expand_dims({'zc': [zc]}) if isinstance(level, (float, int)) else zeta
    zeta = assign_compatible_coords(zeta, ds).squeeze('zc')

    # FFT core dimensions require single chunks in y/x. Only rechunk when needed.
    if hasattr(zeta.data, 'chunks'):
        needs_xy_single_chunk = any(
            len(zeta.chunksizes.get(dim, ())) != 1
            for dim in ('yc', 'xc')
            if dim in zeta.dims
        )
        if needs_xy_single_chunk:
            global _RECHUNK_WARNED
            if not _RECHUNK_WARNED:
                chunks_info = {
                    dim: zeta.chunksizes.get(dim, ())
                    for dim in ('yc', 'xc')
                    if dim in zeta.dims
                }
                warnings.warn(
                    "Center finding triggered runtime rechunk on horizontal dims "
                    f"({chunks_info}) to satisfy FFT core-dimension requirements. "
                    "This may be slow. Consider using a dedicated center dataset "
                    "with lat/lon=-1 (or y/x single chunks).",
                    stacklevel=2,
                )
                _RECHUNK_WARNED = True
            zeta = zeta.chunk({'yc': -1, 'xc': -1})
    return zeta


# =============================================================================
# Center finding algorithms
# =============================================================================

def _find_centroid_iterative(
    var: np.ndarray,
    dx: float,
    dy: float,
    rough_center_idx: tuple[int, int] | None = None,
    radius: float = 100e3,
    max_iter: int = 5,
) -> np.ndarray:
    """
    Find center using iterative weighted centroid with periodic BC.
    
    The algorithm:
    1. Find initial guess (global maximum or provided index)
    2. Roll field to center the guess at array midpoint
    3. Compute weighted centroid within search radius
    4. Update guess and repeat until convergence
    
    Parameters
    ----------
    var : np.ndarray
        2D field with shape (ny, nx). Should be positive at center.
    dx, dy : float
        Grid spacing (m).
    rough_center_idx : tuple (y_idx, x_idx), optional
        Initial guess indices. If None, uses argmax.
    radius : float
        Search radius for centroid (m).
    max_iter : int
        Maximum iterations.
        
    Returns
    -------
    np.ndarray
        Center indices [x_idx, y_idx] in grid index space.
    """
    ny, nx = var.shape
    if not np.isfinite(var).all():
        raise ValueError(
            "Input field contains NaN or Inf. "
            "Ensure NaN values are filled before center finding."
        )

    signal = var.max() - var.min()
    if signal == 0:
        warnings.warn(
            "Uniform field detected (e.g. initial zeta=0). "
            "Defaulting to domain center.",
            stacklevel=2,
        )
        return np.array([nx / 2.0, ny / 2.0])
    
    # Initial guess
    if rough_center_idx is None:
        flat_idx = np.argmax(var)
        y_idx, x_idx = np.unravel_index(flat_idx, (ny, nx))
    else:
        y_idx, x_idx = rough_center_idx
    
    curr_y_idx, curr_x_idx = int(y_idx), int(x_idx)
    
    # Build local coordinate grid for centroid calculation
    search_grid_r = int(radius / min(dx, dy)) + 5
    search_grid_r = min(search_grid_r, ny // 2, nx // 2)
    y_local, x_local = np.meshgrid(
        np.arange(-search_grid_r, search_grid_r + 1) * dy,
        np.arange(-search_grid_r, search_grid_r + 1) * dx,
        indexing='ij',
    )
    dist_sq_local = x_local**2 + y_local**2
    radius_sq = radius**2
    
    # Fallback to current guess if iterative centroid cannot improve.
    final_x_idx = float(curr_x_idx)
    final_y_idx = float(curr_y_idx)
    
    for _ in range(max_iter):
        # Roll to center current guess at array midpoint
        shift_y = ny // 2 - curr_y_idx
        shift_x = nx // 2 - curr_x_idx
        var_shifted = np.roll(np.roll(var, shift_y, axis=0), shift_x, axis=1)
        
        # Crop to search region
        cy_mid, cx_mid = ny // 2, nx // 2
        y_slice = slice(cy_mid - search_grid_r, cy_mid + search_grid_r + 1)
        x_slice = slice(cx_mid - search_grid_r, cx_mid + search_grid_r + 1)
        var_crop = var_shifted[y_slice, x_slice]
        
        # Compute weights (shift to positive)
        local_min = var_crop.min()
        w = var_crop - local_min
        
        # Apply circular mask
        mask = dist_sq_local <= radius_sq
        w_masked = np.where(mask, w, 0.0)
        
        # Compute centroid offset
        sum_w = w_masked.sum()
        if sum_w <= 0:
            break
        
        offset_x = (x_local * w_masked).sum() / sum_w
        offset_y = (y_local * w_masked).sum() / sum_w
        
        # Convert to global coordinates
        global_cx_idx = (curr_x_idx + offset_x / dx) % nx
        global_cy_idx = (curr_y_idx + offset_y / dy) % ny
        
        final_x_idx = float(global_cx_idx)
        final_y_idx = float(global_cy_idx)
        
        # Update integer center for next iteration
        new_int_x = int(round(global_cx_idx)) % nx
        new_int_y = int(round(global_cy_idx)) % ny
        
        # Check convergence
        if new_int_x == curr_x_idx and new_int_y == curr_y_idx:
            break
        
        curr_x_idx = new_int_x
        curr_y_idx = new_int_y
    
    return np.array([final_x_idx, final_y_idx])


def _find_extremum(
    var: np.ndarray,
    find_max: bool = True,
) -> np.ndarray:
    """
    Find extremum location in periodic domain.
    
    Parameters
    ----------
    var : np.ndarray
        2D field with shape (ny, nx).
    find_max : bool
        If True, find maximum. If False, find minimum.
        
    Returns
    -------
    np.ndarray
        Center indices [x_idx, y_idx] in grid index space.
    """
    ny, nx = var.shape
    if not np.isfinite(var).all():
        raise ValueError(
            "Input field contains NaN or Inf. "
            "Ensure NaN values are filled before center finding."
        )

    signal = var.max() - var.min()
    if signal == 0:
        warnings.warn(
            "Uniform field detected (e.g. initial zeta=0). "
            "Defaulting to domain center.",
            stacklevel=2,
        )
        return np.array([nx / 2.0, ny / 2.0])
    
    if find_max:
        flat_idx = np.argmax(var)
    else:
        flat_idx = np.argmin(var)
    
    y_idx, x_idx = np.unravel_index(flat_idx, (ny, nx))
    
    return np.array([float(x_idx), float(y_idx)])
