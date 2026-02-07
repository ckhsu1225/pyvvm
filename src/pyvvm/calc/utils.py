"""
Utility functions for xarray operations.

This module provides helper functions for common data manipulation tasks,
particularly for working with VVM datasets.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import scipy.fft as sfft

__all__ = [
    'take_along_vertical',
    'solve_poisson_spectral',
]


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


def solve_poisson_spectral(
    zeta: np.ndarray,
    dx: float,
    dy: float,
    fft_workers: int = -1,
) -> np.ndarray:
    """
    Fast periodic Poisson solver using rFFT.
    
    Solves the 2D Poisson equation: ∇²ψ = ζ
    In spectral space: -(kx² + ky²) ψ̂ = ζ̂
    
    Parameters
    ----------
    zeta : np.ndarray
        2D vorticity field with shape (ny, nx).
    dx : float
        Grid spacing in x-direction [m].
    dy : float
        Grid spacing in y-direction [m].
    fft_workers : int, optional
        Number of workers for FFT. Default -1 uses all available cores.
        
    Returns
    -------
    np.ndarray
        2D streamfunction field with same shape as input.
        
    Notes
    -----
    - Assumes periodic boundary conditions in both x and y
    - Input is automatically made zero-mean to ensure solvability
    - Uses real FFT (rfft2) for efficiency
    """
    ny, nx = zeta.shape
    rhs = zeta - zeta.mean()

    # Real FFT wavenumbers
    kx = 2 * np.pi * sfft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * sfft.fftfreq(ny, d=dy)
    K2 = (ky[:, None]**2) + (kx[None, :nx//2 + 1]**2)  # rfft2 shape (ny, nx//2+1)

    rhs_hat = sfft.rfft2(rhs, workers=fft_workers)
    mask = K2 != 0.0
    rhs_hat[~mask] = 0.0
    rhs_hat[mask] *= -1.0 / K2[mask]
    psi = sfft.irfft2(rhs_hat, s=(ny, nx), workers=fft_workers)
    return psi

