"""
Numerical methods using FFT for periodic domains.

This module provides spectral solvers and filters for doubly-periodic grids.
"""

from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from functools import lru_cache

__all__ = [
    'solve_poisson_spectral',
    'periodic_gaussian_smooth',
]


@lru_cache(maxsize=32)
def _spectral_k2(
    ny: int,
    nx: int,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Cached squared total wavenumber on (ky, kx_rfft) grid.
    """
    kx = 2 * np.pi * sfft.rfftfreq(nx, d=dx)  # (nx//2+1,)
    ky = 2 * np.pi * sfft.fftfreq(ny, d=dy)   # (ny,)
    k2 = (ky[:, None] ** 2) + (kx[None, :] ** 2)  # (ny, nx//2+1)
    k2.setflags(write=False)
    return k2


@lru_cache(maxsize=32)
def _poisson_inv_laplacian(
    ny: int,
    nx: int,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Cached inverse Laplacian multiplier for periodic Poisson solve.
    """
    k2 = _spectral_k2(ny, nx, dx, dy)
    inv_lap = np.zeros_like(k2, dtype=np.float64)
    mask = k2 != 0.0
    inv_lap[mask] = -1.0 / k2[mask]
    inv_lap.setflags(write=False)
    return inv_lap


@lru_cache(maxsize=32)
def _gaussian_filter(
    ny: int,
    nx: int,
    dx: float,
    dy: float,
    sigma: float,
) -> np.ndarray:
    """
    Cached spectral Gaussian filter exp(-0.5 * sigma^2 * K2).
    """
    k2 = _spectral_k2(ny, nx, dx, dy)
    filt = np.exp(-0.5 * (sigma ** 2) * k2)
    filt.setflags(write=False)
    return filt


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
    inv_lap = _poisson_inv_laplacian(ny, nx, float(dx), float(dy))

    rhs_hat = sfft.rfft2(rhs, workers=fft_workers)
    rhs_hat *= inv_lap
    psi = sfft.irfft2(rhs_hat, s=(ny, nx), workers=fft_workers)
    return psi


def periodic_gaussian_smooth(
    field2d: np.ndarray,
    dx: float,
    dy: float,
    sigma: float,
    fft_workers: int = -1,
) -> np.ndarray:
    """
    Periodic 2D Gaussian smoothing via FFT.

    This is the exact periodic convolution with a Gaussian kernel.
    Works best for doubly periodic domains.

    Parameters
    ----------
    field2d : np.ndarray
        2D array (ny, nx).
    dx : float
        Grid spacing in x-direction [m].
    dy : float
        Grid spacing in y-direction [m].
    sigma : float
        Gaussian sigma in m.
    fft_workers : int
        FFT workers (scipy.fft), default -1 uses all.

    Returns
    -------
    np.ndarray
        Smoothed 2D array (ny, nx).
    """
    ny, nx = field2d.shape
    filt = _gaussian_filter(ny, nx, float(dx), float(dy), float(sigma))
    F = sfft.rfft2(field2d, workers=fft_workers)
    F *= filt
    out = sfft.irfft2(F, s=(ny, nx), workers=fft_workers)
    return out
