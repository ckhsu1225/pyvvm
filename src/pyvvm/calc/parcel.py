"""
Parcel theory calculations for convective diagnostics.

This module implements lifted parcel calculations for CAPE and CIN.
It is designed to work efficiently with dask arrays via xr.map_blocks.

Functions
---------
build_thes_lut : Build θes lookup table for moist adiabat inversion
compute_cape_cin : Calculate CAPE and CIN for a VVM dataset
"""

import numpy as np
import xarray as xr
from . import formulas as F
from .constants import Cp_d, g
from .._utils import take_along_vertical

__all__ = [
    'build_thes_lut',
    'compute_cape_cin',
]


# ============================================================================
# Lookup Table Construction
# ============================================================================

def build_thes_lut(tgrid: np.ndarray, pgrid: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """
    Build saturation equivalent potential temperature lookup table.

    Creates a 2D lookup table (pressure × temperature) for θes that can be
    used to invert the moist adiabat: given θe, find T at each pressure level.

    Parameters
    ----------
    tgrid : np.ndarray
        1D array of temperatures [K] for the lookup table
    pgrid : np.ndarray
        1D array of pressure levels [Pa]
    alpha : float, optional
        Maximum ratio of es/p for valid calculations, default 0.2

    Returns
    -------
    np.ndarray
        2D array of shape (nz, nt) containing θes values.
        Invalid entries (where es > alpha * p) are set to NaN.
    """
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        es = F.saturation_vapor_pressure(tgrid[None, :])
        qvs = F.saturation_mixing_ratio(pgrid[:, None], es)
        
        # Validity check: es must be small relative to p
        valid = (es < alpha * pgrid[:, None]) & np.isfinite(qvs) & (qvs >= 0)
        
        thes = F.saturation_equivalent_potential_temperature(
            tgrid[None, :], pgrid[:, None], es, qvs
        )
        thes[~valid] = np.nan
        return thes


def _find_last_valid_index(thes_lut: np.ndarray) -> np.ndarray:
    """
    Find the last valid (finite) index in each row of the lookup table.

    Parameters
    ----------
    thes_lut : np.ndarray
        2D θes lookup table of shape (nz, nt)

    Returns
    -------
    np.ndarray
        1D array of shape (nz,) with the last valid index for each level
    """
    finite = np.isfinite(thes_lut)
    last = finite.shape[1] - 1 - np.argmax(finite[:, ::-1], axis=1)
    has = finite.any(axis=1)
    return np.where(has, last, 1).astype(np.int64)


# ============================================================================
# Block Function Factory
# ============================================================================

def _make_cape_cin_block_func(thes_lut: np.ndarray, last: np.ndarray, tgrid: np.ndarray):
    """
    Create a block function for computing CAPE/CIN via xr.map_blocks.

    Parameters
    ----------
    thes_lut : np.ndarray
        2D θes lookup table of shape (nz, nt)
    last : np.ndarray
        1D array of last valid indices for each pressure level
    tgrid : np.ndarray
        1D temperature grid used to build the lookup table

    Returns
    -------
    callable
        Function that takes a Dataset block and returns cape/cin
    """
    def _block(ds_blk):
        pbar = ds_blk['pbar'].values
        pibar = ds_blk['pibar'].values
        zc = ds_blk['zc'].values
        dz = ds_blk['dz'].values

        th = ds_blk['th'].values
        qv = ds_blk['qv'].values
        th_sfc = ds_blk['th_sfc'].values
        qv_sfc = ds_blk['qv_sfc'].values
        the_sfc = ds_blk['the_sfc'].values
        lcl = ds_blk['lcl'].values
        height = ds_blk['height'].values
        sfc_idx = ds_blk['sfc_idx'].values.astype(np.int64)

        height = height[None, :, :]
        sfc_idx = sfc_idx[None, :, :]

        nt, nz, ny, nx = th.shape
        cape = np.zeros((nt, ny, nx), dtype=np.float32)
        cin = np.zeros((nt, ny, nx), dtype=np.float32)

        t_env = F.temperature(pibar[None, :, None, None], th)
        tv_env = F.virtual_temperature(t_env, qv)

        for k in range(nz):
            tv_e = tv_env[:, k, :, :]

            # Dry adiabatic parcel temperature
            t_dry = th_sfc * pibar[k]

            # Moist adiabatic parcel temperature from LUT interpolation
            i1 = int(last[k]) + 1
            xp = thes_lut[k, :i1]
            fp = tgrid[:i1]
            t_moist = np.interp(the_sfc, xp, fp, left=fp[0], right=fp[-1])

            # Choose dry or moist based on LCL
            dry = zc[k] < lcl
            t_p = np.where(dry, t_dry, t_moist)

            # Parcel mixing ratio
            es_p = F.saturation_vapor_pressure(t_p)
            qv_p = np.where(dry, qv_sfc, F.saturation_mixing_ratio(pbar[k], es_p))
            tv_p = F.virtual_temperature(t_p, qv_p)

            b = g * (tv_p - tv_e) / tv_e
            above_ground = (k >= sfc_idx)

            pos = above_ground & (b > 0)
            cape += np.where(pos, b * dz[k], 0.0)

            neg = above_ground & (b < 0) & (zc[k] < (height + 5000.0))
            cin += np.where(neg, b * dz[k], 0.0)

        out = xr.Dataset({
            'cape': (('time', 'yc', 'xc'), cape),
            'cin': (('time', 'yc', 'xc'), cin),
        }).assign_coords(ds_blk.coords)
        return out

    return _block


# ============================================================================
# Main CAPE/CIN Computation
# ============================================================================

def compute_cape_cin(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute CAPE and CIN for a VVM dataset.

    Uses surface-based parcel lifting with:
    - Dry adiabatic ascent below the LCL
    - Moist adiabatic ascent above the LCL (via θes lookup table inversion)
    - Virtual temperature correction for buoyancy

    Parameters
    ----------
    ds : xr.Dataset
        VVM dataset containing:
        - th, qv: potential temperature and mixing ratio
        - pbar, pibar: pressure and Exner function profiles
        - zc, dz: height coordinates and layer thickness
        - topo: terrain height index

    Returns
    -------
    xr.Dataset
        Dataset containing:
        - cape: Convective Available Potential Energy [J/kg]
        - cin: Convective Inhibition [J/kg]

    Notes
    -----
    - CAPE is integrated over all levels where parcel buoyancy is positive
    - CIN is integrated from surface to 5 km above ground where buoyancy is negative
    - Uses xr.map_blocks for dask compatibility
    """
    # Expand time dimension if it doesn't exist
    squeeze_time = 'time' not in ds.dims
    if squeeze_time:
        ds = ds.expand_dims('time')

    # Extract surface values
    topo = ds['topo'] - 1
    sfc_idx = np.maximum(topo, 0).astype(int)

    th_sfc = take_along_vertical(ds['th'], sfc_idx)
    qv_sfc = take_along_vertical(ds['qv'], sfc_idx)
    zc_sfc = take_along_vertical(ds['zc'], sfc_idx)
    p_sfc = take_along_vertical(ds['pbar'], sfc_idx)
    pi_sfc = take_along_vertical(ds['pibar'], sfc_idx)

    # Compute surface parcel properties
    t_sfc = F.temperature(pi_sfc, th_sfc)
    e_sfc = F.vapor_pressure(p_sfc, qv_sfc)
    td_sfc = F.dew_point_temperature(e_sfc)
    tl_sfc = F.lcl_temperature(t_sfc, td_sfc)
    the_sfc = F.equivalent_potential_temperature(t_sfc, p_sfc, qv_sfc, tl_sfc)

    # LCL height
    dt = np.maximum(t_sfc - tl_sfc, 0)
    lcl = (zc_sfc + dt * Cp_d / g).transpose('time', 'yc', 'xc')

    # Build θes lookup table
    pgrid = ds['pbar'].values
    tgrid = np.arange(150.0, 330.1, 0.1)
    thes_lut = build_thes_lut(tgrid, pgrid)
    last = _find_last_valid_index(thes_lut)

    # Prepare working dataset for map_blocks
    work = xr.Dataset({
        'th': ds['th'],
        'qv': ds['qv'],
        'th_sfc': th_sfc,
        'qv_sfc': qv_sfc,
        'the_sfc': the_sfc,
        'lcl': lcl,
        'sfc_idx': sfc_idx,
    }, coords=ds.coords)

    # Create block function and template
    block_func = _make_cape_cin_block_func(
        thes_lut=thes_lut,
        last=last,
        tgrid=tgrid,
    )

    template = xr.Dataset({
        'cape': xr.zeros_like(th_sfc, dtype=np.float32),
        'cin': xr.zeros_like(th_sfc, dtype=np.float32),
    }, coords=ds.coords)

    # Compute via map_blocks
    out = xr.map_blocks(block_func, work, template=template)

    # Squeeze time dimension if it was added
    if squeeze_time:
        out = out.squeeze('time', drop=True)

    # Add metadata
    out['cape'].attrs.update({
        'standard_name': 'convective_available_potential_energy',
        'long_name': 'convective available potential energy',
        'units': 'J kg-1',
    })
    out['cin'].attrs.update({
        'standard_name': 'convective_inhibition',
        'long_name': 'convective inhibition',
        'units': 'J kg-1',
    })

    return out
