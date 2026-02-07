"""
Terrain data loading and processing utilities.

This module provides functions for loading VVM topo data (TOPO.nc)
and integrating it with xarray datasets.
"""

from __future__ import annotations

import logging
import xarray as xr
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    'load_topo',
]

def _find_var_case_insensitive(ds: xr.Dataset, name: str) -> str | None:
    """Find a variable name in a dataset with case-insensitive matching."""
    name_lower = name.lower()
    for var in ds.data_vars:
        if var.lower() == name_lower:
            return var
    return None


def _latlon_to_xcyc(
    da: xr.DataArray,
    coords_xy: dict[str, xr.DataArray],
) -> xr.DataArray:
    """Rename dimensions and assign xy coordinates to a topo variable."""
    return da.rename({'lat': 'yc', 'lon': 'xc'}).assign_coords(coords_xy)


def add_topo_variables(
    ds: xr.Dataset,
    ds_topo: xr.Dataset,
    coords_xy: dict[str, xr.DataArray],
) -> xr.Dataset:
    """
    Add topo index and height as coordinates to a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The main dataset to add topo variables to.
    ds_topo : xr.Dataset
        The topo dataset (TOPO.nc).
    coords_xy : dict
        Dictionary of xc/yc coordinates for alignment.

    Returns
    -------
    xr.Dataset
        The dataset with topo variables added.
    """
    # Find and add topo level index (case-insensitive)
    topo_var = _find_var_case_insensitive(ds_topo, 'topo')
    if topo_var:
        topo = _latlon_to_xcyc(ds_topo[topo_var], coords_xy)
        topo.attrs.update({
            'long_name': 'terrain level index',
            'units': '1',
        })
        ds = ds.assign_coords(topo=topo.astype('int'))

    # Find and add terrain height (case-insensitive)
    height_var = _find_var_case_insensitive(ds_topo, 'height')
    if height_var:
        # Convert from km to m to match zc units
        height = _latlon_to_xcyc(ds_topo[height_var], coords_xy) * 1000.0
        height.attrs.update({
            'long_name': 'terrain height',
            'units': 'm',
        })
        ds = ds.assign_coords(height=height)

    return ds


def add_geographic_coordinates(
    ds: xr.Dataset,
    ds_topo: xr.Dataset,
) -> xr.Dataset:
    """
    Add lon/lat geographic coordinates from TOPO.nc to a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The main dataset to add coordinates to.
    ds_topo : xr.Dataset
        The topo dataset containing lon/lat values.

    Returns
    -------
    xr.Dataset
        The dataset with lon/lat coordinates added.
    """
    coords_to_add = {}

    if 'lon' in ds_topo:
        coords_to_add['lon'] = ('xc', ds_topo['lon'].values)

    if 'lat' in ds_topo:
        coords_to_add['lat'] = ('yc', ds_topo['lat'].values)

    if coords_to_add:
        ds = ds.assign_coords(coords_to_add)

    return ds


def load_topo(
    case_path: str | Path,
    ds: xr.Dataset,
    chunks: dict | None = None,
    topo_filename: str = 'TOPO.nc',
) -> xr.Dataset:
    """
    Load topo data from TOPO.nc and add to dataset.

    This function loads topo index, height, and geographic coordinates
    (lon/lat) from the topo file and adds them to the main dataset.

    Parameters
    ----------
    case_path : str or Path
        Path to the simulation directory.
    ds : xr.Dataset
        The main dataset to add topo data to.
    chunks : dict, optional
        Dask chunking options for loading the topo file. Defaults to None.
    topo_filename : str, optional
        Name of the topo file. Defaults to 'TOPO.nc'.

    Returns
    -------
    xr.Dataset
        The dataset with topo data added (or unchanged if no topo file).
    """
    case_path = Path(case_path)
    topo_path = case_path / topo_filename

    if not topo_path.exists():
        logger.debug(f"TOPO file not found: {topo_path}")
        return ds

    try:
        # Prepare xy coordinates for alignment (only keep xc, yc)
        coords_xy = {k: ds.coords[k] for k in ['xc', 'yc'] if k in ds.coords}

        # Open with lazy mode, then apply valid chunks
        ds_topo = xr.open_dataset(topo_path, chunks={}, decode_times=False)

        # Filter chunks to only include dimensions present in TOPO.nc
        if chunks:
            valid_chunks = {k: v for k, v in chunks.items() if k in ds_topo.dims}
            if valid_chunks:
                ds_topo = ds_topo.chunk(valid_chunks)

        ds = add_topo_variables(ds, ds_topo, coords_xy)
        ds = add_geographic_coordinates(ds, ds_topo)

        logger.debug(f"Loaded topo data from {topo_path}")

    except Exception as e:
        logger.warning(f"Failed to load topo data from {topo_path}: {e}")

    return ds
