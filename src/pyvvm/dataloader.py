"""
VVM Dataset loading and processing utilities.

This module provides the VVMDataset class for loading VVM simulation output
with proper C-grid structure and coordinate handling.
"""

from __future__ import annotations

import re
import logging
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Sequence

from .topo import load_topo
from .config import parse_vvm_setup, parse_fort98

logger = logging.getLogger(__name__)

__all__ = [
    'VVMDataLoader',
]

class VVMDataLoader:
    """
    VVM simulation dataset loader with C-grid support.

    This class handles loading VVM NetCDF output files and restructures
    the data to conform to Arakawa C-grid conventions for use with xgcm.
    """

    # Default file groups to load
    DEFAULT_GROUPS: list[str] = [
        'L.Dynamic',
        'L.Thermodynamic',
        'L.Radiation',
        'C.Surface',
    ]

    # Default chunking for dask arrays
    DEFAULT_CHUNKS: dict[str, int] = {
        'time': 1,
        'lev': 1,
        'lat': -1,
        'lon': -1,
    }

    # C-grid variable dimension mapping
    # Format: 'variable_name': {'original_dim': 'staggered_dim'}
    # Variables not listed here remain at cell centers (xc, yc, zc)
    CGRID_MAPPING: dict[str, dict[str, str]] = {
        'u':    {'xc': 'xb'},
        'v':    {'yc': 'yb'},
        'w':    {'zc': 'zb'},
        'zeta': {'xc': 'xb', 'yc': 'yb'},
        'eta':  {'xc': 'xb', 'zc': 'zb'},
        'xi':   {'yc': 'yb', 'zc': 'zb'},
    }

    def __init__(
        self,
        case_path: str | Path,
        steps: int | slice | Sequence[int] | None = None,
        groups: list[str] | None = None,
        chunks: dict[str, int] | None = None,
    ) -> None:
        """
        Initialize VVM dataset.

        Parameters
        ----------
        case_path : str or Path
            Path to the simulation directory.
        steps : int, list, slice, optional
            Time steps to load (corresponding to filename indices).
            - None: Load all (default)
            - slice(start, stop, step): Load range, e.g., slice(0, 100, 5)
            - [0, 10, 20]: Load specific steps
        groups : list[str], optional
            File groups to load. Defaults to DEFAULT_GROUPS.
        chunks : dict[str, int], optional
            Dask chunk sizes. Defaults to DEFAULT_CHUNKS.
        """
        self.case_path = Path(case_path)
        self.case_name = self.case_path.name
        self.groups = groups if groups is not None else self.DEFAULT_GROUPS
        self.chunks = chunks if chunks is not None else self.DEFAULT_CHUNKS

        # 1. Read configuration (contains DT and NXSAVG)
        self.config = parse_vvm_setup(self.case_path / 'vvm.setup')
        self.profile_data = parse_fort98(self.case_path / 'fort.98')

        # 2. Scan directory for available steps
        self.available_steps = self._scan_available_steps()
        self._available_steps_set = set(self.available_steps)  # For O(1) lookup

        # 3. Filter steps based on user request
        self.selected_steps = self._filter_steps(steps)

        # 4. Load dataset (only open selected files)
        self.ds = self._load_dataset()

    def __repr__(self) -> str:
        """Return string representation of the VVMDataLoader."""
        return (
            f"VVMDataLoader('{self.case_name}', "
            f"steps={len(self.selected_steps)}/{len(self.available_steps)})"
        )

    def _scan_available_steps(self) -> list[int]:
        """Scan archive directory to find all available time step indices."""
        # Use Dynamic files for scanning (assuming they always exist)
        # Filename format: case_name.L.Dynamic-000000.nc
        archive_dir = self.case_path / 'archive'
        pattern = re.compile(rf"{re.escape(self.case_name)}\.L\.Dynamic-(\d+)\.nc")

        steps: list[int] = []
        for p in archive_dir.glob(f"{self.case_name}.L.Dynamic-*.nc"):
            match = pattern.search(p.name)
            if match:
                steps.append(int(match.group(1)))

        return sorted(steps)

    def _filter_steps(
        self, steps: int | slice | Sequence[int] | None
    ) -> list[int]:
        """Convert user input (slice/list) to actual step list."""
        all_steps = np.array(self.available_steps)

        if steps is None:
            return all_steps.tolist()

        if isinstance(steps, slice):
            # Filter by filename index range (not list index)
            # e.g., steps=slice(0, 100) loads files numbered 0 to 100
            start = all_steps[0] if steps.start is None else steps.start
            stop = all_steps[-1] if steps.stop is None else steps.stop
            step_freq = 1 if steps.step is None else steps.step

            # Logic: between start and stop, respecting step interval
            filtered = all_steps[(all_steps >= start) & (all_steps <= stop)]
            return filtered[::step_freq].tolist()

        if isinstance(steps, (list, tuple, np.ndarray)):
            # Intersect with available steps using O(1) set lookup
            return [s for s in steps if s in self._available_steps_set]

        if isinstance(steps, int):
            if steps in self._available_steps_set:
                return [steps]

        return []

    # =========================================================================
    # Dataset Loading Pipeline
    # =========================================================================

    def _load_dataset(self) -> xr.Dataset:
        """
        Build file list from selected_steps and open dataset.
        
        This is the main loading pipeline that orchestrates all loading steps.
        """
        if not self.selected_steps:
            raise ValueError("No valid time steps selected or found!")

        ds = self._open_netcdf_files()
        ds = self._restructure_to_cgrid(ds)
        ds = self._assign_time_coordinate(ds)
        ds = self._assign_global_attrs(ds)
        ds = load_topo(self.case_path, ds, chunks=self.chunks)
        return ds

    def _open_netcdf_files(self) -> xr.Dataset:
        """Open and merge NetCDF files for all selected steps and groups."""
        datasets: list[xr.Dataset] = []

        for group in self.groups:
            # Build explicit file list for selected steps
            file_paths: list[str] = []
            for step in self.selected_steps:
                # Filename format: 000000 (6 digits)
                fname = f"{self.case_name}.{group}-{step:06d}.nc"
                fpath = self.case_path / 'archive' / fname
                if fpath.exists():
                    file_paths.append(str(fpath))

            if not file_paths:
                continue

            try:
                ds_sub = xr.open_mfdataset(
                    file_paths,
                    chunks=self.chunks,
                    parallel=True,
                    coords='minimal',
                    data_vars='minimal',
                    compat='override',
                )
                datasets.append(ds_sub)
            except Exception as e:
                logger.warning(f"Failed to load {group}: {e}")
                continue

        return xr.merge(datasets, compat='override', join='override')

    def _assign_time_coordinate(self, ds: xr.Dataset) -> xr.Dataset:
        """Compute and assign simulation time coordinate."""
        dt_interval = self.config.get('output_interval')
        current_steps = np.array(self.selected_steps)
        sim_time = current_steps * dt_interval

        ds = ds.assign_coords(time=sim_time)
        ds.time.attrs.update({
            'long_name': 'simulation time',
            'units': 'seconds since simulation start',
            'axis': 'T',
        })
        return ds

    def _assign_global_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        """Assign global attributes including Coriolis parameter and geographic reference."""
        ds.attrs['coriolis_parameter'] = self.config.get('f')        # Coriolis parameter (None if not f-plane)
        ds.attrs['reference_latitude'] = self.config.get('RLAT')     # Reference latitude
        ds.attrs['reference_longitude'] = self.config.get('RLON')    # Reference longitude
        return ds

    # =========================================================================
    # C-grid Restructuring Pipeline
    # =========================================================================

    def _restructure_to_cgrid(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Restructure dataset dimensions to conform to Arakawa C-grid layout.

        This enables xgcm to automatically recognize that u is on the xb axis
        while th is on the xc axis.
        """
        ds = self._assign_spatial_coordinates(ds)
        ds = self._apply_cgrid_staggering(ds)
        ds = self._assign_staggered_coordinates(ds)
        ds = self._add_background_profiles(ds)
        ds = self._set_coordinate_attributes(ds)
        
        # Skip ghost layer (zc=0; below surface) which is typically a boundary condition layer
        ds = ds.isel(zc=slice(1, None))
        ds = self._add_vertical_level_indices(ds)

        return ds

    def _assign_spatial_coordinates(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Rename dimensions and assign spatial coordinate values.
        
        This function:
        1. Saves original xc/yc coordinate values before dropping
        2. Drops existing coordinate variables (xc, yc, zc, zb)
        3. Renames dimensions (lon->xc, lat->yc, lev->zc)
        4. Assigns proper coordinate values for center and edge grids
        """
        # IMPORTANT: Save original coordinate values BEFORE dropping
        xc_vals = ds['xc'].values
        yc_vals = ds['yc'].values

        # Drop existing coordinate variables that will be replaced
        ds = ds.drop_vars(['zc', 'zb', 'yc', 'xc'], errors='ignore')

        # Rename dimensions to center-grid convention
        ds = ds.rename({'lon': 'xc', 'lat': 'yc', 'lev': 'zc'})

        # Vertical coordinate values from fort.98
        df_profile = self.profile_data['profile']
        n_lev = self.config.get('NK2')
        zc_vals = df_profile['ZT'].values[:n_lev]  # Thermodynamic height

        # Assign center coordinates
        coords_assign = {
            'xc': (['xc'], xc_vals),
            'yc': (['yc'], yc_vals),
            'zc': (['zc'], zc_vals),
        }
        ds = ds.assign_coords(coords_assign)

        return ds

    def _assign_staggered_coordinates(self, ds: xr.Dataset) -> xr.Dataset:
        """Assign staggered (edge) coordinate and distance metrics after C-grid renaming."""
        dx = self.config.get('DX')
        dy = self.config.get('DYNEW')

        xb_vals = ds['xc'].values + dx / 2.0
        yb_vals = ds['yc'].values + dy / 2.0

        df_profile = self.profile_data['profile']
        n_lev = self.config.get('NK2')
        zb_vals = df_profile['ZZ'].values[:n_lev]  # Interface height
        dz = np.append(0, np.diff(zb_vals))  # Interface thickness (set buttom to 0)

        coords_assign = {
            'xb': (['xb'], xb_vals),
            'yb': (['yb'], yb_vals),
            'zb': (['zb'], zb_vals),
            'dx': dx,
            'dy': dy,
            'dz': (['zc'], dz),
        }
        return ds.assign_coords(coords_assign)

    def _apply_cgrid_staggering(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply C-grid dimension remapping to staggered variables."""
        for var_name, dims_map in self.CGRID_MAPPING.items():
            if var_name in ds:
                try:
                    ds[var_name] = ds[var_name].rename(dims_map)
                except ValueError as e:
                    logger.warning(f"Skipping rename for {var_name}: {e}")
        return ds

    def _add_background_profiles(self, ds: xr.Dataset) -> xr.Dataset:
        """Add background thermodynamic profiles as coordinates."""
        df_profile = self.profile_data['profile']
        df_rhoz = self.profile_data['rhoz']
        n_lev = self.config.get('NK2')

        coords_assign = {
            'rho':    (['zc'], df_profile['RHO'].values[:n_lev]),
            'rhoz':   (['zb'], df_rhoz['RHOZ'].values[:n_lev]),
            'thbar':  (['zc'], df_profile['THBAR'].values[:n_lev]),
            'pbar':   (['zc'], df_profile['PBAR'].values[:n_lev]),
            'pibar':  (['zc'], df_profile['PIBAR'].values[:n_lev]),
            'qvbar':  (['zc'], df_profile['QVBAR'].values[:n_lev]),
        }
        return ds.assign_coords(coords_assign)

    def _add_vertical_level_indices(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Add 1-based vertical level indices used by terrain masking logic.

        These indices stay attached to zc/zb after slicing (including scalar
        selections), so masking code can recover the absolute model level.
        """
        coords_assign: dict[str, tuple[list[str], np.ndarray]] = {}

        if 'zc' in ds.dims:
            coords_assign['kzc'] = (
                ['zc'],
                np.arange(1, ds.sizes['zc'] + 1, dtype=np.int32),
            )
        if 'zb' in ds.dims:
            coords_assign['kzb'] = (
                ['zb'],
                np.arange(1, ds.sizes['zb'] + 1, dtype=np.int32),
            )

        if coords_assign:
            ds = ds.assign_coords(coords_assign)

        if 'kzc' in ds.coords:
            ds.kzc.attrs.update({
                'long_name': '1-based vertical level index at thermodynamic levels',
                'units': '1',
            })
        if 'kzb' in ds.coords:
            ds.kzb.attrs.update({
                'long_name': '1-based vertical level index at interface levels',
                'units': '1',
            })

        return ds

    def _set_coordinate_attributes(self, ds: xr.Dataset) -> xr.Dataset:
        """Set metadata attributes (long_name, units, axis) for all coordinates."""
        # Horizontal coordinates
        ds.xc.attrs.update({
            'long_name': 'x-coordinate at cell center',
            'units': 'm',
            'axis': 'X',
        })
        ds.xb.attrs.update({
            'long_name': 'x-coordinate at cell edge',
            'units': 'm',
            'axis': 'X',
            'c_grid_axis_shift': 0.5,
        })
        ds.yc.attrs.update({
            'long_name': 'y-coordinate at cell center',
            'units': 'm',
            'axis': 'Y',
        })
        ds.yb.attrs.update({
            'long_name': 'y-coordinate at cell edge',
            'units': 'm',
            'axis': 'Y',
            'c_grid_axis_shift': 0.5,
        })

        # Vertical coordinates
        ds.zc.attrs.update({
            'long_name': 'height at thermodynamic levels',
            'units': 'm',
            'axis': 'Z',
            'positive': 'up',
        })
        ds.zb.attrs.update({
            'long_name': 'height at vertical interfaces',
            'units': 'm',
            'axis': 'Z',
            'positive': 'up',
            'c_grid_axis_shift': 0.5,
        })

        # Distance metrics
        ds.dx.attrs.update({
            'long_name': 'grid spacing in x-direction',
            'units': 'm',
        })
        ds.dy.attrs.update({
            'long_name': 'grid spacing in y-direction',
            'units': 'm',
        })
        ds.dz.attrs.update({
            'long_name': 'layer thickness in z-direction',
            'units': 'm',
        })

        # Background profile coordinates
        ds.rho.attrs.update({
            'long_name': 'base state density at cell center',
            'units': 'kg m-3',
        })
        ds.rhoz.attrs.update({
            'long_name': 'base state density at vertical interfaces',
            'units': 'kg m-3',
        })
        ds.thbar.attrs.update({
            'long_name': 'base state potential temperature',
            'units': 'K',
        })
        ds.pbar.attrs.update({
            'long_name': 'base state pressure',
            'units': 'Pa',
        })
        ds.pibar.attrs.update({
            'long_name': 'base state Exner function',
            'units': '1',
        })
        ds.qvbar.attrs.update({
            'long_name': 'base state water vapor mixing ratio',
            'units': 'kg kg-1',
        })

        return ds
