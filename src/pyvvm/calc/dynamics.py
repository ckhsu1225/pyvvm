"""
Dynamics calculations mixin for VVM data.

This module provides the DynamicsMixin class which adds dynamics
diagnostic calculations to the VVM accessor.
"""

from __future__ import annotations

import logging
import numpy as np
import xarray as xr
from ..numerics import solve_poisson_spectral
from .._utils import assign_compatible_coords

logger = logging.getLogger(__name__)

__all__ = [
    'DynamicsMixin',
]


class DynamicsMixin:
    """
    Mixin class providing dynamics calculations for VVM data.
    
    All properties return xr.DataArray with appropriate CF-compliant attributes.
    
    Available Properties
    --------------------
    **Wind Variables**
    
    - ``ws`` : Wind speed [m/s]
    - ``wd`` : Wind direction [deg]
    
    **Integrated Vapor Transport**
    
    - ``ivtu`` : Zonal integrated vapor transport [kg/m/s]
    - ``ivtv`` : Meridional integrated vapor transport [kg/m/s]
    - ``ivt`` : Integrated vapor transport magnitude [kg/m/s]
    
    **Vorticity-Related**
    
    - ``pv`` : Ertel potential vorticity [K m² kg⁻¹ s⁻¹]
    - ``psi`` : Streamfunction [m²/s]
    """

    # =========================================================================
    # Public properties - Wind
    # =========================================================================

    @property
    def ws(self) -> xr.DataArray:
        """Wind speed [m/s]."""
        ds = self._ds
        grid = self.grid

        u = grid.interp(ds['u'], 'X')
        v = grid.interp(ds['v'], 'Y')
        ws = np.sqrt(u**2 + v**2)
        ws = assign_compatible_coords(ws, ds)
        ws.attrs.update({
            'standard_name': 'wind_speed',
            'long_name': 'wind speed',
            'units': 'm s-1',
        })
        return ws.rename('ws')

    @property
    def wd(self) -> xr.DataArray:
        """Wind direction [deg]."""
        ds = self._ds
        grid = self.grid

        u = grid.interp(ds['u'], 'X')
        v = grid.interp(ds['v'], 'Y')
        wd = (270. - np.rad2deg(np.arctan2(v, u))) % 360.
        wd = assign_compatible_coords(wd, ds)
        wd.attrs.update({
            'standard_name': 'wind_direction',
            'long_name': 'wind direction',
            'units': 'deg',
            'valid_range': [0, 360],
        })
        return wd.rename('wd')

    # =========================================================================
    # Public properties - Integrated Vapor Transport
    # =========================================================================

    @property
    def ivtu(self) -> xr.DataArray:
        """Zonal integrated vapor transport [kg/m/s]."""
        ds = self._ds
        grid = self.grid

        rho = ds['rho']
        qv = ds['qv']
        u = grid.interp(ds['u'], 'X')
        integrand = self.mask(rho * qv * u)

        ivtu = grid.integrate(integrand, 'Z')
        ivtu.attrs.update({
            'standard_name': 'zonal_integrated_vapor_transport',
            'long_name': 'zonal integrated vapor transport',
            'units': 'kg m-1 s-1',
        })
        return ivtu.rename('ivtu')

    @property
    def ivtv(self) -> xr.DataArray:
        """Meridional integrated vapor transport [kg/m/s]."""
        ds = self._ds
        grid = self.grid

        rho = ds['rho']
        qv = ds['qv']
        v = grid.interp(ds['v'], 'Y')
        integrand = self.mask(rho * qv * v)

        ivtv = grid.integrate(integrand, 'Z')
        ivtv.attrs.update({
            'standard_name': 'meridional_integrated_vapor_transport',
            'long_name': 'meridional integrated vapor transport',
            'units': 'kg m-1 s-1',
        })
        return ivtv.rename('ivtv')

    @property
    def ivt(self) -> xr.DataArray:
        """Integrated vapor transport magnitude [kg/m/s]."""
        ds = self._ds
        grid = self.grid

        rho = ds['rho']
        qv = ds['qv']
        ws = self.ws
        integrand = self.mask(rho * qv * ws)

        ivt = grid.integrate(integrand, 'Z')
        ivt.attrs.update({
            'standard_name': 'integrated_vapor_transport',
            'long_name': 'integrated vapor transport',
            'units': 'kg m-1 s-1',
        })
        return ivt.rename('ivt')

    # =========================================================================
    # Public properties - Vorticity
    # =========================================================================

    @property
    def pv(self) -> xr.DataArray:
        """Ertel potential vorticity [K m² kg⁻¹ s⁻¹]."""
        self._validate_chunks('pv')
        ds = self._ds
        grid = self.grid

        rho = ds['rho']
        th = self.mask(ds['th'])
        f = ds.attrs.get('coriolis_parameter')
        if f is None:
            logger.info(
                "Coriolis parameter not available. Using f=0 (non-rotating simulation)."
            )
            f = 0.

        zeta = grid.interp(ds['zeta'], ['X', 'Y'])
        eta = grid.interp(-ds['eta'], ['X', 'Z'])
        xi = grid.interp(ds['xi'], ['Y', 'Z'])

        dth_dx = grid.derivative(grid.interp(th, 'X'), 'X')
        dth_dy = grid.derivative(grid.interp(th, 'Y'), 'Y')
        dth_dz = grid.derivative(grid.interp(th, 'Z'), 'Z')

        pv = (xi * dth_dx + eta * dth_dy + (zeta + f) * dth_dz) / rho
        pv.attrs.update({
            'standard_name': 'potential_vorticity',
            'long_name': 'Ertel potential vorticity',
            'units': 'K m2 kg-1 s-1',
        })
        return pv.rename('pv')

    @property
    def psi(self) -> xr.DataArray:
        """Streamfunction from horizontal vorticity [m²/s]."""
        self._validate_chunks('psi')
        ds = self._ds
        grid = self.grid
        
        zeta = self.mask(ds['zeta'])
        zeta = grid.interp(zeta, ['X', 'Y']).fillna(0.0)
        zeta = assign_compatible_coords(zeta, ds)

        dx = ds.coords['dx'].values
        dy = ds.coords['dy'].values

        psi = xr.apply_ufunc(
            solve_poisson_spectral,
            zeta,
            kwargs={'dx': dx, 'dy': dy},
            input_core_dims=[['yc', 'xc']],
            output_core_dims=[['yc', 'xc']],
            vectorize=True,      
            dask='parallelized', 
            output_dtypes=[zeta.dtype]
        )

        psi.attrs.update({
            'standard_name': 'streamfunction',
            'long_name': 'streamfunction',
            'units': 'm2 s-1',
        })
        return psi.rename('psi')
