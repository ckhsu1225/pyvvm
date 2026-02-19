"""
Thermodynamic calculations mixin for VVM data.

This module provides the ThermoMixin class which adds thermodynamic
diagnostic calculations to the VVM accessor.
"""

from __future__ import annotations

import xarray as xr
from .constants import g
from . import formulas as F

__all__ = [
    'ThermoMixin',
]


class ThermoMixin:
    """
    Mixin class providing thermodynamic calculations for VVM data.
    
    All properties return xr.DataArray with appropriate CF-compliant attributes.
    
    Available Properties
    --------------------
    **Temperature Variables**
    
    - ``t`` : Air temperature [K]
    - ``tv`` : Virtual temperature [K]
    - ``td`` : Dew point temperature [K]
    - ``tl`` : Lifting condensation level temperature [K]
    
    **Potential Temperature Variables**
    
    - ``thv`` : Virtual potential temperature [K]
    - ``the`` : Equivalent potential temperature [K]
    - ``thes`` : Saturation equivalent potential temperature [K]
    
    **Moisture Variables**
    
    - ``e`` : Vapor pressure [Pa]
    - ``es`` : Saturation vapor pressure [Pa]
    - ``qvs`` : Saturation mixing ratio [kg/kg]
    - ``rh`` : Relative humidity [1]
    
    **Static Energy Variables**
    
    - ``sd`` : Dry static energy [J/kg]
    - ``hm`` : Moist static energy [J/kg]
    - ``hms`` : Saturation moist static energy [J/kg]
    
    **Stability Variables**
    
    - ``b`` : Buoyancy [m/s²]
    - ``n2`` : Brunt-Väisälä frequency squared [s⁻²]
    
    **Column-Integrated Variables**
    
    - ``cwv`` : Column water vapor [mm]
    - ``lwp`` : Liquid water path [mm]
    - ``iwp`` : Ice water path [mm]
    - ``crh`` : Column relative humidity [1]
    
    **Derived Products**
    
    - ``cape_cin`` : CAPE and CIN from parcel analysis
    """

    # =========================================================================
    # Private calculation methods (thin wrappers around formulas)
    # =========================================================================

    def _calc_t(self, pi, th):
        return F.temperature(pi, th)

    def _calc_e(self, p, qv):
        return F.vapor_pressure(p, qv)

    def _calc_es(self, pi, th):
        t = F.temperature(pi, th)
        return F.saturation_vapor_pressure(t)

    def _calc_qvs(self, p, pi, th):
        es = self._calc_es(pi, th)
        return F.saturation_mixing_ratio(p, es)

    def _calc_td(self, p, qv):
        e = F.vapor_pressure(p, qv)
        return F.dew_point_temperature(e)

    def _calc_tl(self, p, pi, th, qv):
        t = F.temperature(pi, th)
        td = self._calc_td(p, qv)
        return F.lcl_temperature(t, td)

    def _calc_rh(self, p, pi, th, qv):
        e = F.vapor_pressure(p, qv)
        es = self._calc_es(pi, th)
        return F.relative_humidity(e, es)

    def _calc_tv(self, pi, th, qv, qc, qi, qr):
        t = F.temperature(pi, th)
        return F.virtual_temperature(t, qv, qc, qi, qr)

    def _calc_thv(self, th, qv, qc, qi, qr):
        return F.virtual_potential_temperature(th, qv, qc, qi, qr)

    def _calc_the(self, p, pi, th, qv):
        t = F.temperature(pi, th)
        tl = self._calc_tl(p, pi, th, qv)
        return F.equivalent_potential_temperature(t, p, qv, tl)

    def _calc_thes(self, p, pi, th):
        t = F.temperature(pi, th)
        es = F.saturation_vapor_pressure(t)
        qvs = F.saturation_mixing_ratio(p, es)
        return F.saturation_equivalent_potential_temperature(t, p, es, qvs)

    def _calc_sd(self, z, pi, th):
        t = F.temperature(pi, th)
        return F.dry_static_energy(t, z)

    def _calc_hm(self, z, pi, th, qv):
        t = F.temperature(pi, th)
        return F.moist_static_energy(t, z, qv)

    def _calc_hms(self, z, p, pi, th):
        t = F.temperature(pi, th)
        qvs = self._calc_qvs(p, pi, th)
        return F.saturation_moist_static_energy(t, z, qvs)

    def _calc_b(self, thbar, qvbar, th, qv, qc, qi, qr):
        thv_bar = F.virtual_potential_temperature(thbar, qvbar, 0, 0, 0)
        thv = F.virtual_potential_temperature(th, qv, qc, qi, qr)
        thv_prime = thv - thv_bar
        return g * thv_prime / thv_bar

    # =========================================================================
    # Public properties - Temperature
    # =========================================================================

    @property
    def t(self) -> xr.DataArray:
        """Air temperature [K]."""
        ds = self._ds
        t = self._calc_t(ds['pibar'], ds['th'])
        t.attrs.update({
            'standard_name': 'air_temperature',
            'long_name': 'air temperature',
            'units': 'K',
        })
        return t.rename('t')

    @property
    def tv(self) -> xr.DataArray:
        """Virtual temperature [K]."""
        ds = self._ds
        tv = self._calc_tv(ds['pibar'], ds['th'], ds['qv'], ds['qc'], ds['qi'], ds['qr'])
        tv.attrs.update({
            'standard_name': 'virtual_temperature',
            'long_name': 'virtual temperature',
            'units': 'K',
        })
        return tv.rename('tv')

    @property
    def td(self) -> xr.DataArray:
        """Dew point temperature [K]."""
        ds = self._ds
        td = self._calc_td(ds['pbar'], ds['qv'])
        td.attrs.update({
            'standard_name': 'dew_point_temperature',
            'long_name': 'dew point temperature',
            'units': 'K',
        })
        return td.rename('td')

    @property
    def tl(self) -> xr.DataArray:
        """Lifting condensation level temperature [K]."""
        ds = self._ds
        tl = self._calc_tl(ds['pbar'], ds['pibar'], ds['th'], ds['qv'])
        tl.attrs.update({
            'standard_name': 'lifting_condensation_level_temperature',
            'long_name': 'lifting condensation level temperature',
            'units': 'K',
        })
        return tl.rename('tl')

    # =========================================================================
    # Public properties - Potential Temperature
    # =========================================================================

    @property
    def thv(self) -> xr.DataArray:
        """Virtual potential temperature [K]."""
        ds = self._ds
        thv = self._calc_thv(ds['th'], ds['qv'], ds['qc'], ds['qi'], ds['qr'])
        thv.attrs.update({
            'standard_name': 'virtual_potential_temperature',
            'long_name': 'virtual potential temperature',
            'units': 'K',
        })
        return thv.rename('thv')

    @property
    def the(self) -> xr.DataArray:
        """Equivalent potential temperature [K]."""
        ds = self._ds
        the = self._calc_the(ds['pbar'], ds['pibar'], ds['th'], ds['qv'])
        the.attrs.update({
            'standard_name': 'equivalent_potential_temperature',
            'long_name': 'equivalent potential temperature',
            'units': 'K',
        })
        return the.rename('the')

    @property
    def thes(self) -> xr.DataArray:
        """Saturation equivalent potential temperature [K]."""
        ds = self._ds
        thes = self._calc_thes(ds['pbar'], ds['pibar'], ds['th'])
        thes.attrs.update({
            'standard_name': 'saturation_equivalent_potential_temperature',
            'long_name': 'saturation equivalent potential temperature',
            'units': 'K',
        })
        return thes.rename('thes')

    # =========================================================================
    # Public properties - Moisture
    # =========================================================================

    @property
    def e(self) -> xr.DataArray:
        """Vapor pressure [Pa]."""
        ds = self._ds
        e = self._calc_e(ds['pbar'], ds['qv'])
        e.attrs.update({
            'standard_name': 'vapor_pressure',
            'long_name': 'vapor pressure',
            'units': 'Pa',
        })
        return e.rename('e')

    @property
    def es(self) -> xr.DataArray:
        """Saturation vapor pressure [Pa]."""
        ds = self._ds
        es = self._calc_es(ds['pibar'], ds['th'])
        es.attrs.update({
            'standard_name': 'saturation_vapor_pressure',
            'long_name': 'saturation vapor pressure',
            'units': 'Pa',
        })
        return es.rename('es')

    @property
    def qvs(self) -> xr.DataArray:
        """Saturation mixing ratio [kg/kg]."""
        ds = self._ds
        qvs = self._calc_qvs(ds['pbar'], ds['pibar'], ds['th'])
        qvs.attrs.update({
            'standard_name': 'saturation_mixing_ratio',
            'long_name': 'saturation mixing ratio',
            'units': 'kg kg-1',
        })
        return qvs.rename('qvs')

    @property
    def rh(self) -> xr.DataArray:
        """Relative humidity [1]."""
        ds = self._ds
        rh = self._calc_rh(ds['pbar'], ds['pibar'], ds['th'], ds['qv'])
        rh.attrs.update({
            'standard_name': 'relative_humidity',
            'long_name': 'relative humidity',
            'units': '1',
        })
        return rh.rename('rh')

    # =========================================================================
    # Public properties - Static Energy
    # =========================================================================

    @property
    def sd(self) -> xr.DataArray:
        """Dry static energy [J/kg]."""
        ds = self._ds
        sd = self._calc_sd(ds['zc'], ds['pibar'], ds['th'])
        sd.attrs.update({
            'standard_name': 'dry_static_energy',
            'long_name': 'dry static energy',
            'units': 'J kg-1',
        })
        return sd.rename('sd')

    @property
    def hm(self) -> xr.DataArray:
        """Moist static energy [J/kg]."""
        ds = self._ds
        hm = self._calc_hm(ds['zc'], ds['pibar'], ds['th'], ds['qv'])
        hm.attrs.update({
            'standard_name': 'moist_static_energy',
            'long_name': 'moist static energy',
            'units': 'J kg-1',
        })
        return hm.rename('hm')

    @property
    def hms(self) -> xr.DataArray:
        """Saturation moist static energy [J/kg]."""
        ds = self._ds
        hms = self._calc_hms(ds['zc'], ds['pbar'], ds['pibar'], ds['th'])
        hms.attrs.update({
            'standard_name': 'saturation_moist_static_energy',
            'long_name': 'saturation moist static energy',
            'units': 'J kg-1',
        })
        return hms.rename('hms')

    # =========================================================================
    # Public properties - Stability
    # =========================================================================

    @property
    def b(self) -> xr.DataArray:
        """Buoyancy [m/s²]."""
        ds = self._ds
        b = self._calc_b(ds['thbar'], ds['qvbar'], ds['th'], ds['qv'], ds['qc'], ds['qi'], ds['qr'])
        b.attrs.update({
            'standard_name': 'buoyancy',
            'long_name': 'buoyancy',
            'units': 'm s-2',
        })
        return b.rename('b')

    @property
    def n2(self) -> xr.DataArray:
        """Brunt-Väisälä frequency squared [s⁻²]."""
        self._validate_chunks('n2')
        ds = self._ds
        grid = self.grid

        thv_bar = F.virtual_potential_temperature(ds['thbar'], ds['qvbar'], 0, 0, 0)
        thv = grid.interp(self.mask(self.thv), 'Z')
        dthv_dz = grid.derivative(thv, 'Z')

        n2 = (g / thv_bar) * dthv_dz
        n2.attrs.update({
            'standard_name': 'brunt_vaisala_frequency_squared',
            'long_name': 'Brunt-Vaisala frequency squared',
            'units': 's-2',
        })
        return n2.rename('n2')

    # =========================================================================
    # Public properties - Column-Integrated
    # =========================================================================

    @property
    def cwv(self) -> xr.DataArray:
        """Column water vapor [mm]."""
        ds = self._ds
        grid = self.grid

        qv = ds['qv']
        rho = ds['rho']
        integrand = self.mask(qv * rho)

        cwv = grid.integrate(integrand, 'Z')
        cwv.attrs.update({
            'standard_name': 'column_water_vapor',
            'long_name': 'column water vapor',
            'units': 'mm',
        })
        return cwv.rename('cwv')

    @property
    def lwp(self) -> xr.DataArray:
        """Liquid water path [mm]."""
        ds = self._ds
        grid = self.grid

        ql = ds['qc'] + ds['qr']
        rho = ds['rho']
        integrand = self.mask(ql * rho)

        lwp = grid.integrate(integrand, 'Z')
        lwp.attrs.update({
            'standard_name': 'liquid_water_path',
            'long_name': 'liquid water path',
            'units': 'mm',
        })
        return lwp.rename('lwp')

    @property
    def iwp(self) -> xr.DataArray:
        """Ice water path [mm]."""
        ds = self._ds
        grid = self.grid

        qi = ds['qi']
        rho = ds['rho']
        integrand = self.mask(qi * rho)

        iwp = grid.integrate(integrand, 'Z')
        iwp.attrs.update({
            'standard_name': 'ice_water_path',
            'long_name': 'ice water path',
            'units': 'mm',
        })
        return iwp.rename('iwp')

    @property
    def crh(self) -> xr.DataArray:
        """Column relative humidity [1]."""
        ds = self._ds
        grid = self.grid

        cwv = self.cwv
        qvs = self.qvs
        rho = ds['rho']
        integrand = self.mask(qvs * rho)
        cwvs = grid.integrate(integrand, 'Z')

        crh = cwv / cwvs
        crh.attrs.update({
            'standard_name': 'column_relative_humidity',
            'long_name': 'column relative humidity',
            'units': '1',
        })
        return crh.rename('crh')

    # =========================================================================
    # Public properties - Derived Products
    # =========================================================================

    @property
    def cape_cin(self) -> xr.Dataset:
        """CAPE and CIN from parcel analysis."""
        self._validate_chunks('cape_cin')
        from .parcel import compute_cape_cin
        return compute_cape_cin(self._ds)
