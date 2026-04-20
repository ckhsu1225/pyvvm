"""
Thermodynamic calculations mixin for VVM data.

This module provides the ThermoMixin class which adds thermodynamic
diagnostic calculations to the VVM accessor.
"""

from __future__ import annotations

import numpy as np
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
    **Latent Heats**

    - ``Lv`` : Latent heat of vaporization [J/kg]
    - ``Lf`` : Latent heat of fusion [J/kg]
    - ``Ls`` : Latent heat of sublimation [J/kg]

    **Temperature Variables**
    
    - ``t`` : Air temperature [K]
    - ``tv`` : Virtual temperature [K]
    - ``td`` : Dew point temperature [K]
    - ``tl`` : Lifting condensation level temperature [K]
    
    **Potential Temperature Variables**
    
    - ``thv`` : Virtual potential temperature [K]
    - ``the`` : Equivalent potential temperature [K]
    - ``thes`` : Saturation equivalent potential temperature [K]
    - ``thei`` : Equivalent potential temperature with respect to ice [K]
    
    **Moisture Variables**
    
    - ``e`` : Vapor pressure [Pa]
    - ``esl`` : Saturation vapor pressure with respect to liquid water [Pa]
    - ``esi`` : Saturation vapor pressure with respect to ice [Pa]
    - ``qvsl`` : Saturation mixing ratio with respect to liquid water [kg/kg]
    - ``qvsi`` : Saturation mixing ratio with respect to ice [kg/kg]
    - ``rhl`` : Relative humidity with respect to liquid water [1]
    - ``rhi`` : Relative humidity with respect to ice [1]
    
    **Static Energy Variables**
    
    - ``sd`` : Dry static energy [J/kg]
    - ``hm`` : Moist static energy [J/kg]
    - ``hms`` : Saturation moist static energy [J/kg]
    - ``hf`` : Frozen moist static energy [J/kg]
    
    **Entropy and Gibbs Free Energy Variables**
    
    - ``s`` : Specific entropy [J/kg/K]
    - ``gv`` : Specific Gibbs free energy of water vapor [J/kg]
    - ``gl`` : Specific Gibbs free energy of liquid water [J/kg]
    - ``gi`` : Specific Gibbs free energy of ice [J/kg]
    
    **Stability Variables**
    
    - ``b`` : Buoyancy [m/s²]
    - ``n2`` : Brunt-Väisälä frequency squared [s⁻²]
    - ``cape_cin`` : CAPE and CIN from parcel analysis [J/kg]
    
    **Column-Integrated Variables**
    
    - ``cwv`` : Column water vapor [mm]
    - ``lwp`` : Liquid water path [mm]
    - ``iwp`` : Ice water path [mm]
    - ``crh`` : Column relative humidity [1]
    
    """

    # =========================================================================
    # Private calculation methods (thin wrappers around formulas)
    # =========================================================================

    def _calc_qv(self, qv):
        """Ensure non-negative water vapor mixing ratio."""
        return np.maximum(qv, 0.0)

    def _calc_Lv(self, pi, th):
        t = F.temperature(pi, th)
        return F.latent_heat_of_vaporization(t)

    def _calc_Lf(self, pi, th):
        t = F.temperature(pi, th)
        return F.latent_heat_of_fusion(t)

    def _calc_Ls(self, pi, th):
        t = F.temperature(pi, th)
        return F.latent_heat_of_sublimation(t)

    def _calc_t(self, pi, th):
        return F.temperature(pi, th)

    def _calc_e(self, p, qv):
        return F.vapor_pressure(p, qv)

    def _calc_es(self, pi, th, phase):
        t = F.temperature(pi, th)
        return F.saturation_vapor_pressure(t, phase)

    def _calc_qvs(self, p, pi, th, phase):
        es = self._calc_es(pi, th, phase)
        return F.saturation_mixing_ratio(p, es)

    def _calc_rh(self, p, pi, th, qv, phase):
        e = F.vapor_pressure(p, qv)
        es = self._calc_es(pi, th, phase)
        return F.relative_humidity(e, es)

    def _calc_td(self, p, qv):
        e = F.vapor_pressure(p, qv)
        return F.dew_point_temperature(e)

    def _calc_tl(self, p, pi, th, qv):
        t = F.temperature(pi, th)
        td = self._calc_td(p, qv)
        return F.lcl_temperature(t, td)

    def _calc_tv(self, pi, th, qv, qc, qi, qr):
        t = F.temperature(pi, th)
        return F.virtual_temperature(t, qv, qc, qi, qr)

    def _calc_thv(self, th, qv, qc, qi, qr):
        return F.virtual_potential_temperature(th, qv, qc, qi, qr)

    def _calc_the(self, p, pi, th, qv):
        tl = self._calc_tl(p, pi, th, qv)
        return F.equivalent_potential_temperature(th, p, pi, qv, tl)

    def _calc_thes(self, p, pi, th):
        t = F.temperature(pi, th)
        es = F.saturation_vapor_pressure(t)
        qvs = F.saturation_mixing_ratio(p, es)
        return F.saturation_equivalent_potential_temperature(t, p, es, qvs)

    def _calc_thei(self, p, pi, th, qv, qc, qi, qr):
        t = F.temperature(pi, th)
        return F.ice_equivalent_potential_temperature(t, p, qv, qc, qi, qr)

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

    def _calc_hf(self, z, pi, th, qv, qi):
        t = F.temperature(pi, th)
        return F.frozen_moist_static_energy(t, z, qv, qi)

    def _calc_s(self, p, pi, th, qv, qc, qi, qr):
        t = F.temperature(pi, th)
        return F.specific_entropy(t, p, qv, qc, qi, qr)

    def _calc_gv(self, p, pi, th, qv):
        t = F.temperature(pi, th)
        return F.specific_gibbs_free_energy_of_water_vapor(t, p, qv)

    def _calc_gl(self, pi, th):
        t = F.temperature(pi, th)
        return F.specific_gibbs_free_energy_of_liquid_water(t)

    def _calc_gi(self, pi, th):
        t = F.temperature(pi, th)
        return F.specific_gibbs_free_energy_of_ice(t)

    # =========================================================================
    # Public properties - Latent heat
    # =========================================================================

    @property
    def Lv(self) -> xr.DataArray:
        """Latent heat of vaporization [J/kg]."""
        ds = self._ds
        Lv = self._calc_Lv(ds['pibar'], ds['th'])
        Lv.attrs.update({
            'standard_name': 'latent_heat_of_vaporization',
            'long_name': 'latent heat of vaporization',
            'units': 'J kg-1',
        })
        return Lv.rename('Lv')

    @property
    def Lf(self) -> xr.DataArray:
        """Latent heat of fusion [J/kg]."""
        ds = self._ds
        Lf = self._calc_Lf(ds['pibar'], ds['th'])
        Lf.attrs.update({
            'standard_name': 'latent_heat_of_fusion',
            'long_name': 'latent heat of fusion',
            'units': 'J kg-1',
        })
        return Lf.rename('Lf')

    @property
    def Ls(self) -> xr.DataArray:
        """Latent heat of sublimation [J/kg]."""
        ds = self._ds
        Ls = self._calc_Ls(ds['pibar'], ds['th'])
        Ls.attrs.update({
            'standard_name': 'latent_heat_of_sublimation',
            'long_name': 'latent heat of sublimation',
            'units': 'J kg-1',
        })
        return Ls.rename('Ls')

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
        qv = self._calc_qv(ds['qv'])
        tv = self._calc_tv(ds['pibar'], ds['th'], qv, ds['qc'], ds['qi'], ds['qr'])
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
        qv = self._calc_qv(ds['qv'])
        td = self._calc_td(ds['pbar'], qv)
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
        qv = self._calc_qv(ds['qv'])
        tl = self._calc_tl(ds['pbar'], ds['pibar'], ds['th'], qv)
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
        qv = self._calc_qv(ds['qv'])
        thv = self._calc_thv(ds['th'], qv, ds['qc'], ds['qi'], ds['qr'])
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
        qv = self._calc_qv(ds['qv'])
        the = self._calc_the(ds['pbar'], ds['pibar'], ds['th'], qv)
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

    @property
    def thei(self) -> xr.DataArray:
        """Equivalent potential temperature with respect to ice [K]."""
        ds = self._ds
        qv = self._calc_qv(ds['qv'])
        thei = self._calc_thei(ds['pbar'], ds['pibar'], ds['th'], qv, ds['qc'], ds['qi'], ds['qr'])
        thei.attrs.update({
            'standard_name': 'ice_equivalent_potential_temperature',
            'long_name': 'equivalent potential temperature with respect to ice',
            'units': 'K',
        })
        return thei.rename('thei')

    # =========================================================================
    # Public properties - Moisture
    # =========================================================================

    @property
    def e(self) -> xr.DataArray:
        """Vapor pressure [Pa]."""
        ds = self._ds
        qv = self._calc_qv(ds['qv'])
        e = self._calc_e(ds['pbar'], qv)
        e.attrs.update({
            'standard_name': 'vapor_pressure',
            'long_name': 'vapor pressure',
            'units': 'Pa',
        })
        return e.rename('e')

    @property
    def esl(self) -> xr.DataArray:
        """Saturation vapor pressure with respect to liquid water [Pa]."""
        ds = self._ds
        esl = self._calc_es(ds['pibar'], ds['th'], phase='liquid')
        esl.attrs.update({
            'standard_name': 'saturation_vapor_pressure_liquid',
            'long_name': 'saturation vapor pressure with respect to liquid water',
            'units': 'Pa',
        })
        return esl.rename('esl')

    @property
    def esi(self) -> xr.DataArray:
        """Saturation vapor pressure with respect to ice [Pa]."""
        ds = self._ds
        esi = self._calc_es(ds['pibar'], ds['th'], phase='ice')
        esi.attrs.update({
            'standard_name': 'saturation_vapor_pressure_ice',
            'long_name': 'saturation vapor pressure with respect to ice',
            'units': 'Pa',
        })
        return esi.rename('esi')

    @property
    def qvsl(self) -> xr.DataArray:
        """Saturation mixing ratio with respect to liquid water [kg/kg]."""
        ds = self._ds
        qvsl = self._calc_qvs(ds['pbar'], ds['pibar'], ds['th'], phase='liquid')
        qvsl.attrs.update({
            'standard_name': 'saturation_mixing_ratio_liquid',
            'long_name': 'saturation mixing ratio with respect to liquid water',
            'units': 'kg kg-1',
        })
        return qvsl.rename('qvsl')

    @property
    def qvsi(self) -> xr.DataArray:
        """Saturation mixing ratio with respect to ice [kg/kg]."""
        ds = self._ds
        qvsi = self._calc_qvs(ds['pbar'], ds['pibar'], ds['th'], phase='ice')
        qvsi.attrs.update({
            'standard_name': 'saturation_mixing_ratio_ice',
            'long_name': 'saturation mixing ratio with respect to ice',
            'units': 'kg kg-1',
        })
        return qvsi.rename('qvsi')

    @property
    def rhl(self) -> xr.DataArray:
        """Relative humidity with respect to liquid water [1]."""
        ds = self._ds
        qv = self._calc_qv(ds['qv'])
        rhl = self._calc_rh(ds['pbar'], ds['pibar'], ds['th'], qv, phase='liquid')
        rhl.attrs.update({
            'standard_name': 'relative_humidity_liquid',
            'long_name': 'relative humidity with respect to liquid water',
            'units': '1',
        })
        return rhl.rename('rhl')

    @property
    def rhi(self) -> xr.DataArray:
        """Relative humidity with respect to ice [1]."""
        ds = self._ds
        qv = self._calc_qv(ds['qv'])
        rhi = self._calc_rh(ds['pbar'], ds['pibar'], ds['th'], qv, phase='ice')
        rhi.attrs.update({
            'standard_name': 'relative_humidity_ice',
            'long_name': 'relative humidity with respect to ice',
            'units': '1',
        })
        return rhi.rename('rhi')

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
        qv = self._calc_qv(ds['qv'])
        hm = self._calc_hm(ds['zc'], ds['pibar'], ds['th'], qv)
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

    @property
    def hf(self) -> xr.DataArray:
        """Frozen moist static energy [J/kg]."""
        ds = self._ds
        qv = self._calc_qv(ds['qv'])
        hf = self._calc_hf(ds['zc'], ds['pibar'], ds['th'], qv, ds['qi'])
        hf.attrs.update({
            'standard_name': 'frozen_moist_static_energy',
            'long_name': 'frozen moist static energy',
            'units': 'J kg-1',
        })
        return hf.rename('hf')

    # =========================================================================
    # Public properties - Entropy and Gibbs free energy
    # =========================================================================

    @property
    def s(self) -> xr.DataArray:
        """Specific entropy [J/kg/K]."""
        ds = self._ds
        qv = self._calc_qv(ds['qv'])
        s = self._calc_s(ds['pbar'], ds['pibar'], ds['th'], qv, ds['qc'], ds['qi'], ds['qr'])
        s.attrs.update({
            'standard_name': 'specific_entropy',
            'long_name': 'specific entropy',
            'units': 'J kg-1 K-1',
        })
        return s.rename('s')

    @property
    def gv(self) -> xr.DataArray:
        """Specific Gibbs free energy of water vapor [J/kg]."""
        ds = self._ds
        qv = self._calc_qv(ds['qv'])
        gv = self._calc_gv(ds['pbar'], ds['pibar'], ds['th'], qv)
        gv.attrs.update({
            'standard_name': 'specific_gibbs_free_energy_of_water_vapor',
            'long_name': 'specific Gibbs free energy of water vapor',
            'units': 'J kg-1',
        })
        return gv.rename('gv')

    @property
    def gl(self) -> xr.DataArray:
        """Specific Gibbs free energy of liquid water [J/kg]."""
        ds = self._ds
        gl = self._calc_gl(ds['pibar'], ds['th'])
        gl.attrs.update({
            'standard_name': 'specific_gibbs_free_energy_of_liquid_water',
            'long_name': 'specific Gibbs free energy of liquid water',
            'units': 'J kg-1',
        })
        return gl.rename('gl')

    @property
    def gi(self) -> xr.DataArray:
        """Specific Gibbs free energy of ice [J/kg]."""
        ds = self._ds
        gi = self._calc_gi(ds['pibar'], ds['th'])
        gi.attrs.update({
            'standard_name': 'specific_gibbs_free_energy_of_ice',
            'long_name': 'specific Gibbs free energy of ice',
            'units': 'J kg-1',
        })
        return gi.rename('gi')

    # =========================================================================
    # Public properties - Stability
    # =========================================================================

    @property
    def b(self) -> xr.DataArray:
        """Buoyancy [m/s²]."""
        ds = self._ds
        qv = self._calc_qv(ds['qv'])
        thv = self._calc_thv(ds['th'], qv, ds['qc'], ds['qi'], ds['qr'])
        thv_bar = thv.mean(['xc', 'yc'])
        thv_prime = thv - thv_bar
        b = g * thv_prime / thv_bar

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

    @property
    def cape_cin(self) -> xr.Dataset:
        """CAPE and CIN from parcel analysis."""
        self._validate_chunks('cape_cin')
        from .parcel import compute_cape_cin
        return compute_cape_cin(self._ds)

    # =========================================================================
    # Public properties - Column-Integrated
    # =========================================================================

    @property
    def cwv(self) -> xr.DataArray:
        """Column water vapor [mm]."""
        ds = self._ds
        grid = self.grid

        qv = self._calc_qv(ds['qv'])
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
