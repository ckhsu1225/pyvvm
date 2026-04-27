"""
Pure thermodynamic formulas for atmospheric calculations.

This module contains numpy-based functions that implement fundamental
thermodynamic equations. These are independent of xarray and can be
used with scalars, numpy arrays, or within xr.map_blocks.

All formulas use SI units unless otherwise specified.

References
----------
Bolton, D., 1980: The Computation of Equivalent Potential Temperature.
    Mon. Wea. Rev., 108, 1046–1053,
    https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2.
Bretherton, C. S., P. N. Blossey, and M. Khairoutdinov, 2005:
    An Energy-Balance Analysis of Deep Convective Self-Aggregation above Uniform SST.
    J. Atmos. Sci., 62, 4273–4292,
    https://doi.org/10.1175/JAS3614.1.
D. J.Raymond, 2013: Sources and sinks of entropy in the atmosphere.
    J. Adv. Model. Earth Syst., 5, 755–763,
    https://doi.org/10.1002/jame.20050.
Pauluis, O. M., 2016: The Mean Air Flow as Lagrangian Dynamics Approximation and Its Application to Moist Convection.
    J. Atmos. Sci., 73, 4407–4425,
    https://doi.org/10.1175/JAS-D-15-0284.1.
Ambaum MHP, 2020: Accurate, simple equation for saturated vapour pressure over water and ice.
    QJR Meteorol Soc., 146: 4252–4258,
    https://doi.org/10.1002/qj.3899.
Warren, R.A, 2025: A consistent treatment of mixed-phase saturation for atmospheric thermodynamics.
    QJR Meteorol Soc., 151:e4866.
    https://doi.org/10.1002/qj.4866.
Gu, J.-F., & Tan, Z.-M., 2025: Reconciling the discrepancies of equivalent potential temperatures in atmosphere:
    A general pathway rooted in entropy conservation.
    Journal of Advances in Modeling Earth Systems, 17, e2025MS004985.
    https://doi.org/10.1029/2025MS004985.
"""

import numpy as np
from .constants import (
    Lv0, Lf0, Ls0,
    T0, p0, es0,
    Cpd, Cpl, Cpv, Cpi, Rd, Rv,
    epsilon, kappa, g
)

__all__ = [
    # Latent heat
    'latent_heat_of_vaporization',
    'latent_heat_of_fusion',
    'latent_heat_of_sublimation',

    # Basic thermodynamic variables
    'temperature',
    'vapor_pressure',
    'saturation_vapor_pressure',
    'saturation_mixing_ratio',
    'relative_humidity',
    'dew_point_temperature',
    'lcl_temperature',

    # Virtual quantities
    'virtual_temperature',
    'virtual_potential_temperature',

    # Equivalent potential temperature
    'equivalent_potential_temperature',
    'saturation_equivalent_potential_temperature',
    'ice_equivalent_potential_temperature',

    # Static energies
    'dry_static_energy',
    'moist_static_energy',
    'saturation_moist_static_energy',
    'frozen_moist_static_energy',

    # Enthalpy, Entropy and Gibbs free energy
    'specific_enthalpy',
    'specific_entropy',
    'specific_gibbs_free_energy_of_water_vapor',
    'specific_gibbs_free_energy_of_liquid_water',
    'specific_gibbs_free_energy_of_ice',
]


def _where(condition, x, y):
    """Helper function to apply np.where or xarray's where if available."""
    where = getattr(x, 'where', None)
    if callable(where):
        return where(condition, y)
    return np.where(condition, x, y)

# ============================================================================
# Latent heat
# ============================================================================

def latent_heat_of_vaporization(T):
    """
    Compute temperature-dependent latent heat of vaporization.

    Parameters
    ----------
    T : array_like
        Temperature [K]

    Returns
    -------
    array_like
        Latent heat of vaporization [J kg^-1]

    References
    ----------
    Warren (2025) Eq. (23).
    """
    return Lv0 + (Cpv - Cpl) * (T - T0)


def latent_heat_of_fusion(T):
    """
    Compute temperature-dependent latent heat of fusion.

    Parameters
    ----------
    T : array_like
        Temperature [K]

    Returns
    -------
    array_like
        Latent heat of fusion [J kg^-1]

    References
    ----------
    Warren (2025) Eq. (25).
    """
    return Lf0 + (Cpl - Cpi) * (T - T0)


def latent_heat_of_sublimation(T):
    """
    Compute temperature-dependent latent heat of sublimation.

    Parameters
    ----------
    T : array_like
        Temperature [K]

    Returns
    -------
    array_like
        Latent heat of sublimation [J kg^-1]

    References
    ----------
    Warren (2025) Eq. (24).
    """
    return Ls0 + (Cpv - Cpi) * (T - T0)


# ============================================================================
# Basic Thermodynamic Variables
# ============================================================================

def temperature(pi, th):
    """
    Compute temperature from Exner function and potential temperature.

    Parameters
    ----------
    pi : array_like
        Exner function (p/p0)^kappa [dimensionless]
    th : array_like
        Potential temperature [K]

    Returns
    -------
    array_like
        Temperature [K]
    """
    return th * pi


def vapor_pressure(p, qv):
    """
    Compute vapor pressure from total pressure and mixing ratio.

    Parameters
    ----------
    p : array_like
        Total pressure [Pa]
    qv : array_like
        Water vapor mixing ratio [kg/kg]

    Returns
    -------
    array_like
        Vapor pressure [Pa]
    """
    return (p * qv) / (qv + epsilon)


def saturation_vapor_pressure(T, phase='liquid'):
    """
    Compute saturation vapor pressure over liquid water or ice.

    Parameters
    ----------
    T : array_like
        Temperature [K]
    phase : {'liquid', 'ice'}, optional
        Phase to compute saturation vapor pressure for. Default is 'liquid'.

    Returns
    -------
    array_like
        Saturation vapor pressure [Pa]

    References
    ----------
    Ambaum (2020) Eq. (13), (17).
    """
    if phase == 'liquid':
        Lv = latent_heat_of_vaporization(T)
        return es0 * (T0 / T) ** ((Cpl - Cpv) / Rv) * np.exp(Lv0 / Rv / T0 - Lv / Rv / T)
    
    elif phase == 'ice':
        Ls = latent_heat_of_sublimation(T)
        return es0 * (T0 / T) ** ((Cpi - Cpv) / Rv) * np.exp(Ls0 / Rv / T0 - Ls / Rv / T)
    
    else:
        raise ValueError("Invalid phase. Use 'liquid' or 'ice'.")


def saturation_mixing_ratio(p, es):
    """
    Compute saturation mixing ratio from pressure and saturation vapor pressure.

    Parameters
    ----------
    p : array_like
        Total pressure [Pa]
    es : array_like
        Saturation vapor pressure [Pa]

    Returns
    -------
    array_like
        Saturation mixing ratio [kg/kg]
    """
    return (epsilon * es) / (p - es)


def relative_humidity(e, es):
    """
    Compute relative humidity from vapor pressure and saturation vapor pressure.

    Parameters
    ----------
    e : array_like
        Vapor pressure [Pa]
    es : array_like
        Saturation vapor pressure [Pa]

    Returns
    -------
    array_like
        Relative humidity [fraction, 0-1+]
    """
    return e / es


def dew_point_temperature(e):
    """
    Compute dew point temperature from vapor pressure.

    Inverts the Bolton's formula to find the temperature at which
    air would be saturated given the current vapor pressure.

    Parameters
    ----------
    e : array_like
        Vapor pressure [Pa]

    Returns
    -------
    array_like
        Dew point temperature [K]

    References
    ----------
    Bolton (1980) Eq. (10).
    """
    e_safe = _where(e > 0, e, np.nan) # Avoid log of zero or negative
    ln_ratio = np.log(e_safe / 611.2)
    return (243.5 * ln_ratio) / (17.67 - ln_ratio) + 273.15


def lcl_temperature(T, Td):
    """
    Compute lifting condensation level (LCL) temperature.

    Uses Bolton's (1980) empirical formula for LCL temperature.

    Parameters
    ----------
    T : array_like
        Air temperature [K]
    Td : array_like
        Dew point temperature [K]

    Returns
    -------
    array_like
        LCL temperature [K]

    References
    ----------
    Bolton (1980) Eq. (15).
    """
    return 1.0 / (1.0 / (Td - 56) + np.log(T / Td) / 800) + 56


# ============================================================================
# Virtual Quantities
# ============================================================================

def virtual_temperature(T, qv, qc=0, qi=0, qr=0):
    """
    Compute virtual temperature including hydrometeor loading.

    The virtual temperature is the temperature dry air would need
    to have the same density as moist air with hydrometeors.

    Parameters
    ----------
    T : array_like
        Temperature [K]
    qv : array_like
        Water vapor mixing ratio [kg/kg]
    qc : array_like, optional
        Cloud water mixing ratio [kg/kg], default 0
    qi : array_like, optional
        Ice mixing ratio [kg/kg], default 0
    qr : array_like, optional
        Rain water mixing ratio [kg/kg], default 0

    Returns
    -------
    array_like
        Virtual temperature [K]
    """
    return T * (1.0 + qv / epsilon) / (1.0 + qv + qc + qi + qr)


def virtual_potential_temperature(th, qv, qc=0, qi=0, qr=0):
    """
    Compute virtual potential temperature including hydrometeor loading.

    Parameters
    ----------
    th : array_like
        Potential temperature [K]
    qv : array_like
        Water vapor mixing ratio [kg/kg]
    qc : array_like, optional
        Cloud water mixing ratio [kg/kg], default 0
    qi : array_like, optional
        Ice mixing ratio [kg/kg], default 0
    qr : array_like, optional
        Rain water mixing ratio [kg/kg], default 0

    Returns
    -------
    array_like
        Virtual potential temperature [K]
    """
    return th * (1.0 + qv / epsilon) / (1.0 + qv + qc + qi + qr)


# ============================================================================
# Equivalent Potential Temperature
# ============================================================================

def equivalent_potential_temperature(th, p, pi, qv, Tl):
    """
    Compute pseudo-adiabatic equivalent potential temperature using Bolton's formula.

    This is the temperature a parcel would have if all its moisture
    were condensed out and the latent heat used to warm the parcel,
    then brought adiabatically to 1000 hPa.

    Parameters
    ----------
    th : array_like
        Potential temperature [K]
    p : array_like
        Total pressure [Pa]
    pi : array_like
        Exner function (p/p0)^kappa [dimensionless]
    qv : array_like
        Water vapor mixing ratio [kg/kg]
    Tl : array_like
        LCL temperature [K]

    Returns
    -------
    array_like
        Equivalent potential temperature [K]

    References
    ----------
    Bolton (1980) Eq. (39).
    """
    T = temperature(pi, th)
    e = vapor_pressure(p, qv)
    th_dl = T * (p0 / (p - e))**kappa * (T / Tl)**(0.28 * qv)
    the = th_dl * np.exp((3036.0 / Tl - 1.78) * qv * (1 + 0.448 * qv))
    the = _where(qv > 0, the, th) # If no moisture, equivalent potential temperature = potential temperature
    return the


def saturation_equivalent_potential_temperature(T, p, es, qvs):
    """
    Compute pseudo-adiabatic saturation equivalent potential temperature.

    This is the equivalent potential temperature if the air were saturated.

    Parameters
    ----------
    T : array_like
        Temperature [K]
    p : array_like
        Total pressure [Pa]
    es : array_like
        Saturation vapor pressure [Pa]
    qvs : array_like
        Saturation mixing ratio [kg/kg]

    Returns
    -------
    array_like
        Saturation equivalent potential temperature [K]

    References
    ----------
    Bolton (1980) Modified from Eq. (39) for saturation.
    """
    th_l = T * (p0 / (p - es))**kappa
    return th_l * np.exp((3036.0 / T - 1.78) * qvs * (1 + 0.448 * qvs))


def ice_equivalent_potential_temperature(T, p, qv, qc, qi, qr):
    """
    Compute reversible equivalent potential temperature with respect to ice.

    Parameters
    ----------
    T : array_like
        Temperature [K]
    p : array_like
        Total pressure [Pa]
    qv : array_like
        Water vapor mixing ratio [kg/kg]
    qc : array_like
        Cloud water mixing ratio [kg/kg]
    qi : array_like
        Ice mixing ratio [kg/kg]
    qr : array_like
        Rain water mixing ratio [kg/kg]
    
    Returns
    -------
    array_like
        Ice equivalent potential temperature [K]

    References
    ----------
    Gu & Tan (2025) Eq. (30).
    """
    ql = qc + qr
    qt = qv + qc + qi + qr

    Lf = latent_heat_of_fusion(T)
    Ls = latent_heat_of_sublimation(T)

    e = vapor_pressure(p, qv)
    e_safe = _where(e > 0, e, np.nan) # Avoid log of zero or negative
    pd = p - e

    esl = saturation_vapor_pressure(T, phase='liquid')
    esi = saturation_vapor_pressure(T, phase='ice')

    denominator = Cpd + qt * Cpi
    rhl_term = qv * np.log(e_safe / esi)
    rhl_term = _where(qv > 0, rhl_term, 0) # If no water vapor, this term should be zero

    thei = T * (p0 / pd)**(Rd / denominator) *\
        np.exp((qv * Ls + ql * Lf) / denominator / T -\
               Rv * (rhl_term - ql * np.log(esi / esl)) / denominator)
    return thei


# ============================================================================
# Static Energies
# ============================================================================

def dry_static_energy(T, z):
    """
    Compute dry static energy.

    Parameters
    ----------
    T : array_like
        Temperature [K]
    z : array_like
        Height [m]

    Returns
    -------
    array_like
        Dry static energy [J/kg]
    """
    return Cpd * T + g * z


def moist_static_energy(T, z, qv):
    """
    Compute moist static energy.

    Parameters
    ----------
    T : array_like
        Temperature [K]
    z : array_like
        Height [m]
    qv : array_like
        Water vapor mixing ratio [kg/kg]

    Returns
    -------
    array_like
        Moist static energy [J/kg]
    """
    return Cpd * T + g * z + Lv0 * qv


def saturation_moist_static_energy(T, z, qvs):
    """
    Compute saturation moist static energy.

    Parameters
    ----------
    T : array_like
        Temperature [K]
    z : array_like
        Height [m]
    qvs : array_like
        Saturation mixing ratio [kg/kg]

    Returns
    -------
    array_like
        Saturation moist static energy [J/kg]
    """
    return Cpd * T + g * z + Lv0 * qvs


def frozen_moist_static_energy(T, z, qv, qi):
    """
    Compute frozen moist static energy.

    Parameters
    ----------
    T : array_like
        Temperature [K]
    z : array_like
        Height [m]
    qv : array_like
        Water vapor mixing ratio [kg/kg]
    qi : array_like
        Ice mixing ratio [kg/kg]

    Returns
    -------
    array_like
        Frozen moist static energy [J/kg]
    
    References
    ----------
    Bretherton et al. (2005).
    """
    return Cpd * T + g * z + Lv0 * qv - Lf0 * qi


# ============================================================================
# Enthalpy, Entropy and Gibbs Free Energy
# ============================================================================

def specific_enthalpy(T, qv, qc, qi, qr):
    """
    Compute specific enthalpy of moist air with hydrometeors.

    Parameters
    ----------
    T : array_like
        Temperature [K]
    qv : array_like
        Water vapor mixing ratio [kg/kg]
    qc : array_like
        Cloud water mixing ratio [kg/kg]
    qi : array_like
        Ice mixing ratio [kg/kg]
    qr : array_like
        Rain water mixing ratio [kg/kg]

    Returns
    -------
    array_like
        Specific enthalpy [J kg^-1]
    
    References
    ----------
        Pauluis (2016) Eq. (A1a)-(A1c).
    """
    hd = Cpd * (T - T0)
    hv = Cpv * (T - T0) + Lv0
    hl = Cpl * (T - T0)
    hi = Cpi * (T - T0) - Lf0
    return hd + qv * hv + (qc + qr) * hl + qi * hi


def specific_entropy(T, p, qv, qc, qi, qr):
    """
    Compute specific entropy of moist air with hydrometeors.

    Parameters
    ----------
    T : array_like
        Temperature [K]
    p : array_like
        Total pressure [Pa]
    qv : array_like
        Water vapor mixing ratio [kg/kg]
    qc : array_like
        Cloud water mixing ratio [kg/kg]
    qi : array_like
        Ice mixing ratio [kg/kg]
    qr : array_like
        Rain water mixing ratio [kg/kg]

    Returns
    -------
    array_like
        Specific entropy [J kg^-1 K^-1]
    
    References
    ----------
    Raymond (2013) Eq. (8)-(16).
    """
    e = vapor_pressure(p, qv)
    e_safe = _where(e > 0, e, np.nan) # Avoid log of zero or negative

    sd = Cpd * np.log(T / T0) - Rd * np.log(p / p0)
    sv = Cpv * np.log(T / T0) - Rv * np.log(e_safe / es0) + Lv0 / T0
    sl = Cpl * np.log(T / T0)
    si = Cpi * np.log(T / T0) - Lf0 / T0

    sv = _where(qv > 0, sv, 0) # If no water vapor, its contribution to entropy is zero
    return sd + qv * sv + (qc + qr) * sl + qi * si


def specific_gibbs_free_energy_of_water_vapor(T, p, qv):
    """
    Compute specific Gibbs free energy of water vapor.

    Parameters
    ----------
    T : array_like
        Temperature [K]
    p : array_like
        Total pressure [Pa]
    qv : array_like
        Water vapor mixing ratio [kg/kg]

    Returns
    -------
    array_like
        Specific Gibbs free energy of water vapor [J kg^-1]

    References
    ----------
    Pauluis (2016) Eq. (A3a).
    """
    e = vapor_pressure(p, qv)
    e_safe = _where(e > 0, e, np.nan) # Avoid log of zero or negative
    return Cpv * (T - T0 - T * np.log(T / T0)) + Rv * T * np.log(e_safe / es0) + Lv0 * (1 - T / T0)


def specific_gibbs_free_energy_of_liquid_water(T):
    """
    Compute specific Gibbs free energy of liquid water.

    Parameters
    ----------
    T : array_like
        Temperature [K]

    Returns
    -------
    array_like
        Specific Gibbs free energy of liquid water [J kg^-1]

    References
    ----------
    Pauluis (2016) Eq. (A3b).
    """
    return Cpl * (T - T0 - T * np.log(T / T0))


def specific_gibbs_free_energy_of_ice(T):
    """
    Compute specific Gibbs free energy of ice.

    Parameters
    ----------
    T : array_like
        Temperature [K]

    Returns
    -------
    array_like
        Specific Gibbs free energy of ice [J kg^-1]

    References
    ----------
    Pauluis (2016) Eq. (A3c).
    """
    return Cpi * (T - T0 - T * np.log(T / T0)) - Lf0 * (1 - T / T0)
