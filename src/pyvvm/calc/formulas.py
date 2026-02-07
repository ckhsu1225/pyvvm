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
"""

import numpy as np
from .constants import epsilon, Lv, Cp_d, g, kappa

__all__ = [
    # Basic thermodynamic variables
    'temperature',
    'vapor_pressure',
    'saturation_vapor_pressure',
    'saturation_mixing_ratio',
    'dew_point_temperature',
    'lcl_temperature',
    'relative_humidity',
    # Virtual quantities
    'virtual_temperature',
    'virtual_potential_temperature',
    # Equivalent potential temperature
    'equivalent_potential_temperature',
    'saturation_equivalent_potential_temperature',
    # Static energies
    'dry_static_energy',
    'moist_static_energy',
    'saturation_moist_static_energy',
]


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
        Vapor pressure [Pa], minimum clamped to 1e-10 to avoid log(0)
    """
    e = (p * qv) / (qv + epsilon)
    return np.maximum(e, 1e-10)


def saturation_vapor_pressure(t):
    """
    Compute saturation vapor pressure using Tetens formula.

    Parameters
    ----------
    t : array_like
        Temperature [K]

    Returns
    -------
    array_like
        Saturation vapor pressure [Pa]
    """
    tc = t - 273.15
    return 611.2 * np.exp((17.67 * tc) / (tc + 243.5))


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


def dew_point_temperature(e):
    """
    Compute dew point temperature from vapor pressure.

    Inverts the Tetens formula to find the temperature at which
    air would be saturated given the current vapor pressure.

    Parameters
    ----------
    e : array_like
        Vapor pressure [Pa]

    Returns
    -------
    array_like
        Dew point temperature [K]
    """
    ln_ratio = np.log(e / 611.2)
    return (243.5 * ln_ratio) / (17.67 - ln_ratio) + 273.15


def lcl_temperature(t, td):
    """
    Compute lifting condensation level (LCL) temperature.

    Uses Bolton's (1980) empirical formula for LCL temperature.

    Parameters
    ----------
    t : array_like
        Air temperature [K]
    td : array_like
        Dew point temperature [K]

    Returns
    -------
    array_like
        LCL temperature [K]

    References
    ----------
    Bolton, D. (1980). Eq. (15).
    """
    return 1.0 / (1.0 / (td - 56) + np.log(t / td) / 800) + 56


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


# ============================================================================
# Virtual Quantities
# ============================================================================

def virtual_temperature(t, qv, qc=0, qi=0, qr=0):
    """
    Compute virtual temperature including hydrometeor loading.

    The virtual temperature is the temperature dry air would need
    to have the same density as moist air with hydrometeors.

    Parameters
    ----------
    t : array_like
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
    return t * (1.0 + qv / epsilon) / (1.0 + qv + qc + qi + qr)


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

def equivalent_potential_temperature(t, p, qv, tl):
    """
    Compute equivalent potential temperature using Bolton's formula.

    This is the temperature a parcel would have if all its moisture
    were condensed out and the latent heat used to warm the parcel,
    then brought adiabatically to 1000 hPa.

    Parameters
    ----------
    t : array_like
        Temperature [K]
    p : array_like
        Total pressure [Pa]
    qv : array_like
        Water vapor mixing ratio [kg/kg]
    tl : array_like
        LCL temperature [K]

    Returns
    -------
    array_like
        Equivalent potential temperature [K]

    References
    ----------
    Bolton, D. (1980). Eq. (39).
    """
    e = vapor_pressure(p, qv)
    th_dl = t * (100000.0 / (p - e))**kappa * (t / tl)**(0.28 * qv)
    return th_dl * np.exp((3036.0 / tl - 1.78) * qv * (1 + 0.448 * qv))


def saturation_equivalent_potential_temperature(t, p, es, qvs):
    """
    Compute saturation equivalent potential temperature.

    This is the equivalent potential temperature if the air were saturated.

    Parameters
    ----------
    t : array_like
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
    Bolton, D. (1980). Modified from Eq. (39) for saturation.
    """
    th_l = t * (100000.0 / (p - es))**kappa
    return th_l * np.exp((3036.0 / t - 1.78) * qvs * (1 + 0.448 * qvs))


# ============================================================================
# Static Energies
# ============================================================================

def dry_static_energy(t, z):
    """
    Compute dry static energy.

    Parameters
    ----------
    t : array_like
        Temperature [K]
    z : array_like
        Height [m]

    Returns
    -------
    array_like
        Dry static energy [J/kg]
    """
    return Cp_d * t + g * z


def moist_static_energy(t, z, qv):
    """
    Compute moist static energy.

    Parameters
    ----------
    t : array_like
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
    return Cp_d * t + g * z + Lv * qv / (1.0 + qv)


def saturation_moist_static_energy(t, z, qvs):
    """
    Compute saturation moist static energy.

    Parameters
    ----------
    t : array_like
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
    return Cp_d * t + g * z + Lv * qvs / (1.0 + qvs)
