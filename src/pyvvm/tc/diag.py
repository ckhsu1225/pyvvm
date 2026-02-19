"""
Axisymmetric TC diagnostics on (time, z, r) data.

All functions accept azimuthally-averaged DataArrays with a ``r``
coordinate (bin-center radius in metres) and return DataArrays on the
same grid.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


__all__ = [
    'angular_momentum',
    'inertial_stability',
    'mass_streamfunction',
]


# =============================================================================
# Diagnostics
# =============================================================================


def _require_r(da: xr.DataArray, name: str) -> xr.DataArray:
    """Validate and return the 1-D radial coordinate."""
    if "r" not in da.dims:
        raise ValueError(f"{name} must have 'r' as a dimension.")
    if "r" not in da.coords:
        raise ValueError(f"{name} must have an 'r' coordinate (meters).")
    r = da.coords["r"]
    if r.ndim != 1:
        raise ValueError(f"{name} coordinate 'r' must be 1-D, got ndim={r.ndim}.")
    return r

def angular_momentum(vt: xr.DataArray, f: float) -> xr.DataArray:
    """
    Absolute angular momentum.

    Formula (plain text):
        aam = r * vt + 0.5 * f * r**2

    where:
        r   = radius (m)
        vt  = azimuthal-mean tangential wind (m s-1)
        f   = Coriolis parameter (s-1)
        aam = absolute angular momentum (m2 s-1)

    Parameters
    ----------
    vt : xr.DataArray
        Azimuthal-mean tangential wind with ``r`` coordinate (m).
    f : float
        Coriolis parameter (s⁻¹).

    Returns
    -------
    xr.DataArray
    """
    r = _require_r(vt, "vt")
    # Keep vt dimension order (e.g., time, zc, r) as the output canonical order.
    aam = vt * r + 0.5 * f * r**2
    aam.attrs.update({
        "long_name": "absolute angular momentum",
        "units": "m2 s-1",
    })
    return aam.rename("aam")


def inertial_stability(vt: xr.DataArray, f: float) -> xr.DataArray:
    """
    Inertial stability (squared).

    Formula (plain text):
        zeta = (1/r) * d(r * vt)/dr
        i2   = (f + 2 * vt / r) * (f + zeta)

    where:
        r    = radius (m)
        vt   = azimuthal-mean tangential wind (m s-1)
        f    = Coriolis parameter (s-1)
        zeta = azimuthal-mean vertical vorticity (s-1)
        i2   = inertial stability squared (s-2)

    Parameters
    ----------
    vt : xr.DataArray
        Azimuthal-mean tangential wind with ``r`` coordinate (m).
    f : float
        Coriolis parameter (s⁻¹).

    Returns
    -------
    xr.DataArray
        Inertial stability I² (s⁻²).
    """
    r = _require_r(vt, "vt")
    safe_r = xr.where(r != 0.0, r, np.nan)

    # zeta = (1/r) * d(r * vt)/dr (azimuthal-mean vertical vorticity)
    # Keep vt dimension order (e.g., time, zc, r) through intermediate fields.
    rvt = vt * r
    drvt_dr = rvt.differentiate("r")
    zeta = drvt_dr / safe_r

    i2 = (f + 2.0 * vt / safe_r) * (f + zeta)
    i2.attrs.update({
        "long_name": "inertial stability",
        "units": "s-2",
    })
    return i2.rename("i2")


def mass_streamfunction(
    vr: xr.DataArray,
    rho: xr.DataArray,
) -> xr.DataArray:
    """
    Mass streamfunction (bottom-up integration).

    Formula (plain text):
        psi(r, z) = -2 * pi * integral_0^z [rho(z') * vr(r, z') * r] dz'

    where:
        r   = radius (m)
        vr  = azimuthal-mean radial wind (m s-1)
        rho = background density (kg m-3)
        psi = mass streamfunction (kg s-1)

    Parameters
    ----------
    vr : xr.DataArray
        Azimuthal-mean radial wind with ``r`` coordinate (m).
        Must also have vertical dimension ``zc``.
    rho : xr.DataArray
        Background density profile (kg m⁻³) on the same ``zc`` grid.

    Returns
    -------
    xr.DataArray
        Mass streamfunction (kg s⁻¹).
    """
    r = _require_r(vr, "vr")
    if "zc" not in vr.dims:
        raise ValueError("vr must have vertical dimension 'zc'.")
    if "zc" not in rho.dims:
        raise ValueError("rho must have vertical dimension 'zc'.")
    if not set(rho.dims).issubset(set(vr.dims)):
        raise ValueError(
            f"rho dims {rho.dims} must be a subset of vr dims {vr.dims}."
        )

    try:
        vr, rho = xr.align(vr, rho, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError(
            "vr and rho must share exactly the same 'zc' coordinates."
        ) from exc

    # Keep vr dimension order (e.g., time, zc, r) in the streamfunction output.
    integrand = -2 * np.pi * vr * rho * r
    psi = integrand.cumulative_integrate("zc")

    psi.attrs.update({
        "long_name": "mass streamfunction",
        "units": "kg s-1",
    })
    return psi.rename("psi")
