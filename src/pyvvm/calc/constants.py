"""
Constants for VVM calculations.

This module contains physical constants used in thermodynamic and dynamic calculations.
"""

# ============================================================================
# Fundamental Physical Constants
# ============================================================================

# Gas constants
Rd = 287.0            # Specific gas constant for dry air [J kg^-1 K^-1]
Rv = 461.5            # Specific gas constant for water vapor [J kg^-1 K^-1]
epsilon = Rd / Rv     # Ratio of gas constants ≈ 0.622 [dimensionless]

# Specific heats (at constant pressure)
Cpd = 1004.0          # Specific heat of dry air [J kg^-1 K^-1]
Cpv = 2040.0          # Specific heat of water vapor [J kg^-1 K^-1]
Cpl = 4220.0          # Specific heat of liquid water [J kg^-1 K^-1]
Cpi = 2097.0          # Specific heat of ice [J kg^-1 K^-1]

# Specific heats (at constant volume)
Cvd = 717.0           # Specific heat of dry air at constant volume [J kg^-1 K^-1]

# Ratio of specific heats
kappa = Rd / Cpd      # ≈ 0.286 [dimensionless]

# Latent heats (at 0°C)
Lv0 = 2.501e6         # Latent heat of vaporization [J kg^-1]
Lf0 = 3.337e5         # Latent heat of fusion [J kg^-1]
Ls0 = Lv0 + Lf0       # Latent heat of sublimation [J kg^-1]

# Reference conditions
T0 = 273.16         # Triple point temperature of water [K]
p0 = 1e5            # Reference pressure [Pa]
es0 = 611.655       # Saturation vapor pressure at T0 [Pa]

# Density constants (at reference conditions)
rho_liquid = 999.97   # Density of liquid water [kg m^-3]
rho_ice = 917.0       # Density of ice [kg m^-3]

# ============================================================================
# Earth Constants
# ============================================================================

g = 9.80665           # Gravitational acceleration [m s^-2]
omega = 7.292e-5      # Earth's angular velocity [rad s^-1]
a_earth = 6.371e6     # Earth's mean radius [m]
