"""
Constants for VVM calculations.

This module contains physical constants used in thermodynamic and dynamic calculations.
"""

# ============================================================================
# Fundamental Physical Constants
# ============================================================================

# Gas constants
R_d = 287.05          # Specific gas constant for dry air [J kg^-1 K^-1]
R_v = 461.5           # Specific gas constant for water vapor [J kg^-1 K^-1]
epsilon = R_d / R_v   # Ratio of gas constants ≈ 0.622 [dimensionless]

# Specific heats (at constant pressure)
Cp_d = 1004.7         # Specific heat of dry air [J kg^-1 K^-1]
Cp_v = 1860.1         # Specific heat of water vapor [J kg^-1 K^-1]
Cp_l = 4219.4         # Specific heat of liquid water [J kg^-1 K^-1]
Cp_i = 2090.0         # Specific heat of ice [J kg^-1 K^-1]

# Specific heats (at constant volume)
Cv_d = 717.6          # Specific heat of dry air at constant volume [J kg^-1 K^-1]

# Ratio of specific heats
kappa = R_d / Cp_d    # ≈ 0.286 [dimensionless]

# Latent heats (at 0°C)
Lv = 2.501e6          # Latent heat of vaporization [J kg^-1]
Lf = 3.337e5          # Latent heat of fusion [J kg^-1]
Ls = Lv + Lf          # Latent heat of sublimation [J kg^-1]

# Density constants (at reference conditions)
rho_liquid = 999.97   # Density of liquid water [kg m^-3]
rho_ice = 917.0       # Density of ice [kg m^-3]

# ============================================================================
# Earth Constants
# ============================================================================

g = 9.80665           # Gravitational acceleration [m s^-2]
omega = 7.292e-5      # Earth's angular velocity [rad s^-1]
a_earth = 6.371e6     # Earth's mean radius [m]
