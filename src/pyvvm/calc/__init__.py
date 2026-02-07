"""
Calculation modules for VVM data processing.

This package provides thermodynamic and dynamics calculations for VVM datasets.
The main interface is through the xarray accessor (ds.vvm), which is automatically
registered when importing pyvvm.

Submodules
----------
constants : Physical constants used in calculations
formulas : Pure computational functions (can be used independently)
"""

from . import constants
from . import formulas

__all__ = [
    'constants',
    'formulas',
]
