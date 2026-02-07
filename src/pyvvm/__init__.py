"""
pyvvm - Python VVM Dataset Reader

A package for loading and processing VVM (Vector Vorticity Model) simulation
output with proper C-grid structure and coordinate handling.

Example
-------
>>> import pyvvm
>>> loader = pyvvm.VVMDataLoader('/path/to/case')   # Initialize dataloader
>>> ds = loader.ds   # xarray dataset
>>> ds.vvm.thv       # Virtual potential temperature
"""

# Register xarray accessor (side-effect import)
from .calc import accessor  # noqa: F401

from .cluster import init_client
from .dataloader import VVMDataLoader

__version__ = '0.1.0'

__all__ = [
    'VVMDataLoader',
    'init_client',
]
