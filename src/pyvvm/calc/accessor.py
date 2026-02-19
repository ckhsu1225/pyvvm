"""
xarray accessor for VVM datasets.

This module provides:
- VVMAccessor: Extends xarray.Dataset with VVM-specific calculations and terrain masking
- MaskedDatasetProxy: Proxy object for on-the-fly terrain masking of variables

Usage:
    ds.vvm.thv          # Virtual potential temperature (unmasked)
    ds.vvm.masked.thv   # Virtual potential temperature (terrain masked)
    ds.vvm.mask(ds.u)  # Apply terrain mask to any variable
"""

from __future__ import annotations

import xgcm
import warnings
import xarray as xr
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict

from .dynamics import DynamicsMixin
from .thermodynamics import ThermoMixin

if TYPE_CHECKING:
    from ..tc.accessor import TCAccessor

__all__ = [
    'VVMAccessor',
    'MaskedDatasetProxy',
]


class MaskedDatasetProxy:
    """
    Proxy that applies terrain mask to variables on-the-fly.
    
    This allows transparent access to both raw and computed variables
    with automatic terrain masking applied.
    """
    
    def __init__(self, accessor: 'VVMAccessor') -> None:
        self._accessor = accessor
        self._ds = accessor._ds

    def __getattr__(self, name: str) -> xr.DataArray:
        """
        Retrieve a variable and apply terrain mask automatically.
        
        Resolution order:
        1. Look for variable in the Dataset (e.g., u, v, th)
        2. Look for computed property in Accessor (e.g., thv, hm)
        3. Raise AttributeError if not found
        """
        # Try Dataset first (raw variables)
        if name in self._ds:
            da = self._ds[name]
        
        # Try Accessor computed properties (ThermoMixin etc.)
        elif hasattr(self._accessor, name):
            da = getattr(self._accessor, name)
        
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        # Apply mask if it's a DataArray with spatial dimensions
        if hasattr(da, 'dims'):
            return self._accessor.mask(da)
        
        return da

    def __getitem__(self, key: str) -> xr.DataArray:
        """Support dict-style access: ds.vvm.masked['es']"""
        return getattr(self, key)

    def __dir__(self) -> list[str]:
        """Enable tab completion in Jupyter/IPython."""
        ds_vars = list(self._ds.keys())
        acc_attrs = [a for a in dir(self._accessor) if not a.startswith('_')]
        return sorted(set(ds_vars + acc_attrs))


class ChunkRule(TypedDict):
    """Chunk constraint rule for one diagnostic."""
    axes: tuple[str, ...]
    inputs: tuple[str, ...]
    reason: str


@xr.register_dataset_accessor('vvm')
class VVMAccessor(ThermoMixin, DynamicsMixin):
    """
    xarray accessor for VVM datasets.
    
    Provides:
    - xgcm Grid for C-grid operations
    - Thermodynamic calculations via ThermoMixin
    - Dynamic calculations via DynamicsMixin
    - Chunk validation for diagnostics
    
    Examples
    --------
    >>> ds.vvm.grid          # xgcm Grid object
    >>> ds.vvm.thv           # Virtual potential temperature
    >>> ds.vvm.mask(ds.u)    # Apply terrain mask
    >>> ds.vvm.masked.th     # Masked potential temperature
    """
    
    # C-grid stagger configuration for terrain interpolation
    # Maps horizontal dimension sets to axes for max interpolation
    STAGGER_CONFIG: dict[frozenset[str], tuple[str, ...]] = {
        frozenset({'xb', 'yc'}): ('X',),        # u, eta
        frozenset({'xc', 'yb'}): ('Y',),        # v, xi
        frozenset({'xb', 'yb'}): ('X', 'Y'),    # zeta
    }

    # Dimension aliases by C-grid axis.
    AXIS_DIMS: dict[str, tuple[str, ...]] = {
        'X': ('xc', 'xb'),
        'Y': ('yc', 'yb'),
        'Z': ('zc', 'zb'),
    }

    # Pre-rename aliases (VVMDataset input chunks use lev/lat/lon).
    CHUNK_ALIAS_TO_INPUT: dict[str, str] = {
        'xc': 'lon',
        'xb': 'lon',
        'yc': 'lat',
        'yb': 'lat',
        'zc': 'lev',
        'zb': 'lev',
    }

    # Diagnostic chunk rules. `axes` means those axes must be single-chunk
    # on the listed input variables to avoid dask core-dimension failures.
    CHUNK_RULES: dict[str, ChunkRule] = {
        'n2': {
            'axes': ('Z',),
            'inputs': ('th', 'qv', 'qc', 'qi', 'qr'),
            'reason': 'vertical derivative uses core Z axis',
        },
        'cape_cin': {
            'axes': ('Z',),
            'inputs': ('th', 'qv'),
            'reason': 'apply_ufunc uses zc as core dimension',
        },
        'psi': {
            'axes': ('X', 'Y'),
            'inputs': ('zeta',),
            'reason': 'apply_ufunc uses horizontal core dimensions',
        },
        'pv': {
            'axes': ('Z',),
            'inputs': ('th', 'zeta', 'eta', 'xi'),
            'reason': 'vertical derivative uses core Z axis',
        },
    }

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._ds = xarray_obj
        self._grid: xgcm.Grid | None = None
        self._tc: TCAccessor | None = None

    @property
    def tc(self) -> 'TCAccessor':
        """
        Tropical-cyclone diagnostics namespace.

        Examples
        --------
        >>> ds.vvm.tc.find_center(field='psi', method='centroid')
        """
        if self._tc is None:
            from ..tc.accessor import TCAccessor
            self._tc = TCAccessor(self)
        return self._tc

    def validate_chunks(
        self,
        diagnostics: str | Sequence[str],
        mode: str = 'raise',
    ) -> bool:
        """
        Validate chunk layout for one or more diagnostics.

        Parameters
        ----------
        diagnostics : str or sequence[str]
            Diagnostic name(s), e.g. ``'psi'`` or ``['n2', 'pv']``.
        mode : {'raise', 'warn', 'ignore'}, optional
            - ``'raise'``: raise ValueError on violations (default)
            - ``'warn'``: emit warnings and continue
            - ``'ignore'``: skip violations

        Returns
        -------
        bool
            True when validation completes (including warn/ignore modes).
        """
        names = [diagnostics] if isinstance(diagnostics, str) else list(diagnostics)

        for name in names:
            self._validate_chunks(name, mode=mode)

        return True

    def _validate_chunks(
        self,
        diagnostic: str,
        mode: str = 'raise',
    ) -> None:
        """
        Internal chunk validator used by diagnostics with core-dim constraints.
        """
        if mode not in {'raise', 'warn', 'ignore'}:
            raise ValueError(
                f"Invalid mode='{mode}'. Expected one of: 'raise', 'warn', 'ignore'."
            )

        rule = self.CHUNK_RULES.get(diagnostic)
        if rule is None:
            valid = ', '.join(sorted(self.CHUNK_RULES))
            raise ValueError(
                f"Unknown diagnostic '{diagnostic}'. Available: {valid}"
            )

        required_axes = rule['axes']
        inputs = rule['inputs']
        reason = rule['reason']

        violations: list[tuple[str, str, tuple[int, ...]]] = []

        for var_name in inputs:
            if var_name not in self._ds:
                continue

            da = self._ds[var_name]
            if not hasattr(da.data, 'chunks'):
                continue

            chunksizes = dict(da.chunksizes)
            for axis in required_axes:
                axis_dims = self.AXIS_DIMS[axis]
                for dim in da.dims:
                    if dim not in axis_dims:
                        continue
                    chunks = tuple(chunksizes.get(dim, ()))
                    if len(chunks) != 1:
                        violations.append((var_name, dim, chunks))

        if not violations or mode == 'ignore':
            return

        # Build suggestions on post-rename dimensions (xc/yc/zc) and
        # on VVMDataset input dimensions (lon/lat/lev).
        suggested_runtime: dict[str, int] = {}
        for axis in required_axes:
            for dim in self.AXIS_DIMS[axis]:
                if dim in self._ds.dims:
                    suggested_runtime[dim] = -1

        suggested_input: dict[str, int] = {}
        for dim in suggested_runtime:
            dim_input = self.CHUNK_ALIAS_TO_INPUT.get(dim, dim)
            suggested_input[dim_input] = -1

        detail = ', '.join(
            f"{name}.{dim}={chunks}" for name, dim, chunks in violations
        )
        message = (
            f"Chunk validation failed for diagnostic '{diagnostic}' ({reason}). "
            f"Violations: {detail}. "
            f"Suggested runtime chunks: {suggested_runtime}. "
            f"Suggested VVMDataset chunks: {suggested_input}."
        )

        if mode == 'warn':
            warnings.warn(message, stacklevel=3)
            return

        raise ValueError(message)

    @property
    def grid(self) -> xgcm.Grid:
        """Lazily initialized xgcm Grid for C-grid operations."""
        if self._grid is None:
            dims = set(self._ds.dims)

            # Build only axes that are present in the current Dataset.
            # This keeps diagnostics working after scalar selections
            # like isel(zc=0), which drops the zc dimension.
            coords: dict[str, dict[str, str]] = {}
            boundary: dict[str, str] = {}
            metrics: dict[tuple[str, ...], list[str]] = {}

            if {'xc', 'xb'} <= dims:
                coords['X'] = {'center': 'xc', 'right': 'xb'}
                boundary['X'] = 'periodic'
                if 'dx' in self._ds:
                    metrics[('X',)] = ['dx']

            if {'yc', 'yb'} <= dims:
                coords['Y'] = {'center': 'yc', 'right': 'yb'}
                boundary['Y'] = 'periodic'
                if 'dy' in self._ds:
                    metrics[('Y',)] = ['dy']

            if {'zc', 'zb'} <= dims:
                coords['Z'] = {'center': 'zc', 'outer': 'zb'}
                boundary['Z'] = 'extend'
                if 'dz' in self._ds:
                    metrics[('Z',)] = ['dz']

            self._grid = xgcm.Grid(
                self._ds,
                coords=coords,
                boundary=boundary,
                metrics=metrics,
                autoparse_metadata=False,
            )
        return self._grid

    def get_mask(self, var: xr.DataArray) -> xr.DataArray:
        """
        Generate a 3D terrain mask for the given variable's grid position.
        
        Parameters
        ----------
        var : xr.DataArray
            Variable to generate mask.
            
        Returns
        -------
        xr.DataArray
            Boolean mask where True = above terrain, False = below terrain.
        """
        ds = self._ds

        # Safety check: if no terrain data, skip masking entirely
        if 'topo' not in ds.coords:
            return xr.DataArray(True)  # Scalar True broadcasts to "no mask"

        # Determine spatial dimensions
        spatial_dims = {'xc', 'yc', 'xb', 'yb'}
        dims = frozenset(set(var.dims) & spatial_dims)

        # Interpolate terrain to target grid position
        target_topo = self._calc_target_topo(dims, ds.topo)

        # Determine vertical index coordinate
        if 'kzc' in var.coords:
            return var['kzc'] >= target_topo
        elif 'kzb' in var.coords:
            return var['kzb'] >= target_topo
        else:
            # 2D variable or no vertical dimension
            return xr.DataArray(True)

    def _calc_target_topo(
        self,
        dims: frozenset[str],
        topo: xr.DataArray,
    ) -> xr.DataArray:
        """
        Interpolate terrain to staggered grid positions using max.
        
        For staggered grids, takes the max of neighboring cells to ensure
        conservative masking (no values below any adjacent terrain).
        """
        # Extract horizontal dimensions for lookup
        horiz_dims = frozenset({'xb', 'yb', 'xc', 'yc'} & dims)
        axes = self.STAGGER_CONFIG.get(horiz_dims, ())
        
        # Apply max interpolation for each stagger axis
        result = topo
        for axis in axes:
            result = self.grid.max(result, axis)
        
        return result

    def mask(self, da: xr.DataArray) -> xr.DataArray:
        """
        Apply terrain mask to a DataArray.
        
        Parameters
        ----------
        da : xr.DataArray
            Data array to mask.
            
        Returns
        -------
        xr.DataArray
            Masked data array with NaN below terrain.
            
        Examples
        --------
        >>> u_masked = ds.vvm.mask(ds.u)
        """
        mask = self.get_mask(da)
        return da.where(mask)

    @property
    def masked(self) -> MaskedDatasetProxy:
        """
        Return a proxy object for transparent terrain-masked access.
        
        Examples
        --------
        >>> ds.vvm.masked.th      # Masked potential temperature
        >>> ds.vvm.masked.thv     # Masked virtual potential temperature
        >>> ds.vvm.masked['qv']   # Dict-style access also works
        """
        return MaskedDatasetProxy(self)
