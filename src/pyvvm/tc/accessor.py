"""
TC accessor for VVM datasets.

This module exposes tropical-cyclone specific workflows from ``ds.vvm.tc``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
import numpy as np
import xarray as xr
from typing import TYPE_CHECKING

from .axisym import axisym_mean
from .center import find_tc_center
from .wind import compute_vr_vt
from .metrics import wind_metrics_from_profile
from .._utils import assign_compatible_coords
from .diag import angular_momentum, inertial_stability, mass_streamfunction

if TYPE_CHECKING:
    from ..calc.accessor import VVMAccessor


logger = logging.getLogger(__name__)

__all__ = [
    'TCAccessor',
    'TCMaskedProxy',
]

_TC_WIND_PROPERTIES = {'vr', 'vt', 'wind'}
_TC_MASKED_PROPERTIES = _TC_WIND_PROPERTIES | {'aam', 'i2', 'psi'}


class TCMaskedProxy:
    """
    Proxy that delegates attribute access to TCAccessor with masking enabled.

    Supports computed TC properties (``vr``, ``vt``, ``wind``, ``aam``,
    ``i2``, ``psi``) and the ``azimuth`` method, applying terrain masking
    to the underlying data automatically.

    Examples
    --------
    >>> ds.vvm.tc.masked.vt
    >>> ds.vvm.tc.masked.azimuth('th')
    """

    def __init__(self, tc_accessor: TCAccessor):
        self._tc = tc_accessor

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError(name)
        if name in _TC_MASKED_PROPERTIES:
            return self._tc._compute_property(name, masked=True)
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )

    def azimuth(
        self,
        var_name: str | xr.DataArray,
        radius: float = None,
        dr: float = None,
    ):
        """Azimuthal mean with terrain masking applied."""
        return self._tc.azimuth(var_name, radius=radius, dr=dr, masked=True)



class TCAccessor:
    """
    Tropical-cyclone diagnostics accessor attached to ``ds.vvm.tc``.
    """

    def __init__(self, parent: 'VVMAccessor') -> None:
        self._parent = parent       # VVMAccessor
        self._ds: xr.Dataset = parent._ds
        self._track: xr.Dataset | None = None
        self._center_source: xr.Dataset | None = None
        self._masked_proxy = TCMaskedProxy(self)

        # Default spatial parameters (metres).
        self.default_radius = 300e3
        self.default_dr = 2e3

    def set_params(self, radius: float = None, dr: float = None):
        """Modify default spatial parameters for TC diagnostics (metres)."""
        new_radius = self.default_radius if radius is None else float(radius)
        new_dr = self.default_dr if dr is None else float(dr)

        if radius is not None:
            if not np.isfinite(new_radius) or new_radius <= 0.0:
                raise ValueError(
                    f"radius must be a finite positive value (m), got {radius!r}."
                )
        if dr is not None:
            if not np.isfinite(new_dr) or new_dr <= 0.0:
                raise ValueError(
                    f"dr must be a finite positive value (m), got {dr!r}."
                )
        if (radius is not None or dr is not None) and new_dr > new_radius:
            raise ValueError(
                f"dr must be <= radius (got dr={new_dr}, radius={new_radius})."
            )

        if radius is not None:
            self.default_radius = new_radius
        if dr is not None:
            self.default_dr = new_dr

    @property
    def masked(self) -> TCMaskedProxy:
        """Access masked versions of computed properties via ``ds.vvm.tc.masked``."""
        return self._masked_proxy

    @property
    def track(self) -> xr.Dataset:
        """Current TC track.  Call ``.find_center()`` first."""
        if self._track is None:
            raise ValueError("No track found. Please run `.find_center()` first.")
        return self._track

    # =========================================================================
    # Data resolution helpers
    # =========================================================================

    def _resolve_data(
        self,
        var_name: str | xr.DataArray,
        masked: bool = False,
    ) -> xr.DataArray:
        """
        Resolve a variable name to a DataArray.

        Resolution order:
        1. If already a DataArray, return as-is.
        2. TC-specific computed properties (vr, vt).
        3. Parent accessor computed properties (thv, pv, …).
        4. Raw dataset variables (u, v, th, …).
        """
        if isinstance(var_name, xr.DataArray):
            return var_name

        # TC-specific computed properties.
        if var_name in _TC_WIND_PROPERTIES:
            return self._compute_property(var_name, masked=masked)

        source = self._parent.masked if masked else self._parent

        # Parent accessor computed properties.
        try:
            da = getattr(source, var_name)
            if isinstance(da, xr.DataArray):
                return da
        except AttributeError:
            pass

        # Raw dataset variables.
        if var_name in self._ds:
            da = self._ds[var_name]
            return self._parent.mask(da) if masked else da

        raise ValueError(
            f"Variable '{var_name}' not found in dataset or accessors."
        )

    def _align_to_center(self, da: xr.DataArray) -> xr.DataArray:
        """Interpolate staggered-grid variable to cell centers (xc, yc)."""
        grid = self._parent.grid
        if 'xb' in da.dims:
            da = grid.interp(da, axis='X')
        if 'yb' in da.dims:
            da = grid.interp(da, axis='Y')
        da = assign_compatible_coords(da, self._ds)
        return da

    def _get_coriolis(self) -> float:
        """Resolve Coriolis parameter from dataset attrs."""
        f = self._ds.attrs.get('coriolis_parameter')
        if f is None:
            logger.info(
                "Coriolis parameter not available. Using f=0 (non-rotating)."
            )
            return 0.0
        return float(f)

    @staticmethod
    def _canonical_dims(
        da: xr.DataArray,
        lead: tuple[str, ...] = ('time', 'zc', 'r'),
    ) -> xr.DataArray:
        """Reorder dims: *lead* dims first (if present), then the rest."""
        head = [d for d in lead if d in da.dims]
        tail = [d for d in da.dims if d not in head]
        target = tuple(head + tail)
        return da if da.dims == target else da.transpose(*target)


    # =========================================================================
    # Center finding
    # =========================================================================

    @staticmethod
    def _parse_int_list_attr(value: str | None) -> list[int]:
        """Parse comma-separated attr string into int list."""
        if not value:
            return []
        return [int(v) for v in value.split(',') if v.strip()]

    def _ensure_center_source(self) -> xr.Dataset:
        """
        Auto-prepare a full-chunk dataset for center finding if needed.

        Center finding uses FFT-based operations that require full horizontal
        chunks.  If the primary dataset's zeta (on xb, yb) is not fully
        chunked in the horizontal, a lightweight dataset is loaded
        automatically with only L.Dynamic group.
        """
        if self._center_source is not None:
            return self._center_source

        ds = self._ds

        # Check if zeta exists and is dask-backed.
        zeta = ds.get('zeta')
        if zeta is None or zeta.chunks is None:
            return ds  # numpy-backed or no zeta → use as-is

        # zeta lives on (xb, yb); FFT needs full horizontal chunks.
        xb_full = zeta.chunksizes['xb'] == (ds.sizes['xb'],)
        yb_full = zeta.chunksizes['yb'] == (ds.sizes['yb'],)
        if xb_full and yb_full:
            return ds  # already full horizontal chunks

        # Reload with full horizontal chunks.
        case_path = ds.attrs.get('vvm_case_path')
        if not case_path:
            logger.warning(
                "ds.attrs['vvm_case_path'] not found; "
                "using primary dataset for center finding despite non-full chunks."
            )
            return ds

        from ..dataloader import VVMDataLoader

        steps = self._parse_int_list_attr(ds.attrs.get('vvm_selected_steps'))
        kwargs: dict = {'case_path': case_path, 'groups': ['L.Dynamic']}
        if steps:
            kwargs['steps'] = steps

        loader = VVMDataLoader(
            **kwargs,
            chunks={'time': 1, 'lev': 1, 'lat': -1, 'lon': -1},
        )
        self._center_source = loader.ds
        return self._center_source

    def find_center(
        self,
        field: str = 'psi',
        method: str = 'extremum',
        level: float | tuple[float, float] = 1000.0,
        sigma: float = 50e3,
        radius: float = 100e3,
    ) -> xr.Dataset:
        """Find and cache TC center track."""
        center_ds = self._ensure_center_source()
        track = find_tc_center(
            center_ds,
            field=field,
            method=method,
            level=level,
            sigma=sigma,
            radius=radius,
        )
        # Compute eagerly to avoid repeated scheduling.
        self._track = track.compute()
        return self._track

    # =========================================================================
    # Computed property dispatch
    # =========================================================================

    def _compute_property(self, name: str, masked: bool = False):
        """Dispatch computed TC properties."""
        if name in _TC_WIND_PROPERTIES:
            wind_ds = self._compute_vr_vt(masked=masked)
            if name == 'wind':
                return wind_ds
            return wind_ds[name]
        if name == 'aam':
            vt_bar = self.azimuth('vt', masked=masked)
            return self._canonical_dims(
                angular_momentum(vt_bar, self._get_coriolis())
            )
        if name == 'i2':
            vt_bar = self.azimuth('vt', masked=masked)
            return self._canonical_dims(
                inertial_stability(vt_bar, self._get_coriolis())
            )
        if name == 'psi':
            vr_bar = self.azimuth('vr', masked=masked)
            if 'rho' not in vr_bar.coords:
                raise ValueError(
                    "Azimuthal-mean vr does not carry 'rho' coord. "
                    "Ensure the input dataset has rho as a coordinate."
                )
            rho = vr_bar.coords['rho']
            return self._canonical_dims(
                mass_streamfunction(vr_bar, rho)
            )
        raise ValueError(f"Unknown TC property: '{name}'")

    def _compute_vr_vt(
        self, masked: bool = False,
    ) -> xr.Dataset:
        """Compute radial / tangential wind from u, v."""
        u = self._resolve_data('u', masked=masked)
        v = self._resolve_data('v', masked=masked)
        u = self._align_to_center(u)
        v = self._align_to_center(v)
        return compute_vr_vt(u, v, self.track)

    @property
    def wind(self) -> xr.Dataset:
        """Dataset containing vr and vt (compute once for both)."""
        return self._compute_property('wind')

    @property
    def vr(self) -> xr.DataArray:
        """Radial wind (positive outward)."""
        return self._compute_property('vr')

    @property
    def vt(self) -> xr.DataArray:
        """Tangential wind (positive cyclonic)."""
        return self._compute_property('vt')

    # =========================================================================
    # Azimuthal averaging
    # =========================================================================

    def azimuth(
        self,
        var_name: str | xr.DataArray,
        radius: float = None,
        dr: float = None,
        masked: bool = False,
    ) -> xr.DataArray:
        """
        Compute azimuthal mean of a variable around the TC center.

        Handles staggered grids, periodic boundaries, and moving centers.
        """
        r_max = radius if radius is not None else self.default_radius
        dr_val = dr if dr is not None else self.default_dr

        da = self._resolve_data(var_name, masked=masked)

        return axisym_mean(
            da_in=da,
            track=self.track,
            r_max=r_max,
            dr=dr_val,
        )

    def wind_metrics(
        self,
        wind: xr.DataArray,
        thresholds: Sequence[float] = (17.0, 25.0, 33.0),
        radius: float = None,
        dr: float = None,
    ) -> xr.Dataset:
        """
        Axisymmetric TC wind size/intensity metrics.

        For best performance, persist or compute the wind DataArray
        before calling this method to avoid deep dask graph overhead::

            ws = ds.vvm.ws.sel(zc=1000, method='nearest').persist()
            metrics = ds.vvm.tc.wind_metrics(ws)

        Parameters
        ----------
        wind : xr.DataArray
            Wind field on the model grid (e.g. ``ds.vvm.ws``, ``ds.vvm.tc.vt``).
            Select vertical levels before passing if needed.
        thresholds : sequence of float, optional
            Threshold wind speeds (m s-1) for threshold radii.
        radius : float, optional
            Maximum azimuthal-mean radius (m). Defaults to accessor setting.
        dr : float, optional
            Radial bin width (m). Defaults to accessor setting.

        Returns
        -------
        xr.Dataset
            Dataset containing ``vmax``, ``rmw`` and threshold radii
            (e.g., ``r17``, ``r25``, ``r33``) on dimensions ``time`` and/or ``zc``.
        """
        if not isinstance(wind, xr.DataArray):
            raise TypeError(
                f"wind must be an xr.DataArray, got {type(wind).__name__}. "
                "Use e.g. ds.vvm.ws.persist() or ds.vvm.tc.vt."
            )
        threshold_values = tuple(float(v) for v in thresholds)
        wind_label = getattr(wind, 'name', None) or 'custom'

        wind_r = self.azimuth(wind, radius=radius, dr=dr)

        out = wind_metrics_from_profile(wind_r, thresholds=threshold_values)

        for name, da in out.data_vars.items():
            out[name] = self._canonical_dims(da, lead=('time', 'zc'))

        out.attrs.update({
            'wind': wind_label,
            'thresholds': threshold_values,
            'radius': float(radius if radius is not None else self.default_radius),
            'dr': float(dr if dr is not None else self.default_dr),
        })
        return out

    # =========================================================================
    # Axisymmetric diagnostics (properties)
    # =========================================================================

    @property
    def aam(self) -> xr.DataArray:
        """Absolute angular momentum on (time, z, r)."""
        return self._compute_property('aam')

    @property
    def i2(self) -> xr.DataArray:
        """Inertial stability I² on (time, z, r)."""
        return self._compute_property('i2')

    @property
    def psi(self) -> xr.DataArray:
        """Mass streamfunction on (time, z, r)."""
        return self._compute_property('psi')
