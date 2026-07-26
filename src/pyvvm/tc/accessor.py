"""Thin xarray convenience layer for TC tracking and cylindrical remapping."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import xarray as xr

from .center import find_tc_center
from .cylindrical import (
    CylindricalGridSpec,
    remap_dataarray,
    remap_dataset as _remap_dataset,
)

if TYPE_CHECKING:
    from ..calc.accessor import VVMAccessor


logger = logging.getLogger(__name__)

__all__ = [
    "TCAccessor",
    "TCMaskedProxy",
]


class TCMaskedProxy:
    """Terrain-masked cylindrical-remapping convenience proxy."""

    def __init__(self, tc_accessor: TCAccessor):
        self._tc = tc_accessor

    def remap(
        self,
        var_name: str | xr.DataArray,
        *,
        spec: CylindricalGridSpec,
        x_dim: str | None = None,
        y_dim: str | None = None,
    ) -> xr.DataArray:
        """Remap a terrain-masked field to ``(theta, r)``."""
        return self._tc.remap(
            var_name,
            spec=spec,
            masked=True,
            x_dim=x_dim,
            y_dim=y_dim,
        )


class TCAccessor:
    """TC tracking and cylindrical remapping attached to ``ds.vvm.tc``."""

    def __init__(self, parent: VVMAccessor) -> None:
        self._parent = parent
        self._ds: xr.Dataset = parent._ds
        self._track: xr.Dataset | None = None
        self._center_source: xr.Dataset | None = None
        self._masked_proxy = TCMaskedProxy(self)

    @property
    def masked(self) -> TCMaskedProxy:
        """Access terrain-masked cylindrical remapping."""
        return self._masked_proxy

    @property
    def track(self) -> xr.Dataset:
        """Current TC track.  Call ``find_center()`` first."""
        if self._track is None:
            raise ValueError("No track found. Please run `find_center()` first.")
        return self._track

    def _resolve_data(
        self,
        var_name: str | xr.DataArray,
        *,
        masked: bool = False,
    ) -> xr.DataArray:
        """Resolve a raw, computed, or custom field for remapping."""
        if isinstance(var_name, xr.DataArray):
            return self._parent.mask(var_name) if masked else var_name
        if not isinstance(var_name, str):
            raise TypeError(
                "var_name must be a variable name or xr.DataArray, "
                f"got {type(var_name).__name__}."
            )

        source = self._parent.masked if masked else self._parent

        try:
            da = getattr(source, var_name)
            if isinstance(da, xr.DataArray):
                return da
        except AttributeError:
            pass

        if var_name in self._ds:
            da = self._ds[var_name]
            return self._parent.mask(da) if masked else da

        raise ValueError(
            f"Variable {var_name!r} not found in dataset or parent accessor."
        )

    @staticmethod
    def _parse_int_list_attr(value: str | None) -> list[int]:
        """Parse a comma-separated attribute into integer values."""
        if not value:
            return []
        return [int(item) for item in value.split(",") if item.strip()]

    def _ensure_center_source(self) -> xr.Dataset:
        """Prepare a full-horizontal-chunk Dataset for center finding."""
        if self._center_source is not None:
            return self._center_source

        ds = self._ds
        zeta = ds.get("zeta")
        if zeta is None or zeta.chunks is None:
            return ds

        xb_full = zeta.chunksizes["xb"] == (ds.sizes["xb"],)
        yb_full = zeta.chunksizes["yb"] == (ds.sizes["yb"],)
        if xb_full and yb_full:
            return ds

        case_path = ds.attrs.get("vvm_case_path")
        if not case_path:
            logger.warning(
                "ds.attrs['vvm_case_path'] not found; using the primary "
                "Dataset for center finding despite non-full horizontal chunks."
            )
            return ds

        from ..dataloader import VVMDataLoader

        steps = self._parse_int_list_attr(ds.attrs.get("vvm_selected_steps"))
        kwargs: dict = {"case_path": case_path, "groups": ["L.Dynamic"]}
        if steps:
            kwargs["steps"] = steps

        loader = VVMDataLoader(
            **kwargs,
            chunks={"time": 1, "lev": 1, "lat": -1, "lon": -1},
        )
        self._center_source = loader.ds
        return self._center_source

    def find_center(
        self,
        field: str = "psi",
        method: str = "extremum",
        level: float | tuple[float, float] = 1000.0,
        sigma: float = 50e3,
        radius: float = 100e3,
    ) -> xr.Dataset:
        """Find, eagerly evaluate, and cache the TC center track."""
        track = find_tc_center(
            self._ensure_center_source(),
            field=field,
            method=method,
            level=level,
            sigma=sigma,
            radius=radius,
        )
        self._track = track.compute()
        return self._track

    def remap(
        self,
        var_name: str | xr.DataArray,
        *,
        spec: CylindricalGridSpec,
        masked: bool = False,
        x_dim: str | None = None,
        y_dim: str | None = None,
    ) -> xr.DataArray:
        """Remap a raw, computed, or custom field to ``(theta, r)``.

        The cached TC track supplies the cylindrical center.  Staggered
        horizontal dimensions are sampled directly from their native source
        coordinates.
        """
        da = self._resolve_data(var_name, masked=masked)
        return remap_dataarray(
            da,
            self.track,
            spec=spec,
            x_dim=x_dim,
            y_dim=y_dim,
        )

    def remap_dataset(
        self,
        *,
        spec: CylindricalGridSpec,
        variables: str | Sequence[str] | None = None,
    ) -> xr.Dataset:
        """Remap selected or all horizontally gridded raw variables."""
        return _remap_dataset(
            self._ds,
            self.track,
            spec=spec,
            variables=variables,
        )
