"""Vector rotation on a sampled cylindrical ``(theta, r)`` grid.

These functions operate after Cartesian-to-cylindrical remapping.  Both
Cartesian components must already have been sampled at the same cylindrical
target points; no source-grid interpolation or TC-track lookup is performed
here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr


__all__ = [
    "rotate_vector",
    "rotate_wind",
    "rotate_vorticity",
]


EtaConvention = Literal["vvm", "physical_y"]


def _validate_cylindrical_component(
    component: xr.DataArray,
    argument_name: str,
) -> None:
    """Validate the part of the cylindrical DataArray contract used here."""
    if not isinstance(component, xr.DataArray):
        raise TypeError(
            f"{argument_name} must be an xr.DataArray, "
            f"got {type(component).__name__}."
        )

    dtype = np.dtype(component.dtype)
    if not (
        np.issubdtype(dtype, np.integer)
        or np.issubdtype(dtype, np.floating)
    ):
        raise TypeError(
            f"{argument_name} must have a real numeric dtype, got {dtype}."
        )

    for dim in ("theta", "r"):
        if dim not in component.dims:
            raise ValueError(
                f"{argument_name} must include cylindrical dimension {dim!r}."
            )
        if dim not in component.coords:
            raise ValueError(
                f"{argument_name} must have a coordinate for dimension {dim!r}."
            )

        coord = component.coords[dim]
        if coord.dims != (dim,):
            raise ValueError(
                f"{argument_name} coordinate {dim!r} must be one-dimensional "
                f"on ({dim!r},), got dims {coord.dims}."
            )

        values = np.asarray(coord.values)
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(
                f"{argument_name} coordinate {dim!r} must be numeric, "
                f"got {values.dtype}."
            )
        values = values.astype(np.float64, copy=False)
        if values.size < 1 or not np.isfinite(values).all():
            raise ValueError(
                f"{argument_name} coordinate {dim!r} must contain at least "
                "one finite value."
            )
        if values.size > 1 and not (np.diff(values) > 0.0).all():
            raise ValueError(
                f"{argument_name} coordinate {dim!r} must be strictly increasing."
            )

        if dim == "r" and (values < 0.0).any():
            raise ValueError(
                f"{argument_name} coordinate 'r' must be non-negative."
            )
        if (
            dim == "theta"
            and values.size > 1
            and values[-1] - values[0] >= 2.0 * np.pi
        ):
            raise ValueError(
                f"{argument_name} coordinate 'theta' must span less than 2*pi "
                "and be expressed in radians."
            )


def _align_components(
    x_component: xr.DataArray,
    y_component: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Validate and exactly align two components without silent broadcasting."""
    _validate_cylindrical_component(x_component, "x_component")
    _validate_cylindrical_component(y_component, "y_component")

    if set(x_component.dims) != set(y_component.dims):
        raise ValueError(
            "x_component and y_component must have the same dimensions; "
            f"got {x_component.dims} and {y_component.dims}."
        )

    y_component = y_component.transpose(*x_component.dims)
    try:
        x_component, y_component = xr.align(
            x_component,
            y_component,
            join="exact",
            copy=False,
        )
    except ValueError as exc:
        raise ValueError(
            "x_component and y_component must have exactly matching coordinates."
        ) from exc

    for coord_name in ("center_x", "center_y"):
        x_coord = x_component.coords.get(coord_name)
        y_coord = y_component.coords.get(coord_name)
        if (x_coord is None) != (y_coord is None):
            raise ValueError(
                f"x_component and y_component must either both define "
                f"{coord_name!r} or both omit it."
            )
        if x_coord is not None and not x_coord.equals(y_coord):
            raise ValueError(
                f"x_component and y_component coordinate {coord_name!r} "
                "must match exactly."
            )

    x_units = x_component.attrs.get("units")
    y_units = y_component.attrs.get("units")
    if x_units and y_units and x_units != y_units:
        raise ValueError(
            "x_component and y_component must use the same units; "
            f"got {x_units!r} and {y_units!r}."
        )

    return x_component, y_component


def _set_component_metadata(
    out: xr.Dataset,
    *,
    radial_name: str,
    tangential_name: str,
    units: str | None,
    radial_long_name: str,
    tangential_long_name: str,
) -> xr.Dataset:
    """Attach common cylindrical-vector metadata."""
    radial_attrs = {
        "long_name": radial_long_name,
        "positive": "outward",
    }
    tangential_attrs = {
        "long_name": tangential_long_name,
        "positive": "counter-clockwise",
    }
    if units:
        radial_attrs["units"] = units
        tangential_attrs["units"] = units

    out[radial_name].attrs = radial_attrs
    out[tangential_name].attrs = tangential_attrs
    out.attrs["vector_rotation"] = "cartesian_to_cylindrical_at_target"
    return out


def rotate_vector(
    x_component: xr.DataArray,
    y_component: xr.DataArray,
    *,
    radial_name: str = "radial",
    tangential_name: str = "tangential",
    mask_origin: bool = True,
) -> xr.Dataset:
    """Rotate Cartesian vector components on a cylindrical target grid.

    Parameters
    ----------
    x_component, y_component : xr.DataArray
        Physical Cartesian x and y components already sampled at exactly the
        same cylindrical target points.  Both arrays must contain ``theta`` and
        ``r`` dimensions with matching coordinates.  ``theta`` is in radians,
        measured counter-clockwise from the positive x-axis.
    radial_name, tangential_name : str, optional
        Names for the returned radial and tangential components.
    mask_origin : bool, optional
        Set both components to NaN at ``r == 0``, where the cylindrical basis
        is undefined.  Enabled by default.

    Returns
    -------
    xr.Dataset
        Radial (positive outward) and tangential (positive
        counter-clockwise) components.  Dask-backed inputs remain lazy.

    Notes
    -----
    The rotation is applied after remapping:

    ``radial = x*cos(theta) + y*sin(theta)``

    ``tangential = -x*sin(theta) + y*cos(theta)``
    """
    if not isinstance(radial_name, str) or not radial_name:
        raise ValueError("radial_name must be a non-empty string.")
    if not isinstance(tangential_name, str) or not tangential_name:
        raise ValueError("tangential_name must be a non-empty string.")
    if radial_name == tangential_name:
        raise ValueError("radial_name and tangential_name must be different.")
    if not isinstance(mask_origin, (bool, np.bool_)):
        raise TypeError("mask_origin must be a boolean.")

    x_component, y_component = _align_components(x_component, y_component)
    out_dtype = np.result_type(
        x_component.dtype,
        y_component.dtype,
        np.float32,
    )
    x_work = x_component.astype(out_dtype)
    y_work = y_component.astype(out_dtype)
    # Evaluate trigonometric functions at the coordinate's native precision,
    # then cast the small basis arrays to the vector result dtype.  In
    # particular, this avoids turning float64 values such as pi/2 into a
    # lower-precision angle before taking the cosine.
    theta = x_work.coords["theta"]
    cos_theta = np.cos(theta).astype(out_dtype)
    sin_theta = np.sin(theta).astype(out_dtype)

    radial = x_work * cos_theta + y_work * sin_theta
    tangential = -x_work * sin_theta + y_work * cos_theta

    if mask_origin:
        away_from_origin = x_work.coords["r"] > 0.0
        radial = radial.where(away_from_origin)
        tangential = tangential.where(away_from_origin)

    radial = radial.rename(radial_name).transpose(*x_component.dims)
    tangential = tangential.rename(tangential_name).transpose(*x_component.dims)
    units = x_component.attrs.get("units") or y_component.attrs.get("units")

    return _set_component_metadata(
        xr.Dataset(
            {
                radial_name: radial,
                tangential_name: tangential,
            }
        ),
        radial_name=radial_name,
        tangential_name=tangential_name,
        units=units,
        radial_long_name="radial vector component",
        tangential_long_name="tangential vector component",
    )


def rotate_wind(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    mask_origin: bool = True,
) -> xr.Dataset:
    """Rotate remapped Cartesian wind into radial and tangential wind.

    ``u`` and ``v`` must first be remapped independently from their native VVM
    C-grid locations to the same ``(theta, r)`` target grid.
    """
    out = rotate_vector(
        u,
        v,
        radial_name="radial_wind",
        tangential_name="tangential_wind",
        mask_origin=mask_origin,
    )
    units = out["radial_wind"].attrs.get("units", "m s-1")
    return _set_component_metadata(
        out,
        radial_name="radial_wind",
        tangential_name="tangential_wind",
        units=units,
        radial_long_name="radial wind",
        tangential_long_name="tangential wind",
    )


def rotate_vorticity(
    xi: xr.DataArray,
    eta: xr.DataArray,
    *,
    eta_convention: EtaConvention = "vvm",
    mask_origin: bool = True,
) -> xr.Dataset:
    """Rotate remapped horizontal vorticity into cylindrical components.

    Parameters
    ----------
    xi, eta : xr.DataArray
        Remapped VVM horizontal-vorticity variables on the same ``(theta, r)``
        grid.  With the default ``eta_convention="vvm"``, ``xi`` is treated as
        the physical x component and ``-eta`` as the physical y component.
    eta_convention : {"vvm", "physical_y"}, optional
        Set to ``"physical_y"`` only when the supplied ``eta`` array already
        represents the physical positive-y component.
    mask_origin : bool, optional
        Set both rotated components to NaN at ``r == 0``.
    """
    if eta_convention not in ("vvm", "physical_y"):
        raise ValueError(
            "eta_convention must be 'vvm' or 'physical_y', "
            f"got {eta_convention!r}."
        )
    if not isinstance(eta, xr.DataArray):
        raise TypeError(
            f"eta must be an xr.DataArray, got {type(eta).__name__}."
        )

    physical_y = -eta if eta_convention == "vvm" else eta
    physical_y.attrs = dict(eta.attrs)
    out = rotate_vector(
        xi,
        physical_y,
        radial_name="radial_vorticity",
        tangential_name="tangential_vorticity",
        mask_origin=mask_origin,
    )
    units = out["radial_vorticity"].attrs.get("units", "s-1")
    out = _set_component_metadata(
        out,
        radial_name="radial_vorticity",
        tangential_name="tangential_vorticity",
        units=units,
        radial_long_name="radial horizontal vorticity",
        tangential_long_name="tangential horizontal vorticity",
    )
    out.attrs["eta_convention"] = eta_convention
    return out
