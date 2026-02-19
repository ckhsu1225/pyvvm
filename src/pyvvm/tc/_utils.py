"""
Shared utilities for the TC analysis package.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def wrap_min(d: np.ndarray, L: float) -> np.ndarray:
    """Periodic shortest distance in [-L/2, L/2)."""
    return (d + 0.5 * L) % L - 0.5 * L


def resolve_track(
    track: xr.Dataset,
    da: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize track into flat numpy arrays aligned with *da*'s time.

    Returns
    -------
    cx_all : np.ndarray, shape (nt,)
        Center x-coordinates in metres.
    cy_all : np.ndarray, shape (nt,)
        Center y-coordinates in metres.
    """
    if "time" in track.dims:
        aligned = track[["x", "y"]].sel(time=da["time"])
        cx_all = np.asarray(aligned["x"].values, dtype=np.float64)
        cy_all = np.asarray(aligned["y"].values, dtype=np.float64)
    else:
        nt = da.sizes.get("time", 1)
        cx_all = np.full(nt, float(track["x"].squeeze().values), dtype=np.float64)
        cy_all = np.full(nt, float(track["y"].squeeze().values), dtype=np.float64)

    return cx_all, cy_all
