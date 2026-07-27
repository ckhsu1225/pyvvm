"""Tests for the xgcm-backed VVM grid."""

from __future__ import annotations

import unittest

import numpy as np
import xarray as xr

from pyvvm.calc.accessor import VVMAccessor


class VVMGridTests(unittest.TestCase):
    def test_grid_uses_periodic_horizontal_and_extended_vertical_padding(
        self,
    ) -> None:
        ds = xr.Dataset(
            data_vars={
                "x_face": ("xb", np.arange(4.0)),
                "y_face": ("yb", np.arange(3.0) * 2.0),
                "z_center": ("zc", np.array([1.0, 3.0])),
            },
            coords={
                "xc": np.arange(4.0),
                "xb": np.arange(4.0) + 0.5,
                "yc": np.arange(3.0),
                "yb": np.arange(3.0) + 0.5,
                "zc": np.array([0.5, 1.5]),
                "zb": np.array([0.0, 1.0, 2.0]),
                "dx": 1.0,
                "dy": 2.0,
                "dz": ("zc", np.ones(2)),
            },
        )

        accessor = VVMAccessor(ds)
        grid = accessor.grid

        np.testing.assert_allclose(
            grid.interp(ds["x_face"], "X"),
            [1.5, 0.5, 1.5, 2.5],
        )
        np.testing.assert_allclose(
            grid.interp(ds["y_face"], "Y"),
            [2.0, 1.0, 3.0],
        )
        np.testing.assert_allclose(
            grid.interp(ds["z_center"], "Z"),
            [1.0, 2.0, 3.0],
        )
        self.assertIs(grid, accessor.grid)


if __name__ == "__main__":
    unittest.main()
