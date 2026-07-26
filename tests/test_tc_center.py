"""Tests for the supported TC center-finding methods."""

from __future__ import annotations

import inspect
import unittest

import numpy as np
import xarray as xr

from pyvvm.tc.accessor import TCAccessor
from pyvvm.tc.center import _get_track, find_tc_center


def _peaked_field() -> xr.DataArray:
    values = np.zeros((2, 5, 5), dtype=np.float64)
    values[0, 1, 2] = 10.0
    values[1, 3, 0] = 20.0
    return xr.DataArray(
        values,
        dims=("time", "yc", "xc"),
        coords={
            "time": [0, 1],
            "yc": 200.0 + np.arange(5) * 20.0,
            "xc": 100.0 + np.arange(5) * 10.0,
            "dx": 10.0,
            "dy": 20.0,
        },
        name="prepared_center_field",
    )


class CenterMethodTests(unittest.TestCase):
    def test_extremum_track_is_unchanged(self) -> None:
        actual = _get_track(_peaked_field(), method="extremum")

        np.testing.assert_allclose(actual["x"], [120.0, 100.0])
        np.testing.assert_allclose(actual["y"], [220.0, 260.0])
        self.assertEqual(set(actual.data_vars), {"x", "y"})
        self.assertEqual(actual.attrs["method"], "extremum")

    def test_centroid_track_is_unchanged_for_isolated_peaks(self) -> None:
        actual = _get_track(
            _peaked_field(),
            method="centroid",
            radius=30.0,
        )

        np.testing.assert_allclose(actual["x"], [120.0, 100.0])
        np.testing.assert_allclose(actual["y"], [220.0, 260.0])
        self.assertEqual(set(actual.data_vars), {"x", "y"})
        self.assertEqual(actual.attrs["method"], "centroid")

    def test_adaptive_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "field must be 'zeta' or 'psi'",
        ):
            find_tc_center(xr.Dataset(), field="adaptive")

    def test_distance_threshold_is_no_longer_public(self) -> None:
        self.assertNotIn(
            "distance_threshold",
            inspect.signature(find_tc_center).parameters,
        )
        self.assertNotIn(
            "distance_threshold",
            inspect.signature(TCAccessor.find_center).parameters,
        )


if __name__ == "__main__":
    unittest.main()
