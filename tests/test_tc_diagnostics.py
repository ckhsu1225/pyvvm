"""Tests for explicit TC diagnostics on user-prepared radial profiles."""

from __future__ import annotations

import unittest

import dask.array as dask_array
import numpy as np
import xarray as xr

import pyvvm.tc as tc
from pyvvm.tc import (
    angular_momentum,
    inertial_stability,
    mass_streamfunction,
    wind_metrics,
)


class WindMetricsTests(unittest.TestCase):
    def test_metrics_are_computed_from_the_supplied_radial_profile(self) -> None:
        profile = xr.DataArray(
            [0.0, 20.0, 10.0, 0.0],
            dims=("r",),
            coords={"r": [0.0, 10.0, 20.0, 30.0]},
            name="quadrant_mean_wind",
            attrs={"units": "m s-1"},
        )

        actual = wind_metrics(profile, thresholds=(15.0, 5.0))

        self.assertEqual(set(actual.data_vars), {"vmax", "rmw", "r15", "r5"})
        self.assertAlmostEqual(float(actual["vmax"]), 20.0)
        self.assertAlmostEqual(float(actual["rmw"]), 10.0)
        self.assertAlmostEqual(float(actual["r15"]), 15.0)
        self.assertAlmostEqual(float(actual["r5"]), 25.0)
        self.assertEqual(actual.attrs["source_wind"], "quadrant_mean_wind")
        self.assertEqual(actual.attrs["thresholds"], (15.0, 5.0))
        self.assertNotIn("azimuthal", actual["vmax"].attrs["long_name"])

    def test_dask_profiles_remain_lazy(self) -> None:
        values = np.array(
            [
                [0.0, 20.0, 10.0, 0.0],
                [0.0, 30.0, 20.0, 0.0],
            ],
            dtype=np.float32,
        )
        profile = xr.DataArray(
            dask_array.from_array(values, chunks=(1, 2)),
            dims=("time", "r"),
            coords={"time": [0, 1], "r": [0.0, 10.0, 20.0, 30.0]},
            name="wind_profile",
        )

        actual = wind_metrics(profile, thresholds=(15.0,))

        self.assertIsInstance(actual["vmax"].data, dask_array.Array)
        computed = actual.compute()
        np.testing.assert_allclose(computed["vmax"], [20.0, 30.0])
        np.testing.assert_allclose(computed["rmw"], [10.0, 10.0])
        np.testing.assert_allclose(computed["r15"], [15.0, 22.5])

    def test_profile_contract_is_validated(self) -> None:
        with self.assertRaisesRegex(TypeError, "xr.DataArray"):
            wind_metrics(np.ones(4))

        missing_radius = xr.DataArray(np.ones(4), dims=("sample",))
        with self.assertRaisesRegex(ValueError, "dimension 'r'"):
            wind_metrics(missing_radius)

        descending_radius = xr.DataArray(
            np.ones(3),
            dims=("r",),
            coords={"r": [2.0, 1.0, 0.0]},
        )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            wind_metrics(descending_radius)


class ExplicitDiagnosticTests(unittest.TestCase):
    def test_diagnostics_consume_explicit_profiles(self) -> None:
        radius = [500.0, 1500.0, 2500.0]
        tangential_wind = xr.DataArray(
            [10.0, 12.0, 11.0],
            dims=("r",),
            coords={"r": radius},
            name="tangential_wind",
        )
        radial_wind = xr.DataArray(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
            dims=("zc", "r"),
            coords={"zc": [500.0, 1500.0], "r": radius},
            name="radial_wind",
        )
        density = xr.DataArray(
            [1.0, 0.8],
            dims=("zc",),
            coords={"zc": radial_wind["zc"]},
            name="rho",
        )

        aam = angular_momentum(tangential_wind, 5.0e-5)
        i2 = inertial_stability(tangential_wind, 5.0e-5)
        streamfunction = mass_streamfunction(radial_wind, density)

        self.assertEqual(aam.name, "aam")
        self.assertEqual(i2.name, "i2")
        self.assertEqual(streamfunction.name, "psi")
        self.assertEqual(aam.dims, ("r",))
        self.assertEqual(i2.dims, ("r",))
        self.assertEqual(streamfunction.dims, ("zc", "r"))

    def test_removed_cartesian_apis_are_not_public(self) -> None:
        removed = {
            "axisym_mean",
            "polar_derivatives",
            "polar_geometry",
            "decompose_vector",
            "compute_vr_vt",
            "compute_vort_rt",
            "wind_metrics_from_profile",
        }

        self.assertTrue(removed.isdisjoint(tc.__all__))
        for name in removed:
            self.assertFalse(hasattr(tc, name), name)

    def test_accessor_no_longer_exposes_implicit_diagnostics(self) -> None:
        accessor = xr.Dataset().vvm.tc
        removed = {
            "set_params",
            "azimuth",
            "polar_derivatives",
            "wind",
            "vr",
            "vt",
            "vorticity",
            "vort_r",
            "vort_t",
            "wind_metrics",
            "aam",
            "i2",
            "psi",
        }

        for name in removed:
            self.assertFalse(hasattr(accessor, name), name)
            self.assertFalse(hasattr(accessor.masked, name), name)


if __name__ == "__main__":
    unittest.main()
