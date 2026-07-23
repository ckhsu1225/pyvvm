"""Tests for the xarray and Dask cylindrical remapping interfaces."""

from __future__ import annotations

import unittest

import dask.array as dask_array
import numpy as np
import xarray as xr

from pyvvm.tc.cylindrical import (
    CylindricalGridSpec,
    cylindrical_target_coordinates,
    remap_dataarray,
    remap_dataset,
)


def _linear_dataarray() -> tuple[xr.DataArray, xr.Dataset]:
    time = np.array([0.0, 1.0])
    zc = np.array([500.0, 1000.0])
    x = np.arange(7, dtype=np.float64)
    y = np.arange(7, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    values = np.empty((time.size, zc.size, y.size, x.size), dtype=np.float32)
    for t in range(time.size):
        for k in range(zc.size):
            values[t, k] = 2.0 * xx - 3.0 * yy + 10.0 * t + 100.0 * k

    field = xr.DataArray(
        values,
        dims=("time", "zc", "yc", "xc"),
        coords={"time": time, "zc": zc, "yc": y, "xc": x},
        name="linear",
        attrs={"units": "K"},
    )
    track = xr.Dataset(
        {
            "x": ("time", [2.0, 3.0]),
            "y": ("time", [2.5, 3.0]),
        },
        coords={"time": time},
    )
    return field, track


def _small_spec() -> CylindricalGridSpec:
    return CylindricalGridSpec(
        r=[0.0, 0.5, 1.25],
        theta=[0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi],
        boundary="nan",
    )


class DataArrayRemapTests(unittest.TestCase):
    def test_moving_center_preserves_leading_dimensions_and_metadata(self) -> None:
        field, track = _linear_dataarray()
        spec = _small_spec()

        actual = remap_dataarray(field, track, spec=spec)

        self.assertEqual(actual.dims, ("time", "zc", "theta", "r"))
        self.assertEqual(actual.shape, (2, 2, 4, 3))
        self.assertEqual(actual.name, "linear")
        self.assertEqual(actual.attrs["units"], "K")
        np.testing.assert_array_equal(actual["zc"], field["zc"])
        np.testing.assert_allclose(actual["center_x"], track["x"])
        np.testing.assert_allclose(actual["center_y"], track["y"])

        for t in range(field.sizes["time"]):
            x_target, y_target = cylindrical_target_coordinates(
                spec,
                float(track["x"].values[t]),
                float(track["y"].values[t]),
            )
            for k in range(field.sizes["zc"]):
                expected = (
                    2.0 * x_target
                    - 3.0 * y_target
                    + 10.0 * t
                    + 100.0 * k
                )
                np.testing.assert_allclose(
                    actual.values[t, k],
                    expected,
                    rtol=2.0e-6,
                    atol=2.0e-5,
                )

    def test_dask_input_stays_lazy_and_reuses_stencil_across_z_chunks(self) -> None:
        field, track = _linear_dataarray()
        chunked = field.copy(
            data=dask_array.from_array(field.values, chunks=(2, 1, 3, 2))
        )
        spec = _small_spec()

        actual = remap_dataarray(chunked, track, spec=spec)

        self.assertIsInstance(actual.data, dask_array.Array)
        self.assertEqual(actual.chunksizes["time"], (1, 1))
        self.assertEqual(actual.chunksizes["zc"], (1, 1))
        self.assertEqual(actual.chunksizes["theta"], (4,))
        self.assertEqual(actual.chunksizes["r"], (3,))

        graph_keys = actual.data.__dask_graph__().keys()
        stencil_keys = [
            key
            for key in graph_keys
            if str(key[0] if isinstance(key, tuple) else key).startswith(
                "build_cylindrical_stencil-"
            )
        ]
        self.assertEqual(len(stencil_keys), field.sizes["time"])

        expected = remap_dataarray(field, track, spec=spec)
        xr.testing.assert_allclose(actual.compute(), expected)

    def test_surface_and_derived_dataarray_need_no_variable_name_or_z(self) -> None:
        field, _ = _linear_dataarray()
        surface = (2.0 * field.isel(zc=0, drop=True) + 7.0).rename("derived")
        spec = _small_spec()

        actual = remap_dataarray(surface, (2.5, 2.5), spec=spec)

        self.assertEqual(actual.dims, ("time", "theta", "r"))
        self.assertEqual(actual.name, "derived")
        self.assertNotIn("zc", actual.dims)
        x_target, y_target = cylindrical_target_coordinates(spec, 2.5, 2.5)
        for t in range(surface.sizes["time"]):
            expected = 2.0 * (
                2.0 * x_target - 3.0 * y_target + 10.0 * t
            ) + 7.0
            np.testing.assert_allclose(
                actual.values[t], expected, rtol=2.0e-6, atol=2.0e-5
            )

    def test_explicit_custom_horizontal_dimension_names(self) -> None:
        lon = np.arange(6, dtype=np.float64)
        lat = np.arange(6, dtype=np.float64)
        xx, yy = np.meshgrid(lon, lat)
        field = xr.DataArray(
            dask_array.from_array(xx + 4.0 * yy, chunks=(3, 2)),
            dims=("lat", "lon"),
            coords={"lat": lat, "lon": lon},
        )
        spec = _small_spec()

        actual = remap_dataarray(
            field,
            (2.5, 2.5),
            spec=spec,
            x_dim="lon",
            y_dim="lat",
        )

        x_target, y_target = cylindrical_target_coordinates(spec, 2.5, 2.5)
        self.assertIsInstance(actual.data, dask_array.Array)
        np.testing.assert_allclose(
            actual.compute(),
            x_target + 4.0 * y_target,
        )

    def test_scalar_time_coordinate_selects_one_center(self) -> None:
        field, track = _linear_dataarray()
        snapshot = field.isel(time=1, zc=0)

        actual = remap_dataarray(snapshot, track, spec=_small_spec())
        expected = remap_dataarray(
            snapshot.drop_vars("time"),
            (float(track["x"].values[1]), float(track["y"].values[1])),
            spec=_small_spec(),
        )

        self.assertNotIn("time", actual.dims)
        self.assertEqual(float(actual["time"]), float(snapshot["time"]))
        xr.testing.assert_allclose(actual.drop_vars("time"), expected)

    def test_time_dependent_center_without_input_time_is_rejected(self) -> None:
        field, track = _linear_dataarray()

        with self.assertRaisesRegex(ValueError, "time-dependent center"):
            remap_dataarray(
                field.isel(time=0, zc=0, drop=True),
                track,
                spec=_small_spec(),
            )


class DatasetRemapTests(unittest.TestCase):
    @staticmethod
    def _mixed_dataset() -> xr.Dataset:
        time = np.array([0.0, 1.0])
        zc = np.array([500.0, 1000.0])
        zb = np.array([0.0, 750.0, 1250.0])
        xc = np.arange(6, dtype=np.float64)
        xb = xc + 0.5
        yc = np.arange(6, dtype=np.float64)

        th_values = np.broadcast_to(
            np.arange(4, dtype=np.float32).reshape(2, 2, 1, 1),
            (2, 2, 6, 6),
        ).copy()
        eta_values = np.broadcast_to(
            (10.0 + np.arange(6, dtype=np.float32)).reshape(2, 3, 1, 1),
            (2, 3, 6, 6),
        ).copy()
        rain_values = np.broadcast_to(
            np.array([1.0, 2.0], dtype=np.float32).reshape(2, 1, 1),
            (2, 6, 6),
        ).copy()

        return xr.Dataset(
            {
                "th": (("time", "zc", "yc", "xc"), th_values),
                "eta": (("time", "zb", "yc", "xb"), eta_values),
                "rain": (("time", "yc", "xc"), rain_values),
                "profile": (("zc",), [1.0, 2.0]),
            },
            coords={
                "time": time,
                "zc": zc,
                "zb": zb,
                "xc": xc,
                "xb": xb,
                "yc": yc,
            },
            attrs={"case": "test"},
        )

    def test_optional_variables_keep_mixed_vertical_grids_separate(self) -> None:
        source = self._mixed_dataset()
        spec = _small_spec()

        actual = remap_dataset(source, (2.5, 2.5), spec=spec)

        self.assertEqual(set(actual.data_vars), {"th", "eta", "rain"})
        self.assertEqual(actual["th"].dims, ("time", "zc", "theta", "r"))
        self.assertEqual(actual["eta"].dims, ("time", "zb", "theta", "r"))
        self.assertEqual(actual["rain"].dims, ("time", "theta", "r"))
        self.assertNotIn("profile", actual)
        self.assertEqual(actual.attrs["case"], "test")
        np.testing.assert_allclose(actual["th"].isel(time=1, zc=0), 2.0)
        np.testing.assert_allclose(actual["eta"].isel(time=0, zb=2), 12.0)
        np.testing.assert_allclose(actual["rain"].isel(time=1), 2.0)

    def test_variable_selection_and_validation(self) -> None:
        source = self._mixed_dataset()

        actual = remap_dataset(
            source,
            (2.5, 2.5),
            spec=_small_spec(),
            variables="rain",
        )
        self.assertEqual(set(actual.data_vars), {"rain"})

        with self.assertRaisesRegex(ValueError, "not found"):
            remap_dataset(
                source,
                (2.5, 2.5),
                spec=_small_spec(),
                variables=["missing"],
            )
        with self.assertRaisesRegex(ValueError, "Cannot infer"):
            remap_dataset(
                source,
                (2.5, 2.5),
                spec=_small_spec(),
                variables=["profile"],
            )

    def test_tc_accessor_accepts_names_and_custom_dataarrays(self) -> None:
        source = self._mixed_dataset()
        track = xr.Dataset(
            {
                "x": ("time", [2.5, 2.5]),
                "y": ("time", [2.5, 2.5]),
            },
            coords={"time": source["time"]},
        )
        tc = source.vvm.tc
        tc._track = track

        named = tc.remap("rain", spec=_small_spec())
        custom = tc.remap((source["rain"] + 3.0).rename("custom"), spec=_small_spec())
        dataset = tc.remap_dataset(spec=_small_spec(), variables=["rain"])

        np.testing.assert_allclose(named.isel(time=0), 1.0)
        np.testing.assert_allclose(custom.isel(time=1), 5.0)
        self.assertEqual(set(dataset.data_vars), {"rain"})


if __name__ == "__main__":
    unittest.main()
