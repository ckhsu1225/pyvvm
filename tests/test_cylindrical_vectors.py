"""Tests for vector rotation on cylindrical target grids."""

from __future__ import annotations

import unittest

import dask.array as dask_array
import numpy as np
import xarray as xr

from pyvvm.tc import (
    CylindricalGridSpec,
    remap_dataset,
    rotate_vector,
    rotate_vorticity,
    rotate_wind,
)


def _component(
    values: np.ndarray | dask_array.Array,
    *,
    theta: np.ndarray,
    radius: np.ndarray,
    name: str,
    units: str = "m s-1",
) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("theta", "r"),
        coords={"theta": theta, "r": radius},
        name=name,
        attrs={"units": units},
    )


class CylindricalVectorRotationTests(unittest.TestCase):
    def test_cartesian_basis_is_rotated_at_target_theta(self) -> None:
        theta = np.arange(4, dtype=np.float64) * (0.5 * np.pi)
        radius = np.array([500.0, 1500.0])
        x = _component(
            np.ones((4, 2), dtype=np.float32),
            theta=theta,
            radius=radius,
            name="x",
        )
        y = _component(
            np.zeros((4, 2), dtype=np.float32),
            theta=theta,
            radius=radius,
            name="y",
        )

        actual = rotate_vector(x, y)

        expected_radial = np.broadcast_to(np.cos(theta)[:, None], x.shape)
        expected_tangential = np.broadcast_to(-np.sin(theta)[:, None], x.shape)
        np.testing.assert_allclose(
            actual["radial"], expected_radial, atol=1.0e-7
        )
        np.testing.assert_allclose(
            actual["tangential"], expected_tangential, atol=1.0e-7
        )
        self.assertEqual(actual["radial"].dims, x.dims)
        self.assertEqual(actual["tangential"].dims, x.dims)
        self.assertEqual(actual["radial"].dtype, np.dtype(np.float32))
        self.assertEqual(
            actual.attrs["vector_rotation"],
            "cartesian_to_cylindrical_at_target",
        )

    def test_origin_is_masked_unless_explicitly_requested(self) -> None:
        theta = np.array([0.0, 0.5 * np.pi])
        radius = np.array([0.0, 1000.0])
        x = _component(
            np.ones((2, 2), dtype=np.float64),
            theta=theta,
            radius=radius,
            name="x",
        )
        y = _component(
            np.ones((2, 2), dtype=np.float64),
            theta=theta,
            radius=radius,
            name="y",
        )

        masked = rotate_vector(x, y)
        unmasked = rotate_vector(x, y, mask_origin=False)

        self.assertTrue(np.isnan(masked["radial"].sel(r=0.0)).all())
        self.assertTrue(np.isnan(masked["tangential"].sel(r=0.0)).all())
        self.assertTrue(np.isfinite(unmasked["radial"].sel(r=0.0)).all())
        self.assertTrue(np.isfinite(unmasked["tangential"].sel(r=0.0)).all())

    def test_dask_inputs_remain_lazy_and_keep_leading_dimensions(self) -> None:
        theta = np.arange(4, dtype=np.float64) * (0.5 * np.pi)
        radius = np.array([500.0, 1500.0, 2500.0])
        values = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
        x = xr.DataArray(
            dask_array.from_array(values, chunks=(1, 2, 3)),
            dims=("time", "theta", "r"),
            coords={"time": [0, 1], "theta": theta, "r": radius},
            attrs={"units": "m s-1"},
        )
        y = xr.zeros_like(x)

        actual = rotate_wind(x, y)

        self.assertIsInstance(actual["radial_wind"].data, dask_array.Array)
        self.assertIsInstance(actual["tangential_wind"].data, dask_array.Array)
        self.assertEqual(
            actual["radial_wind"].dims,
            ("time", "theta", "r"),
        )
        self.assertEqual(actual["radial_wind"].chunks, x.chunks)
        expected_vr = x.compute() * np.cos(x["theta"])
        expected_vt = -x.compute() * np.sin(x["theta"])
        xr.testing.assert_allclose(
            actual["radial_wind"].compute(),
            expected_vr,
            atol=1.0e-6,
        )
        xr.testing.assert_allclose(
            actual["tangential_wind"].compute(),
            expected_vt,
            atol=1.0e-6,
        )

    def test_wind_wrapper_sets_names_and_metadata(self) -> None:
        theta = np.array([0.0, 0.5 * np.pi])
        radius = np.array([1000.0])
        u = _component(
            np.array([[3.0], [3.0]]),
            theta=theta,
            radius=radius,
            name="u",
        )
        v = _component(
            np.array([[4.0], [4.0]]),
            theta=theta,
            radius=radius,
            name="v",
        )

        actual = rotate_wind(u, v)

        self.assertEqual(
            set(actual.data_vars),
            {"radial_wind", "tangential_wind"},
        )
        np.testing.assert_allclose(actual["radial_wind"][:, 0], [3.0, 4.0])
        np.testing.assert_allclose(
            actual["tangential_wind"][:, 0],
            [4.0, -3.0],
        )
        self.assertEqual(
            actual["radial_wind"].attrs["long_name"],
            "radial wind",
        )
        self.assertEqual(
            actual["tangential_wind"].attrs["long_name"],
            "tangential wind",
        )
        self.assertEqual(actual["radial_wind"].attrs["positive"], "outward")
        self.assertEqual(
            actual["tangential_wind"].attrs["positive"],
            "counter-clockwise",
        )
        self.assertEqual(actual["radial_wind"].attrs["units"], "m s-1")

    def test_vorticity_wrapper_applies_vvm_eta_sign(self) -> None:
        theta = np.array([0.0, 0.5 * np.pi])
        radius = np.array([1000.0])
        xi = _component(
            np.ones((2, 1)),
            theta=theta,
            radius=radius,
            name="xi",
            units="s-1",
        )
        eta = _component(
            np.full((2, 1), 2.0),
            theta=theta,
            radius=radius,
            name="eta",
            units="s-1",
        )

        vvm = rotate_vorticity(xi, eta)
        physical_y = rotate_vorticity(
            xi,
            eta,
            eta_convention="physical_y",
        )

        np.testing.assert_allclose(
            vvm["radial_vorticity"][:, 0],
            [1.0, -2.0],
        )
        np.testing.assert_allclose(
            vvm["tangential_vorticity"][:, 0],
            [-2.0, -1.0],
        )
        np.testing.assert_allclose(
            physical_y["radial_vorticity"][:, 0],
            [1.0, 2.0],
        )
        np.testing.assert_allclose(
            physical_y["tangential_vorticity"][:, 0],
            [2.0, -1.0],
        )
        self.assertEqual(vvm.attrs["eta_convention"], "vvm")
        self.assertEqual(vvm["radial_vorticity"].attrs["units"], "s-1")

    def test_components_must_share_dimensions_coordinates_and_units(self) -> None:
        theta = np.array([0.0, 1.0])
        radius = np.array([1000.0, 2000.0])
        x = _component(
            np.ones((2, 2)),
            theta=theta,
            radius=radius,
            name="x",
        )

        mismatched_theta = _component(
            np.ones((2, 2)),
            theta=np.array([0.0, 1.1]),
            radius=radius,
            name="y",
        )
        with self.assertRaisesRegex(ValueError, "matching coordinates"):
            rotate_vector(x, mismatched_theta)

        missing_r = xr.DataArray(
            np.ones(2),
            dims=("theta",),
            coords={"theta": theta},
        )
        with self.assertRaisesRegex(ValueError, "dimension 'r'"):
            rotate_vector(x, missing_r)

        mismatched_units = _component(
            np.ones((2, 2)),
            theta=theta,
            radius=radius,
            name="y",
            units="s-1",
        )
        with self.assertRaisesRegex(ValueError, "same units"):
            rotate_vector(x, mismatched_units)


class RemapThenRotateTests(unittest.TestCase):
    def test_native_staggered_wind_and_vorticity_are_rotated_after_remap(
        self,
    ) -> None:
        xc = np.arange(7, dtype=np.float64)
        xb = xc + 0.5
        yc = np.arange(7, dtype=np.float64)
        yb = yc + 0.5
        shape = (7, 7)
        source = xr.Dataset(
            {
                "u": (
                    ("yc", "xb"),
                    np.full(shape, 2.0, dtype=np.float32),
                    {"units": "m s-1"},
                ),
                "v": (
                    ("yb", "xc"),
                    np.full(shape, -1.0, dtype=np.float32),
                    {"units": "m s-1"},
                ),
                "xi": (
                    ("yb", "xc"),
                    np.full(shape, 3.0, dtype=np.float32),
                    {"units": "s-1"},
                ),
                "eta": (
                    ("yc", "xb"),
                    np.full(shape, 4.0, dtype=np.float32),
                    {"units": "s-1"},
                ),
            },
            coords={"xc": xc, "xb": xb, "yc": yc, "yb": yb},
        )
        spec = CylindricalGridSpec(
            r=[0.5, 1.0],
            theta=[0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi],
            boundary="nan",
        )

        cylindrical = remap_dataset(
            source,
            (3.0, 3.0),
            spec=spec,
            variables=["u", "v", "xi", "eta"],
        )
        wind = rotate_wind(cylindrical["u"], cylindrical["v"])
        vorticity = rotate_vorticity(
            cylindrical["xi"],
            cylindrical["eta"],
        )

        cos_theta = np.broadcast_to(
            np.cos(spec.theta)[:, None],
            spec.shape,
        )
        sin_theta = np.broadcast_to(
            np.sin(spec.theta)[:, None],
            spec.shape,
        )
        np.testing.assert_allclose(
            wind["radial_wind"],
            2.0 * cos_theta - sin_theta,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            wind["tangential_wind"],
            -2.0 * sin_theta - cos_theta,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            vorticity["radial_vorticity"],
            3.0 * cos_theta - 4.0 * sin_theta,
            atol=2.0e-6,
        )
        np.testing.assert_allclose(
            vorticity["tangential_vorticity"],
            -3.0 * sin_theta - 4.0 * cos_theta,
            atol=2.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
