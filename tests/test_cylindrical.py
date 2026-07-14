"""Tests for the low-level cylindrical remapping kernels."""

from __future__ import annotations

import unittest

import numpy as np

from pyvvm.tc.cylindrical import (
    CylindricalGridSpec,
    apply_cylindrical_stencil,
    build_cylindrical_stencil,
    cylindrical_target_coordinates,
)


class CylindricalGridSpecTests(unittest.TestCase):
    def test_cell_centred_coordinates(self) -> None:
        spec = CylindricalGridSpec.from_spacing(r_max=6.0, dr=2.0, ntheta=4)

        self.assertEqual(spec.nr, 3)
        self.assertEqual(spec.shape, (4, 3))
        np.testing.assert_allclose(spec.r, [1.0, 3.0, 5.0])
        np.testing.assert_allclose(
            spec.theta,
            [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi],
            atol=1.0e-15,
        )

    def test_invalid_specification(self) -> None:
        invalid_kwargs = (
            {"r_max": 0.0, "dr": 1.0, "ntheta": 4},
            {"r_max": 1.0, "dr": 2.0, "ntheta": 4},
            {"r_max": 2.0, "dr": 1.0, "ntheta": 0},
            {"r_max": 2.0, "dr": 1.0, "ntheta": 4, "method": "cubic"},
            {"r_max": 2.0, "dr": 1.0, "ntheta": 4, "boundary": "clip"},
            {"r_max": 2.0, "dr": 1.0, "ntheta": 4, "nan_policy": "zero"},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CylindricalGridSpec.from_spacing(**kwargs)

        with self.assertRaises(TypeError):
            CylindricalGridSpec.from_spacing(r_max=2.0, dr=1.0, ntheta=4.5)
        with self.assertRaises(TypeError):
            CylindricalGridSpec.from_spacing(r_max=2.0, dr=1.0, ntheta=True)

    def test_explicit_nonuniform_coordinates(self) -> None:
        radius_input = np.array([0.0, 1.0, 2.5, 5.0])
        theta_input = np.deg2rad([-90.0, 0.0, 30.0, 170.0])
        spec = CylindricalGridSpec(r=radius_input, theta=theta_input)

        self.assertEqual(spec.shape, (4, 4))
        np.testing.assert_array_equal(spec.r, radius_input)
        np.testing.assert_array_equal(spec.theta, theta_input)

        radius_input[0] = 99.0
        self.assertEqual(float(spec.r[0]), 0.0)
        with self.assertRaises(ValueError):
            spec.r[0] = 1.0

    def test_invalid_explicit_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            CylindricalGridSpec(r=[-1.0, 1.0], theta=[0.0])
        with self.assertRaises(ValueError):
            CylindricalGridSpec(r=[1.0, 1.0], theta=[0.0])
        with self.assertRaises(ValueError):
            CylindricalGridSpec(r=[1.0], theta=[0.0, 2.0 * np.pi])

    def test_target_coordinates_follow_theta_convention(self) -> None:
        spec = CylindricalGridSpec.from_spacing(r_max=2.0, dr=2.0, ntheta=4)
        x_target, y_target = cylindrical_target_coordinates(spec, 10.0, 20.0)

        np.testing.assert_allclose(x_target[:, 0], [11.0, 10.0, 9.0, 10.0])
        np.testing.assert_allclose(y_target[:, 0], [20.0, 21.0, 20.0, 19.0])


class CylindricalStencilTests(unittest.TestCase):
    def test_explicit_nonuniform_targets_are_remapped(self) -> None:
        x = np.arange(8, dtype=np.float64)
        y = np.arange(8, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        field = 3.0 * xx - 2.0 * yy + 4.0
        spec = CylindricalGridSpec(
            r=[0.0, 0.75, 2.25],
            theta=np.deg2rad([-60.0, 0.0, 35.0, 150.0]),
            boundary="nan",
        )
        stencil = build_cylindrical_stencil(
            x,
            y,
            center_x=3.5,
            center_y=3.5,
            spec=spec,
        )

        actual = apply_cylindrical_stencil(field, stencil)
        x_target, y_target = cylindrical_target_coordinates(spec, 3.5, 3.5)
        expected = 3.0 * x_target - 2.0 * y_target + 4.0

        self.assertEqual(actual.shape, (4, 3))
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)

    def test_linear_interpolation_is_exact_for_linear_field(self) -> None:
        x = np.arange(7, dtype=np.float64) * 2.0
        y = 10.0 + np.arange(6, dtype=np.float64) * 3.0
        xx, yy = np.meshgrid(x, y)
        field = 2.0 * xx + 3.0 * yy + 5.0
        spec = CylindricalGridSpec.from_spacing(
            r_max=4.0,
            dr=2.0,
            ntheta=8,
            boundary="nan",
        )
        stencil = build_cylindrical_stencil(
            x,
            y,
            center_x=6.0,
            center_y=17.5,
            spec=spec,
        )

        actual = apply_cylindrical_stencil(field, stencil)
        x_target, y_target = cylindrical_target_coordinates(spec, 6.0, 17.5)
        expected = 2.0 * x_target + 3.0 * y_target + 5.0

        self.assertEqual(actual.shape, spec.shape)
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-5)

    def test_arbitrary_leading_dimensions_are_preserved(self) -> None:
        x = np.arange(7, dtype=np.float64)
        y = np.arange(7, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        base = (xx + 2.0 * yy).astype(np.float32)
        scales = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
        offsets = np.arange(6, dtype=np.float32).reshape(2, 3) * 10.0
        values = (
            scales[..., None, None] * base
            + offsets[..., None, None]
        )
        spec = CylindricalGridSpec.from_spacing(
            r_max=3.0,
            dr=1.0,
            ntheta=8,
            boundary="nan",
        )
        stencil = build_cylindrical_stencil(
            x,
            y,
            center_x=3.0,
            center_y=3.0,
            spec=spec,
        )

        actual = apply_cylindrical_stencil(values, stencil)
        x_target, y_target = cylindrical_target_coordinates(spec, 3.0, 3.0)
        target_base = x_target + 2.0 * y_target
        expected = (
            scales[..., None, None] * target_base
            + offsets[..., None, None]
        )

        self.assertEqual(actual.shape, (2, 3, spec.ntheta, spec.nr))
        self.assertEqual(actual.dtype, np.dtype(np.float32))
        np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=2.0e-5)

    def test_surface_time_dimension_needs_no_vertical_axis(self) -> None:
        x = np.arange(5, dtype=np.float64)
        y = np.arange(5, dtype=np.float64)
        surface = np.stack(
            (
                np.ones((5, 5), dtype=np.float32),
                np.full((5, 5), 2.0, dtype=np.float32),
            )
        )
        spec = CylindricalGridSpec.from_spacing(r_max=2.0, dr=1.0, ntheta=4)
        stencil = build_cylindrical_stencil(
            x,
            y,
            center_x=2.0,
            center_y=2.0,
            spec=spec,
        )

        actual = apply_cylindrical_stencil(surface, stencil)

        self.assertEqual(actual.shape, (2, 4, 2))
        np.testing.assert_allclose(actual[0], 1.0)
        np.testing.assert_allclose(actual[1], 2.0)

    def test_periodic_boundary_wraps_continuously_in_index_space(self) -> None:
        x = np.arange(4, dtype=np.float64)
        y = np.arange(4, dtype=np.float64)
        field = np.broadcast_to(x, (4, 4)).copy()
        spec = CylindricalGridSpec.from_spacing(r_max=1.0, dr=1.0, ntheta=4)
        right_stencil = build_cylindrical_stencil(
            x,
            y,
            center_x=3.75,
            center_y=1.5,
            spec=spec,
        )
        left_stencil = build_cylindrical_stencil(
            x,
            y,
            center_x=-0.25,
            center_y=1.5,
            spec=spec,
        )

        right = apply_cylindrical_stencil(field, right_stencil)
        left = apply_cylindrical_stencil(field, left_stencil)

        np.testing.assert_allclose(right, left)
        self.assertAlmostEqual(float(right[0, 0]), 0.25, places=6)
        self.assertAlmostEqual(float(right[2, 0]), 2.25, places=6)

    def test_nonperiodic_targets_outside_extent_are_nan(self) -> None:
        x = np.arange(4, dtype=np.float64)
        y = np.arange(4, dtype=np.float64)
        field = np.ones((4, 4), dtype=np.float32)
        spec = CylindricalGridSpec.from_spacing(
            r_max=2.0,
            dr=2.0,
            ntheta=4,
            boundary="nan",
        )
        stencil = build_cylindrical_stencil(
            x,
            y,
            center_x=0.0,
            center_y=0.0,
            spec=spec,
        )

        actual = apply_cylindrical_stencil(field, stencil)

        np.testing.assert_allclose(actual[:2], 1.0)
        self.assertTrue(np.isnan(actual[2:, 0]).all())

    def test_nearest_neighbour_uses_closest_grid_point(self) -> None:
        x = np.arange(5, dtype=np.float64)
        y = np.arange(5, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)
        field = 10.0 * yy + xx
        spec = CylindricalGridSpec.from_spacing(
            r_max=1.2,
            dr=1.2,
            ntheta=4,
            method="nearest",
            boundary="nan",
        )
        stencil = build_cylindrical_stencil(
            x,
            y,
            center_x=2.0,
            center_y=2.0,
            spec=spec,
        )

        actual = apply_cylindrical_stencil(field, stencil)

        np.testing.assert_allclose(actual[:, 0], [23.0, 32.0, 21.0, 12.0])

    def test_nan_policy_propagate_and_omit(self) -> None:
        x = np.arange(4, dtype=np.float64)
        y = np.arange(4, dtype=np.float64)
        field = np.zeros((4, 4), dtype=np.float32)
        field[1, 1] = np.nan
        field[1, 2] = 3.0
        field[2, 1] = 5.0
        field[2, 2] = 7.0
        dr = np.sqrt(2.0)
        spec = CylindricalGridSpec.from_spacing(
            r_max=dr,
            dr=dr,
            ntheta=8,
            boundary="nan",
        )
        stencil = build_cylindrical_stencil(
            x,
            y,
            center_x=1.0,
            center_y=1.0,
            spec=spec,
        )

        propagated = apply_cylindrical_stencil(
            field,
            stencil,
            nan_policy="propagate",
        )
        omitted = apply_cylindrical_stencil(
            field,
            stencil,
            nan_policy="omit",
        )

        self.assertTrue(np.isnan(propagated[1, 0]))
        self.assertAlmostEqual(float(omitted[1, 0]), 5.0, places=6)

    def test_spec_nan_policy_is_used_by_default(self) -> None:
        x = np.arange(4, dtype=np.float64)
        y = np.arange(4, dtype=np.float64)
        field = np.full((4, 4), np.nan, dtype=np.float32)
        field[1, 2] = 6.0
        spec = CylindricalGridSpec.from_spacing(
            r_max=1.0,
            dr=1.0,
            ntheta=4,
            boundary="nan",
            nan_policy="omit",
        )
        stencil = build_cylindrical_stencil(
            x,
            y,
            center_x=1.5,
            center_y=1.0,
            spec=spec,
        )

        actual = apply_cylindrical_stencil(field, stencil)

        # theta=0 lands exactly on (x=2, y=1); zero-weight NaN neighbours
        # must neither propagate nor affect the renormalization.
        self.assertAlmostEqual(float(actual[0, 0]), 6.0, places=6)

    def test_coordinate_and_shape_validation(self) -> None:
        spec = CylindricalGridSpec.from_spacing(r_max=1.0, dr=1.0, ntheta=4)
        with self.assertRaises(ValueError):
            build_cylindrical_stencil(
                [0.0, 1.0, 3.0],
                [0.0, 1.0, 2.0],
                center_x=1.0,
                center_y=1.0,
                spec=spec,
            )

        stencil = build_cylindrical_stencil(
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
            center_x=1.0,
            center_y=1.0,
            spec=spec,
        )
        with self.assertRaises(ValueError):
            apply_cylindrical_stencil(np.ones((4, 4)), stencil)


if __name__ == "__main__":
    unittest.main()
