"""Tests for the intentionally reduced thermodynamic API."""

from __future__ import annotations

import unittest

from pyvvm.calc import formulas
from pyvvm.calc.thermodynamics import ThermoMixin


class RemovedThermodynamicAPITests(unittest.TestCase):
    def test_unused_accessor_properties_and_wrappers_are_removed(self) -> None:
        removed_names = ("thei", "h", "gv", "gl", "gi")

        for name in removed_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(ThermoMixin, name))
                self.assertFalse(hasattr(ThermoMixin, f"_calc_{name}"))

    def test_unused_pure_formulas_are_removed(self) -> None:
        removed_names = (
            "ice_equivalent_potential_temperature",
            "specific_enthalpy",
            "specific_gibbs_free_energy_of_water_vapor",
            "specific_gibbs_free_energy_of_liquid_water",
            "specific_gibbs_free_energy_of_ice",
        )

        for name in removed_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(formulas, name))
                self.assertNotIn(name, formulas.__all__)

    def test_entropy_api_is_retained(self) -> None:
        self.assertTrue(hasattr(ThermoMixin, "s"))
        self.assertTrue(hasattr(ThermoMixin, "_calc_s"))
        self.assertTrue(hasattr(formulas, "specific_entropy"))
        self.assertIn("specific_entropy", formulas.__all__)


if __name__ == "__main__":
    unittest.main()
