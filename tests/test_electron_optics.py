"""Relativistic beam physics reference tests.

Compared against textbook values from:
  - Kirkland, "Advanced Computing in Electron Microscopy" (2nd ed), App. C
  - de Graef, "Introduction to Conventional TEM" Table 2.1
  - NIST/CODATA constants
"""
import numpy as np
import pytest

from electron_optics import (
    wavelength_from_voltage,
    interaction_constant_from_voltage,
)


# (voltage [V], expected wavelength [Å]) — Kirkland App. C, ±1e-4 Å
WAVELENGTH_REF = [
    ( 60e3, 0.04866),
    (100e3, 0.03701),
    (120e3, 0.03349),
    (200e3, 0.02508),
    (300e3, 0.01969),
    (1e6,   0.00872),
]


@pytest.mark.parametrize("V, lam_ref", WAVELENGTH_REF)
def test_relativistic_wavelength(V, lam_ref):
    assert wavelength_from_voltage(V) == pytest.approx(lam_ref, abs=2e-4)


# (voltage, expected sigma [rad/(V·Å)])  — de Graef Table 2.1 / Kirkland
# Note: de Graef gives σ in rad/(V·nm); convert by /10.
SIGMA_REF_V_A = [
    (100e3, 9.2442e-4),
    (200e3, 7.2883e-4),
    (300e3, 6.5258e-4),
]


@pytest.mark.parametrize("V, sigma_ref", SIGMA_REF_V_A)
def test_interaction_constant_V_A(V, sigma_ref):
    sigma = interaction_constant_from_voltage(V, potential_units="V_A")
    # 0.3% tolerance accounts for differences in CODATA vintage
    assert sigma == pytest.approx(sigma_ref, rel=3e-3)


def test_sigma_unit_conversion_V_A_vs_V_nm():
    """σ in V·nm units must be exactly 10× σ in V·Å units."""
    for V in (100e3, 200e3, 300e3):
        s_A = interaction_constant_from_voltage(V, "V_A")
        s_nm = interaction_constant_from_voltage(V, "V_nm")
        assert s_nm == pytest.approx(10.0 * s_A, rel=1e-12)


def test_sigma_invalid_units_raises():
    with pytest.raises(ValueError):
        interaction_constant_from_voltage(200e3, potential_units="V_m")


def test_wavelength_monotone_in_voltage():
    """Higher accelerating voltage ⇒ shorter wavelength."""
    lams = [wavelength_from_voltage(V) for V in (60e3, 100e3, 200e3, 300e3, 1e6)]
    assert all(a > b for a, b in zip(lams, lams[1:]))


def test_non_relativistic_limit_small_V():
    """At very low voltages the relativistic correction is negligible,
    so λ ≈ h/sqrt(2mE) ≈ 12.264 / sqrt(V).  (The code's formula
    intentionally includes the 0.9788e-6 V² correction.)"""
    V = 1.0  # 1 V test case, correction term ~1e-6
    lam_code = wavelength_from_voltage(V)
    lam_nonrel = 12.264 / np.sqrt(V)
    assert lam_code == pytest.approx(lam_nonrel, rel=1e-5)
