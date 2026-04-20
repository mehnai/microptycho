"""Aberration function and CTF tests."""
import numpy as np
import pytest

from microscope import Microscope
from microptycho import MicroPtycho


def _kgrid(N, dx):
    kx = np.fft.fftfreq(N, dx)
    return np.meshgrid(kx, kx)


# ---------------------------- Aberration function --------------------------- #

def test_chi_zero_at_k_zero():
    scope = Microscope(voltage=200e3, alpha=0.010,
                       C1=100.0, A1=50.0, B2=80.0, C3=1e7)
    KX, KY = _kgrid(64, 0.5)
    chi = scope.aberration_function(KX, KY)
    assert chi[0, 0] == 0.0


def test_chi_pure_defocus_is_parabolic():
    """Pure C1: chi = π λ C1 k² (azimuthally symmetric)."""
    scope = Microscope(voltage=200e3, alpha=0.010, C1=100.0)
    KX, KY = _kgrid(64, 0.5)
    chi = scope.aberration_function(KX, KY)
    expected = np.pi * scope.wavelength * scope.C1 * (KX ** 2 + KY ** 2)
    assert np.max(np.abs(chi - expected)) < 1e-13


def test_chi_pure_Cs_quartic():
    """Pure C3: chi = (π/2) λ³ C3 k⁴."""
    scope = Microscope(voltage=200e3, alpha=0.010, C3=1e7)
    KX, KY = _kgrid(64, 0.5)
    chi = scope.aberration_function(KX, KY)
    expected = 0.5 * np.pi * scope.wavelength ** 3 * scope.C3 * (KX ** 2 + KY ** 2) ** 2
    # chi reaches ~10^3 rad at the corner; accept 10^-12 rel roundoff (k**4 vs (k²)²
    # computed via sqrt differs at the ulp level).
    assert np.allclose(chi, expected, rtol=1e-12, atol=1e-12)


def test_chi_astigmatism_axis_dependence():
    """Pure A1: chi has cos(2(φ-φ_A1)) azimuthal dependence ⇒ same |k| gives
    opposite signs at orthogonal azimuths."""
    scope = Microscope(voltage=200e3, alpha=0.010, A1=100.0, phi_A1=0.0)
    # k in +x direction vs +y direction (same |k|)
    KX = np.array([[0.1, 0.0]])
    KY = np.array([[0.0, 0.1]])
    chi = scope.aberration_function(KX, KY)
    assert chi[0, 0] == pytest.approx(-chi[0, 1], rel=1e-12)


def test_chi_coma_odd_symmetry():
    """Pure B2: cos(φ - φ_B2). With φ_B2=0, chi(+kx) = -chi(-kx), i.e. odd in kx."""
    scope = Microscope(voltage=200e3, alpha=0.010, B2=100.0, phi_B2=0.0)
    KX = np.array([[0.1, -0.1]])
    KY = np.zeros_like(KX)
    chi = scope.aberration_function(KX, KY)
    assert chi[0, 0] == pytest.approx(-chi[0, 1], rel=1e-12)


# --------------------- Contrast transfer function --------------------------- #

def test_ctf_is_sin_chi():
    scope = Microscope(voltage=200e3, alpha=0.010, C1=100.0, C3=1e7)
    KX, KY = _kgrid(64, 0.5)
    assert np.max(np.abs(scope.ctf(KX, KY) - np.sin(scope.aberration_function(KX, KY)))) < 1e-13


def test_ctf_magnitude_bounded_by_one():
    scope = Microscope(voltage=200e3, alpha=0.010, C1=1000.0, A1=500.0, C3=2e7)
    KX, KY = _kgrid(128, 0.43)
    ctf = scope.ctf(KX, KY)
    assert ctf.max() <= 1.0 + 1e-14
    assert ctf.min() >= -1.0 - 1e-14


# ---------------------- Transfer function vs propagation ------------------- #

def test_transfer_function_is_unit_modulus():
    scope = Microscope(voltage=200e3, alpha=0.010, C1=100.0, A1=50.0, B2=80.0, C3=1e7)
    KX, KY = _kgrid(64, 0.5)
    H = scope.transfer_function(KX, KY)
    assert np.max(np.abs(np.abs(H) - 1.0)) < 1e-13


def test_defocus_vs_fresnel_propagation_equivalence():
    """Applying H = exp(-i π λ C1 k²) (pure-defocus aberration with convention
    H=exp(-iχ)) is the same as Fresnel-propagating by -C1: both move the focus
    by -C1. This confirms the sign convention is self-consistent."""
    N, dx = 64, 0.43
    KX, KY = _kgrid(N, dx)
    C1 = 100.0
    scope = Microscope(voltage=200e3, alpha=0.010, C1=C1)
    H_aberr = scope.transfer_function(KX, KY)                      # exp(-i π λ C1 k²)
    H_prop  = MicroPtycho.fresnel_propagator(KX, KY, dz=+C1,
                                             lam=scope.wavelength)  # exp(-i π λ C1 k²)
    assert np.max(np.abs(H_aberr - H_prop)) < 1e-13


# --------------- Microscope probe equals MicroPtycho ideal probe ----------- #

def test_unaberrated_microscope_probe_matches_microptycho_probe():
    mp = MicroPtycho(N=128, dx=0.43, voltage=200e3)
    probe_mp = mp.construct_probe(alpha=0.010)
    probe_scope = Microscope(voltage=200e3, alpha=0.010).construct_probe(N=128, dx=0.43)
    assert np.max(np.abs(probe_mp - probe_scope)) < 1e-13


def test_microscope_presets_roundtrip_wavelength():
    for scope in (Microscope.TEM_200kV(), Microscope.aberration_corrected_200kV(),
                  Microscope.TEM_300kV(), Microscope.aberration_corrected_300kV()):
        assert scope.wavelength == pytest.approx(Microscope.wavelength_from_voltage(scope.voltage))


# ---------- Apply CTF is a linear real-space convolution ------------------- #

def test_apply_ctf_linearity():
    rng = np.random.default_rng(0)
    scope = Microscope(voltage=200e3, alpha=0.010, C1=100.0, C3=1e7)
    a = rng.normal(size=(64, 64))
    b = rng.normal(size=(64, 64))
    out_sum = scope.apply_ctf(a + 3.0 * b, dx=0.43)
    out_a   = scope.apply_ctf(a, dx=0.43)
    out_b   = scope.apply_ctf(b, dx=0.43)
    assert np.max(np.abs(out_sum - (out_a + 3.0 * out_b))) < 1e-10
