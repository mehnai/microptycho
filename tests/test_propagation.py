"""Fresnel propagation, transmission, and FFT invariants."""
import numpy as np
import pytest

from microptycho import MicroPtycho


# ------------------------------ Fresnel kernel ----------------------------- #

def test_fresnel_dc_phase_is_zero():
    """The k=0 component should have zero phase (identity on average)."""
    H00 = MicroPtycho.fresnel_propagator(np.array([[0.0]]), np.array([[0.0]]), dz=50.0, lam=0.025)
    assert H00[0, 0] == pytest.approx(1.0 + 0.0j)


def test_fresnel_unit_modulus():
    """Free-space propagation is lossless: |H(k)| = 1 everywhere."""
    N, dx = 64, 0.43
    kx = np.fft.fftfreq(N, dx)
    KX, KY = np.meshgrid(kx, kx)
    H = MicroPtycho.fresnel_propagator(KX, KY, dz=25.0, lam=0.025)
    assert np.max(np.abs(np.abs(H) - 1.0)) < 1e-12


def test_fresnel_propagation_is_reversible():
    """H(+dz) * H(-dz) = 1 — the propagator is unitary in k-space."""
    N, dx, dz, lam = 64, 0.5, 30.0, 0.025
    kx = np.fft.fftfreq(N, dx)
    KX, KY = np.meshgrid(kx, kx)
    H_fwd = MicroPtycho.fresnel_propagator(KX, KY, +dz, lam)
    H_inv = MicroPtycho.fresnel_propagator(KX, KY, -dz, lam)
    assert np.max(np.abs(H_fwd * H_inv - 1.0)) < 1e-14


def test_fresnel_composition():
    """H(dz1) * H(dz2) = H(dz1 + dz2)."""
    N, dx, lam = 64, 0.5, 0.025
    kx = np.fft.fftfreq(N, dx)
    KX, KY = np.meshgrid(kx, kx)
    H1 = MicroPtycho.fresnel_propagator(KX, KY, 10.0, lam)
    H2 = MicroPtycho.fresnel_propagator(KX, KY, 17.5, lam)
    H3 = MicroPtycho.fresnel_propagator(KX, KY, 27.5, lam)
    assert np.max(np.abs(H1 * H2 - H3)) < 1e-13


def test_fresnel_preserves_total_intensity():
    """Parseval + |H|=1 ⇒ ∑|ψ|² is conserved across free-space propagation."""
    rng = np.random.default_rng(0)
    N, dx, lam, dz = 64, 0.43, 0.025, 30.0
    kx = np.fft.fftfreq(N, dx)
    KX, KY = np.meshgrid(kx, kx)
    H = MicroPtycho.fresnel_propagator(KX, KY, dz, lam)
    psi = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    psi_out = np.fft.ifft2(np.fft.fft2(psi) * H)
    before = np.sum(np.abs(psi) ** 2)
    after = np.sum(np.abs(psi_out) ** 2)
    assert after == pytest.approx(before, rel=1e-12)


# --------------------------- Transmission function -------------------------- #

def test_transmission_is_unit_modulus_for_real_V():
    """t(r) = exp(iσV) — for real σ,V the transmission has |t|=1 (no absorption)."""
    rng = np.random.default_rng(0)
    V = rng.normal(size=(32, 32)) * 30.0
    t = MicroPtycho.transmission_function_from_sigma(V, sigma=0.007288)
    assert np.max(np.abs(np.abs(t) - 1.0)) < 1e-14


def test_transmission_weak_phase_limit():
    """For small σV, t ≈ 1 + iσV + O((σV)²)."""
    V = np.linspace(-1.0, 1.0, 16)
    sigma = 1e-4
    t = MicroPtycho.transmission_function_from_sigma(V, sigma=sigma)
    approx = 1 + 1j * sigma * V
    # second-order error ~ (σV)²/2
    assert np.max(np.abs(t - approx)) < (sigma * V.max()) ** 2


# --------------------------- Multislice consistency ------------------------- #

def test_multislice_single_slice_matches_manual_TH():
    """propagate_wavefunction with 1 slice ≡ IFFT(FFT(ψ·T)·H)."""
    mp = MicroPtycho(N=64, dx=0.43, voltage=200e3)
    probe = mp.construct_probe(alpha=0.010)
    V = 0.02 * np.exp(-(mp.X ** 2 + mp.Y ** 2) / 8.0)
    mp.set_potentials(V[np.newaxis, ...])
    T = mp.transmission_function(V)
    H = mp.fresnel_propagator(mp.KX, mp.KY, dz=2.0, lam=mp.wavelength)
    manual = np.fft.ifft2(np.fft.fft2(probe * T) * H)
    auto = mp.propagate_wavefunction(probe, dz=2.0)
    assert np.max(np.abs(manual - auto)) < 1e-10


def test_multislice_multiple_slices_ordered_correctly():
    """For n slices the code should apply T₁ H T₂ H … Tₙ H in that order."""
    rng = np.random.default_rng(1)
    mp = MicroPtycho(N=64, dx=0.43, voltage=200e3)
    probe = mp.construct_probe(alpha=0.010)
    V_stack = rng.normal(size=(4, mp.N, mp.N)) * 0.1
    mp.set_potentials(V_stack)

    psi = probe.copy()
    H = mp.fresnel_propagator(mp.KX, mp.KY, dz=2.0, lam=mp.wavelength)
    for k in range(V_stack.shape[0]):
        psi = psi * mp.transmission_function(V_stack[k])
        psi = np.fft.ifft2(np.fft.fft2(psi) * H)
    auto = mp.propagate_wavefunction(probe, dz=2.0)
    assert np.max(np.abs(psi - auto)) < 1e-10


def test_zero_potential_multislice_is_pure_propagation():
    """With V=0 the stack is just n free-space propagations."""
    mp = MicroPtycho(N=64, dx=0.43, voltage=200e3)
    probe = mp.construct_probe(alpha=0.010)
    mp.set_potentials(np.zeros((3, mp.N, mp.N)))
    H = mp.fresnel_propagator(mp.KX, mp.KY, dz=2.0, lam=mp.wavelength)
    expected = probe.copy()
    for _ in range(3):
        expected = np.fft.ifft2(np.fft.fft2(expected) * H)
    auto = mp.propagate_wavefunction(probe, dz=2.0)
    assert np.max(np.abs(expected - auto)) < 1e-12


def test_unitarity_of_full_multislice():
    """The entire multislice operator is unitary (real V ⇒ |t|=1, free-space |H|=1),
    so the total probe energy is preserved."""
    rng = np.random.default_rng(2)
    mp = MicroPtycho(N=64, dx=0.43, voltage=200e3)
    probe = mp.construct_probe(alpha=0.010)
    mp.set_potentials(rng.normal(size=(5, mp.N, mp.N)) * 0.5)
    exit_wave = mp.propagate_wavefunction(probe, dz=2.0)
    assert np.sum(np.abs(exit_wave) ** 2) == pytest.approx(
        np.sum(np.abs(probe) ** 2), rel=1e-10
    )


# ------------------------------ FFT invariants ------------------------------ #

# -------------------------- Phase ramp remover ----------------------------- #

def test_remove_phase_ramp_is_idempotent():
    """Applying the ramp remover twice should be identical to once."""
    from microptycho import MicroPtycho as MP
    mp = MP(N=96, dx=0.43, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe = mp.extract_probe_patch(patch_size=24)
    once = MP._remove_phase_ramp(probe, dx=0.43)
    twice = MP._remove_phase_ramp(once, dx=0.43)
    assert np.max(np.abs(twice - once)) < 1e-12


def test_remove_phase_ramp_recovers_injected_ramp():
    """If we multiply a field by exp(i 2π q·r) and then run the ramp remover,
    we should get the original field back (up to a global phase)."""
    from microptycho import MicroPtycho as MP
    mp = MP(N=96, dx=0.43, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe = mp.extract_probe_patch(patch_size=24)

    N, dx = 24, 0.43
    xg = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xg, xg)
    kx, ky = 0.03, -0.05
    tilted = probe * np.exp(1j * 2 * np.pi * (kx * X + ky * Y))
    recovered = MP.align_global_phase(MP._remove_phase_ramp(tilted, dx=dx), probe)
    rel = np.max(np.abs(recovered - probe)) / np.max(np.abs(probe))
    assert rel < 1e-10


def test_remove_phase_ramp_is_noop_for_flat_phase():
    """A field with a flat phase (no ramp) should be returned unchanged
    apart from roundoff."""
    from microptycho import MicroPtycho as MP
    rng = np.random.default_rng(0)
    amp = np.abs(rng.normal(size=(32, 32))) + 0.1  # strictly positive, flat phase
    field = amp.astype(complex)
    out = MP._remove_phase_ramp(field, dx=0.5)
    assert np.max(np.abs(out - field)) < 1e-10


def test_parseval_theorem_numpy_convention():
    """Under numpy's (unnormalized forward FFT) convention:
       sum |ψ|² = (1/N²) sum |FFT(ψ)|²."""
    rng = np.random.default_rng(0)
    N = 32
    psi = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    P = np.fft.fft2(psi)
    lhs = np.sum(np.abs(psi) ** 2)
    rhs = np.sum(np.abs(P) ** 2) / (N * N)
    assert lhs == pytest.approx(rhs, rel=1e-12)
