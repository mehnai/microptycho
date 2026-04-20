"""Iterative ptychography reconstruction tests.

These tests verify that both the single-slice ePIE and the multislice ePIE
actually converge to the ground truth on noiseless synthetic data, and that
they obey the invariants every ptychography solver should obey:

  - Diffraction residual decreases monotonically (roughly) over iterations.
  - Final reconstructed diffraction intensity matches the measured intensity.
  - Reconstructed object inside the scanned region matches the ground truth
    up to the gauge freedoms (global phase for single-slice, plus per-slice
    phase ramps / global amplitude rescaling for multi-slice).

Reference implementation: Maiden & Rodenburg, Ultramicroscopy 109 (2009) 1256.
"""
import numpy as np
import pytest

from microptycho import MicroPtycho


# -------------- helpers --------------------------------------------------- #

def _scan_mask(shape, positions, dx, patch_size):
    mask = np.zeros(shape, dtype=bool)
    hp = patch_size // 2
    N = shape[0]
    for p in positions:
        ix = int(round(p[0] / dx)) + N // 2
        iy = int(round(p[1] / dx)) + N // 2
        mask[iy - hp:iy + hp, ix - hp:ix + hp] = True
    return mask


def _make_test_object(N, rng):
    x = np.arange(N) - N // 2
    X, Y = np.meshgrid(x, x)
    phase = (0.6 * np.exp(-((X - 10) ** 2 + (Y + 8) ** 2) / 100.0)
             - 0.4 * np.exp(-((X + 12) ** 2 + (Y - 6) ** 2) / 80.0))
    amp = 1.0 + 0.05 * np.exp(-((X - 5) ** 2 + (Y + 5) ** 2) / 200.0)
    return amp * np.exp(1j * phase)


# ------------------------ Single-slice ePIE -------------------------------- #

def test_ePIE_converges_on_synthetic_object():
    """50+ iterations should drive residual down >100× and reconstruct the
    object to <25 mrad mean phase error inside the scanned region."""
    rng = np.random.default_rng(1)
    N, dx, patch = 128, 0.43, 24
    mp = MicroPtycho(N=N, dx=dx, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe_patch = mp.extract_probe_patch(patch_size=patch)

    O_true = _make_test_object(N, rng)
    positions = MicroPtycho.construct_scan_positions(
        scan_margin=0, d_probe=2 * 0.86, overlap=0.6, scan_range=20
    )
    I = MicroPtycho.create_intensities(O_true, probe_patch, dx, positions, patch_size=patch)

    O_init = np.ones_like(O_true)
    probe_init = probe_patch.copy()
    probe_r, O_r, residuals = MicroPtycho.ePIE(
        60, probe_init, O_init, I, positions, dx, patch_size=patch,
        alpha=1.0, beta=0.0, verbose=False,
    )

    # 1) residual drops by >100×
    assert residuals[0] / residuals[-1] > 100

    # 2) reconstructed object is close to truth inside the scanned region
    mask = _scan_mask(O_true.shape, positions, dx, patch)
    aligned = MicroPtycho.align_global_phase(O_r, O_true)
    phase_err = np.abs(np.angle(aligned * np.conj(O_true)))[mask]
    assert phase_err.mean() < 0.025
    assert phase_err.max() < 0.05

    amp_err = np.abs(np.abs(aligned) - np.abs(O_true))[mask]
    assert amp_err.max() < 1e-2


def test_ePIE_forward_residuals_match_intensities_at_convergence():
    """After convergence, the reconstructed (object, probe) pair should
    produce diffraction intensities that match the measured ones."""
    rng = np.random.default_rng(2)
    N, dx, patch = 96, 0.43, 24
    mp = MicroPtycho(N=N, dx=dx, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe_patch = mp.extract_probe_patch(patch_size=patch)
    O_true = _make_test_object(N, rng)
    positions = MicroPtycho.construct_scan_positions(
        scan_margin=0, d_probe=2 * 0.86, overlap=0.6, scan_range=14
    )
    I = MicroPtycho.create_intensities(O_true, probe_patch, dx, positions, patch_size=patch)

    probe_r, O_r, _ = MicroPtycho.ePIE(
        80, probe_patch.copy(), np.ones_like(O_true), I, positions, dx,
        patch_size=patch, alpha=1.0, beta=0.0, verbose=False,
    )
    I_rec = MicroPtycho.create_intensities(O_r, probe_r, dx, positions, patch_size=patch)
    # normalised mean square intensity error
    rel = np.sum((I - I_rec) ** 2) / np.sum(I ** 2)
    assert rel < 1e-4


def test_ePIE_monotonic_residual_trend():
    """Residual is a stochastic ePIE estimate but should broadly decrease —
    the last-5 average should be well below the first-5 average."""
    rng = np.random.default_rng(3)
    N, dx, patch = 96, 0.43, 24
    mp = MicroPtycho(N=N, dx=dx, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe_patch = mp.extract_probe_patch(patch_size=patch)
    O_true = _make_test_object(N, rng)
    positions = MicroPtycho.construct_scan_positions(
        scan_margin=0, d_probe=2 * 0.86, overlap=0.5, scan_range=14
    )
    I = MicroPtycho.create_intensities(O_true, probe_patch, dx, positions, patch_size=patch)
    _, _, residuals = MicroPtycho.ePIE(
        30, probe_patch.copy(), np.ones_like(O_true), I, positions, dx,
        patch_size=patch, alpha=1.0, beta=0.0, verbose=False,
    )
    assert np.mean(residuals[-5:]) < 0.1 * np.mean(residuals[:5])


def test_ePIE_handles_fractional_scan_positions():
    """With sub-pixel scan steps, the reconstruction should still converge."""
    rng = np.random.default_rng(4)
    N, dx, patch = 96, 0.43, 24
    mp = MicroPtycho(N=N, dx=dx, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe_patch = mp.extract_probe_patch(patch_size=patch)
    O_true = _make_test_object(N, rng)
    # Build a scan with intentional fractional offsets
    step = 0.86
    grid1 = np.arange(-14, 14, step)
    X, Y = np.meshgrid(grid1, grid1)
    positions = np.column_stack([X.ravel() + 0.17, Y.ravel() - 0.23])
    I = MicroPtycho.create_intensities(O_true, probe_patch, dx, positions, patch_size=patch)
    probe_r, O_r, residuals = MicroPtycho.ePIE(
        40, probe_patch.copy(), np.ones_like(O_true), I, positions, dx,
        patch_size=patch, alpha=1.0, beta=0.0, verbose=False,
    )
    assert residuals[0] / residuals[-1] > 50


def test_ePIE_probe_recovery_with_noisy_initial_probe():
    """Joint probe + object recovery starting from a noisy probe.

    Probe recovery requires a strongly phase-textured object; a near-unity
    object is ambiguous because |FFT(P·O)|² ≈ |FFT(P)|² and many probes fit
    the same data. We use an object with ~1 rad phase structure so the probe
    is identifiable from the diffraction data.
    """
    rng = np.random.default_rng(5)
    N, dx, patch = 96, 0.43, 24
    mp = MicroPtycho(N=N, dx=dx, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe_patch = mp.extract_probe_patch(patch_size=patch)

    x = np.arange(N) - N // 2
    X, Y = np.meshgrid(x, x)
    phase = 1.2 * np.sin(X / 6.0) * np.cos(Y / 5.0)
    O_true = np.exp(1j * phase)

    positions = MicroPtycho.construct_scan_positions(
        scan_margin=0, d_probe=2 * 0.86, overlap=0.7, scan_range=14
    )
    I = MicroPtycho.create_intensities(O_true, probe_patch, dx, positions, patch_size=patch)

    # Noise scaled to the probe peak amplitude — otherwise random initial
    # noise drowns out the (very faint) cropped probe patch.
    probe_peak = np.sqrt(np.max(np.abs(probe_patch) ** 2))
    noise_level = 0.03 * probe_peak
    probe_init = probe_patch + noise_level * (
        rng.normal(size=probe_patch.shape) + 1j * rng.normal(size=probe_patch.shape)
    )
    probe_r, O_r, residuals = MicroPtycho.ePIE(
        120, probe_init, np.ones_like(O_true), I, positions, dx,
        patch_size=patch, alpha=1.0, beta=1.0, verbose=False,
    )
    assert residuals[0] / residuals[-1] > 100
    probe_aligned = MicroPtycho.align_phase_affine(probe_r, probe_patch, dx=dx)
    rel_err = np.sum(np.abs(probe_aligned - probe_patch) ** 2) / np.sum(np.abs(probe_patch) ** 2)
    assert rel_err < 0.05


# ------------------------ Multislice ePIE --------------------------------- #

def test_multislice_ePIE_single_slice_matches_single_slice_ePIE():
    """For K=1 slices with dz=0 (fresnel=identity) the multislice solver must
    reduce to the single-slice solver for the same problem."""
    rng = np.random.default_rng(6)
    N, dx, patch = 96, 0.43, 24
    mp = MicroPtycho(N=N, dx=dx, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe_patch = mp.extract_probe_patch(patch_size=patch)
    O_true = _make_test_object(N, rng)
    positions = MicroPtycho.construct_scan_positions(
        scan_margin=0, d_probe=2 * 0.86, overlap=0.6, scan_range=14
    )
    I = MicroPtycho.create_intensities(O_true, probe_patch, dx, positions, patch_size=patch)

    fresnel_id = mp.make_fresnel_kernel(shape=(patch, patch), dx=dx, dz=0.0)
    _, O_r, residuals = mp.multislice_ePIE(
        40, probe_patch.copy(), np.ones_like(O_true)[np.newaxis, ...], I, positions,
        fresnel_kernel=fresnel_id, dx=dx, patch_size=patch,
        alpha_0=1.0, beta_0=0.0, tau=1e9, random_seed=0, verbose=False,
    )
    assert residuals[0] / residuals[-1] > 50
    mask = _scan_mask(O_true.shape, positions, dx, patch)
    aligned = MicroPtycho.align_global_phase(O_r[0], O_true)
    phase_err = np.abs(np.angle(aligned * np.conj(O_true)))[mask]
    assert phase_err.mean() < 0.03


def test_multislice_ePIE_converges_on_2slice_object():
    """Two weakly-scattering slices separated by a Fresnel step.  Solver must
    recover the summed projected phase well (the per-slice decomposition has
    known gauge freedom, but the summed phase is invariant)."""
    rng = np.random.default_rng(7)
    N, dx, patch, dz = 96, 0.5, 24, 50.0
    mp = MicroPtycho(N=N, dx=dx, voltage=200e3)
    mp.construct_probe(alpha=0.015)
    probe_patch = mp.extract_probe_patch(patch_size=patch)

    x = np.arange(N) - N // 2
    X, Y = np.meshgrid(x, x)
    ph1 = 0.5 * np.exp(-((X - 4) ** 2 + (Y + 2) ** 2) / 40.0)
    ph2 = 0.3 * np.exp(-((X + 3) ** 2 + (Y - 2) ** 2) / 30.0)
    O_true = np.stack([np.exp(1j * ph1), np.exp(1j * ph2)])

    positions = MicroPtycho.construct_scan_positions(
        scan_margin=0, d_probe=2 * 0.6, overlap=0.7, scan_range=12
    )
    fresnel = mp.make_fresnel_kernel(shape=(patch, patch), dx=dx, dz=dz)
    I, _ = mp.create_multislice_intensities(O_true, probe_patch, dx, positions,
                                             fresnel_kernel=fresnel, patch_size=patch)

    _, O_r, residuals = mp.multislice_ePIE(
        120, probe_patch.copy(), np.ones_like(O_true), I, positions,
        fresnel_kernel=fresnel, dx=dx, patch_size=patch,
        alpha_0=0.1, beta_0=0.0, tau=20, random_seed=0, verbose=False,
    )
    assert residuals[0] / residuals[-1] > 10

    phi_true = np.angle(O_true[0]) + np.angle(O_true[1])
    phi_rec = np.angle(O_r[0]) + np.angle(O_r[1])
    mask = _scan_mask(O_true.shape[1:], positions, dx, patch)
    # remove global phase constant
    offset = np.angle(np.sum(np.exp(1j * (phi_rec - phi_true)) * mask))
    err = np.abs(((phi_rec - phi_true - offset + np.pi) % (2 * np.pi)) - np.pi)
    assert err[mask].mean() < 0.05


def test_multislice_ePIE_matches_intensities_at_convergence():
    rng = np.random.default_rng(8)
    N, dx, patch, dz = 96, 0.5, 24, 50.0
    mp = MicroPtycho(N=N, dx=dx, voltage=200e3)
    mp.construct_probe(alpha=0.015)
    probe_patch = mp.extract_probe_patch(patch_size=patch)
    x = np.arange(N) - N // 2
    X, Y = np.meshgrid(x, x)
    ph1 = 0.5 * np.exp(-(X ** 2 + Y ** 2) / 80.0)
    O_true = np.stack([np.exp(1j * ph1), np.exp(1j * 0.5 * ph1)])

    positions = MicroPtycho.construct_scan_positions(
        scan_margin=0, d_probe=2 * 0.6, overlap=0.6, scan_range=10
    )
    fresnel = mp.make_fresnel_kernel(shape=(patch, patch), dx=dx, dz=dz)
    I, _ = mp.create_multislice_intensities(O_true, probe_patch, dx, positions,
                                             fresnel_kernel=fresnel, patch_size=patch)

    probe_r, O_r, _ = mp.multislice_ePIE(
        150, probe_patch.copy(), np.ones_like(O_true), I, positions,
        fresnel_kernel=fresnel, dx=dx, patch_size=patch,
        alpha_0=0.1, beta_0=0.0, tau=20, random_seed=0, verbose=False,
    )
    I_rec, _ = mp.create_multislice_intensities(O_r, probe_r, dx, positions,
                                                 fresnel_kernel=fresnel, patch_size=patch)
    rel = np.sum((I - I_rec) ** 2) / np.sum(I ** 2)
    assert rel < 5e-3
