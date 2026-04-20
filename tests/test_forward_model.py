"""Forward model — diffraction intensity generation.

These tests cross-check `create_intensities` / `create_multislice_intensities`
against an independent reference implementation written from scratch following
Rodenburg & Maiden (2019, Springer chapter) and Kirkland (2nd ed, Ch. 6):

    I_j(q) = | FFT2{ P(r - r_j) · O(r) } |^2                    (single-slice)
    ψ_j  = P(r - r_j)
    for k in 1..K:  ψ_j <- IFFT2{ FFT2{ ψ_j · O_k(r) } · H(dz) }
    I_j(q) = | FFT2{ ψ_j } |^2                                   (multi-slice)
"""
import numpy as np
import pytest

from microptycho import MicroPtycho


# --------- Independent reference forward model (single-slice) -------------- #

def _reference_single_slice(obj, probe_patch, int_positions_px, patch_size):
    out = []
    hp = patch_size // 2
    N = obj.shape[0]
    c = N // 2
    for (ix, iy) in int_positions_px:
        patch = obj[c + iy - hp: c + iy + hp, c + ix - hp: c + ix + hp]
        psi = probe_patch * patch
        out.append(np.abs(np.fft.fft2(psi)) ** 2)
    return np.array(out)


def test_single_slice_forward_matches_reference():
    """Independent textbook formula agrees to numerical precision."""
    rng = np.random.default_rng(7)
    N, dx, patch = 96, 0.43, 24
    mp = MicroPtycho(N=N, dx=dx, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe_patch = mp.extract_probe_patch(patch_size=patch)

    obj = 1.0 + 0.2 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
    int_pos = [(0, 0), (3, 2), (-4, 1), (5, -3), (-2, -2)]
    positions = np.array([(ix * dx, iy * dx) for (ix, iy) in int_pos])

    I_mp = MicroPtycho.create_intensities(obj, probe_patch, dx, positions, patch_size=patch)
    I_ref = _reference_single_slice(obj, probe_patch, int_pos, patch)
    assert np.max(np.abs(I_mp - I_ref)) < 1e-10


def test_forward_intensity_is_nonnegative():
    rng = np.random.default_rng(0)
    mp = MicroPtycho(N=64, dx=0.43, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe_patch = mp.extract_probe_patch(patch_size=16)
    obj = rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
    I = MicroPtycho.create_intensities(obj, probe_patch, 0.43,
                                       np.array([[0, 0], [1, 2], [-1, 1]]),
                                       patch_size=16)
    assert I.min() >= 0


def test_forward_intensity_flat_object_is_shift_invariant():
    """With O ≡ 1 everywhere, diffraction intensity is a pure |FFT(probe_patch)|²
    and must be the same at every scan position (probe shift only changes phase)."""
    mp = MicroPtycho(N=64, dx=0.43, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe_patch = mp.extract_probe_patch(patch_size=16)
    obj = np.ones((64, 64), dtype=complex)
    positions = np.array([[0.0, 0.0], [1.3, -0.7], [2.5, 2.5]])
    I = MicroPtycho.create_intensities(obj, probe_patch, 0.43, positions, patch_size=16)
    for i in range(1, len(I)):
        assert np.max(np.abs(I[i] - I[0])) < 1e-9


def test_forward_energy_conservation_flat_object():
    """|P|² integrated = sum of diffraction intensity / N² (Parseval)."""
    mp = MicroPtycho(N=64, dx=0.43, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe_patch = mp.extract_probe_patch(patch_size=16)
    obj = np.ones((64, 64), dtype=complex)
    I = MicroPtycho.create_intensities(obj, probe_patch, 0.43,
                                       np.array([[0.0, 0.0]]), patch_size=16)
    Np = 16
    sum_real = np.sum(np.abs(probe_patch) ** 2)
    sum_diff = np.sum(I[0]) / (Np ** 2)
    assert sum_real == pytest.approx(sum_diff, rel=1e-12)


# --------- Independent reference for multislice forward -------------------- #

def _reference_multislice(O_stack, probe_patch, int_positions_px, fresnel_kernel, patch_size):
    out = []
    K = O_stack.shape[0]
    hp = patch_size // 2
    N = O_stack.shape[1]
    c = N // 2
    for (ix, iy) in int_positions_px:
        psi = probe_patch.copy()
        for k in range(K):
            patch = O_stack[k, c + iy - hp: c + iy + hp, c + ix - hp: c + ix + hp]
            psi = psi * patch
            psi = np.fft.ifft2(np.fft.fft2(psi) * fresnel_kernel)
        out.append(np.abs(np.fft.fft2(psi)) ** 2)
    return np.array(out)


def test_multislice_forward_matches_reference():
    rng = np.random.default_rng(3)
    N, dx, patch, dz = 96, 0.43, 24, 20.0
    mp = MicroPtycho(N=N, dx=dx, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe_patch = mp.extract_probe_patch(patch_size=patch)
    fresnel = mp.make_fresnel_kernel(shape=(patch, patch), dx=dx, dz=dz)

    O_stack = np.stack([
        np.exp(1j * 0.05 * rng.normal(size=(N, N))),
        np.exp(1j * 0.05 * rng.normal(size=(N, N))),
        np.exp(1j * 0.05 * rng.normal(size=(N, N))),
    ])
    int_pos = [(0, 0), (2, -1), (-3, 2)]
    positions = np.array([(ix * dx, iy * dx) for (ix, iy) in int_pos])
    I_mp, _ = mp.create_multislice_intensities(O_stack, probe_patch, dx, positions,
                                                fresnel_kernel=fresnel, patch_size=patch)
    I_ref = _reference_multislice(O_stack, probe_patch, int_pos, fresnel, patch)
    assert np.max(np.abs(I_mp - I_ref)) < 1e-10


def test_multislice_with_trivial_objects_equals_free_space():
    """If every slice has O=1 (no scattering), the multislice forward model
    gives the same intensity as a single free-space propagation stack."""
    N, dx, patch, dz = 96, 0.43, 24, 20.0
    mp = MicroPtycho(N=N, dx=dx, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe_patch = mp.extract_probe_patch(patch_size=patch)
    fresnel = mp.make_fresnel_kernel(shape=(patch, patch), dx=dx, dz=dz)

    K = 4
    O_stack = np.ones((K, N, N), dtype=complex)
    pos = np.array([[0.0, 0.0]])
    I, _ = mp.create_multislice_intensities(O_stack, probe_patch, dx, pos,
                                            fresnel_kernel=fresnel, patch_size=patch)
    # Expected: same probe, propagated K times in free space, then |FFT|²
    psi = probe_patch.copy()
    for _ in range(K):
        psi = np.fft.ifft2(np.fft.fft2(psi) * fresnel)
    I_expected = np.abs(np.fft.fft2(psi)) ** 2
    assert np.max(np.abs(I[0] - I_expected)) < 1e-10


def test_multislice_collapses_to_single_slice_at_zero_dz():
    """When dz = 0 the Fresnel kernel is identity, so a K-slice multislice
    forward is equivalent to the single-slice forward with O_eff = ∏ O_k."""
    rng = np.random.default_rng(9)
    N, dx, patch = 96, 0.43, 24
    mp = MicroPtycho(N=N, dx=dx, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe_patch = mp.extract_probe_patch(patch_size=patch)
    fresnel_zero = mp.make_fresnel_kernel(shape=(patch, patch), dx=dx, dz=0.0)
    assert np.allclose(fresnel_zero, 1.0)

    K = 3
    O_stack = np.exp(1j * 0.1 * rng.normal(size=(K, N, N)))
    O_prod = np.prod(O_stack, axis=0)
    pos = np.array([[0.0, 0.0], [1.3, -0.9]])

    I_multi, _ = mp.create_multislice_intensities(
        O_stack, probe_patch, dx, pos, fresnel_kernel=fresnel_zero, patch_size=patch
    )
    I_single = MicroPtycho.create_intensities(O_prod, probe_patch, dx, pos, patch_size=patch)
    assert np.max(np.abs(I_multi - I_single)) < 1e-10


def test_probe_shape_mismatch_raises():
    mp = MicroPtycho(N=64, dx=0.43, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    bad = np.ones((20, 20), dtype=complex)
    with pytest.raises(ValueError):
        MicroPtycho.create_intensities(np.ones((64, 64), dtype=complex),
                                       bad, 0.43, np.array([[0.0, 0.0]]), patch_size=16)
