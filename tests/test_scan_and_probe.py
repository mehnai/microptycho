"""Scan position geometry, probe construction, and shift invariants."""
import numpy as np
import pytest

from microptycho import MicroPtycho


# ---------------------- Grid and coordinate conventions --------------------- #

def test_real_space_grid_is_centered():
    N, dx = 32, 0.5
    X, Y, KX, KY = MicroPtycho.make_grid(N, dx)
    # centered at 0 to within dx
    assert X.min() == -N // 2 * dx
    assert X.max() == (N // 2 - 1) * dx
    # row index ↔ y (varies along axis 0); col index ↔ x
    assert np.all(X[:, 0] == X[0, 0])   # x constant down a column
    assert np.all(Y[0, :] == Y[0, 0])   # y constant along a row


def test_k_grid_uses_fftfreq():
    N, dx = 16, 0.5
    _, _, KX, KY = MicroPtycho.make_grid(N, dx)
    kx_expected = np.fft.fftfreq(N, dx)
    assert np.allclose(KX[0, :], kx_expected)
    assert np.allclose(KY[:, 0], kx_expected)


# ------------------------- Probe aperture behaviour ------------------------- #

def test_probe_energy_preserved_under_ifft():
    """Ideal top-hat aperture: sum |probe|² = (sum aperture²) / N²."""
    mp = MicroPtycho(N=64, dx=0.43, voltage=200e3)
    probe = mp.construct_probe(alpha=0.010)
    k = np.sqrt(mp.KX ** 2 + mp.KY ** 2)
    kmax = 0.010 / mp.wavelength
    aperture = (k <= kmax).astype(float)
    expected = np.sum(aperture ** 2) / (mp.N ** 2)
    assert np.sum(np.abs(probe) ** 2) == pytest.approx(expected, rel=1e-12)


def test_probe_is_diffraction_limited():
    """Aperture radius → real-space Airy disk radius.  r_airy ≈ 0.61 λ/α.
    Full probe intensity must be well-localized within ~5× Airy radius."""
    mp = MicroPtycho(N=256, dx=0.43, voltage=200e3)
    probe = mp.construct_probe(alpha=0.010)
    intensity = np.abs(np.fft.fftshift(probe)) ** 2
    # Centroid of intensity should be at centre pixel
    N = mp.N
    xgrid = (np.arange(N) - N // 2) * mp.dx
    X, Y = np.meshgrid(xgrid, xgrid)
    mx = np.sum(X * intensity) / np.sum(intensity)
    my = np.sum(Y * intensity) / np.sum(intensity)
    # Even-N grids have an asymmetric frequency set (includes -Nyquist, excludes +Nyquist),
    # which shifts the centroid by at most ~dx/N ≈ milli-pixel.  Accept up to 0.1 × dx.
    assert abs(mx) < 0.1 * mp.dx
    assert abs(my) < 0.1 * mp.dx
    # 90% of energy within 5× Airy radius (~10 Å at 200 kV, α=10 mrad)
    r_airy = 0.61 * mp.wavelength / 0.010
    mask = X ** 2 + Y ** 2 < (5 * r_airy) ** 2
    assert np.sum(intensity * mask) / np.sum(intensity) > 0.85


# --------------------- Scan position / patch geometry ---------------------- #

def test_patch_slices_center():
    """A position (0,0) picks out the centre patch of an even-sized array."""
    ys, xs = MicroPtycho._patch_slices((64, 64), 0, 0, 16)
    assert ys == slice(24, 40)
    assert xs == slice(24, 40)


def test_patch_slices_out_of_bounds_raises():
    with pytest.raises(ValueError):
        MicroPtycho._patch_slices((32, 32), 20, 0, 16)


def test_patch_size_must_be_positive_even():
    for bad in (0, -4, 5, 7):
        with pytest.raises(ValueError):
            MicroPtycho._validate_patch_size(bad)


def test_fractional_shift_geometry_matches_manual_shift():
    """Forward-model intensity at a fractional-pixel position should equal
    the intensity computed by shifting the probe by the same amount."""
    rng = np.random.default_rng(0)
    N, dx, patch = 128, 0.43, 32
    mp = MicroPtycho(N=N, dx=dx, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe_patch = mp.extract_probe_patch(patch_size=patch)

    obj = 1.0 + 0.1 * (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N)))
    pos = np.array([[2.4, -1.9]])
    I_ref = MicroPtycho.create_intensities(obj, probe_patch, dx, pos, patch_size=patch)

    # Manual: extract integer-centre patch, shift the probe by the fractional amount
    ix = int(np.round(pos[0, 0] / dx)); iy = int(np.round(pos[0, 1] / dx))
    hp = patch // 2
    c = N // 2
    patch_obj = obj[c + iy - hp:c + iy + hp, c + ix - hp:c + ix + hp]
    frac_x = (pos[0, 0] / dx - ix) * dx
    frac_y = (pos[0, 1] / dx - iy) * dx
    probe_KX, probe_KY = MicroPtycho.make_k_grid(probe_patch.shape, dx)
    pr = MicroPtycho._shift_field(probe_patch, frac_x, frac_y, dx, probe_KX, probe_KY)
    I_manual = np.abs(np.fft.fft2(pr * patch_obj)) ** 2

    assert np.max(np.abs(I_ref[0] - I_manual)) < 1e-10


def test_shift_field_by_integer_pixel_is_circular_roll():
    """shifting by exactly n·dx in each direction should equal np.roll."""
    rng = np.random.default_rng(0)
    N, dx = 32, 0.5
    field = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    KX, KY = MicroPtycho.make_k_grid((N, N), dx)
    shifted = MicroPtycho._shift_field(field, 3 * dx, -2 * dx, dx, KX, KY)
    # shift_x = +3·dx ⇒ field moves to the right by 3 columns ⇒ np.roll axis=1 by +3
    expected = np.roll(field, shift=(-2, 3), axis=(0, 1))
    assert np.max(np.abs(shifted - expected)) < 1e-10


def test_shift_field_identity_when_zero():
    rng = np.random.default_rng(0)
    N, dx = 16, 0.5
    field = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    out = MicroPtycho._shift_field(field, 0.0, 0.0, dx)
    assert np.array_equal(field, out)


# ----------------------------- Scan construction --------------------------- #

def test_construct_scan_positions_overlap_and_range():
    """Step size = d_probe*(1-overlap); positions stay inside scan_range."""
    pos = MicroPtycho.construct_scan_positions(
        scan_margin=2, d_probe=2 * 0.86, overlap=0.4, scan_range=20
    )
    step = 2 * 0.86 * (1 - 0.4)
    # all positions within (margin-range, range-margin)
    assert pos[:, 0].min() >= -20 + 2 - 1e-9
    assert pos[:, 0].max() <=  20 - 2 + step
    # Step size reflected in x-difference between adjacent x-values
    xs = np.unique(np.round(pos[:, 0], 6))
    deltas = np.diff(xs)
    assert np.allclose(deltas, step, rtol=1e-6)


def test_probe_patch_extraction_shape():
    mp = MicroPtycho(N=64, dx=0.43, voltage=200e3)
    mp.construct_probe(alpha=0.010)
    patch = mp.extract_probe_patch(patch_size=24)
    assert patch.shape == (24, 24)


def test_probe_patch_extraction_requires_probe():
    mp = MicroPtycho(N=64, dx=0.43)
    with pytest.raises(ValueError):
        mp.extract_probe_patch(patch_size=24)
