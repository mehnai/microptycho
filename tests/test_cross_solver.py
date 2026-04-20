"""Cross-validation against independent ptychography solvers.

We can't pull in PyNX, ptypy, py4DSTEM, etc., in an offline environment.
Instead we implement two reference algorithms from scratch directly from the
primary literature and check that MicroPtycho's ePIE gives consistent answers:

  1. Vanilla Maiden & Rodenburg ePIE  (Ultramicroscopy 109, 1256 (2009))
     — the textbook formulation *without* the (1-ρ)+ρ regulariser used
     internally by MicroPtycho.  Both solvers must converge to the same
     object up to the global-phase gauge.

  2. Difference Map (Thibault et al., Science 321, 379 (2008))
     — a structurally different (non-gradient, projection-based) algorithm.
     On a clean problem the DM fixed point and ePIE fixed point have to
     agree to within gauge freedom.

This is the closest thing to an apples-to-apples comparison we can do
without external deps, and it's a genuine independent check: the two
reference algorithms share no code with MicroPtycho.
"""
import numpy as np
import pytest

from microptycho import MicroPtycho


# -------------------------------------------------------------------------- #
#  Reference implementations (written from scratch against the papers)       #
# -------------------------------------------------------------------------- #

def _crop_patch(obj, pos, dx, patch_size):
    """Extract the object patch with the same integer-rounding convention
    used by MicroPtycho (so the two solvers see exactly the same data)."""
    N = obj.shape[0]
    ix = int(round(pos[0] / dx))
    iy = int(round(pos[1] / dx))
    hp = patch_size // 2
    c = N // 2
    return (slice(c + iy - hp, c + iy + hp),
            slice(c + ix - hp, c + ix + hp),
            (pos[0] / dx - ix) * dx,
            (pos[1] / dx - iy) * dx)


def _shift_probe(probe, shift_x, shift_y, dx):
    if shift_x == 0.0 and shift_y == 0.0:
        return probe
    kx = np.fft.fftfreq(probe.shape[1], dx)
    ky = np.fft.fftfreq(probe.shape[0], dx)
    KX, KY = np.meshgrid(kx, ky)
    ramp = np.exp(-1j * 2 * np.pi * (shift_x * KX + shift_y * KY))
    return np.fft.ifft2(np.fft.fft2(probe) * ramp)


def reference_ePIE(n_iter, probe, obj, intensity, positions, dx, patch_size,
                   alpha=1.0, beta=0.0, seed=0):
    """Canonical Maiden & Rodenburg 2009 ePIE — no (1-ρ)+ρ regulariser."""
    rng = np.random.default_rng(seed)
    amp = np.sqrt(np.maximum(intensity, 0.0))
    idx = np.arange(len(positions))
    residuals = []
    eps = 1e-12
    for _ in range(n_iter):
        rng.shuffle(idx)
        res = 0.0
        for j in idx:
            ys, xs, fx, fy = _crop_patch(obj, positions[j], dx, patch_size)
            patch = obj[ys, xs]
            pr = _shift_probe(probe, fx, fy, dx)
            psi = pr * patch
            Psi = np.fft.fft2(psi)
            res += np.sum((amp[j] - np.abs(Psi)) ** 2)
            Psi_new = amp[j] * Psi / (np.abs(Psi) + 1e-8)
            delta = np.fft.ifft2(Psi_new) - psi

            max_pr2 = np.max(np.abs(pr) ** 2) + eps
            obj[ys, xs] = patch + alpha * np.conj(pr) / max_pr2 * delta
            if beta != 0:
                max_pa2 = np.max(np.abs(patch) ** 2) + eps
                probe_update = beta * np.conj(patch) / max_pa2 * delta
                probe += _shift_probe(probe_update, -fx, -fy, dx)
        residuals.append(res)
    return probe, obj, residuals


def reference_DM(n_iter, probe, obj, intensity, positions, dx, patch_size, seed=0):
    """Difference Map for ptychography (Thibault 2008, Thibault 2009).

    State variable: z[j] = exit wave at position j.
    π_F = Fourier projection (enforce measured amplitude).
    π_R = overlap projection (enforce z_j = P(r-r_j) · O(r) for all j
          simultaneously by least-squares reassembly of O from {z_j}).

    DM update (Thibault 2008 notation, β=1):
        z = z + π_F(2·π_R(z) - z) - π_R(z)
    """
    rng = np.random.default_rng(seed)
    amp = np.sqrt(np.maximum(intensity, 0.0))
    J = len(positions)
    # Initial exit waves
    z = np.zeros((J, patch_size, patch_size), dtype=complex)
    slices = []
    shifted_probes = []
    for j in range(J):
        ys, xs, fx, fy = _crop_patch(obj, positions[j], dx, patch_size)
        slices.append((ys, xs))
        pr = _shift_probe(probe, fx, fy, dx)
        shifted_probes.append(pr)
        z[j] = pr * obj[ys, xs]

    eps = 1e-8
    residuals = []

    for _ in range(n_iter):
        # π_R: reassemble O from current z, keep probe fixed.
        num = np.zeros_like(obj)
        den = np.zeros(obj.shape)
        for j in range(J):
            ys, xs = slices[j]
            pr = shifted_probes[j]
            num[ys, xs] += np.conj(pr) * z[j]
            den[ys, xs] += np.abs(pr) ** 2
        obj_new = num / (den + 1e-12)
        # compute π_R(z) = P · π_R_obj
        pi_R = np.empty_like(z)
        for j in range(J):
            ys, xs = slices[j]
            pi_R[j] = shifted_probes[j] * obj_new[ys, xs]

        # π_F on (2·π_R - z)
        y = 2 * pi_R - z
        pi_F = np.empty_like(y)
        res = 0.0
        for j in range(J):
            Y = np.fft.fft2(y[j])
            Z = np.fft.fft2(z[j])
            res += np.sum((amp[j] - np.abs(Z)) ** 2)
            pi_F[j] = np.fft.ifft2(amp[j] * Y / (np.abs(Y) + eps))
        z = z + pi_F - pi_R
        residuals.append(res)
        obj = obj_new  # carry forward for final reporting

    return probe, obj, residuals


# -------------------------------------------------------------------------- #
#  Helpers                                                                   #
# -------------------------------------------------------------------------- #

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


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(42)
    N, dx, patch = 96, 0.43, 24
    mp = MicroPtycho(N=N, dx=dx, voltage=200e3)
    mp.construct_probe(alpha=0.012)
    probe_patch = mp.extract_probe_patch(patch_size=patch)

    O_true = _make_test_object(N, rng)
    positions = MicroPtycho.construct_scan_positions(
        scan_margin=0, d_probe=2 * 0.86, overlap=0.6, scan_range=14
    )
    I = MicroPtycho.create_intensities(O_true, probe_patch, dx,
                                       positions, patch_size=patch)
    return dict(N=N, dx=dx, patch=patch, probe=probe_patch,
                O_true=O_true, positions=positions, I=I)


# -------------------------------------------------------------------------- #
#  Tests                                                                     #
# -------------------------------------------------------------------------- #

def test_microptycho_ePIE_matches_reference_ePIE(problem):
    """MicroPtycho.ePIE and the textbook ePIE must reconstruct the same
    object up to a global phase."""
    N, dx, patch = problem["N"], problem["dx"], problem["patch"]
    probe = problem["probe"]
    I, positions, O_true = problem["I"], problem["positions"], problem["O_true"]

    # MicroPtycho solver
    _, O_mp, _ = MicroPtycho.ePIE(
        60, probe.copy(), np.ones_like(O_true), I, positions, dx,
        patch_size=patch, alpha=1.0, beta=0.0, verbose=False,
    )
    # Reference textbook ePIE
    _, O_ref, _ = reference_ePIE(
        60, probe.copy(), np.ones_like(O_true), I, positions, dx, patch,
        alpha=1.0, beta=0.0, seed=0,
    )

    mask = _scan_mask(O_true.shape, positions, dx, patch)
    # align reference to truth (global phase), then align MicroPtycho to reference
    O_ref_a = MicroPtycho.align_global_phase(O_ref, O_true)
    O_mp_a  = MicroPtycho.align_global_phase(O_mp, O_ref_a)
    diff = np.abs(O_mp_a - O_ref_a)[mask]
    assert diff.mean() < 0.02
    assert diff.max() < 0.1


def test_microptycho_ePIE_matches_reference_DM(problem):
    """MicroPtycho.ePIE (gradient-based) and Difference Map (projection-based)
    — two algorithmically distinct solvers — should agree on the answer."""
    N, dx, patch = problem["N"], problem["dx"], problem["patch"]
    probe = problem["probe"]
    I, positions, O_true = problem["I"], problem["positions"], problem["O_true"]

    _, O_mp, _ = MicroPtycho.ePIE(
        80, probe.copy(), np.ones_like(O_true), I, positions, dx,
        patch_size=patch, alpha=1.0, beta=0.0, verbose=False,
    )
    _, O_dm, _ = reference_DM(
        80, probe.copy(), np.ones_like(O_true), I, positions, dx, patch,
    )

    mask = _scan_mask(O_true.shape, positions, dx, patch)
    O_mp_a = MicroPtycho.align_global_phase(O_mp, O_true)
    O_dm_a = MicroPtycho.align_global_phase(O_dm, O_true)
    diff_mp = np.abs(np.angle(O_mp_a * np.conj(O_true)))[mask]
    diff_dm = np.abs(np.angle(O_dm_a * np.conj(O_true)))[mask]
    # both solvers should converge to <50 mrad mean phase error
    assert diff_mp.mean() < 0.05
    assert diff_dm.mean() < 0.05
    # and they should agree with each other
    cross = np.abs(np.angle(O_mp_a * np.conj(O_dm_a)))[mask]
    assert cross.mean() < 0.05


def test_solvers_agree_on_final_intensity(problem):
    """Forward-propagating each solver's reconstructed object should
    reproduce the measured intensity — a basic consistency check that
    any correct ptychography solver must pass."""
    N, dx, patch = problem["N"], problem["dx"], problem["patch"]
    probe = problem["probe"]
    I, positions, O_true = problem["I"], problem["positions"], problem["O_true"]

    _, O_mp, _ = MicroPtycho.ePIE(
        80, probe.copy(), np.ones_like(O_true), I, positions, dx,
        patch_size=patch, alpha=1.0, beta=0.0, verbose=False,
    )
    _, O_ref, _ = reference_ePIE(
        80, probe.copy(), np.ones_like(O_true), I, positions, dx, patch,
        alpha=1.0, beta=0.0, seed=0,
    )
    _, O_dm, _ = reference_DM(
        80, probe.copy(), np.ones_like(O_true), I, positions, dx, patch,
    )
    for name, O in [("MicroPtycho", O_mp), ("ref_ePIE", O_ref), ("DM", O_dm)]:
        I_rec = MicroPtycho.create_intensities(O, probe, dx, positions,
                                               patch_size=patch)
        rel = np.sum((I - I_rec) ** 2) / np.sum(I ** 2)
        assert rel < 1e-3, f"{name}: relative intensity error {rel:.2e} too high"


def test_ePIE_converges_from_random_initialisation(problem):
    """Good ptychography algorithms should escape random initial conditions.
    Verify MicroPtycho does — if it can only converge from ground truth, that
    would indicate the forward model / algorithm is secretly baking in the
    answer."""
    N, dx, patch = problem["N"], problem["dx"], problem["patch"]
    probe = problem["probe"]
    I, positions, O_true = problem["I"], problem["positions"], problem["O_true"]
    rng = np.random.default_rng(123)

    random_obj = np.ones_like(O_true) * (1 + 0.1 * (
        rng.normal(size=O_true.shape) + 1j * rng.normal(size=O_true.shape)
    ))
    _, O_r, residuals = MicroPtycho.ePIE(
        120, probe.copy(), random_obj, I, positions, dx,
        patch_size=patch, alpha=1.0, beta=0.0, verbose=False,
    )
    assert residuals[0] / residuals[-1] > 50
    mask = _scan_mask(O_true.shape, positions, dx, patch)
    aligned = MicroPtycho.align_global_phase(O_r, O_true)
    err = np.abs(np.angle(aligned * np.conj(O_true)))[mask]
    assert err.mean() < 0.05
