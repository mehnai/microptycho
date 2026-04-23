"""
Ptychography with Microscope Aberrations
=========================================
Full simulation pipeline:
  0. electron_optics  — relativistic beam physics
  1. Load potentials  — center-crop Z1.npy slices
  2. Microscope       — define electron optics with aberrations
  3. Multislice       — propagate probe through crystal
  4. Scanning         — generate diffraction patterns
  5. ePIE             — reconstruct object and probe
  6. CTF imaging      — simulate conventional HRTEM image

All figures are saved to the 'testrun/' folder.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no windows, just files
import matplotlib.pyplot as plt

from electron_optics import wavelength_from_voltage, interaction_constant_from_voltage
from crystalmaker import CrystalMaker
from microptycho import MicroPtycho
from microscope import Microscope

os.makedirs("testrun", exist_ok=True)


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}", flush=True)

def step(msg):
    print(f"  >> {msg}", flush=True)

def save(name):
    path = os.path.join("testrun", name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f"     saved → {path}", flush=True)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Ptychography Tutorial")
print("=" * 60)
print("  Output folder: testrun/")

rng = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# High-level demo knobs. The derived quantities below (a_lat, sigma,
# scan_range, patch_size, dz) are all computed from these so the demo scales
# cleanly from a *sparse* 36-atom sample to a dense thousands-of-atoms sample.
# ---------------------------------------------------------------------------
DEMO = {
    "N": 256,               # simulation grid size (pixels)
    "dx": 0.43,             # Å per pixel
    "n_iter": 300,          # ePIE iterations (joint probe+object)
    "voltage": 200e3,       # beam voltage (V)
    # --- sample knobs ------------------------------------------------------
    "nx": 6,                # unit cells along x
    "ny": 6,                # unit cells along y
    "nz": 1,                # unit cells along z (=> n_slices for sc)
    "structure": "sc",      # 'sc' | 'bcc' | 'fcc' | 'diamond'
    "Z1": 31,               # atomic number (amplitude weight)
    "rotation_deg": 45.0,   # in-plane lattice rotation (0 = axis-aligned)
    # --- derived knobs (None → auto) --------------------------------------
    "sample_fill": 0.7,     # fraction of FOV the sample should span
    "sigma_frac": 0.30,     # sigma / in-plane spacing (atom visual size)
    "scan_fill": 0.8,       # fraction of sample span the scan covers
    "patch_min_A": 20.0,    # minimum probe-patch size in Å
}

# ---- Structures: atoms/unit-cell and min in-plane spacing as fraction of a
#      (used to compute a sensible Gaussian sigma and a_lat from the knobs).
ATOMS_PER_CELL = {"sc": 1, "bcc": 2, "fcc": 4, "diamond": 8}
MIN_IN_PLANE_SPACING_FRAC = {"sc": 1.0, "bcc": 0.707, "fcc": 0.5, "diamond": 0.354}
# How many atom rows a cell contributes along a crystal axis in [001] projection
ROWS_PER_CELL_AXIS = {"sc": 1, "bcc": 1, "fcc": 2, "diamond": 4}

# ---- Derive everything from the knobs --------------------------------------
_structure = DEMO["structure"]
_fov = DEMO["N"] * DEMO["dx"]
_n_atoms_side = max(DEMO["nx"], DEMO["ny"]) * ROWS_PER_CELL_AXIS[_structure]

# Rotation factor: when the lattice is rotated by θ, the axis-aligned
# bounding box of the atom positions grows by (|cos θ| + |sin θ|) along
# each axis (max at θ=45° → √2). Account for this in the fitting so
# rotated lattices still fit inside the scan-safe region.
_rot_rad = np.deg2rad(DEMO.get("rotation_deg", 0.0))
_rot_scale = abs(np.cos(_rot_rad)) + abs(np.sin(_rot_rad))
_n_atoms_eff = 1 + (_n_atoms_side - 1) * _rot_scale  # effective side for fitting

# Solve jointly so sample-span + probe-patch ≤ margin·FOV. The probe patch
# is set to ~2× the in-plane atom spacing (so each probe position touches
# several atoms) but no smaller than `patch_min_A`. `margin` is kept
# conservative (0.8) so the outer atoms sit well inside the scan-safe
# region — near the scan boundary corners get few probe placements and
# reconstruct weakly.
_margin = 0.8
_patch_min = DEMO["patch_min_A"]
# Case A: patch clamps to 2·spacing → (n_eff+1)·spacing ≤ margin·FOV
_spacing_A = _margin * _fov / (_n_atoms_eff + 1)
if 2 * _spacing_A >= _patch_min:
    _in_plane_spacing = _spacing_A
    _patch_A = 2 * _spacing_A
else:
    # Case B: patch clamps to patch_min → (n_eff-1)·spacing = margin·FOV - patch_min
    _in_plane_spacing = max(
        (_margin * _fov - _patch_min) / max(_n_atoms_eff - 1, 1),
        0.5,
    )
    _patch_A = _patch_min

a_lat = _in_plane_spacing / MIN_IN_PLANE_SPACING_FRAC[_structure]
atom_sigma = DEMO["sigma_frac"] * _in_plane_spacing

# Slice spacing: one slice per atomic layer in z (thin enough vs λ/α² depth
# resolution so multislice and single-slice agree for dz ≪ λ/α²).
dz = a_lat
n_slices_expected = DEMO["nz"]
n_atoms_total = DEMO["nx"] * DEMO["ny"] * DEMO["nz"] * ATOMS_PER_CELL[_structure]

patch_size = int(np.ceil(_patch_A / DEMO["dx"] / 2.0) * 2)

# Scan range: cover the atoms with margin, but stay inside the
# patch-safe region (|scan_pos| + patch_half ≤ N·dx/2). Rotated
# lattice bounding box grows by `_rot_scale` along each axis. Pad by
# an extra full spacing so the outermost atoms get scan positions on
# BOTH sides (otherwise they're under-sampled and reconstruct weakly).
_sample_half_span = (_n_atoms_side - 1) / 2.0 * _in_plane_spacing * _rot_scale
_scan_max = (DEMO["N"] / 2 - patch_size / 2) * DEMO["dx"]
scan_range = min(
    _sample_half_span + 1.5 * _in_plane_spacing,
    0.98 * _scan_max,
)

# Scan step: 1/6 the in-plane spacing. For sparse samples dense scans are
# crucial — each probe position only sees atoms strongly when it lands
# near one, so oversample positions to cover every atom with several
# probe placements. Denser scanning also improves probe/object
# disambiguation in the sparse regime.
scan_step = max(_in_plane_spacing / 6.0, 2 * DEMO["dx"])

print(f"  Config: N={DEMO['N']}, dx={DEMO['dx']} Å, FOV={_fov:.1f} Å")
print(f"  Sample: {DEMO['nx']}×{DEMO['ny']}×{DEMO['nz']} {_structure} cells "
      f"→ {n_atoms_total} atoms, rotated {DEMO.get('rotation_deg', 0.0):.1f}°")
print(f"  Derived: a_lat={a_lat:.2f} Å, spacing={_in_plane_spacing:.2f} Å, "
      f"σ={atom_sigma:.2f} Å, dz={dz:.2f} Å")
print(f"           patch_size={patch_size}px ({patch_size*DEMO['dx']:.1f} Å), "
      f"scan_range=±{scan_range:.1f} Å, step={scan_step:.2f} Å")
print(f"  ePIE iterations: {DEMO['n_iter']}")


# ---------------------------------------------------------------------------
# Module 0: electron_optics — beam physics
# ---------------------------------------------------------------------------
section("Module 0: electron_optics — beam physics")
step("Computing wavelength and interaction constant at common voltages...")

voltages = [60e3, 100e3, 200e3, 300e3]
print(f"\n  {'Voltage':>10}  {'λ (pm)':>8}  {'σ (V⁻¹Å⁻²)':>14}  {'σ×10 (V_nm)':>14}")
print("  " + "-" * 55)
for V in voltages:
    lam = wavelength_from_voltage(V)
    sigma_A = interaction_constant_from_voltage(V, potential_units='V_A')
    sigma_nm = interaction_constant_from_voltage(V, potential_units='V_nm')
    print(f"  {V/1e3:>9.0f}kV  {lam*100:>7.4f}pm  {sigma_A:>14.6f}  {sigma_nm:>14.6f}")
print()
print("  Note: MicroPtycho uses V_nm units by default.")


# ---------------------------------------------------------------------------
# Step 1: Build crystal and generate projected potentials
# ---------------------------------------------------------------------------
section(f"Step 1: Build {_structure} crystal ({n_atoms_total} atoms) "
        f"and generate projected potentials")
step(f"Tiling {_structure} supercell (a={a_lat:.2f} Å)...")

cm = CrystalMaker(lattice_constant=a_lat, Z1=DEMO["Z1"], structure=_structure)
cm.tile(nx=DEMO["nx"], ny=DEMO["ny"], nz=DEMO["nz"])
if DEMO.get("rotation_deg", 0.0) != 0.0:
    cm.rotate_xy(DEMO["rotation_deg"])
print(f"  Supercell: {cm.supercell.shape[0]} atoms  |  "
      f"lattice constant: {cm.lattice_constant} Å ({_structure}) "
      f"| rotation: {DEMO.get('rotation_deg', 0.0):.1f}°")

step("Initialising simulation grid...")
mp = MicroPtycho(N=DEMO["N"], dx=DEMO["dx"], voltage=DEMO["voltage"])
print(f"  Grid: {mp.N}x{mp.N}, pixel size = {mp.dx} Å")

step(f"Projecting atoms onto slices (dz={dz:.2f} Å, σ={atom_sigma:.2f} Å)...")
V = cm.create_potentials(mp.X, mp.Y, dz=dz, sigma=atom_sigma)
mp.set_potentials(V)
print(f"  Potentials: {V.shape[0]} slices of {V.shape[1]}x{V.shape[2]} px")
print(f"  Value range: [{V.min():.4f}, {V.max():.4f}]")

step("Plotting projected potentials...")
n_slices = V.shape[0]
slices_to_show = [0, n_slices // 4, n_slices // 2, 3 * n_slices // 4]

fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
extent = [mp.X.min(), mp.X.max(), mp.Y.min(), mp.Y.max()]

for i, s in enumerate(slices_to_show):
    im = axes[i].imshow(V[s], cmap='inferno', extent=extent)
    axes[i].set_title(f'Slice {s}')
    axes[i].set_xlabel('x (Å)')
    if i == 0:
        axes[i].set_ylabel('y (Å)')

im = axes[4].imshow(V.sum(axis=0), cmap='inferno', extent=extent)
axes[4].set_title('Total projection')
axes[4].set_xlabel('x (Å)')
plt.colorbar(im, ax=axes[4], shrink=0.8)
plt.suptitle('Crystal Projected Potentials (GaAs)', fontsize=13)
plt.tight_layout()
save("01_potentials.png")

step("Building transmission function O = exp(i·σ·V)...")
sigma = mp.interaction_constant
print(f"  σ = {sigma:.6f}  (at {mp.voltage/1e3:.0f} kV, units='{mp.potential_units}')")

O_true = np.exp(1j * sigma * V)  # shape: (n_slices, N, N)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
extent = [mp.X.min(), mp.X.max(), mp.Y.min(), mp.Y.max()]

axes[0].imshow(np.abs(O_true).sum(axis=0), cmap='gray', extent=extent)
axes[0].set_title('Object amplitude (sum over slices)')
axes[0].set_xlabel('x (Å)')
axes[0].set_ylabel('y (Å)')

im = axes[1].imshow(np.angle(O_true).sum(axis=0), cmap='twilight', extent=extent)
axes[1].set_title('Object phase (sum over slices)')
axes[1].set_xlabel('x (Å)')
plt.colorbar(im, ax=axes[1], label='Phase (rad)')
plt.suptitle('True Object (Transmission Function)', fontsize=13)
plt.tight_layout()
save("02_transmission.png")


# ---------------------------------------------------------------------------
# Step 2: Set up the microscope with aberrations
# ---------------------------------------------------------------------------
section("Step 2: Microscope with aberrations")
step("Constructing aberrated microscope (C1=100Å, A1=50Å, B2=80Å, Cs=1mm)...")

scope = Microscope(
    voltage=DEMO["voltage"],
    alpha=0.010,
    C1=100.0,
    A1=50.0,
    phi_A1=np.pi / 6,
    B2=80.0,
    phi_B2=0.0,
    C3=1e7,
)
print(f"  {scope}")
print(f"  Wavelength: {scope.wavelength:.5f} Å  (200 kV)")

step("Comparing preset microscope configurations...")
presets = [
    ("TEM 200kV",          Microscope.TEM_200kV()),
    ("AC-STEM 200kV",      Microscope.aberration_corrected_200kV()),
    ("TEM 300kV",          Microscope.TEM_300kV()),
    ("AC-STEM 300kV",      Microscope.aberration_corrected_300kV()),
    ("Our aberrated scope", scope),
]
print(f"\n  {'Name':<22}  {'λ (pm)':>8}  {'α (mrad)':>9}  {'Cs (mm)':>8}  {'C1 (Å)':>8}")
print("  " + "-" * 63)
for name, s in presets:
    print(f"  {name:<22}  {s.wavelength*100:>7.4f}pm  {s.alpha*1e3:>8.1f}  {s.C3/1e7:>8.3f}  {s.C1:>8.1f}")

step("Plotting probe, aberration function, and CTF...")
scope.plot_probe(N=mp.N, dx=mp.dx, patch_size=48)
save("03_probe.png")

scope.plot_aberration_function(N=mp.N, dx=mp.dx)
save("04_aberration_function.png")

scope.plot_ctf(N=mp.N, dx=mp.dx)
save("05_ctf.png")

scope.plot_ctf_1d(N=mp.N, dx=mp.dx)
save("06_ctf_1d.png")

step("Constructing ideal and aberrated probes for comparison...")
scope_ideal = Microscope(voltage=DEMO["voltage"], alpha=scope.alpha)
probe_ideal = scope_ideal.construct_probe(N=mp.N, dx=mp.dx)
probe_aberrated = scope.construct_probe_for(mp)

vis_patch_size = 48
c = mp.N // 2
hp = vis_patch_size // 2
extent = np.array([-hp, hp, -hp, hp]) * mp.dx

ideal_patch = np.fft.fftshift(probe_ideal)[c-hp:c+hp, c-hp:c+hp]
aberr_patch = np.fft.fftshift(probe_aberrated)[c-hp:c+hp, c-hp:c+hp]

fig, axes = plt.subplots(2, 2, figsize=(10, 9))

axes[0, 0].imshow(np.abs(ideal_patch)**2, cmap='hot', extent=extent)
axes[0, 0].set_title('Ideal probe — intensity')
axes[0, 0].set_ylabel('y (Å)')

axes[0, 1].imshow(np.angle(ideal_patch), cmap='twilight', extent=extent, vmin=-np.pi, vmax=np.pi)
axes[0, 1].set_title('Ideal probe — phase')

axes[1, 0].imshow(np.abs(aberr_patch)**2, cmap='hot', extent=extent)
axes[1, 0].set_title('Aberrated probe — intensity')
axes[1, 0].set_xlabel('x (Å)')
axes[1, 0].set_ylabel('y (Å)')

axes[1, 1].imshow(np.angle(aberr_patch), cmap='twilight', extent=extent, vmin=-np.pi, vmax=np.pi)
axes[1, 1].set_title('Aberrated probe — phase')
axes[1, 1].set_xlabel('x (Å)')

plt.suptitle('Ideal vs Aberrated Probe (C1=100Å, A1=50Å, B2=80Å, Cs=1mm)', fontsize=13)
plt.tight_layout()
save("07_ideal_vs_aberrated_probe.png")

step("Plotting gallery: effect of each aberration independently...")
aberrations = [
    ("Ideal (no aberrations)", {}),
    ("Defocus only (C1=200Å)", dict(C1=200.0)),
    ("Astigmatism only (A1=150Å)", dict(A1=150.0, phi_A1=0)),
    ("Coma only (B2=300Å)", dict(B2=300.0, phi_B2=0)),
    ("Cs only (1 mm)", dict(C3=1e7)),
]

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
gallery_patch_size = 48
c = mp.N // 2
hp = gallery_patch_size // 2
extent = np.array([-hp, hp, -hp, hp]) * mp.dx

for col, (label, params) in enumerate(aberrations):
    s = Microscope(voltage=DEMO["voltage"], alpha=scope.alpha, **params)
    p = s.construct_probe(N=mp.N, dx=mp.dx)
    patch = np.fft.fftshift(p)[c-hp:c+hp, c-hp:c+hp]
    axes[0, col].imshow(np.abs(patch)**2, cmap='hot', extent=extent)
    axes[0, col].set_title(label, fontsize=9)
    axes[1, col].imshow(np.angle(patch), cmap='twilight', extent=extent, vmin=-np.pi, vmax=np.pi)

axes[0, 0].set_ylabel('Intensity\ny (Å)')
axes[1, 0].set_ylabel('Phase\ny (Å)')
plt.suptitle('Effect of Individual Aberrations on the Electron Probe', fontsize=13)
plt.tight_layout()
save("08_aberration_gallery.png")


# ---------------------------------------------------------------------------
# Step 3: Multislice wave propagation
# ---------------------------------------------------------------------------
section("Step 3: Multislice wave propagation")
step("Propagating aberrated and ideal probes through the crystal...")

exit_wave = mp.propagate_wavefunction(probe_aberrated, dz=dz)
exit_wave_ideal = mp.propagate_wavefunction(probe_ideal, dz=dz)
print(f"  Exit wave shape: {exit_wave.shape}")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
extent = [mp.X.min(), mp.X.max(), mp.Y.min(), mp.Y.max()]

axes[0, 0].imshow(np.abs(np.fft.fftshift(probe_aberrated))**2, cmap='hot', extent=extent)
axes[0, 0].set_title('Aberrated probe (input)')
axes[0, 0].set_ylabel('Aberrated y (Å)')

axes[0, 1].imshow(np.abs(np.fft.fftshift(exit_wave))**2, cmap='hot', extent=extent)
axes[0, 1].set_title('Exit wave intensity')

axes[0, 2].imshow(np.angle(np.fft.fftshift(exit_wave)), cmap='twilight', extent=extent)
axes[0, 2].set_title('Exit wave phase')

axes[1, 0].imshow(np.abs(np.fft.fftshift(probe_ideal))**2, cmap='hot', extent=extent)
axes[1, 0].set_title('Ideal probe (input)')
axes[1, 0].set_ylabel('Ideal y (Å)')

axes[1, 1].imshow(np.abs(np.fft.fftshift(exit_wave_ideal))**2, cmap='hot', extent=extent)
axes[1, 1].set_title('Exit wave intensity')

axes[1, 2].imshow(np.angle(np.fft.fftshift(exit_wave_ideal)), cmap='twilight', extent=extent)
axes[1, 2].set_title('Exit wave phase')

for ax in axes.flat:
    ax.set_xlabel('x (Å)')
plt.suptitle('Multislice Propagation Through Crystal', fontsize=13)
plt.tight_layout()
save("09_multislice_propagation.png")


# ---------------------------------------------------------------------------
# Step 4: Scanning and diffraction pattern generation
# ---------------------------------------------------------------------------
section("Step 4: Scanning and diffraction pattern generation")
step("Constructing scan positions...")

grid_positions = MicroPtycho.construct_scan_positions(
    scan_margin=0,
    d_probe=scan_step / (1 - 0.3),  # invert step = d_probe*(1-overlap)
    overlap=0.3,
    scan_range=scan_range,
)
# Jitter scan positions: a perfectly regular raster aliases with the atom
# lattice in sparse ptychography, producing a moiré grid artifact in the
# reconstruction. A uniform random shift of ±scan_step/3 breaks the
# symmetry without hurting probe-overlap coverage. Keep it inside the
# patch-safe region.
_jitter_mag = scan_step / 3.0
_jitter_rng = np.random.default_rng(11)
grid_positions = grid_positions + _jitter_rng.uniform(
    -_jitter_mag, _jitter_mag, size=grid_positions.shape
)
# Clip so every jittered position still satisfies |pos| ≤ scan_max.
_scan_max_abs = (DEMO["N"] / 2 - patch_size / 2 - 1) * DEMO["dx"]
grid_positions = np.clip(grid_positions, -_scan_max_abs, _scan_max_abs)
print(f"  Scan positions: {len(grid_positions)} (scan_range=±{scan_range:.1f} Å, "
      f"step={scan_step:.2f} Å, jitter=±{_jitter_mag:.2f} Å)")

mp.plot_scan_positions(grid_positions, probe_radius=0.86)
save("10_scan_positions.png")

probe_patch = mp.extract_probe_patch(patch_size=patch_size)
print(f"  Probe patch shape: {probe_patch.shape} ({patch_size*mp.dx:.1f} Å wide)")
depth_resolution = mp.wavelength / (scope.alpha**2)
print(f"  Axial resolution estimate λ/α² ≈ {depth_resolution:.1f} Å (dz = {dz:.1f} Å)")

step("Building Fresnel propagation kernel...")
fresnel_kernel = mp.make_fresnel_kernel(
    KX=np.fft.fftfreq(patch_size, mp.dx)[np.newaxis, :] * np.ones((patch_size, 1)),
    KY=np.fft.fftfreq(patch_size, mp.dx)[:, np.newaxis] * np.ones((1, patch_size)),
    dz=dz,
)

step(f"Generating diffraction patterns for {len(grid_positions)} scan positions...")
intensity, interwaves = mp.create_multislice_intensities(
    O_true, probe_patch, mp.dx, grid_positions,
    fresnel_kernel=fresnel_kernel, patch_size=patch_size
)
print(f"  Output: {intensity.shape}  ({intensity.shape[0]} patterns of {intensity.shape[1]}x{intensity.shape[2]} px)")

fig, axes = plt.subplots(2, 5, figsize=(16, 6))
for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(np.log1p(np.fft.fftshift(intensity[i])), cmap='inferno')
    ax.set_title(f'Pos {i}', fontsize=9)
    ax.axis('off')
plt.suptitle('Diffraction Patterns (log scale) — Aberrated Probe', fontsize=13)
plt.tight_layout()
save("11_diffraction_patterns.png")


# ---------------------------------------------------------------------------
# Step 5: Multislice ePIE reconstruction
# ---------------------------------------------------------------------------
section("Step 5: Multislice ePIE reconstruction")
step(f"Initialising reconstruction ({n_slices} slices, flat object prior)...")

n_slices = O_true.shape[0]
O_init = np.ones_like(O_true)

# ---- Probe Fourier-amplitude template ----
# Experimentally, |F(probe)| is measured from a *vacuum* diffraction
# pattern — one scan position with no sample, so the detector records
# |F(probe)|². This is a routine STEM calibration step and requires no
# knowledge of aberrations χ(k) (it uses probe amplitude only). In this
# simulation we reproduce that by computing |F(probe_patch_true)|. The
# aberration *phase* χ(k) is still unknown to the algorithm — only the
# amplitude is.
probe_fourier_support = np.abs(np.fft.fft2(probe_patch))

# Probe init: start from the *ideal* (unaberrated) probe patch. ePIE
# will then refine the phase inside the aperture to recover χ(k). With
# the amplitude template above, the probe cannot absorb object features
# (which would need amplitude changes); it can only move phase.
_ideal_patch_hp = patch_size // 2
_c = mp.N // 2
probe_ideal_patch = np.fft.fftshift(probe_ideal)[
    _c - _ideal_patch_hp: _c + _ideal_patch_hp,
    _c - _ideal_patch_hp: _c + _ideal_patch_hp,
].copy()
probe_init = probe_ideal_patch.copy()

step(f"Running {DEMO['n_iter']} ePIE iterations over {len(grid_positions)} positions...")
probe_recon, O_recon, residuals = mp.multislice_ePIE(
    n_iter=DEMO["n_iter"],
    probe=probe_init,
    O=O_init,
    intensity=intensity,
    grid_positions=grid_positions,
    fresnel_kernel=fresnel_kernel,
    dx=mp.dx,
    patch_size=patch_size,
    # Joint probe+object reconstruction.  The amplitude-locked Fourier
    # aperture collapses the probe to one-phase-per-k-bin, so probe
    # update cannot absorb object features. Warmup lets the object
    # settle first (otherwise probe tries to explain a random O).
    alpha_0=3.0,                           # push hard: probe dominates
                                           #   diffraction, so object
                                           #   gradients are weak
    beta_0=0.2,                            # amplitude-locked → probe can
                                           #   only learn phase (χ)
    tau=80,
    object_constraint='phase_nonneg',      # V ≥ 0 ⇒ phase ≥ 0 (exact)
    rho_object=0.05,                       # low Tikhonov, rely on constraint
    rho_probe=0.3,
    probe_fourier_support=probe_fourier_support,
    probe_warmup_iters=0,
    random_seed=7,
)

# Align per-slice global phase for fair visual comparison against ground truth.
O_recon_aligned = O_recon.copy()
for k in range(n_slices):
    O_recon_aligned[k] = mp.align_phase_affine(
        O_recon_aligned[k],
        O_true[k],
        dx=mp.dx,
    )

print(f"\n  Initial residual : {residuals[0]:.4e}")
print(f"  Final residual   : {residuals[-1]:.4e}")
print(f"  Improvement      : {residuals[0]/residuals[-1]:.2f}x")

step("Saving reconstruction figures...")
mp.plot_residuals(residuals)
save("12_residuals.png")

recon_proj = np.angle(O_recon_aligned).sum(axis=0)
true_proj = np.angle(O_true).sum(axis=0)
# Lock both panels to the TRUE phase range — twilight is cyclic, so a
# recon that slightly overshoots true's range displays as alternating
# sign even when the atoms are at correct positions with correct phase.
_vmin, _vmax = float(true_proj.min()), float(true_proj.max())
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im0 = axes[0].imshow(true_proj, cmap='twilight', vmin=_vmin, vmax=_vmax)
axes[0].set_title("True projected phase (sum over slices)")
plt.colorbar(im0, ax=axes[0], label='phase (rad)')
im1 = axes[1].imshow(recon_proj, cmap='twilight', vmin=_vmin, vmax=_vmax)
axes[1].set_title("Reconstructed projected phase (sum over slices)")
plt.colorbar(im1, ax=axes[1], label='phase (rad)')
plt.tight_layout()
save("13_recon_projected_phase.png")

mp.plot_multislice_reconstruction(O_recon_aligned, O_true)
save("14_multislice_reconstruction.png")

probe_recon_aligned = mp.align_phase_affine(probe_recon, probe_patch, dx=mp.dx)

extent = np.array([-patch_size//2, patch_size//2,
                   -patch_size//2, patch_size//2]) * mp.dx

fig, axes = plt.subplots(2, 2, figsize=(10, 9))

axes[0, 0].imshow(np.abs(probe_patch)**2, cmap='hot', extent=extent)
axes[0, 0].set_title('True probe — intensity')
axes[0, 0].set_ylabel('y (Å)')

axes[0, 1].imshow(np.angle(probe_patch), cmap='twilight', extent=extent, vmin=-np.pi, vmax=np.pi)
axes[0, 1].set_title('True probe — phase')

axes[1, 0].imshow(np.abs(probe_recon)**2, cmap='hot', extent=extent)
axes[1, 0].set_title('Reconstructed probe — intensity')
axes[1, 0].set_xlabel('x (Å)')
axes[1, 0].set_ylabel('y (Å)')

axes[1, 1].imshow(np.angle(probe_recon_aligned), cmap='twilight', extent=extent, vmin=-np.pi, vmax=np.pi)
axes[1, 1].set_title('Reconstructed probe — phase')
axes[1, 1].set_xlabel('x (Å)')

plt.suptitle('Probe Reconstruction (ideal init → aberrated probe)', fontsize=13)
plt.tight_layout()
save("15_probe_reconstruction.png")

probe_mse = np.mean(np.abs(probe_recon_aligned - probe_patch)**2)
print(f"  Probe MSE: {probe_mse:.4e}")

slices_to_compare = sorted(set([0, n_slices // 2, n_slices - 1]))
fig, axes = plt.subplots(2, len(slices_to_compare), figsize=(5 * len(slices_to_compare), 8))
axes = np.atleast_2d(axes)
if axes.shape[0] == 1:  # single-slice: subplots returns (2,) not (2,1)
    axes = axes.reshape(2, 1)
# Lock colormap limits to the TRUE phase range for each slice so both
# panels show the same signal level. Auto-scaling per-panel makes the
# recon look alternating-sign when it just has a slightly wider range
# than truth (twilight is cyclic: symmetric low/high both look dark).
for col, s in enumerate(slices_to_compare):
    vmin, vmax = float(np.angle(O_true[s]).min()), float(np.angle(O_true[s]).max())
    axes[0, col].imshow(np.angle(O_true[s]), cmap='twilight', vmin=vmin, vmax=vmax)
    axes[0, col].set_title(f'True — slice {s}')
    axes[1, col].imshow(np.angle(O_recon_aligned[s]), cmap='twilight', vmin=vmin, vmax=vmax)
    axes[1, col].set_title(f'Reconstructed — slice {s}')

axes[0, 0].set_ylabel('True phase')
axes[1, 0].set_ylabel('Recon phase')
plt.suptitle('Slice-by-Slice Phase Comparison', fontsize=13)
plt.tight_layout()
save("16_slice_comparison.png")


# ---------------------------------------------------------------------------
# Step 6: Applying the CTF to an image
# ---------------------------------------------------------------------------
section("Step 6: Applying the CTF to an image")
step("Simulating HRTEM images via CTF convolution...")

perfect_image = V.sum(axis=0)
aberrated_image = scope.apply_ctf(perfect_image, dx=mp.dx)

scope_scherzer = Microscope(voltage=200e3, alpha=0.010, C1=-60.0, C3=1e7)
scherzer_image = scope_scherzer.apply_ctf(perfect_image, dx=mp.dx)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
extent = [mp.X.min(), mp.X.max(), mp.Y.min(), mp.Y.max()]

axes[0].imshow(perfect_image, cmap='gray', extent=extent)
axes[0].set_title('Perfect projected potential')
axes[0].set_ylabel('y (Å)')

axes[1].imshow(np.real(scherzer_image), cmap='gray', extent=extent)
axes[1].set_title('Scherzer defocus (C1=-60Å, Cs=1mm)')

axes[2].imshow(np.real(aberrated_image), cmap='gray', extent=extent)
axes[2].set_title('Full aberrations (C1=100, A1=50, B2=80, Cs=1mm)')

for ax in axes:
    ax.set_xlabel('x (Å)')
plt.suptitle('Conventional HRTEM Image Simulation via CTF', fontsize=13)
plt.tight_layout()
save("17_ctf_imaging.png")


# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("  All done!")
print(f"  17 figures saved to testrun/")
print(f"{'='*60}\n")
