# microptycho — working notes

## Tutorial demo (`tutorial.py`)

`DEMO` dict holds *high-level* knobs only (`nx`, `ny`, `nz`, `structure`,
`sample_fill`, `sigma_frac`, `scan_fill`, `patch_min_A`). Everything else
(`a_lat`, `atom_sigma`, `dz`, `patch_size`, `scan_range`, `scan_step`) is
derived from them so the demo scales cleanly from a sparse 36-atom sample
(`nx=ny=6, nz=1, structure='sc'`) to a dense thousands-of-atoms sample
(`nx=ny=15, nz=5, structure='fcc'`).

How the derivation works:

- **FOV** = `N * dx`. The sample is always centred; the code auto-tiles
  exactly `nx·ny·nz` unit cells (`atoms_per_cell` from
  `ATOMS_PER_CELL[structure]`) and picks `a_lat` so sample + probe-patch
  fits inside `0.9·FOV`.
- **Probe patch** defaults to `2·in_plane_spacing` (so each probe position
  touches several atoms) or `patch_min_A`, whichever is larger.
- **`scan_range`** is clamped to `(N/2 − patch_size/2)·dx` automatically —
  no more ePIE-only-reconstructs-a-disc bug when `N`/`dx`/`patch_size`
  change.
- **`atom_sigma = sigma_frac · in_plane_spacing`** (default `sigma_frac=0.3`
  → atoms ~30 % of nearest-neighbour spacing → visible, non-merging).
- **`dz = a_lat`** (one slice per atomic layer in z). Check against
  `λ/α²` depth resolution — for the 200 kV, α=10 mrad demo, that's ~250 Å,
  so dz≲20 Å is well below the depth-resolution floor.

To add a new structure: update `ATOMS_PER_CELL`, `MIN_IN_PLANE_SPACING_FRAC`,
and `ROWS_PER_CELL_AXIS` dicts at the top of `tutorial.py`, and add a
`_<name>_unit_cell` to `CrystalMaker`.

**Lattice rotation.** `DEMO['rotation_deg']` rotates the tiled supercell
in the xy plane via `CrystalMaker.rotate_xy`. Default is 45° —
deliberately off-axis to prove the reconstruction is lattice-orientation
agnostic. The geometry derivation scales the effective side of the
bounding box by `|cos θ| + |sin θ|` (→ √2 at 45°) so the rotated
sample + probe patch still fits inside the scan-safe region.

## Sparse multislice ePIE (the 36-atom demo) — BOTH probe and object

**Multislice ptychography by definition reconstructs both probe AND
object.** Do not fix the probe. The demo reconstructs them jointly
starting from `χ(k) = 0` (no aberration prior).

Prior knowledge required:
- `α` (convergence semi-angle) and `λ` (= voltage). Microscope
  settings; always known.
- `|F(probe_patch)|` — the probe's Fourier-amplitude envelope. In a
  real experiment this is a **vacuum diffraction pattern**: scan one
  position with no sample and the detector records `|F(probe)|²`. A
  routine STEM calibration step that contains no aberration info,
  since aberrations are pure phase `exp(-iχ)` with `|·|=1`.

What ePIE learns from scratch:
- The aberration phase `χ(k)` inside the aperture (defocus `C1`,
  astigmatism `A1`, coma `B2`, `Cs`, ...)
- The full object `O(r)`

Regularizers that make this work for sparse data:

- **Amplitude-locked Fourier-aperture projection.** Pass
  `probe_fourier_support = |F(probe_patch)|` to `multislice_ePIE`. The
  helper `_project_probe_to_aperture` enforces `F(probe) = template ·
  exp(iφ(k))` after every probe update — only the phase inside the
  aperture is free. That's exactly `sum(aperture)` real DoFs (one phase
  per k-bin), and those phases *are* χ(k). With this on, the probe
  cannot absorb object features (which would require amplitude
  changes). `normalize_probe` is auto-skipped when this is on (it would
  fight the lock each epoch).
- **`object_constraint='phase_nonneg'`** — for `O = exp(i·σ·V)` with
  `V ≥ 0` this is physically exact (`|O|=1 ∧ phase≥0`). Strictly tighter
  than `phase_only`; kills the phase-sign ambiguity that produces
  random "negative atom" artefacts.
- **Scan jitter** — a perfectly regular raster aliases with the atom
  lattice into a moiré pattern. Add `±scan_step/3` uniform jitter.
- **Dense scanning** — `scan_step ≤ ⅙·in-plane-spacing` (not the usual
  ~0.7× probe diameter). Each atom needs multiple probe placements, and
  dense scanning is the single most effective knob for cleaning up
  Nyquist-scale ringing between atoms in the object reconstruction.
- **Aggressive `alpha_0`** — the probe dominates the diffraction
  pattern for sparse samples, so the *object's contribution* to residual
  is small. In the 36-atom demo, going from identity-O to true-O drops
  residual from ~44 to 0 while going from ideal-probe to true-probe
  drops it from ~1280 to ~44 — object is 3% of probe's signal. Object
  updates therefore have weak per-position gradients; push `alpha_0` to
  ~2–3 with low `rho_object` (~0.05) so the object actually moves.
- **`probe_warmup_iters=0`** — with amplitude-lock, the probe cannot
  damage itself, so no warmup needed. The probe can update from iter 1.

Tutorial defaults that produce clean joint convergence on 36 atoms:
`alpha_0=3.0, beta_0=0.2, tau=80, rho_object=0.05, rho_probe=0.3,
object_constraint='phase_nonneg', probe_fourier_support=|F(probe_patch)|,
n_iter=300`. Residual drops cleanly 48 → 22 with no oscillation; both
the 6×6 atom grid (correct phase magnitude, correct positions) and the
aberration rings in the probe phase are visible.

Other `multislice_ePIE` knobs worth knowing: `alpha_0`, `beta_0` (step
sizes), `tau` (decay), `rho_object`, `rho_probe` (Tikhonov-like
denominator terms), `normalize_probe` (Parseval-based probe-energy pin;
auto-skipped when aperture lock is on), `remove_probe_phase_ramp`
(removes the probe-tilt vs object-ramp gauge).

## ePIE regularization (`microptycho.py`)

`_apply_object_constraint` (line ~148) already exists and accepts
`'unit' | 'phase' | 'phase_only' | 'unit_modulus'`. All four variants project
the object onto the unit circle (`O ← exp(i·angle(O))`).

**Use `object_constraint='phase_only'` whenever the object is
`O = exp(i·σ·V)`** (i.e. any simulation built via `create_potentials` →
`exp(1j*sigma*V)`). It's not a heuristic — it's exactly true by construction,
so enforcing it just removes a gauge freedom. Dramatically lowers residuals
and stabilizes probe reconstruction.

Other knobs on `multislice_ePIE` / `ePIE`: `alpha_0`, `beta_0` (step sizes),
`tau` (decay), `rho_object`, `rho_probe` (Tikhonov-like denominator terms,
defaults 0.2), `normalize_probe` (Parseval-based probe-energy pin),
`remove_probe_phase_ramp` (removes the probe-tilt vs object-ramp gauge).

## Probe / object gauge freedoms to remember

These cause reconstructions to "look wrong" even when residuals are fine:
- **Probe translation ↔ object translation** — a shifted probe with an
  opposite-shifted object gives identical diffraction patterns. If the probe
  comes out drifted, check sample coverage first (most common cause).
- **Probe linear phase ramp ↔ object linear phase ramp** — handled by
  `remove_probe_phase_ramp=True` (on by default).
- **Probe/object modulus split** — fixed by `normalize_probe=True` plus
  `object_constraint='phase_only'` for pure-phase objects.

## Git / workflow

Feature branch for this line of work: `claude/generalize-reconstruction-code-hQoHo`.
Standard flow is develop on branch → commit → push → merge to `main` → push main.
