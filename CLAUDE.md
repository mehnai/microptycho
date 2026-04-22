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

Note on sparse ptychography: with ~36 atoms, each probe position sees
fewer atoms, so the probe update is less well constrained. Expect a
noisier reconstructed probe even though the object (atom positions) comes
out cleanly. The gauge-freedom fixes (`phase_only`, probe normalization,
phase-ramp removal) are essential in this regime.

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
