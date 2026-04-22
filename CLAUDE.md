# microptycho — working notes

## Tutorial demo (`tutorial.py`)

`DEMO` dict drives the simulation. Coupled constraints worth knowing:

- **FOV** = `N * dx`. The GaAs supercell must fully cover the FOV — tiling only the
  scan footprint creates a hard sample edge inside the grid and ruins ePIE
  reconstruction. Compute `n_tile = ceil(fov / a_gaas) + 1`.
- **scan_range** must satisfy `scan_range <= (N/2 - patch_size/2) * dx`. Good results
  come from keeping `scan_range ≈ 0.4 * FOV` (what the original N=128 demo used).
  When you change `N` or `dx`, scale `scan_range` proportionally — otherwise
  ePIE only reconstructs a tiny central disc of the sample.
- **Atom count** is dominated by `nz`. Reducing `nz` (e.g. 40 → 5) cuts total atoms
  ~8× while keeping the in-plane lattice pattern identical — prefer this over
  shrinking `nx, ny`, which would leave the FOV partially empty.
- **`sigma`** in `create_potentials(..., sigma=...)` is atom Gaussian width in Å.
  Default 0.7 keeps atoms distinct for GaAs (a=5.65 Å). Going much above ~1.0
  merges adjacent atoms within a unit cell and destroys atomic contrast.

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

Feature branch for this line of work: `claude/lattice-resolution-atom-count-WRthg`.
Standard flow is develop on branch → commit → push → merge to `main` → push main.
