"""Basic sanity tests for the crystal/potential generator."""
import numpy as np
import pytest

from crystalmaker import CrystalMaker
from microptycho import MicroPtycho


def test_diamond_unit_cell_has_8_atoms():
    cm = CrystalMaker(structure='diamond')
    assert cm.unit_cell.shape == (8, 4)


def test_fcc_unit_cell_has_4_atoms():
    cm = CrystalMaker(structure='fcc')
    assert cm.unit_cell.shape == (4, 4)


def test_bcc_unit_cell_has_2_atoms():
    cm = CrystalMaker(structure='bcc')
    assert cm.unit_cell.shape == (2, 4)


def test_sc_unit_cell_has_1_atom():
    cm = CrystalMaker(structure='sc')
    assert cm.unit_cell.shape == (1, 4)


def test_sc_tile_produces_nx_ny_nz_atoms():
    cm = CrystalMaker(lattice_constant=10.0, Z1=14, structure='sc')
    cm.tile(nx=6, ny=6, nz=1)
    assert cm.supercell.shape == (36, 4)


def test_invalid_structure_raises():
    with pytest.raises(ValueError):
        CrystalMaker(structure='hcp')


def test_tile_atom_count_diamond():
    cm = CrystalMaker.gallium_arsenide(nx=20, ny=20, nz=40)
    assert cm.supercell.shape[0] == 8 * 20 * 20 * 40


def test_create_potentials_requires_tile():
    cm = CrystalMaker(structure='fcc')  # no tile() call
    mp = MicroPtycho(N=32, dx=0.5)
    with pytest.raises(ValueError):
        cm.create_potentials(mp.X, mp.Y)


def test_create_potentials_shape_and_slicing():
    mp = MicroPtycho(N=64, dx=0.43)
    cm = CrystalMaker.silicon(nx=4, ny=4, nz=4)
    V = cm.create_potentials(mp.X, mp.Y, dz=2.0)
    # slicing along z covers the whole supercell
    sc_z = cm.supercell[:, 2]
    n_slices_expected = max(int(np.ceil((sc_z.max() - sc_z.min()) / 2.0)), 1)
    assert V.shape == (n_slices_expected, 64, 64)
    assert V.min() >= 0.0
    assert V.max() > 0.0


def test_potentials_respect_atom_centring():
    """When `center=True`, atoms are centred in the xy plane, so the
    potential integrated across all slices peaks near the origin."""
    mp = MicroPtycho(N=128, dx=0.43)
    cm = CrystalMaker.silicon(nx=4, ny=4, nz=4)
    V = cm.create_potentials(mp.X, mp.Y, dz=2.0, center=True)
    proj = V.sum(axis=0)
    # Half the image centered at origin must account for most of the mass
    N = proj.shape[0]
    inner = proj[N // 4: 3 * N // 4, N // 4: 3 * N // 4]
    assert inner.sum() > 0.5 * proj.sum()
