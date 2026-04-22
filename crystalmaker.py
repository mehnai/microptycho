import numpy as np
import matplotlib.pyplot as plt


class CrystalMaker:
    """Generates crystal structures and toy projected potentials.

    Supported structures: 'diamond', 'fcc', 'bcc'.

    The potential model uses Gaussian atoms, so it is appropriate for
    qualitative ptychography simulations rather than quantitative electron
    scattering.
    """

    STRUCTURES = ('diamond', 'fcc', 'bcc', 'sc')

    def __init__(self, lattice_constant=5.43, Z1=14, Z2=None, structure='diamond',
                 reference_atomic_number=14.0):
        if structure not in self.STRUCTURES:
            raise ValueError(f"structure must be one of {self.STRUCTURES}, got '{structure}'")
        self.lattice_constant = lattice_constant
        self.Z1 = Z1
        self.Z2 = Z2 if Z2 is not None else Z1
        self.structure = structure
        self.reference_atomic_number = reference_atomic_number
        self.unit_cell = self._make_unit_cell()
        self.supercell = None
        self.potentials = None

    def _make_unit_cell(self):
        a = self.lattice_constant
        Z1, Z2 = self.Z1, self.Z2
        if self.structure == 'diamond':
            return self._diamond_unit_cell(a, Z1, Z2)
        elif self.structure == 'fcc':
            return self._fcc_unit_cell(a, Z1)
        elif self.structure == 'bcc':
            return self._bcc_unit_cell(a, Z1)
        elif self.structure == 'sc':
            return self._sc_unit_cell(a, Z1)

    @staticmethod
    def _diamond_unit_cell(a, Z1, Z2):
        return np.array([
            [0,       0,       0,       Z1],
            [a/4,     a/4,     a/4,     Z1],
            [a/2,     a/2,     0,       Z1],
            [3*a/4,   3*a/4,   a/4,     Z1],
            [a/2,     0,       a/2,     Z2],
            [3*a/4,   a/4,     3*a/4,   Z2],
            [0,       a/2,     a/2,     Z2],
            [a/4,     3*a/4,   3*a/4,   Z2],
        ])

    @staticmethod
    def _fcc_unit_cell(a, Z):
        """Face-centered cubic: 4 atoms per unit cell."""
        return np.array([
            [0,     0,     0,     Z],
            [a/2,   a/2,   0,     Z],
            [a/2,   0,     a/2,   Z],
            [0,     a/2,   a/2,   Z],
        ])

    @staticmethod
    def _bcc_unit_cell(a, Z):
        """Body-centered cubic: 2 atoms per unit cell."""
        return np.array([
            [0,     0,     0,     Z],
            [a/2,   a/2,   a/2,   Z],
        ])

    @staticmethod
    def _sc_unit_cell(a, Z):
        """Simple cubic: 1 atom per unit cell. Useful for sparse toy lattices."""
        return np.array([
            [0,     0,     0,     Z],
        ])

    def tile(self, nx=5, ny=5, nz=10):
        tiled = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    shift = np.array([i, j, k, 0]) * self.lattice_constant
                    tiled.append(self.unit_cell + shift)
        self.supercell = np.vstack(tiled)
        return self.supercell

    def create_potentials(self, X, Y, dz=2.0, sigma=0.7, amplitude=10, center=True,
                          atomic_scale=True, save_path=None):
        if self.supercell is None:
            raise ValueError("Call tile() first to generate a supercell.")

        sc = self.supercell.copy()

        if center:
            xy_center = (sc[:, :2].max(axis=0) + sc[:, :2].min(axis=0)) / 2
            sc[:, :2] -= xy_center

        z_min = float(np.min(sc[:, 2]))
        z_max = float(np.max(sc[:, 2]))
        total_thickness = z_max - z_min
        n_slices = max(int(np.ceil(total_thickness / dz)), 1)

        Z = np.zeros((n_slices, *X.shape))

        for atom in sc:
            x, y, z, atomic_num = atom
            if total_thickness > 0:
                slice_idx = int(np.floor((z - z_min) / dz))
            else:
                slice_idx = 0
            slice_idx = min(slice_idx, n_slices - 1)
            atom_amplitude = amplitude * self._atom_weight(atomic_num) if atomic_scale else amplitude
            Z[slice_idx] += self._gaussian_envelope(X, Y, x, y, atom_amplitude, sigma)

        self.potentials = Z

        if save_path is not None:
            np.save(save_path, Z)

        return Z

    @staticmethod
    def _gaussian_envelope(x, y, x_off, y_off, A, c):
        return A * np.exp(-(((x - x_off)**2 + (y - y_off)**2) / (2 * c**2)))

    def _atom_weight(self, atomic_num):
        return float(atomic_num) / self.reference_atomic_number

    # ---- convenience constructors ----

    @classmethod
    def silicon(cls, nx=5, ny=5, nz=10):
        cm = cls(lattice_constant=5.43, Z1=14)
        cm.tile(nx, ny, nz)
        return cm

    @classmethod
    def gallium_arsenide(cls, nx=5, ny=5, nz=10):
        cm = cls(lattice_constant=5.65, Z1=31, Z2=33)
        cm.tile(nx, ny, nz)
        return cm

    @classmethod
    def copper(cls, nx=5, ny=5, nz=10):
        cm = cls(lattice_constant=3.61, Z1=29, structure='fcc')
        cm.tile(nx, ny, nz)
        return cm

    @classmethod
    def aluminum(cls, nx=5, ny=5, nz=10):
        cm = cls(lattice_constant=4.05, Z1=13, structure='fcc')
        cm.tile(nx, ny, nz)
        return cm

    @classmethod
    def gold(cls, nx=5, ny=5, nz=10):
        cm = cls(lattice_constant=4.08, Z1=79, structure='fcc')
        cm.tile(nx, ny, nz)
        return cm

    @classmethod
    def iron(cls, nx=5, ny=5, nz=10):
        cm = cls(lattice_constant=2.87, Z1=26, structure='bcc')
        cm.tile(nx, ny, nz)
        return cm

    @classmethod
    def tungsten(cls, nx=5, ny=5, nz=10):
        cm = cls(lattice_constant=3.16, Z1=74, structure='bcc')
        cm.tile(nx, ny, nz)
        return cm

    # ---- plotting ----

    def plot_projected_potential(self, extent=None, cmap='gray'):
        if self.potentials is None:
            raise ValueError("Call create_potentials() first.")
        V_proj = np.sum(self.potentials, axis=0)
        plt.figure(figsize=(6, 6))
        plt.imshow(V_proj, extent=extent, origin='lower', cmap=cmap)
        plt.colorbar(label='Projected potential (a.u.)')
        plt.title('Projected potential')
        plt.xlabel('x (Å)')
        plt.ylabel('y (Å)')
        plt.show()

    def plot_slice(self, slice_idx=0, cmap='gray'):
        if self.potentials is None:
            raise ValueError("Call create_potentials() first.")
        plt.figure(figsize=(6, 6))
        plt.imshow(self.potentials[slice_idx], cmap=cmap)
        plt.colorbar()
        plt.title(f'Potential slice {slice_idx}')
        plt.show()
