import numpy as np
import matplotlib.pyplot as plt

from electron_optics import wavelength_from_voltage


class Microscope:
    """Electron microscope simulation with aberrations for ptychography.

    Simulates the effect of lens aberrations (defocus, astigmatism, coma,
    spherical aberration) on an electron probe. Generates aberrated probes
    compatible with MicroPtycho.

    All lengths are in Angstroms. All angles are in radians.

    Parameters
    ----------
    voltage : float
        Accelerating voltage in Volts (default 200 kV).
    alpha : float
        Convergence semi-angle in radians (default 10 mrad).
    C1 : float
        Defocus in Angstroms. Positive = overfocus.
    A1 : float
        Two-fold astigmatism magnitude in Angstroms.
    phi_A1 : float
        Astigmatism azimuthal angle in radians.
    B2 : float
        Axial coma magnitude in Angstroms.
    phi_B2 : float
        Coma azimuthal angle in radians.
    C3 : float
        Third-order spherical aberration (Cs) in Angstroms.
        Note: 1 mm = 1e7 Angstroms.
    """

    def __init__(self, voltage=200e3, alpha=0.010,
                 C1=0.0, A1=0.0, phi_A1=0.0,
                 B2=0.0, phi_B2=0.0, C3=0.0):
        self.voltage = voltage
        self.wavelength = self.wavelength_from_voltage(voltage)
        self.alpha = alpha
        self.C1 = C1
        self.A1 = A1
        self.phi_A1 = phi_A1
        self.B2 = B2
        self.phi_B2 = phi_B2
        self.C3 = C3

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def defocus(self):
        return self.C1

    @defocus.setter
    def defocus(self, val):
        self.C1 = val

    @property
    def Cs(self):
        return self.C3

    @Cs.setter
    def Cs(self, val):
        self.C3 = val

    # ------------------------------------------------------------------ #
    #  Core physics
    # ------------------------------------------------------------------ #

    @staticmethod
    def wavelength_from_voltage(voltage):
        """Relativistic de Broglie wavelength in Angstroms."""
        return wavelength_from_voltage(voltage)

    def aberration_function(self, KX, KY):
        """Compute the aberration function chi(k, phi) on a k-space grid.

        Returns
        -------
        chi : ndarray
            Aberration phase (same shape as KX, KY).
        """
        k = np.sqrt(KX**2 + KY**2)
        phi = np.arctan2(KY, KX)
        lam = self.wavelength

        chi = (np.pi * lam * self.C1 * k**2
               + np.pi * lam * self.A1 * k**2 * np.cos(2 * (phi - self.phi_A1))
               + (2 * np.pi / 3) * lam**2 * self.B2 * k**3 * np.cos(phi - self.phi_B2)
               + (np.pi / 2) * lam**3 * self.C3 * k**4)
        return chi

    def transfer_function(self, KX, KY):
        """Complex aberration phase plate H(k) = exp(-i * chi)."""
        return np.exp(-1j * self.aberration_function(KX, KY))

    def ctf(self, KX, KY):
        """Contrast Transfer Function (weak-phase-object approximation).

        Returns sin(chi), the imaginary part of the transfer function
        relevant for phase contrast imaging.
        """
        return np.sin(self.aberration_function(KX, KY))

    # ------------------------------------------------------------------ #
    #  Probe construction
    # ------------------------------------------------------------------ #

    def construct_probe(self, N=256, dx=0.43, KX=None, KY=None):
        """Create an aberrated probe compatible with MicroPtycho.

        Parameters
        ----------
        N : int
            Grid size.
        dx : float
            Pixel size in Angstroms.
        KX, KY : ndarray, optional
            Fourier-space grids. If None, generated from N and dx using
            the same convention as MicroPtycho.make_grid.

        Returns
        -------
        probe : ndarray, shape (N, N)
            Complex probe in real space.
        """
        if KX is None or KY is None:
            kx = np.fft.fftfreq(N, dx)
            KX, KY = np.meshgrid(kx, kx)

        k_radius = np.sqrt(KX**2 + KY**2)
        kmax = self.alpha / self.wavelength
        aperture = np.where(k_radius <= kmax, 1.0, 0.0)

        H = self.transfer_function(KX, KY)
        probe = np.fft.ifft2(aperture * H)
        return probe

    def construct_probe_for(self, ptycho, sync_beam=True, sync_interaction_constant=True):
        """Build an aberrated probe and assign it to a MicroPtycho instance.

        Parameters
        ----------
        ptycho : MicroPtycho
            The ptychography simulation object.
        sync_beam : bool
            When True, update the MicroPtycho wavelength to match this
            microscope before constructing the probe.
        sync_interaction_constant : bool
            When syncing the beam, also refresh the MicroPtycho interaction
            constant using the same accelerating voltage.

        Returns
        -------
        probe : ndarray
            The aberrated probe (also set as ptycho.probe).
        """
        if sync_beam:
            interaction_constant = None if sync_interaction_constant else ptycho.interaction_constant
            ptycho.set_beam_energy(self.voltage, interaction_constant=interaction_constant)
        probe = self.construct_probe(N=ptycho.N, dx=ptycho.dx,
                                     KX=ptycho.KX, KY=ptycho.KY)
        ptycho.probe = probe
        return probe

    # ------------------------------------------------------------------ #
    #  Image-domain CTF application
    # ------------------------------------------------------------------ #

    def apply_ctf(self, image, N=None, dx=0.43):
        """Apply the microscope transfer function to a real-space image.

        Parameters
        ----------
        image : ndarray
            Real-space image (2D).
        dx : float
            Pixel size in Angstroms.

        Returns
        -------
        image_out : ndarray
            Image after application of the aberration transfer function.
        """
        if N is None:
            N = image.shape[0]
        kx = np.fft.fftfreq(N, dx)
        KX, KY = np.meshgrid(kx, kx)
        H = self.transfer_function(KX, KY)
        return np.fft.ifft2(np.fft.fft2(image) * H)

    # ------------------------------------------------------------------ #
    #  Visualization
    # ------------------------------------------------------------------ #

    def _make_k_grid(self, N, dx):
        kx = np.fft.fftfreq(N, dx)
        return np.meshgrid(kx, kx)

    def plot_aberration_function(self, N=256, dx=0.43):
        """Plot the aberration function chi(k) in Fourier space."""
        KX, KY = self._make_k_grid(N, dx)
        chi = self.aberration_function(KX, KY)
        chi_shifted = np.fft.fftshift(chi)

        k_extent = 1 / (2 * dx)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(chi_shifted, cmap='twilight',
                       extent=[-k_extent, k_extent, -k_extent, k_extent])
        ax.set_xlabel(r'$k_x$ ($\AA^{-1}$)')
        ax.set_ylabel(r'$k_y$ ($\AA^{-1}$)')
        ax.set_title(r'Aberration function $\chi(k)$')
        plt.colorbar(im, ax=ax, label='Phase (rad)')
        plt.tight_layout()
        plt.show()

    def plot_ctf(self, N=256, dx=0.43):
        """Plot the 2D Contrast Transfer Function."""
        KX, KY = self._make_k_grid(N, dx)
        ctf = self.ctf(KX, KY)
        ctf_shifted = np.fft.fftshift(ctf)

        k_extent = 1 / (2 * dx)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(ctf_shifted, cmap='RdBu_r', vmin=-1, vmax=1,
                       extent=[-k_extent, k_extent, -k_extent, k_extent])
        ax.set_xlabel(r'$k_x$ ($\AA^{-1}$)')
        ax.set_ylabel(r'$k_y$ ($\AA^{-1}$)')
        ax.set_title('Contrast Transfer Function')
        plt.colorbar(im, ax=ax, label=r'$\sin(\chi)$')
        plt.tight_layout()
        plt.show()

    def plot_ctf_1d(self, N=256, dx=0.43):
        """Plot radially averaged CTF vs spatial frequency."""
        KX, KY = self._make_k_grid(N, dx)
        k = np.sqrt(KX**2 + KY**2)
        ctf = self.ctf(KX, KY)

        k_flat = k.ravel()
        ctf_flat = ctf.ravel()
        sort_idx = np.argsort(k_flat)
        k_sorted = k_flat[sort_idx]
        ctf_sorted = ctf_flat[sort_idx]

        n_bins = N // 2
        k_max = k_sorted[-1]
        bin_edges = np.linspace(0, k_max, n_bins + 1)
        k_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ctf_avg = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (k_sorted >= bin_edges[i]) & (k_sorted < bin_edges[i + 1])
            if mask.any():
                ctf_avg[i] = ctf_sorted[mask].mean()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(k_centers, ctf_avg, 'b-', linewidth=1)
        ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
        kmax = self.alpha / self.wavelength
        ax.axvline(kmax, color='r', linewidth=0.5, linestyle='--', label='Aperture cutoff')
        ax.set_xlabel(r'Spatial frequency ($\AA^{-1}$)')
        ax.set_ylabel(r'CTF [$\sin(\chi)$]')
        ax.set_title('Radial Contrast Transfer Function')
        ax.legend()
        ax.set_xlim(0, k_max)
        plt.tight_layout()
        plt.show()

    def plot_probe(self, N=256, dx=0.43, patch_size=48):
        """Plot the aberrated probe intensity and phase."""
        probe = self.construct_probe(N=N, dx=dx)
        probe_shifted = np.fft.fftshift(probe)
        c = N // 2
        hp = patch_size // 2
        patch = probe_shifted[c - hp:c + hp, c - hp:c + hp]

        extent = np.array([-hp, hp, -hp, hp]) * dx

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        im0 = axes[0].imshow(np.abs(patch)**2, cmap='hot', extent=extent)
        axes[0].set_title('Probe intensity')
        axes[0].set_xlabel(r'x ($\AA$)')
        axes[0].set_ylabel(r'y ($\AA$)')
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(np.angle(patch), cmap='twilight', extent=extent,
                             vmin=-np.pi, vmax=np.pi)
        axes[1].set_title('Probe phase')
        axes[1].set_xlabel(r'x ($\AA$)')
        axes[1].set_ylabel(r'y ($\AA$)')
        plt.colorbar(im1, ax=axes[1], label='Phase (rad)')

        plt.suptitle(f'Aberrated probe  |  C1={self.C1:.0f}Å  A1={self.A1:.0f}Å  '
                     f'B2={self.B2:.0f}Å  C3={self.C3:.1e}Å')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    #  Preset microscopes
    # ------------------------------------------------------------------ #

    @classmethod
    def TEM_200kV(cls):
        """Typical uncorrected TEM at 200 kV. Cs = 1 mm."""
        return cls(voltage=200e3, alpha=0.010, C3=1e7)

    @classmethod
    def aberration_corrected_200kV(cls):
        """Aberration-corrected STEM at 200 kV. Cs ~ 0, alpha = 25 mrad."""
        return cls(voltage=200e3, alpha=0.025, C3=0.0)

    @classmethod
    def TEM_300kV(cls):
        """Typical uncorrected TEM at 300 kV. Cs = 1.2 mm."""
        return cls(voltage=300e3, alpha=0.010, C3=1.2e7)

    @classmethod
    def aberration_corrected_300kV(cls):
        """Aberration-corrected STEM at 300 kV. Cs ~ 0, alpha = 30 mrad."""
        return cls(voltage=300e3, alpha=0.030, C3=0.0)

    def __repr__(self):
        return (f"Microscope(voltage={self.voltage:.0f}, alpha={self.alpha:.4f}, "
                f"C1={self.C1}, A1={self.A1}, B2={self.B2}, C3={self.C3})")
