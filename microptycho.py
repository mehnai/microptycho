import numpy as np
import matplotlib.pyplot as plt

from electron_optics import interaction_constant_from_voltage, wavelength_from_voltage


class MicroPtycho:
    """Miniature ptychography simulation and reconstruction package."""

    def __init__(self, N=256, dx=0.43, V=None, voltage=200e3, wavelength=None,
                 interaction_constant=None, potential_units="V_nm"):
        self.N = N
        self.dx = dx
        self.voltage = voltage
        self.potential_units = potential_units
        self.wavelength = wavelength if wavelength is not None else wavelength_from_voltage(voltage)
        if interaction_constant is None:
            interaction_constant = interaction_constant_from_voltage(voltage, potential_units=potential_units)
        self.interaction_constant = interaction_constant
        self.X, self.Y, self.KX, self.KY = self.make_grid(N, dx)
        self.potentials = V
        self.probe = None

    def set_potentials(self, V):
        self.potentials = V

    def set_beam_energy(self, voltage, interaction_constant=None, potential_units=None):
        if potential_units is not None:
            self.potential_units = potential_units
        self.voltage = voltage
        self.wavelength = wavelength_from_voltage(voltage)
        if interaction_constant is None:
            interaction_constant = interaction_constant_from_voltage(
                voltage,
                potential_units=self.potential_units,
            )
        self.interaction_constant = interaction_constant

    # ------------------------------------------------------------------ #
    #  Grid & propagation utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def wavelength_from_voltage(voltage):
        return wavelength_from_voltage(voltage)

    @staticmethod
    def interaction_constant_from_voltage(voltage, potential_units="V_nm"):
        return interaction_constant_from_voltage(voltage, potential_units=potential_units)

    @staticmethod
    def make_grid(N, dx):
        x_grid = np.arange(-N // 2, N // 2) * dx
        X, Y = np.meshgrid(x_grid, x_grid)
        KX, KY = MicroPtycho.make_k_grid((N, N), dx)
        return X, Y, KX, KY

    @staticmethod
    def make_k_grid(shape, dx):
        if isinstance(shape, int):
            shape = (shape, shape)
        ky = np.fft.fftfreq(shape[0], dx)
        kx = np.fft.fftfreq(shape[1], dx)
        return np.meshgrid(kx, ky)

    @staticmethod
    def fresnel_propagator(k_x, k_y, dz, lam=0.02508):
        return np.exp(-1j * np.pi * lam * dz * (k_x**2 + k_y**2))

    def make_fresnel_kernel(self, KX=None, KY=None, dz=2.0, shape=None, dx=None):
        if KX is None or KY is None:
            if shape is None:
                KX = self.KX
                KY = self.KY
            else:
                dx = self.dx if dx is None else dx
                KX, KY = self.make_k_grid(shape, dx)
        return self.fresnel_propagator(KX, KY, dz, lam=self.wavelength)

    def shift(self, shiftx, shifty):
        return np.exp(-1j * 2 * np.pi * (shiftx * self.KX + shifty * self.KY))

    @staticmethod
    def gaussian_envelope(x, y, x_off, y_off, A, c):
        return A * np.exp(-(((x - x_off)**2 + (y - y_off)**2) / (2 * c**2)))

    @staticmethod
    def transmission_function_from_sigma(V, sigma):
        return np.exp(1j * sigma * V)

    def transmission_function(self, V, sigma=None):
        if sigma is None:
            sigma = self.interaction_constant
        return self.transmission_function_from_sigma(V, sigma)

    @staticmethod
    def _validate_patch_size(patch_size):
        if patch_size <= 0 or patch_size % 2 != 0:
            raise ValueError("patch_size must be a positive even integer.")

    @staticmethod
    def _validate_probe_shape(probe, patch_size):
        if probe.shape != (patch_size, patch_size):
            raise ValueError(
                f"probe shape {probe.shape} does not match patch_size={patch_size}."
            )

    @staticmethod
    def _patch_slices(array_shape, center_x, center_y, patch_size):
        MicroPtycho._validate_patch_size(patch_size)
        hp = patch_size // 2
        cx = array_shape[1] // 2 + center_x
        cy = array_shape[0] // 2 + center_y
        x0, x1 = cx - hp, cx + hp
        y0, y1 = cy - hp, cy + hp
        if x0 < 0 or y0 < 0 or x1 > array_shape[1] or y1 > array_shape[0]:
            raise ValueError(
                f"scan position ({center_x}, {center_y}) with patch_size={patch_size} "
                f"falls outside object bounds {array_shape}."
            )
        return slice(y0, y1), slice(x0, x1)

    @staticmethod
    def _scan_patch_geometry(position, dx, patch_size, array_shape):
        pixel_position = np.asarray(position, dtype=float) / dx
        integer_position = np.rint(pixel_position).astype(int)
        fractional_position = pixel_position - integer_position
        y_slice, x_slice = MicroPtycho._patch_slices(
            array_shape,
            integer_position[0],
            integer_position[1],
            patch_size,
        )
        shift_x = fractional_position[0] * dx
        shift_y = fractional_position[1] * dx
        return y_slice, x_slice, shift_x, shift_y

    @staticmethod
    def _shift_field(field, shift_x, shift_y, dx, KX=None, KY=None):
        if np.isclose(shift_x, 0.0) and np.isclose(shift_y, 0.0):
            return field.copy()
        if KX is None or KY is None:
            KX, KY = MicroPtycho.make_k_grid(field.shape, dx)
        phase_ramp = np.exp(-1j * 2 * np.pi * (shift_x * KX + shift_y * KY))
        return np.fft.ifft2(np.fft.fft2(field) * phase_ramp)

    @staticmethod
    def _apply_object_constraint(field, constraint):
        if constraint is None:
            return field
        if constraint in ("unit", "phase", "phase_only", "unit_modulus"):
            return np.exp(1j * np.angle(field))
        raise ValueError(
            "object_constraint must be one of None, 'unit', 'phase', "
            "'phase_only', or 'unit_modulus'."
        )

    @staticmethod
    def _inverse_transmission(transmission, eps):
        return np.conj(transmission) / np.maximum(np.abs(transmission)**2, eps)

    @staticmethod
    def _normalize_probe_energy(probe, target_energy, eps=1e-12):
        current_energy = np.sum(np.abs(probe)**2)
        if current_energy <= eps or target_energy <= eps:
            return probe
        return probe * np.sqrt(target_energy / current_energy)

    @staticmethod
    def _remove_phase_ramp(field, dx, eps=1e-12):
        """Remove the best-fit linear phase ramp from a complex field.

        Uses the standard maximum-likelihood momentum estimator:
            q_x = arg( Σ  ψ(x+dx) · ψ*(x) ) / (2π dx)
        Summing the complex products before taking the angle avoids the
        phase-wrapping / random-phase-tail issues that break a naive
        weighted mean of angle(·) values.
        """
        if field.size == 0:
            return field
        dx_prod = field[:, 1:] * np.conj(field[:, :-1])
        dy_prod = field[1:, :] * np.conj(field[:-1, :])
        sum_x = np.sum(dx_prod)
        sum_y = np.sum(dy_prod)
        if np.abs(sum_x) < eps and np.abs(sum_y) < eps:
            return field
        qx = np.angle(sum_x) / (2 * np.pi * dx)
        qy = np.angle(sum_y) / (2 * np.pi * dx)

        ny, nx = field.shape
        x = (np.arange(nx) - nx // 2) * dx
        y = (np.arange(ny) - ny // 2) * dx
        X, Y = np.meshgrid(x, y)
        return field * np.exp(-1j * 2 * np.pi * (qx * X + qy * Y))

    @staticmethod
    def align_global_phase(field, reference):
        """Align the global phase of field to reference via least-squares overlap."""
        overlap = np.vdot(reference, field)
        if np.isclose(np.abs(overlap), 0.0):
            return field
        return field * np.exp(-1j * np.angle(overlap))

    @staticmethod
    def align_phase_affine(field, reference, dx, weight_floor=1e-12):
        """
        Align global phase + linear phase ramp in `field` to `reference`.
        Useful for multislice ptychography where affine phase gauge freedom
        can make visually identical reconstructions look mismatched.
        """
        if field.shape != reference.shape:
            raise ValueError("field and reference must have identical shapes.")
        ny, nx = field.shape
        x = (np.arange(nx) - nx // 2) * dx
        y = (np.arange(ny) - ny // 2) * dx
        X, Y = np.meshgrid(x, y)

        phase_diff = np.angle(field * np.conj(reference))
        weight = np.maximum(np.abs(field) * np.abs(reference), weight_floor)
        A = np.column_stack((X.ravel(), Y.ravel(), np.ones(field.size)))
        Aw = A * np.sqrt(weight.ravel())[:, None]
        bw = phase_diff.ravel() * np.sqrt(weight.ravel())
        coeffs, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
        ax, ay, c = coeffs
        return field * np.exp(-1j * (ax * X + ay * Y + c))

    # ------------------------------------------------------------------ #
    #  Wave propagation
    # ------------------------------------------------------------------ #

    def propagate_wavefunction(self, input_wavefunction, KX=None, KY=None,
                               interaction_constant=None, dz=2.0):
        if self.potentials is None:
            raise ValueError("Set potentials first (pass V to constructor or assign self.potentials).")
        KX = KX if KX is not None else self.KX
        KY = KY if KY is not None else self.KY
        if interaction_constant is None:
            interaction_constant = self.interaction_constant
        slices = self.potentials.shape[0]
        psi = input_wavefunction.copy()
        for k in range(slices):
            V_k = self.potentials[k]
            T_k = self.transmission_function(V_k, interaction_constant)
            psi *= T_k
            psi = np.fft.ifft2(np.fft.fft2(psi) * self.fresnel_propagator(KX, KY, dz=dz, lam=self.wavelength))
        return psi

    def propagate(self, potential, input_wavefunction, KX=None, KY=None,
                  interaction_constant=None, dz=2.0):
        KX = KX if KX is not None else self.KX
        KY = KY if KY is not None else self.KY
        if interaction_constant is None:
            interaction_constant = self.interaction_constant
        psi = input_wavefunction.copy()
        T_k = self.transmission_function(potential, interaction_constant)
        psi *= T_k
        psi = np.fft.ifft2(np.fft.fft2(psi) * self.fresnel_propagator(KX, KY, dz=dz, lam=self.wavelength))
        return psi

    # ------------------------------------------------------------------ #
    #  Probe construction & scanning
    # ------------------------------------------------------------------ #

    def construct_probe(self, alpha=0.010, wavelength=None):
        if wavelength is None:
            wavelength = self.wavelength
        k_radius = np.sqrt(self.KX**2 + self.KY**2)
        kmax = alpha / wavelength
        aperture = np.where(k_radius <= kmax, 1, 0)
        probe = np.fft.ifft2(aperture)
        self.probe = probe
        return probe

    def extract_probe_patch(self, patch_size=24):
        if self.probe is None:
            raise ValueError("Call construct_probe() first.")
        self._validate_patch_size(patch_size)
        probe_shifted = np.fft.fftshift(self.probe)
        c = self.N // 2
        hp = patch_size // 2
        return probe_shifted[c - hp:c + hp, c - hp:c + hp]

    def tile_probe(self, grid_positions):
        if self.probe is None:
            raise ValueError("Call construct_probe() first.")
        probes = self.probe[:, :, np.newaxis].copy()
        for pos in grid_positions:
            shifted_probe = self._shift_field(self.probe, pos[0], pos[1], self.dx, self.KX, self.KY)
            probes = np.dstack((probes, shifted_probe))
        return probes

    def diffract_probe(self, probes, KX=None, KY=None):
        if self.potentials is None:
            raise ValueError("Set potentials first.")
        KX = KX if KX is not None else self.KX
        KY = KY if KY is not None else self.KY
        output = np.zeros_like(probes)
        for i in range(probes.shape[2]):
            output[:, :, i] = self.propagate_wavefunction(probes[:, :, i], KX, KY)
        return output

    # ------------------------------------------------------------------ #
    #  Scan positions
    # ------------------------------------------------------------------ #

    @staticmethod
    def construct_scan_positions(scan_margin=5, d_probe=2 * 0.86, overlap=0.3,
                                  scan_range=54):
        step = d_probe * (1 - overlap)
        scan_x = np.arange(-scan_range + scan_margin, scan_range - scan_margin, step)
        scan_y = np.arange(-scan_range + scan_margin, scan_range - scan_margin, step)
        scan_xx, scan_yy = np.meshgrid(scan_x, scan_y)
        positions = np.column_stack([scan_xx.ravel(), scan_yy.ravel()])
        return positions

    @staticmethod
    def isolate_grid_positions(scan_positions, sep, num=10):
        offsets = np.linspace(-((num - 1) / 2), (num - 1) / 2, num)
        grid_positions = []
        for iy in offsets:
            for ix in offsets:
                target = np.array([ix * sep, iy * sep])
                dists = np.linalg.norm(scan_positions - target, axis=1)
                grid_positions.append(scan_positions[np.argmin(dists)])
        return np.array(grid_positions)

    # ------------------------------------------------------------------ #
    #  Forward model — intensity generation
    # ------------------------------------------------------------------ #

    @staticmethod
    def create_true_object(A, V, sigma=0.0073):
        return A * np.exp(1j * sigma * V)

    @staticmethod
    def create_intensities(true_object, probe, dx, grid_positions, patch_size=24):
        MicroPtycho._validate_patch_size(patch_size)
        MicroPtycho._validate_probe_shape(probe, patch_size)
        Np = patch_size
        probe_KX, probe_KY = MicroPtycho.make_k_grid(probe.shape, dx)
        intensity = []
        for pos in grid_positions:
            y_slice, x_slice, shift_x, shift_y = MicroPtycho._scan_patch_geometry(
                pos, dx, Np, true_object.shape
            )
            patch = true_object[y_slice, x_slice]
            probe_at_position = MicroPtycho._shift_field(
                probe, shift_x, shift_y, dx, probe_KX, probe_KY
            )
            psi = probe_at_position * patch
            Psi = np.fft.fft2(psi)
            intensity.append(np.abs(Psi)**2)
        return np.array(intensity)

    def create_multislice_intensities(self, O_true, probe, dx, grid_positions,
                                       fresnel_kernel=None, patch_size=24,
                                       return_interwaves=False):
        if fresnel_kernel is None:
            fresnel_kernel = self.make_fresnel_kernel(shape=probe.shape, dx=dx)
        self._validate_patch_size(patch_size)
        self._validate_probe_shape(probe, patch_size)
        if fresnel_kernel.shape != probe.shape:
            raise ValueError("fresnel_kernel must match the probe patch shape.")
        Np = patch_size
        probe_KX, probe_KY = self.make_k_grid(probe.shape, dx)
        intensity = []
        interwaves = [] if return_interwaves else None
        for pos in grid_positions:
            slice_waves = [] if return_interwaves else None
            y_slice, x_slice, shift_x, shift_y = self._scan_patch_geometry(
                pos, dx, Np, O_true.shape[1:]
            )
            psi = self._shift_field(probe, shift_x, shift_y, dx, probe_KX, probe_KY)
            for i in range(O_true.shape[0]):
                patch = O_true[i, y_slice, x_slice]
                if return_interwaves:
                    slice_waves.append(psi.copy())
                psi = psi * patch
                psi = np.fft.ifft2(np.fft.fft2(psi) * fresnel_kernel)
            if return_interwaves:
                interwaves.append(slice_waves)
            Psi = np.fft.fft2(psi)
            intensity.append(np.abs(Psi)**2)
        if return_interwaves:
            interwaves = np.array(interwaves)
        return np.array(intensity), interwaves

    # ------------------------------------------------------------------ #
    #  Reconstruction — single-slice ePIE
    # ------------------------------------------------------------------ #

    @staticmethod
    def ePIE(n_iter, initial_probe, initial_object, intensity, grid_positions,
             dx, patch_size=24, alpha=1.0, beta=1.0,
             object_constraint=None, rho_object=0.2, rho_probe=0.2,
             normalize_probe=True, verbose=True, remove_probe_phase_ramp=True):
        MicroPtycho._validate_patch_size(patch_size)
        MicroPtycho._validate_probe_shape(initial_probe, patch_size)
        if len(intensity) != len(grid_positions):
            raise ValueError("intensity and grid_positions must have the same length.")
        positions = np.arange(len(intensity))
        residuals = []
        probe_KX, probe_KY = MicroPtycho.make_k_grid(initial_probe.shape, dx)
        eps = 1e-12
        target_probe_energy = np.sum(np.abs(initial_probe)**2)
        measured_amplitude = np.sqrt(np.maximum(intensity, 0.0))

        for i in range(n_iter):
            np.random.shuffle(positions)
            res = 0
            for pos in positions:
                y_slice, x_slice, shift_x, shift_y = MicroPtycho._scan_patch_geometry(
                    grid_positions[pos], dx, patch_size, initial_object.shape
                )
                patch = initial_object[y_slice, x_slice]
                shifted_probe = MicroPtycho._shift_field(
                    initial_probe, shift_x, shift_y, dx, probe_KX, probe_KY
                )
                psi = shifted_probe * patch
                Psi = np.fft.fft2(psi)
                Psi_new = measured_amplitude[pos] * Psi / (np.abs(Psi) + 1e-8)
                psi_new = np.fft.ifft2(Psi_new)
                error = psi_new - psi

                probe_intensity = np.abs(shifted_probe)**2
                patch_intensity = np.abs(patch)**2
                patch_update = alpha * np.conj(shifted_probe) / (
                    (1 - rho_object) * np.max(probe_intensity) + rho_object * probe_intensity + eps
                ) * error
                patch_updated = MicroPtycho._apply_object_constraint(
                    patch + patch_update,
                    object_constraint,
                )
                initial_object[y_slice, x_slice] = patch_updated

                if beta != 0:
                    probe_update = beta * np.conj(patch) / (
                        (1 - rho_probe) * np.max(patch_intensity) + rho_probe * patch_intensity + eps
                    ) * error
                    initial_probe += MicroPtycho._shift_field(
                        probe_update,
                        -shift_x,
                        -shift_y,
                        dx,
                        probe_KX,
                        probe_KY,
                    )
                res += np.sum((measured_amplitude[pos] - np.abs(Psi))**2)

            if normalize_probe and beta != 0:
                initial_probe = MicroPtycho._normalize_probe_energy(
                    initial_probe,
                    target_probe_energy,
                    eps=eps,
                )
            if beta != 0 and remove_probe_phase_ramp:
                initial_probe = MicroPtycho._remove_phase_ramp(initial_probe, dx=dx, eps=eps)
            residuals.append(res)
            if verbose:
                print(f"Iteration {i + 1}/{n_iter}, Residual: {res:.4e}")
        return initial_probe, initial_object, residuals

    # ------------------------------------------------------------------ #
    #  Reconstruction — multi-slice ePIE
    # ------------------------------------------------------------------ #

    def multislice_ePIE(self, n_iter, probe, O, intensity, grid_positions,
                        fresnel_kernel=None, dx=None, patch_size=24,
                        alpha_0=1e-3, beta_0=None, tau=10,
                        object_constraint=None, rho_object=0.2, rho_probe=0.2,
                        normalize_probe=True, remove_probe_phase_ramp=True,
                        random_seed=None,
                        verbose=True):
        if fresnel_kernel is None:
            fresnel_kernel = self.make_fresnel_kernel(shape=probe.shape, dx=self.dx if dx is None else dx)
        if dx is None:
            dx = self.dx
        self._validate_patch_size(patch_size)
        self._validate_probe_shape(probe, patch_size)
        if fresnel_kernel.shape != probe.shape:
            raise ValueError("fresnel_kernel must match the probe patch shape.")
        if len(intensity) != len(grid_positions):
            raise ValueError("intensity and grid_positions must have the same length.")
        if beta_0 is None:
            beta_0 = alpha_0
        positions = np.arange(len(intensity))
        rng = np.random.default_rng(random_seed)
        residuals = []
        probe_KX, probe_KY = self.make_k_grid(probe.shape, dx)
        eps = 1e-12
        target_probe_energy = np.sum(np.abs(probe)**2)
        measured_amplitude = np.sqrt(np.maximum(intensity, 0.0))

        for i in range(n_iter):
            rng.shuffle(positions)
            iter_residual = 0.0
            alpha = alpha_0 / (1 + i / tau)
            beta = beta_0 / (1 + i / tau)
            for j in positions:
                y_slice, x_slice, shift_x, shift_y = self._scan_patch_geometry(
                    grid_positions[j], dx, patch_size, O.shape[1:]
                )

                # --- Forward pass ---
                slice_waves = []
                shifted_probe = self._shift_field(probe, shift_x, shift_y, dx, probe_KX, probe_KY)
                psi = shifted_probe.copy()
                for k in range(O.shape[0]):
                    patch = O[k, y_slice, x_slice]
                    slice_waves.append(psi.copy())
                    psi = psi * patch
                    psi = np.fft.ifft2(np.fft.fft2(psi) * fresnel_kernel)

                # --- Magnitude constraint ---
                Psi = np.fft.fft2(psi)
                iter_residual += np.sum((measured_amplitude[j] - np.abs(Psi))**2)
                Psi_new = measured_amplitude[j] * Psi / (np.abs(Psi) + 1e-8)
                psi_corrected = np.fft.ifft2(Psi_new)

                # --- Backward pass ---
                for k in reversed(range(O.shape[0])):
                    psi_corrected = np.fft.ifft2(
                        np.fft.fft2(psi_corrected) * np.conj(fresnel_kernel)
                    )
                    patch = O[k, y_slice, x_slice].copy()
                    illum = slice_waves[k]
                    error = psi_corrected - illum * patch
                    illum_intensity = np.abs(illum)**2
                    patch_update = (
                        alpha
                        * np.conj(illum)
                        / (
                            (1 - rho_object) * np.max(illum_intensity)
                            + rho_object * illum_intensity
                            + eps
                        )
                        * error
                    )
                    patch_updated = self._apply_object_constraint(
                        patch + patch_update,
                        object_constraint,
                    )
                    O[k, y_slice, x_slice] = patch_updated
                    if k == 0 and beta != 0:
                        patch_intensity = np.abs(patch)**2
                        probe_update = beta * np.conj(patch) / (
                            (1 - rho_probe) * np.max(patch_intensity)
                            + rho_probe * patch_intensity
                            + eps
                        ) * error
                        probe += self._shift_field(
                            probe_update, -shift_x, -shift_y, dx, probe_KX, probe_KY
                        )
                    # Backpropagation should use the same (pre-update) transmission
                    # used to generate `error` above. Using the updated patch here
                    # mixes forward and backward models and slows convergence.
                    psi_corrected = self._inverse_transmission(patch, eps) * psi_corrected

            if normalize_probe and beta != 0:
                probe = self._normalize_probe_energy(probe, target_probe_energy, eps=eps)

            if beta != 0 and remove_probe_phase_ramp:
                # Remove linear phase tilt from probe to break probe-object ramp degeneracy.
                # A tilt in the probe is indistinguishable from a conjugate ramp in the object
                # because it leaves all diffraction patterns unchanged. Removing it each epoch
                # prevents the object from accumulating a compensating phase ramp.
                probe = self._remove_phase_ramp(probe, dx=dx, eps=eps)

            residuals.append(iter_residual)
            if verbose:
                print(f"Iteration {i + 1}/{n_iter} completed. Residual: {iter_residual:.6e}")

        return probe, O, residuals

    # ------------------------------------------------------------------ #
    #  Plotting helpers
    # ------------------------------------------------------------------ #

    def plot_probe(self, patch_size=None):
        if self.probe is None:
            raise ValueError("Call construct_probe() first.")
        probe_show = np.fft.fftshift(self.probe)
        if patch_size is not None:
            c = self.N // 2
            hp = patch_size // 2
            probe_show = probe_show[c - hp:c + hp, c - hp:c + hp]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(np.abs(probe_show)**2, cmap='inferno')
        axes[0].set_title('Probe intensity')
        axes[1].imshow(np.angle(probe_show), cmap='twilight')
        axes[1].set_title('Probe phase')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_scan_positions(grid_positions, probe_radius=None):
        fig, ax = plt.subplots(figsize=(6, 6))
        for pos in grid_positions:
            if probe_radius is not None:
                circle = plt.Circle(pos, radius=probe_radius, fill=False, alpha=0.3)
                ax.add_patch(circle)
            ax.plot(pos[0], pos[1], 'r.', markersize=3)
        ax.set_aspect('equal')
        ax.set_xlabel('x (Å)')
        ax.set_ylabel('y (Å)')
        ax.set_title('Scan positions')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_reconstruction(obj, obj_true=None, cmap='gray'):
        n_plots = 2 if obj_true is None else 4
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 2:
            axes[0].imshow(np.abs(obj), cmap=cmap)
            axes[0].set_title('Reconstructed amplitude')
            axes[1].imshow(np.angle(obj), cmap=cmap)
            axes[1].set_title('Reconstructed phase')
        else:
            axes[0].imshow(np.abs(obj), cmap=cmap)
            axes[0].set_title('Reconstructed amplitude')
            axes[1].imshow(np.angle(obj), cmap=cmap)
            axes[1].set_title('Reconstructed phase')
            axes[2].imshow(np.abs(obj_true), cmap=cmap)
            axes[2].set_title('True amplitude')
            axes[3].imshow(np.angle(obj_true), cmap=cmap)
            axes[3].set_title('True phase')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_multislice_reconstruction(O, O_true=None, cmap='gray'):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(np.sum(np.abs(O)**2, axis=0), cmap=cmap)
        axes[0].set_title('Reconstructed (sum |O|²)')
        axes[1].imshow(np.angle(np.sum(O, axis=0)), cmap=cmap)
        axes[1].set_title('Reconstructed (sum phase)')
        plt.tight_layout()
        plt.show()

        if O_true is not None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(np.sum(np.abs(O_true)**2, axis=0), cmap=cmap)
            axes[0].set_title('True (sum |O|²)')
            axes[1].imshow(np.angle(np.sum(O_true, axis=0)), cmap=cmap)
            axes[1].set_title('True (sum phase)')
            plt.tight_layout()
            plt.show()

    @staticmethod
    def plot_residuals(residuals):
        plt.figure()
        plt.semilogy(range(1, len(residuals) + 1), residuals)
        plt.xlabel('Iteration')
        plt.ylabel('Residual')
        plt.title('Convergence')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_diffraction_patterns(intensity, ncols=10, cmap='inferno'):
        n = intensity.shape[0]
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
        axes = np.atleast_2d(axes)
        for i in range(n):
            ax = axes[i // ncols, i % ncols]
            ax.imshow(np.log1p(intensity[i]), cmap=cmap)
            ax.axis('off')
        for i in range(n, nrows * ncols):
            axes[i // ncols, i % ncols].axis('off')
        plt.tight_layout()
        plt.show()
