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
        if constraint in ("phase_nonneg", "phase_positive"):
            # Physically exact for pure-phase objects built as exp(i·σ·V)
            # with V ≥ 0: forces |O|=1 and phase ≥ 0. Strongly regularizes
            # sparse ptychography by killing the phase-sign ambiguity that
            # phase_only still allows.
            phi = np.angle(field)
            return np.exp(1j * np.maximum(phi, 0.0))
        raise ValueError(
            "object_constraint must be one of None, 'unit', 'phase', "
            "'phase_only', 'unit_modulus', or 'phase_nonneg'."
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
    def _project_probe_to_aperture(probe, aperture):
        """Project a probe onto {F(probe) = aperture · exp(iφ), φ ∈ ℝ}.

        The idealized aberrated probe satisfies F(probe) = aperture(k) ·
        exp(-iχ(k)), so |F(probe)| equals the 0/1 aperture and only the
        phase χ is a free parameter. Forcing |F(probe)| = aperture is
        strictly stronger than multiplying by the mask (which only zeros
        k outside the aperture) and collapses the probe's degrees of
        freedom from `sum(aperture)` complex numbers to `sum(aperture)`
        *real* numbers (one phase per in-aperture bin). Critical for
        sparse ptychography where the probe update is weak and otherwise
        absorbs object features.
        """
        F = np.fft.fft2(probe)
        mag = np.abs(F)
        # Where |F| ~ 0, the phase is undefined — use 1 (which the
        # aperture-multiplication then zeros out anyway).
        safe_mag = np.where(mag > 1e-20, mag, 1.0)
        F_new = aperture * (F / safe_mag)
        return np.fft.ifft2(F_new)

    # Order kept as documented in `_krivanek_basis_at`. The wave-aberration
    # uses Krivanek/Haider notation with the radial coordinate normalized by
    # the aperture radius, so each coefficient is "phase contributed at the
    # aperture edge" in radians.
    KRIVANEK_LABELS = (
        'C1',                # defocus, ρ²
        'A1_cos', 'A1_sin',  # 2-fold astig, ρ²·cos2θ / ρ²·sin2θ
        'B2_cos', 'B2_sin',  # axial coma, ρ³·cosθ / ρ³·sinθ
        'A2_cos', 'A2_sin',  # 3-fold astig, ρ³·cos3θ / ρ³·sin3θ
        'C3',                # spherical, ρ⁴
        'S3_cos', 'S3_sin',  # axial star, ρ⁴·cos2θ / ρ⁴·sin2θ
        'A3_cos', 'A3_sin',  # 4-fold astig, ρ⁴·cos4θ / ρ⁴·sin4θ
    )

    @staticmethod
    def _krivanek_basis_at(kx, ky, kmax, mask):
        """Krivanek aberration basis evaluated at the in-aperture k-points.

        Returns a (n_in_aperture, 12) matrix B such that χ(k) ≈ B @ c, with
        c the coefficient vector (units: radians of phase at the aperture
        edge). The radial polynomials are *monomials* in ρ = k/kmax (the
        STEM convention), not orthogonal Zernikes — this means the columns
        are correlated, but the coefficients map directly to physical
        Krivanek/Haider aberrations (C1, A1, B2, A2, C3, S3, A3).
        """
        kx_in = kx[mask]
        ky_in = ky[mask]
        k = np.sqrt(kx_in**2 + ky_in**2)
        rho = k / max(kmax, 1e-12)
        theta = np.arctan2(ky_in, kx_in)
        rho2, rho3, rho4 = rho**2, rho**3, rho**4
        return np.stack([
            rho2,
            rho2 * np.cos(2 * theta), rho2 * np.sin(2 * theta),
            rho3 * np.cos(theta),     rho3 * np.sin(theta),
            rho3 * np.cos(3 * theta), rho3 * np.sin(3 * theta),
            rho4,
            rho4 * np.cos(2 * theta), rho4 * np.sin(2 * theta),
            rho4 * np.cos(4 * theta), rho4 * np.sin(4 * theta),
        ], axis=1)

    @staticmethod
    def _project_probe_to_krivanek(probe, template, basis, mask, c=None):
        """Project a probe onto {F = template · exp(iχ), χ = basis @ c}.

        Strong aberrations (e.g. Cs=1mm contributing ~6 rad of phase at
        the aperture edge) wrap several times across the aperture, so a
        naive arg() of F(probe)/template aliases. We unwrap the in-aperture
        phase in 2D (skimage's Goldstein-style algorithm), then solve the
        Krivanek-monomial LSQ in one shot — recovers canonical coefficients
        to machine precision regardless of initial state.

        The `c` argument is accepted for API symmetry but is no longer used;
        the unwrap-then-LSQ path is independent of warm-start.

        Returns (probe_new, c_new).
        """
        from skimage.restoration import unwrap_phase
        F = np.fft.fft2(probe)
        safe_template = np.where(np.abs(template) > 1e-20, template, 1.0)
        # Unwrap on an fftshifted layout so the aperture is a single
        # connected region in the centre of the array (Goldstein needs
        # connectivity). Mask the outside so unwrap doesn't try to
        # unwrap noise.
        phase_full = np.angle(F / safe_template)
        phase_shift = np.fft.fftshift(phase_full)
        mask_shift = np.fft.fftshift(mask)
        masked = np.ma.masked_array(phase_shift, ~mask_shift)
        unwrapped_shift = np.asarray(unwrap_phase(masked).filled(0.0))
        unwrapped = np.fft.ifftshift(unwrapped_shift)
        target = unwrapped[mask]
        # Add a temporary constant column to the basis so the LSQ can
        # absorb the global 2π·N offset that unwrap_phase picks
        # arbitrarily. Without this, c1 and c3 grow huge to fake a
        # constant via cancelling ρ² and ρ⁴ terms — meaningless when
        # interpreted as physical aberration coefficients. The constant
        # is discarded after fitting (a global probe phase is a free
        # gauge that doesn't affect diffraction).
        basis_with_const = np.column_stack([basis, np.ones(basis.shape[0])])
        c_full, *_ = np.linalg.lstsq(basis_with_const, target, rcond=None)
        c_new = c_full[:-1]
        F_fit = np.zeros_like(F)
        F_fit[mask] = template[mask] * np.exp(1j * (basis @ c_new))
        return np.fft.ifft2(F_fit), c_new

    @staticmethod
    def _remove_phase_ramp(field, dx, eps=1e-12):
        """Remove the best-fit linear phase ramp from a complex field.

        Uses a Fourier-moment k-centroid estimator:
            q_hat = Σ k · |F(P)(k)|² / Σ |F(P)(k)|²
        A linear phase ramp exp(i 2π q·r) in real space is a rigid shift
        of |F(P)|² in k-space, so the centroid is well-defined, wrap-free,
        and robust to strongly-aberrated probes where local phase
        gradients wrap over ±π within a single pixel (which biases naive
        finite-difference estimators such as angle(P(r+dx)·P*(r))).

        Matches the approach used by PyNX's `ZeroPhaseRamp` and
        PSI's fold_slice probe recentering.
        """
        if field.size == 0:
            return field
        F = np.fft.fftshift(np.fft.fft2(field))
        weight = np.abs(F) ** 2
        ny, nx = field.shape
        # Even-N FFT grids include the negative Nyquist but not the positive
        # one, so Σ k·|F|² is not exactly zero for real-valued signals due to
        # that unpaired bin. Drop the Nyquist row/column before the moment.
        if nx % 2 == 0:
            weight[:, 0] = 0.0
        if ny % 2 == 0:
            weight[0, :] = 0.0
        total = np.sum(weight)
        if total < eps:
            return field
        kx = np.fft.fftshift(np.fft.fftfreq(nx, dx))
        ky = np.fft.fftshift(np.fft.fftfreq(ny, dx))
        KX, KY = np.meshgrid(kx, ky)
        qx = np.sum(KX * weight) / total
        qy = np.sum(KY * weight) / total

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

    @staticmethod
    def align_translation(field, reference):
        """
        Remove an integer-pixel real-space translation between `field`
        and `reference`. ePIE has a translation gauge freedom — the
        reconstructed object and probe can drift together by any
        vector. With a periodic lattice this lets the algorithm
        converge to a position offset by an integer number of lattice
        periods from the ground truth, which visually "loses" a row
        of atoms at one edge while gaining ghost rows at the other.

        Uses complex cross-correlation and selects the peak in |xcorr|
        to find the shift, then rolls `field` onto the reference.
        Wrap-aware (the roll is exact for a periodic FOV).
        """
        if field.shape != reference.shape:
            raise ValueError("field and reference must have identical shapes.")
        # Use complex cross-correlation and maximize |xcorr| so the
        # estimated shift is invariant to unknown global phase between
        # `field` and `reference`. Correlating wrapped phase maps
        # directly can bias the peak on strongly periodic lattices and
        # produce the apparent "lost edge row / extra ghost row" effect.
        F1 = np.fft.fft2(field)
        F2 = np.fft.fft2(reference)
        xcorr = np.fft.fftshift(np.fft.ifft2(F1 * np.conj(F2)))
        xcorr_abs = np.abs(xcorr)
        peak_y, peak_x = np.unravel_index(np.argmax(xcorr_abs), xcorr_abs.shape)
        shift_y = peak_y - field.shape[0] // 2
        shift_x = peak_x - field.shape[1] // 2
        if shift_y == 0 and shift_x == 0:
            return field
        return np.roll(field, (-shift_y, -shift_x), axis=(-2, -1))

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
        # Target probe energy is derived from the measured data via Parseval's
        # theorem: Σ|FFT(ψ)|² = N² · Σ|ψ|² = N² · Σ|P|² · ⟨|O|²⟩ ≈ N² · Σ|P|²
        # for an |O|≈1 object. Using Σ|initial_probe|² here would lock in any
        # noise injected into the initial probe guess (PyNX and fold_slice
        # derive the target from data for the same reason).
        patch_pixels = initial_probe.shape[0] * initial_probe.shape[1]
        target_probe_energy = float(np.mean(np.sum(intensity, axis=(-2, -1))) / patch_pixels)
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
                        object_phase_shrink=0.0, probe_update_clip=0.0,
                        normalize_probe=True, remove_probe_phase_ramp=True,
                        probe_fourier_support=None, probe_warmup_iters=0,
                        probe_phase='free',
                        random_seed=None,
                        verbose=True):
        """
        Regularization knobs (on top of the usual ePIE options):

        probe_fourier_support : bool ndarray or None
            Boolean mask the size of the probe patch (k-space layout matching
            `np.fft.fft2(probe)`, i.e. *not* fftshifted). When set, the probe
            is projected onto the support after every probe update: for an
            aberrated convergent probe this enforces the physical aperture
            |k| ≤ α/λ. Essential for sparse samples where the probe update
            signal is weak and the probe tends to soak up noise.
        probe_warmup_iters : int
            Keep the probe fixed (beta=0) for this many iterations, giving
            the object a chance to settle before probe/object ambiguity
            kicks in. Recommended for sparse samples.
        """
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
        if probe_fourier_support is not None and probe_fourier_support.shape != probe.shape:
            raise ValueError("probe_fourier_support must have the probe shape.")
        if object_phase_shrink < 0:
            raise ValueError("object_phase_shrink must be >= 0.")
        if probe_update_clip < 0:
            raise ValueError("probe_update_clip must be >= 0.")
        if probe_phase not in ('free', 'parameterized'):
            raise ValueError("probe_phase must be 'free' or 'parameterized'.")
        if probe_phase == 'parameterized' and probe_fourier_support is None:
            raise ValueError(
                "probe_phase='parameterized' requires probe_fourier_support "
                "(the |F(probe)| amplitude template) — the parametric fit "
                "constrains only the in-aperture phase."
            )
        if beta_0 is None:
            beta_0 = alpha_0
        positions = np.arange(len(intensity))
        rng = np.random.default_rng(random_seed)
        residuals = []
        probe_KX, probe_KY = self.make_k_grid(probe.shape, dx)
        # Krivanek aberration basis (built once; mask + basis matrix don't
        # change between iters). The fit coefficients carry across iters so
        # each projection stays in the unwrapped basin (see
        # _project_probe_to_krivanek).
        krivanek_basis = None
        krivanek_mask = None
        krivanek_coefs = None
        if probe_phase == 'parameterized':
            krivanek_mask = probe_fourier_support > (probe_fourier_support.max() * 1e-6)
            k_in = np.sqrt(probe_KX[krivanek_mask]**2 + probe_KY[krivanek_mask]**2)
            kmax_eff = float(k_in.max()) if k_in.size else 1.0
            krivanek_basis = self._krivanek_basis_at(
                probe_KX, probe_KY, kmax_eff, krivanek_mask
            )
            krivanek_coefs = np.zeros(krivanek_basis.shape[1])
        eps = 1e-12
        # See ePIE: target probe energy is derived from measured data via
        # Parseval, not from the (possibly noisy) initial probe.
        patch_pixels = probe.shape[0] * probe.shape[1]
        target_probe_energy = float(np.mean(np.sum(intensity, axis=(-2, -1))) / patch_pixels)
        measured_amplitude = np.sqrt(np.maximum(intensity, 0.0))

        # Apply Fourier support once up-front so the initial probe is
        # consistent with the aperture constraint.
        if probe_fourier_support is not None:
            probe = self._project_probe_to_aperture(probe, probe_fourier_support)

        for i in range(n_iter):
            rng.shuffle(positions)
            iter_residual = 0.0
            alpha = alpha_0 / (1 + i / tau)
            beta = 0.0 if i < probe_warmup_iters else beta_0 / (1 + i / tau)
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
                        if probe_update_clip > 0:
                            mag = np.abs(probe_update)
                            probe_update = np.where(
                                mag > probe_update_clip,
                                probe_update * (probe_update_clip / (mag + eps)),
                                probe_update,
                            )
                        probe += self._shift_field(
                            probe_update, -shift_x, -shift_y, dx, probe_KX, probe_KY
                        )
                    # Backpropagation should use the same (pre-update) transmission
                    # used to generate `error` above. Using the updated patch here
                    # mixes forward and backward models and slows convergence.
                    psi_corrected = self._inverse_transmission(patch, eps) * psi_corrected

            # Free-phase amplitude lock per iter — F(probe) = template ·
            # exp(iφ(k)), φ unconstrained per k-bin. The parametric
            # snap (if requested) happens once after the loop, see below.
            if beta != 0 and probe_fourier_support is not None:
                probe = self._project_probe_to_aperture(probe, probe_fourier_support)

            # Amplitude-locked Fourier projection fixes the probe's total
            # energy to sum(|template|²)/N² by Parseval, so a separate
            # normalize step would fight the lock each iter. Skip it.
            if normalize_probe and beta != 0 and probe_fourier_support is None:
                probe = self._normalize_probe_energy(probe, target_probe_energy, eps=eps)

            if beta != 0 and remove_probe_phase_ramp:
                # Remove linear phase tilt from probe to break probe-object ramp degeneracy.
                # A tilt in the probe is indistinguishable from a conjugate ramp in the object
                # because it leaves all diffraction patterns unchanged. Removing it each epoch
                # prevents the object from accumulating a compensating phase ramp.
                probe = self._remove_phase_ramp(probe, dx=dx, eps=eps)
                # After phase-ramp removal, re-lock |F(probe)| to template
                # (the ramp removal is an in-real-space phase shift, which
                # doesn't change |F(probe)| in principle, but a fractional
                # real-space shift does alter the discrete |F| slightly).
                if probe_fourier_support is not None:
                    probe = self._project_probe_to_aperture(probe, probe_fourier_support)

            if object_phase_shrink > 0 and object_constraint in ("phase_nonneg", "phase_positive"):
                phi = np.maximum(np.angle(O) - object_phase_shrink, 0.0)
                O = np.exp(1j * phi)

            residuals.append(iter_residual)
            if verbose:
                print(f"Iteration {i + 1}/{n_iter} completed. Residual: {iter_residual:.6e}")

        # Post-hoc parametric snap. Free-phase ePIE converges to the
        # best per-k probe (best residual). The Krivanek snap then fits
        # those ~hundreds of free phases to 12 physical coefficients,
        # denoising the probe at the cost of a slightly higher
        # data-fit residual. The trade-off is bias-vs-variance: snap
        # gives a probe that's a *valid* aberrated probe by
        # construction, suitable for reporting microscope aberrations
        # or feeding into downstream simulations.
        if probe_phase == 'parameterized':
            probe, krivanek_coefs = self._project_probe_to_krivanek(
                probe, probe_fourier_support,
                krivanek_basis, krivanek_mask,
            )
            if verbose:
                print("Fitted Krivanek aberration coefficients (rad at aperture edge):")
                for label, value in zip(self.KRIVANEK_LABELS, krivanek_coefs):
                    print(f"  {label:>7s} = {value:+.4f}")

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
