import numpy as np


_PLANCK_CONSTANT = 6.62607015e-34
_ELECTRON_MASS = 9.1093837015e-31
_ELEMENTARY_CHARGE = 1.602176634e-19
_SPEED_OF_LIGHT = 299792458.0


def wavelength_from_voltage(voltage):
    """Relativistic electron wavelength in Angstroms."""
    return 12.264 / np.sqrt(voltage + 0.9788e-6 * voltage**2)


def interaction_constant_from_voltage(voltage, potential_units="V_nm"):
    """Relativistic electron interaction constant.

    Parameters
    ----------
    voltage : float
        Accelerating voltage in Volts.
    potential_units : {"V_A", "V_nm"}
        Units assumed for the projected potential used in the transmission
        function. The historical code path used V*nm, so that remains the
        default for backwards-compatible magnitudes.
    """
    lam_angstrom = wavelength_from_voltage(voltage)
    lam_meters = lam_angstrom * 1e-10
    gamma = 1.0 + (_ELEMENTARY_CHARGE * voltage) / (_ELECTRON_MASS * _SPEED_OF_LIGHT**2)
    sigma_per_angstrom = (
        2
        * np.pi
        * gamma
        * _ELECTRON_MASS
        * _ELEMENTARY_CHARGE
        * lam_meters
        / _PLANCK_CONSTANT**2
        * 1e-10
    )

    if potential_units == "V_A":
        return sigma_per_angstrom
    if potential_units == "V_nm":
        return sigma_per_angstrom * 10.0
    raise ValueError("potential_units must be 'V_A' or 'V_nm'.")
