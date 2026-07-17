import numpy as np
from scipy.interpolate import PchipInterpolator

_E_ELECTRON_MASS = 9.1093837015e-31   # kg (kept consistent with original MATLAB constant)
_EV_TO_J = 1.602e-19


def _channel(eV, energy_grid, cross_section_grid, energy_loss_eV, e_range):
    """
    Computes cross-section (m^2) and energy loss (eV) for a single channel,
    given the electron energy `eV`. Returns (0.0, 0.0) if eV is outside the
    valid range for this channel (matches original MATLAB behavior of
    padding with zeros to keep array sizes aligned).
    """
    lo, hi = e_range
    if not (lo < eV < hi):
        return 0.0, 0.0
    interpolator = PchipInterpolator(energy_grid, cross_section_grid)
    cross_section_m2 = float(interpolator(eV)) / 10000.0  # cm^2 -> m^2
    return cross_section_m2, energy_loss_eV


def cross_section_inelastic_O2(v5):
    """
    Given incident electron velocity vector v5, computes O2 inelastic
    cross sections and the associated energy-loss-equivalent velocity^2
    terms, for use in Monte Carlo collision sampling.

    Returns
    -------
    Qex_O2  : ndarray, ground-state excitation cross sections [m^2]
    Qex_O2p : ndarray, O2+ production cross sections [m^2]
    Qex_O2c : ndarray, O2- attachment cross sections [m^2] (currently a
              placeholder — no attachment data implemented, always 0)
    v_O2, v_O2p, v_O2c : ndarrays, velocity^2 equivalent of energy loss [m^2/s^2]
    """
    v = np.linalg.norm(v5)
    Ec = 0.5 * _E_ELECTRON_MASS * v**2
    eV = Ec / 1.6021765e-19

    Qex_O2, E_O2 = [], []
    Qex_O2p, E_O2p = [], []

    # --- O2 ground-state excitation channels ---

    # B3 state <- ground (X3)  (labelled C3sigma_u- in Physics of the Aurora and Airglow)
    q, e = _channel(eV,
                     [9.74460, 15.1278, 20.1081, 30.4191, 51.0347, 92.5320,
                      147.414, 267.280, 404.335, 551.535],
                     [3.20345e-16, 6.44153e-16, 9.01688e-16, 7.92276e-16,
                      6.27701e-16, 4.14933e-16, 2.88851e-16, 1.90941e-16,
                      1.36405e-16, 1.02621e-16],
                     6.120, e_range=(10, 550))
    Qex_O2.append(q); E_O2.append(e)

    # a1 state <- ground (X3)
    q, e = _channel(eV,
                     [1.47321, 1.83076, 2.71848, 3.53725, 6.09391, 10.7195,
                      16.5424, 23.0255, 36.3334, 50.4421, 73.5572, 99.0133,
                      144.719, 206.684],
                     [5.78351e-19, 1.62394e-18, 3.79688e-18, 6.51773e-18,
                      9.06796e-18, 8.12475e-18, 5.92736e-18, 4.21872e-18,
                      2.25575e-18, 1.27204e-18, 5.82882e-19, 2.06392e-19,
                      1.16319e-19, 8.27644e-20],
                     0.977, e_range=(1, 205))
    Qex_O2.append(q); E_O2.append(e)

    # b1 state <- ground (X3)
    q, e = _channel(eV,
                     [1.73211, 2.00713, 2.85972, 3.44639, 6.11530, 10.2591,
                      17.1658, 30.9949, 54.4567, 63.1765, 71.3378, 89.1277,
                      120.515, 151.578],
                     [3.26740e-20, 1.62256e-19, 7.83344e-19, 1.27806e-18,
                      1.87204e-18, 1.86118e-18, 1.46602e-18, 1.01376e-18,
                      6.16131e-19, 3.39226e-19, 1.68456e-19, 6.79347e-20,
                      3.11529e-20, 2.27788e-20],
                     1.627, e_range=(1, 150))
    Qex_O2.append(q); E_O2.append(e)

    # A3+C3+c1 states <- ground (X3), unresolved triplet
    q, e = _channel(eV,
                     [7.52587, 9.18439, 11.1265, 20.0687, 34.2123, 53.8234,
                      87.0564, 152.826, 283.216, 496.862],
                     [6.52899e-18, 1.48255e-17, 1.79210e-17, 1.05704e-17,
                      5.14950e-18, 2.72630e-18, 1.52424e-18, 8.99335e-19,
                      5.30286e-19, 2.96189e-19],
                     4.262, e_range=(7.6, 496))
    Qex_O2.append(q); E_O2.append(e)

    # --- O2+ production channels ---

    # O2+ ground state (X2)
    q, e = _channel(eV,
                     [16.2285, 31.3318, 52.1857, 79.7560, 107.446, 136.806, 167.823],
                     [0.139095e-16, 0.847252e-16, 1.47442e-16, 1.74421e-16,
                      1.79449e-16, 1.77169e-16, 1.70020e-16],
                     12.072, e_range=(16.3, 167))
    Qex_O2p.append(q); E_O2p.append(e)

    # O2+ B4 state (production from ground; emission observed B4->A4 only)
    q, e = _channel(eV,
                     [18.0957, 22.3092, 30.5386, 43.2876, 81.1131, 168.761,
                      389.860, 1149.76, 2848.04],
                     [6.35375e-19, 2.31013e-18, 7.05480e-18, 1.46780e-17,
                      2.00923e-17, 1.80957e-17, 1.23285e-17, 6.81292e-18,
                      3.63585e-18],
                     18.171, e_range=(18.1, 2800))
    Qex_O2p.append(q); E_O2p.append(e)

    # --- O2- attachment: not implemented, placeholder only ---
    Qex_O2c = np.array([0.0])
    E_O2c = np.array([0.0])

    # --- convert energy loss [eV] -> velocity^2 equivalent [m^2/s^2] ---
    E_O2  = np.array(E_O2)  * _EV_TO_J
    E_O2p = np.array(E_O2p) * _EV_TO_J
    E_O2c = E_O2c * _EV_TO_J

    v_O2  = 2.0 * E_O2  / _E_ELECTRON_MASS
    v_O2p = 2.0 * E_O2p / _E_ELECTRON_MASS
    v_O2c = 2.0 * E_O2c / _E_ELECTRON_MASS

    return (np.array(Qex_O2), np.array(Qex_O2p), Qex_O2c,
            v_O2, v_O2p, v_O2c)