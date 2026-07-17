import numpy as np
from scipy.interpolate import PchipInterpolator

_E_ELECTRON_MASS = 9.1093837015e-31   # kg (kept consistent with original MATLAB constant)
_EV_TO_J = 1.602e-19


def _channel(eV, energy_grid, cross_section_grid, energy_loss_eV, e_range=None):
    """
    Computes cross-section (m^2) and energy loss (eV) for a single channel,
    given the electron energy `eV`. Returns (0.0, 0.0) if eV is outside the
    valid range for this channel (matches original MATLAB behavior of
    padding with zeros to keep array sizes aligned).
    """
    lo, hi = (energy_grid[0], energy_grid[-1]) if e_range is None else e_range
    if not (lo < eV < hi):
        return 0.0, 0.0
    interpolator = PchipInterpolator(energy_grid, cross_section_grid)
    cross_section_cm2 = float(interpolator(eV))
    cross_section_m2 = cross_section_cm2 / 10000.0  # cm^2 -> m^2
    return cross_section_m2, energy_loss_eV


def cross_section_inelastic_N2(v5):
    """
    Given incident electron velocity vector v5, computes N2 inelastic
    cross sections and the associated energy-loss-equivalent velocity^2
    terms, for use in Monte Carlo collision sampling.

    Returns
    -------
    Qex_N2  : ndarray, ground-state excitation cross sections [m^2]
    Qex_N2i : ndarray, N2+ excited-state cross sections [m^2] (currently unused downstream)
    Qex_N2p : ndarray, N2+ production cross sections [m^2]
    v_N2    : ndarray, velocity^2 equivalent of energy loss for Qex_N2 channels [m^2/s^2]
    v_N2p   : ndarray, velocity^2 equivalent of energy loss for Qex_N2p channels [m^2/s^2]
    v_N2i   : ndarray, velocity^2 equivalent of energy loss for Qex_N2i channels [m^2/s^2]
    """
    v = np.linalg.norm(v5)
    Ec = 0.5 * _E_ELECTRON_MASS * v**2   # kinetic energy [J]
    eV = Ec / 1.6021765e-19              # kinetic energy [eV]

    Qex_N2, E_N2 = [], []
    Qex_N2p, E_N2p = [], []
    Qex_N2i, E_N2i = [], []

    # --- N2 ground-state excitation channels ---

    # A3 state <- ground (X1sigma)
    q, e = _channel(eV,
                     [10, 12, 15, 17, 21, 30, 50],
                     [15e-18, 18e-18, 20e-18, 21e-18, 15e-18, 6e-18, 4e-18],
                     6.1688)
    Qex_N2.append(q); E_N2.append(e)

    # B3 state <- ground (X1sigma)
    q, e = _channel(eV,
                     [10, 12, 15, 17, 20, 30, 50],
                     [25e-18, 31e-18, 22e-18, 19e-18, 13e-18, 8e-18, 3e-18],
                     7.3532)
    Qex_N2.append(q); E_N2.append(e)

    # C3 state <- ground (X1sigma)
    q, e = _channel(eV,
                     [12, 13, 14, 15, 19, 24, 30, 50],
                     [0.3e-17, 1.9e-17, 3.9e-17, 3.6e-17, 1.9e-17, 1e-17, 0.6e-17, 0.2e-17],
                     11.032)
    Qex_N2.append(q); E_N2.append(e)

    # C3 <- B3   NOTE: target-state assumption issue, see caveat above
    q, e = _channel(eV,
                     [11.4, 13, 14, 36, 140, 439, 1024],
                     [1.5e-19, 2.8e-18, 1e-17, 1.3e-18, 5.6e-20, 7.4e-21, 2.6e-21],
                     3.6788)
    Qex_N2.append(q); E_N2.append(e)

    # B3 <- A3   NOTE: same caveat as above
    q, e = _channel(eV,
                     [8.3, 8.8, 10.3, 12.8, 14.5, 16.9, 26.7],
                     [1.1e-18, 6.7e-18, 1.2e-17, 8e-18, 1.2e-17, 7.5e-18, 3.6e-18],
                     1.1844)
    Qex_N2.append(q); E_N2.append(e)

    # a1 state <- ground (X1sigma)
    q, e = _channel(eV,
                     [10, 15.4, 39, 403, 1645],
                     [3.5e-19, 1.7e-18, 1e-18, 1.1e-19, 3e-20],
                     8.5489)
    Qex_N2.append(q); E_N2.append(e)

    # --- N2+ production channels ---

    # N2+ ground state (X2sigma)
    pX2eV = [19.5888, 23.3916, 27.5576, 34.9814, 45.8446, 53.1593, 66.2993,
              76.0362, 85.6641, 98.4044, 120.155, 151.098, 209.725, 252.790,
              305.229, 404.099, 506.155, 605.280, 704.478, 803.857, 903.273, 1000]
    pX2Qex = np.array([0.181311, 2.56943, 6.21522, 10.3626, 15.6405, 19.4106,
                        21.7943, 23.1733, 24.1750, 25.1753, 25.0393, 24.3958,
                        22.6071, 20.8258, 19.1659, 16.6035, 14.2913, 12.6093,
                        11.1789, 10.3773, 9.70149, 8.90137]) * 1e-17
    q, e = _channel(eV, pX2eV, pX2Qex, 15.591)
    Qex_N2p.append(q); E_N2p.append(e)

    # N2+ A2 state
    q, e = _channel(eV,
                     [19.7, 34, 85.6, 496],
                     [6.5e-18, 3.2e-17, 6e-17, 3.3e-17],
                     16.699)
    Qex_N2p.append(q); E_N2p.append(e)

    # N2+ B2 state
    q, e = _channel(eV,
                     [17, 45, 77.6, 114, 247.5],
                     [0.35e-18, 21.4e-18, 26.5e-18, 27e-18, 21.7e-18],
                     18.751)
    Qex_N2p.append(q); E_N2p.append(e)

    # --- N2+ ion channels (currently unused downstream, kept for completeness) ---

    # (N2)+ A2 <- ground (X2sigma)
    q, e = _channel(eV,
                     [18, 33, 89, 234, 444],
                     [1.2e-18, 8e-18, 1.4e-17, 1.1e-17, 6.9e-18],
                     16.699)
    Qex_N2i.append(q); E_N2i.append(e)

    # (N2)+ B2 <- ground (X2sigma)
    q, e = _channel(eV,
                     [18, 19, 33.6, 49, 101.5, 360, 1171],
                     [1.2e-19, 5.1e-19, 1e-17, 1.6e-17, 1.8e-17, 1.2e-17, 5e-18],
                     18.751)
    Qex_N2i.append(q); E_N2i.append(e)

    # --- convert energy loss [eV] -> velocity^2 equivalent [m^2/s^2] ---
    E_N2  = np.array(E_N2)  * _EV_TO_J
    E_N2p = np.array(E_N2p) * _EV_TO_J
    E_N2i = np.array(E_N2i) * _EV_TO_J

    v_N2  = 2.0 * E_N2  / _E_ELECTRON_MASS
    v_N2p = 2.0 * E_N2p / _E_ELECTRON_MASS
    v_N2i = 2.0 * E_N2i / _E_ELECTRON_MASS

    return (np.array(Qex_N2), np.array(Qex_N2i), np.array(Qex_N2p),
            v_N2, v_N2p, v_N2i)