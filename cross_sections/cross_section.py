def cross_section(v, S_N2, S_O2):
    """
    Given incident electron velocity vector v, computes N2 and O2 cross sections
    for use in Monte Carlo collision sampling.

    Parameters
    ----------
    v : ndarray
        Incident electron velocity vector [m/s]
    S_N2 : function
        Function to compute N2 cross section given electron energy [eV]
    S_O2 : function
        Function to compute O2 cross section given electron energy [eV]

    Returns
    -------
    Q_N2 : float
        N2 cross section [m^2]
    Q_O2 : float
        O2 cross section [m^2]
    """
    v_magnitude = np.linalg.norm(v)
    Ec = 0.5 * _ME * v_magnitude**2  # kinetic energy in J
    eV = Ec / _E_CHARGE  # kinetic energy in eV

    Q_N2 = S_N2(eV)  # N2 cross section in m^2
    Q_O2 = S_O2(eV)  # O2 cross section in m^2

    return Q_N2, Q_O2