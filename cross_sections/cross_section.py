import numpy as np
from numba import njit
_E_CHARGE = 1.602176634e-19
_ME = 9.1093837015e-31

@njit(cavhe = True)
def cross_section(v, alive):
    """
    Given incident electron velocity vector v, computes N2 and O2 cross sections
    for use in Monte Carlo collision sampling.

    Parameters
    ----------
    v : ndarray
        array of incident electron velocity vectors [m/s] (N,3)
    alive : ndarray
        boolean array indicating which electrons are still alive (N,)

    Returns
    -------
    Q_N2 : float
        N2 cross section [m^2]
    Q_O2 : float
        O2 cross section [m^2]
    """
    N = v.shape[0]
    S_N2 = np.zeros(N)
    S_O2 = np.zeros(N)

    for i in range(N):
        if not alive[i]:
            continue
        vx = v[i, 0]
        vy = v[i, 1]
        vz = v[i, 2]
        Ec = 0.5 * _ME * (vx**2 + vy**2 + vz**2)/_E_CHARGE  # Kinetic energy in eV

        # N2 cross section      (following provided matlab code)
        if Ec < 3.5:
            p1 = 1.67e-16
            p2 = -1.588e-16
            p3 = 5.574e-16
            S_N2[i] = p1*Ec*Ec + p2*Ec + p3

        elif Ec < 5.25:
            p1 = -4.769e-16
            p2 = 3.676e-15
            S_N2[i] = p1*Ec + p2

        else:
            a = 1.11e-15
            b = -0.005322
            c = 1.02e-16
            d = -0.0002517
            S_N2[i] = a*np.exp(b*Ec) + c*np.exp(d*Ec)

    # O2 cross section      (following provided matlab code)
        if Ec < 14.5:
            p1 = -1.063e-19
            p2 = 3.503e-18
            p3 = -3.998e-17
            p4 = 2.134e-16
            p5 = 4.225e-16

            S_O2[i] = (
                p1*Ec**4
                + p2*Ec**3
                + p3*Ec**2
                + p4*Ec
                + p5
            )

        elif Ec < 50:
            p1 = -1.032e-17
            p2 = 1.244e-15
            S_O2[i] = p1*Ec + p2

        else:
            a = 9.619e-15
            b = -0.6549
            c = -9.802e-19
            S_O2[i] = a*Ec**b + c
    
    return S_N2, S_O2