import numpy as np
from scipy.interpolate import PchipInterpolator

_ME = 9.1093837015e-31   # kg (kept consistent with original MATLAB constant)
_EV_TO_J = 1.602176634e-19

class _Channel:
    """One inelastic transition: precomputed interpolator + validity range + energy loss."""
    __slots__ = ("interp", "lo", "hi", "loss_eV")
    
    def __init__(self, energy_grid, cross_section_grid, e_range, energy_loss_eV):
        self.interp = PchipInterpolator(energy_grid, cross_section_grid)
        self.loss_eV = energy_loss_eV
        self.lo, self.hi = e_range
    
    def evaluate(self, eV):
        """
        Evaluate the cross-section at given electron energies.

        Parameters
        ----------
        eV: (N,) array of electron energie in eV

        Returns
        -------
        (N,) array of cross-sections m^2, zero outside the valid range
        """
        out = np.zeros_like(eV)
        mask = (eV > self.lo) & (eV < self.hi)
        if np.any(mask):
            out[mask] = self.interp(eV[mask]) / 10000.0  # cm^2 -> m^2
        return out
    

# Creating channel instances for N2 inelastic transitions once 


# Reaction O2 + e- -> O2* + e- (ground-state excitation)
_O2_CHANNELS = [
    # B3 state <- ground (X3)  (labelled C3sigma_u- in Physics of the Aurora and Airglow)
    _Channel([9.74460,15.1278,20.1081,30.4191,51.0347,92.5320,147.414,267.280,404.335,551.535],
             [3.20345e-16,6.44153e-16,9.01688e-16,7.92276e-16,6.27701e-16,4.14933e-16,
              2.88851e-16,1.90941e-16,1.36405e-16,1.02621e-16],
             (10,550), 6.120),
    
    # a1 state <- ground (X3)
    _Channel([1.47321,1.83076,2.71848,3.53725,6.09391,10.7195,16.5424,23.0255,36.3334,
              50.4421,73.5572,99.0133,144.719,206.684],
             [5.78351e-19,1.62394e-18,3.79688e-18,6.51773e-18,9.06796e-18,8.12475e-18,
              5.92736e-18,4.21872e-18,2.25575e-18,1.27204e-18,5.82882e-19,2.06392e-19,
              1.16319e-19,8.27644e-20],
             (1,205), 0.977),
    
    # b1 state <- ground (X3)
    _Channel([1.73211,2.00713,2.85972,3.44639,6.11530,10.2591,17.1658,30.9949,54.4567,
              63.1765,71.3378,89.1277,120.515,151.578],
             [3.26740e-20,1.62256e-19,7.83344e-19,1.27806e-18,1.87204e-18,1.86118e-18,
              1.46602e-18,1.01376e-18,6.16131e-19,3.39226e-19,1.68456e-19,6.79347e-20,
              3.11529e-20,2.27788e-20],
             (1,150), 1.627),
    
    # A3+C3+c1 states <- ground (X3), unresolved triplet
    _Channel([7.52587,9.18439,11.1265,20.0687,34.2123,53.8234,87.0564,152.826,283.216,496.862],
             [6.52899e-18,1.48255e-17,1.79210e-17,1.05704e-17,5.14950e-18,2.72630e-18,
              1.52424e-18,8.99335e-19,5.30286e-19,2.96189e-19],
             (7.6,496), 4.262),
]

#Reaction O2 + e- -> O2+ + 2e- (ionization)
_O2P_CHANNELS = [
    # O2+ ground state (X2)
    _Channel([16.2285,31.3318,52.1857,79.7560,107.446,136.806,167.823],
             [0.139095e-16,0.847252e-16,1.47442e-16,1.74421e-16,1.79449e-16,1.77169e-16,1.70020e-16],
             (16.3,167), 12.072),
    # O2+ B4 state (production from ground; emission observed B4->A4 only)
    _Channel([18.0957,22.3092,30.5386,43.2876,81.1131,168.761,389.860,1149.76,2848.04],
             [6.35375e-19,2.31013e-18,7.05480e-18,1.46780e-17,2.00923e-17,1.80957e-17,
              1.23285e-17,6.81292e-18,3.63585e-18],
             (18.1,2800), 18.171),
]

# Reaction O2 + e- = O2- (capture)
_O2C_LOSS_EV = 0.0      #No implementation 


def cross_section_inelastic_O2(eV):
    """
    Compute the total inelastic cross-section for O2 at given electron energies.

    Parameters
    ----------
    eV : ndarray
        array of incident electron energies [eV] (N,)

    Returns
    -------
    Qex_O2  : (n_ch1, N) cross sections [m^2], ground-state excitation
    Qex_O2p : (n_ch2, N) cross sections [m^2], O2+ production
    Qex_O2c : (n_ch3, N) cross sections [m^2], O2- capture
    loss_o2  : (n_ch1,) energy loss [J] for each O2 excitation channel
    loss_o2p : (n_ch2,) energy loss [J] for each O2+ production channel
    loss_o2c : (n_ch3,) energy loss [J] for each O2- capture channel
    """

    Qex_O2 = np.stack([channel.evaluate(eV) for channel in _O2_CHANNELS], axis=0)
    Qex_O2p = np.stack([channel.evaluate(eV) for channel in _O2P_CHANNELS], axis=0)
    Qex_O2c = np.zeros((1, eV.shape[0]))  # Placeholder for O2- capture cross-section

    loss_o2 = np.array([channel.loss_eV for channel in _O2_CHANNELS]) * _EV_TO_J
    loss_o2p = np.array([channel.loss_eV for channel in _O2P_CHANNELS]) * _EV_TO_J
    loss_o2c = np.array([_O2C_LOSS_EV]) # not implemented
    
    return Qex_O2, Qex_O2p, Qex_O2c, loss_o2, loss_o2p, loss_o2c
