import numpy as np
from scipy.interpolate import PchipInterpolator

_ME = 9.1093837015e-31   # kg (kept consistent with original MATLAB constant)
_EV_TO_J = 1.602176634e-19

class _Channel:
    """One inelastic transition: precomputed interpolator + validity range + energy loss."""
    __slots__ = ("interp", "lo", "hi", "loss_eV", "color_uint")
    
    def __init__(self, energy_grid, cross_section_grid, e_range, energy_loss_eV, color = "#00000000"):
        self.interp = PchipInterpolator(energy_grid, cross_section_grid)
        self.loss_eV = energy_loss_eV
        self.lo, self.hi = e_range
        self.color_uint = np.uint32(int(color.lstrip("#"), 16))
    
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


# Reaction N2 + e- -> N2* + e- (ground-state excitation)
_N2_CHANNELS = [
    # A3 state to ground (X1epsilon) A³Σu⁺ 
    _Channel([10,12,15,17,21,30,50],
             [15e-18,18e-18,20e-18,21e-18,15e-18,6e-18,4e-18], (10,50), 6.1688),  #invisible, UV
    # B3 state to ground (X1epsilon) B³Πg
    _Channel([10,12,15,17,20,30,50],
             [25e-18,31e-18,22e-18,19e-18,13e-18,8e-18,3e-18], (10,50), 7.3532),    # invisible
    # C3 state to ground (X1epsilon) C³Πu
    _Channel([12,13,14,15,19,24,30,50],
             [0.3e-17,1.9e-17,3.9e-17,3.6e-17,1.9e-17,1e-17,0.6e-17,0.2e-17], (12,50), 11.032), 
    # C3 to B3 (C³Πu -> B³Πg) N2 SECOND POSITIVE 
    _Channel([11.4,13,14,36,140,439,1024],
             [1.5e-19,2.8e-18,1e-17,1.3e-18,5.6e-20,7.4e-21,2.6e-21], (11.4,1024), 3.6788, "#6600CCFF"),    #purple,  visible
    # B3 to A3 (B³Πg -> A³Σu⁺) N2 FIRST POSITIVE
    _Channel([8.3,8.8,10.3,12.8,14.5,16.9,26.7],
             [1.1e-18,6.7e-18,1.2e-17,8e-18,1.2e-17,7.5e-18,3.6e-18], (8.3,26.7), 1.1844, "#FF3300FF"),  #red, visible
    # a1 state to ground (X1epsilon) a¹Πg
    _Channel([10,15.4,39,403,1645],
             [3.5e-19,1.7e-18,1e-18,1.1e-19,3e-20], (10,1645), 8.5489),
]

#Reaction N2 + e- -> N2+ + 2e- (ionization)
_N2P_CHANNELS = [
    # production of N2+ in ground state (X2) N2⁺ X²Σg⁺
    _Channel([19.5888,23.3916,27.5576,34.9814,45.8446,53.1593,66.2993,76.0362,
              85.6641,98.4044,120.155,151.098,209.725,252.790,305.229,404.099,
              506.155,605.280,704.478,803.857,903.273,1000],
             np.array([0.181311,2.56943,6.21522,10.3626,15.6405,19.4106,21.7943,
                       23.1733,24.1750,25.1753,25.0393,24.3958,22.6071,20.8258,
                       19.1659,16.6035,14.2913,12.6093,11.1789,10.3773,9.70149,
                       8.90137]) * 1e-17,
             (19.6,1000), 15.591),  
    # production of N2+ at A2 state N2⁺ A²Πu
    _Channel([19.7,34,85.6,496], [6.5e-18,3.2e-17,6e-17,3.3e-17], (19.7,496), 16.699),
    # production of N2+ at B2 state N2⁺ B²Σu⁺   N2+ first negative
    _Channel([17,45,77.6,114,247.5], [0.35e-18,21.4e-18,26.5e-18,27e-18,21.7e-18],
             (17,247.5), 18.751, "#3300FFFF"),  #blue, visible
]

# Reaction N2+ + e- -> (N2+)* + e- (ionization of N2+)
_N2I_CHANNELS = [
    # N2+ A2 state to ground (X2) N2⁺ A²Πu
    _Channel([18,33,89,234,444], [1.2e-18,8e-18,1.4e-17,1.1e-17,6.9e-18], (18,444), 16.699),
    # (N2)+ B2 state to ground (X2epsilon)
    _Channel([18,19,33.6,49,101.5,360,1171],
             [1.2e-19,5.1e-19,1e-17,1.6e-17,1.8e-17,1.2e-17,5e-18], (18,1171), 18.751),
]


def cross_section_inelastic_N2(eV):
    """
    Compute the total inelastic cross-section for N2 at given electron energies using precomputed interpolators for various excitation and ionization channels.

    Parameters
    ----------
    eV : ndarray
        array of incident electron energies [eV] (N,)

    Returns
    -------
    Qex_N2  : (n_ch1, N) cross sections [m^2], ground-state excitation
    Qex_N2p : (n_ch2, N) cross sections [m^2], N2+ ion excitation (unused downstream)
    Qex_N2i : (n_ch3, N) cross sections [m^2], N2+ production
    loss_n2  : (n_ch1,) energy loss [eV] for each N2 excitation channel
    loss_n2p : (n_ch3,) energy loss [eV] for each N2+ production channel
    loss_n2i : (n_ch2,) energy loss [eV] for each N2+ ion excitation channel
    col_n2    : (n_ch1,) emission color [uint32] for each N2 excitation channel
    col_n2p   : (n_ch2,) emission color [uint32] for each N2+ ion excitation channel
    col_n2i   : (n_ch3,) emission color [uint32] for each N2+ production channel
    """

    Qex_N2 = np.stack([channel.evaluate(eV) for channel in _N2_CHANNELS], axis=0)
    Qex_N2p = np.stack([channel.evaluate(eV) for channel in _N2P_CHANNELS], axis=0)
    Qex_N2i = np.stack([channel.evaluate(eV) for channel in _N2I_CHANNELS], axis=0)
    
    loss_n2 = np.array([channel.loss_eV for channel in _N2_CHANNELS]) 
    loss_n2p = np.array([channel.loss_eV for channel in _N2P_CHANNELS]) 
    loss_n2i = np.array([channel.loss_eV for channel in _N2I_CHANNELS]) 

    col_n2 = np.array([channel.color_uint for channel in _N2_CHANNELS]) 
    col_n2p = np.array([channel.color_uint for channel in _N2P_CHANNELS]) 
    col_n2i = np.array([channel.color_uint for channel in _N2I_CHANNELS])

    return Qex_N2, Qex_N2p, Qex_N2i, loss_n2, loss_n2p, loss_n2i, col_n2, col_n2p, col_n2i
