# Gas inside the Planeterrella
import numpy as np
class Gas:
    def __init__(self, pressure : float, temperature : float, names : np.ndarray, fractions : np.ndarray):
        self.P = pressure
        self.T = temperature
        self.names = names
        self.fractions = fractions