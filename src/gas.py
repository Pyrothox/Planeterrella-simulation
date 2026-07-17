# Gas inside the Planeterrella
import numpy as np
class Gas:
    def __init__(self, pressure : float, temperature : float, names : list, fractions : list):
        self.P = pressure
        self.T = temperature
        self.names = names
        self.fractions = fractions