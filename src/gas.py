# Gas inside the Planeterrella
import numpy as np
class Gas:
    # store the properties of the gas inside the Planeterrella. 
    def __init__(self, pressure : float, temperature : float, names : list, fractions : list):
        self.P = pressure
        self.T = temperature
        self.names = names      # for now only O2 and N2 are supported. 
        self.fractions = fractions