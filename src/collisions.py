import numpy as np
from src.particles import Electrons
from cross_sections import cross_section_inelastic_N2, cross_section_inelastic_O2, cross_section 
_E_CHARGE = 1.602176634e-19
_ME = 9.1093837015e-31
_R_GAS = 8.314
_NA = 6.022e23

class CollisionEngine:
    def __init__(self, gas):
        self.gas = gas
        self.pressure = gas.P
        self.temperature = gas.T
        self.rng = np.random.default_rng()  # Random number generator for collision probabilities
        self.col_n2 = 0.0  # Placeholder for N2 collision cross-section
        self.col_o2 = 0.0  # Placeholder for O2 collision cross-section
        self.fracN2 = gas.fractions[gas.names.index("N2")]
        self.fracO2 = gas.fractions[gas.names.index("O2")]
        self.nN2 = self.fracN2 * self.pressure / (self.temperature * _R_GAS) * _NA  # Number density of N2
        self.nO2 = self.fracO2 * self.pressure / (self.temperature * _R_GAS) * _NA  # Number density of O2

    def collide(self, electrons: Electrons, diagnostics):
        alive = electrons.alive
        pos = electrons.position[alive]
        vel = electrons.velocity[alive]
        dt = electrons.dtarray[alive]

        vel_magnitude = np.linalg.norm(vel, axis=1)
        Ec = 0.5 * _ME * vel_magnitude**2  # Kinetic energy in J
        eV = Ec / _E_CHARGE  # Kinetic energy in eV

        T = self.temperature
        P = self.pressure
        nN2 = self.nN2
        nO2 = self.nO2

        # N2
        