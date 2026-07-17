# Electric and Magnetic fields for the Planeterrella simulation
import numpy as np
from abc import ABC, abstractmethod
from src.geometry import Planeterrella, Dipole

class Fields(ABC):
    @abstractmethod
    def at(self, position: np.ndarray) -> np.ndarray:
        pass

class MagneticField(Fields):
    def __init__(self, planeterrella: Planeterrella):
        self.dipoles: list[Dipole] = planeterrella.dipoles

    def at(self, position: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        Position : (N,3) ndarray

        Returns
        -------
        B : (N,3) ndarray
        """
        B = np.zeros_like(position)

        for dipole in self.dipoles:
            r = position-dipole.position    #(N,3)
            r_norm = np.linalg.norm(r, axis=1)  #(N,)
            m = dipole.moment * dipole.direction    #(3,)
            m_dot_r = r @ m  #(N,)
            mu_0 = 1.25663706212e-6  # T*m/A
            B += mu_0/(4*np.pi) * (3*r*(m_dot_r[:, None])/r_norm[:, None]**5 - m/r_norm[:, None]**3)
        return B

class ElectricField(Fields):
    def __init__(self, planeterrella: Planeterrella, voltage: float):
        self.cathode = planeterrella.cathode
        self.anode = planeterrella.anode
        self.voltage = voltage

    def at(self, position: np.ndarray) -> np.ndarray:
        pass
        #TODO create a simple electric field, static, same as B, and then use interpolation for fast pusher. 
        """
        Parameters
        ----------
        Position : (N,3) ndarray

        Returns
        -------
        E : (N,3) ndarray
        """
        E = np.zeros_like(position)

        # For simplicity, we will assume that the cathode is a point charge and the anode is a grounded sphere.
        # This is not physically accurate but serves as a placeholder for the actual implementation.

        # Cathode as point charge
        r_cathode = position - self.cathode.position
        r_cathode_norm = np.linalg.norm(r_cathode, axis=1)
        q_cathode = 1e-9  # Charge in Coulombs (placeholder value)
        epsilon_0 = 8.854187817e-12  # F/m
        E += q_cathode/(4*np.pi*epsilon_0) * r_cathode/r_cathode_norm[:, None]**3

        # Anode as grounded sphere (no contribution to E field in this simple model)

        return E