# Electric and Magnetic fields for the Planeterrella simulation
import numpy as np
from abc import ABC, abstractmethod
from src.geometry import Planeterrella, Dipole, Sphere, Needle

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

        if isinstance(self.cathode, Sphere) and isinstance(self.anode, Sphere):     # 2 sheres configuration
            eps0 = 8.854187817e-12  # Permittivity of free space
            R1 = self.cathode.radius
            R2 = self.anode.radius
            d = np.linalg.norm(self.anode.position - self.cathode.position)
            V = self.voltage
            Q = 4 * np.pi * eps0 * V / (1/R1 - 1/R2 + 2/d)

        else:       # 1 shere and 1 needle configuration
            if isinstance(self.cathode, Needle):
                needle = self.cathode
            else:
                needle = self.anode
            L = needle.lc + needle.lb
            R = needle.r
            V = self.voltage

            Q = 2 * np.pi * eps0 * V * L / np.log(2 * L / R)

        self.Q = Q


    def at(self, position:np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        Position : (N,3) ndarray

        Returns
        -------
        E : (N,3) ndarray
        """
        # For simplicity, we assume a uniform electric field between the cathode and anode.
        # This is a simplification and may not represent the actual field distribution.
        eps0 = 8.854187817e-12  # Permittivity of free space
        k= 1/(4*np.pi*eps0)

        Q = self.Q
        # charges ponctuelles
        pos1 = self.cathode.position
        pos2 = self.anode.position
        r1 = position - pos1[:,None]  #(N,3)
        r2 = position - pos2[:,None]  #(N,3)
        n1, n2 = np.linalg.norm(r1, axis=1), np.linalg.norm(r2, axis=1)  #(N,)
        E = k * Q * (r1/n1[:,None]**3 - r2/n2[:,None]**3)  #(N,3)
        return E