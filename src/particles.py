import numpy as np
from src.geometry import Needle, Sphere

class Particles:
    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self.position = position
        self.velocity = velocity
        self.alive = np.ones(position.shape[0], dtype=bool)
        self.dt = np.zeros(position.shape[0])  # Initialize dt array with zeros   


class Electrons(Particles):
    def __init__(self, position: np.ndarray, velocity: np.ndarray, cathode: Needle | Sphere, dt:float):
        super().__init__(position, velocity)
        self.cathode = cathode
        self.dtarray = np.full(position.shape[0], dt)  # Initialize dtarray with the same dt for all electrons


