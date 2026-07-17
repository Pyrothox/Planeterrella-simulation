import numpy as np
from src.geometry import Needle, Sphere
class Electrons:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, cathode: Needle | Sphere, dt:float):
        self.position = position
        self.velocity = velocity
        self.alive = np.ones(position.shape[0], dtype=bool)
        self.dtarray = np.full(position.shape[0], dt)  # Initialize dtarray with the same dt for all electrons


