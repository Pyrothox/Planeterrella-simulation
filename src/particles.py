import numpy as np
from src.geometry import Needle, Sphere
class Electrons:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, cathode: Needle | Sphere):
        self.position = position
        self.velocity = velocity
        self.alive = np.ones(position.shape[0], dtype=bool)

    
