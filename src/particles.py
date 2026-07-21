from turtle import position

import numpy as np
from src.geometry import Needle, Sphere
from src.monteCarlo import MonteCarloNeedle, MonteCarloSphere
class Particles:
    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self.position = position
        self.velocity = velocity
        self.alive = np.ones(position.shape[0], dtype=bool) 

    

class Electrons(Particles):
    def __init__(self, N, cathode: Needle | Sphere, dt:float, V:float):
        self.cathode = cathode
        
        #generting random initial positions and velocities for the electrons based on the cathode type
        if isinstance(cathode, Needle):
            initial_settings = [MonteCarloNeedle(cathode, V) for _ in range(N)]
        elif isinstance(cathode, Sphere):
            initial_settings = [MonteCarloSphere(cathode, V) for _ in range(N)]
        else:
            raise ValueError("Unsupported cathode type")
        initial_positions, initial_velocities = zip(*initial_settings)
        initial_positions = np.array(initial_positions)
        initial_velocities = np.array(initial_velocities)

        super().__init__(initial_positions, initial_velocities)

        self.dt = np.full(N, dt)  # Initialize dtarray with the same dt for all electrons
    
    
    
