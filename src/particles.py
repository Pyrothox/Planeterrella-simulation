from turtle import position

import numpy as np
from src.geometry import Planeterrella, Needle, Sphere
from src.monteCarlo import MonteCarloNeedle, MonteCarloSphere
class Particles:
    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self.position = position
        self.velocity = velocity
        self.alive = np.ones(position.shape[0], dtype=bool) 

    

class Electrons(Particles):
    def __init__(self, N, cathode: Needle | Sphere, dt:float, initial_energy:float):
        self.cathode = cathode
        
        #generting random initial positions and velocities for the electrons based on the cathode type
        if isinstance(cathode, Needle):
            initial_settings = [MonteCarloNeedle(cathode, initial_energy) for _ in range(N)]
        elif isinstance(cathode, Sphere):
            initial_settings = [MonteCarloSphere(cathode, initial_energy) for _ in range(N)]
        else:
            raise ValueError("Unsupported cathode type")
        initial_positions, initial_velocities = zip(*initial_settings)
        initial_positions = np.array(initial_positions)
        initial_velocities = np.array(initial_velocities)

        super().__init__(initial_positions, initial_velocities)

        self.dt = np.full(N, dt)  # Initialize dtarray with the same dt for all electrons
        self.cumdt = np.zeros(N)  # Initialize cumulative time array with zeros for all electrons
        self.total_travel_distance = np.zeros(N)  # Initialize total travel distance array with zeros for all electrons
