import numpy as np
from src.geometry import Planeterrella, Needle, Sphere
from src.monteCarlo import MonteCarloNeedle, MonteCarloSphere
class Particles:
    # Base class for particles in the simulation. It stores the positions, velocities, and alive status of the particles.
    def __init__(self, position: np.ndarray, velocity: np.ndarray):
        self.position = position
        self.velocity = velocity
        self.alive = np.ones(position.shape[0], dtype=bool)

    

class Electrons(Particles):
    # represents the electrons in the simulation.
    def __init__(self, N, cathode: Needle | Sphere, dt:float, initial_energy:float, Efield):
        self.cathode = cathode
        
        #generating random initial positions and velocities for the electrons based on the cathode type
        if isinstance(cathode, Needle):
            initial_settings = MonteCarloNeedle(cathode, initial_energy, N, Efield)
        elif isinstance(cathode, Sphere):
            initial_settings = MonteCarloSphere(cathode, initial_energy, N, Efield) 
        else:
            raise ValueError("Unsupported cathode type")
        initial_positions, initial_velocities = initial_settings
        initial_positions = np.array(initial_positions)
        initial_velocities = np.array(initial_velocities)

        super().__init__(initial_positions, initial_velocities)

        self.dt = np.full(N, dt)  # Initialize dtarray with the same dt for all electrons. For now all electrons share the same dt.
        self.cumdt = np.zeros(N)  # Initialize cumulative time array with zeros for all electrons
        self.total_travel_distance = np.zeros(N)  # Initialize total travel distance array with zeros for all electrons
