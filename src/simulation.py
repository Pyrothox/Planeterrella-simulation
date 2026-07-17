from src.experiment import Experiment
from src.particles import Electrons
from src.geometry import Needle, Sphere
from src.monteCarlo import MonteCarloNeedle, MonteCarloSphere
import numpy as np 
class Simulation:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        simSettings = experiment.simSettings
        self.Nparticles = simSettings["Nparticles"]
        self.Nsteps = simSettings["Nsteps"]

    def run(self):
        # Placeholder for simulation logic
        print("Running simulation with the following settings:")
        print(f"Geometry: {self.experiment.geometry}")
        print(f"Gas: {self.experiment.gas}")
        print(f"Simulation Settings: {self.experiment.simSettings}")

        N = self.Nparticles
        initial_positions = np.zeros((N, 3))
        initial_velocities = np.zeros((N, 3))

        V = self.experiment.simSettings["V"]
        cathode = self.experiment.geometry.cathode

        if isinstance(cathode, Needle):
            generator = MonteCarloNeedle
        elif isinstance(cathode, Sphere):
            generator = MonteCarloSphere
        else:
            raise ValueError("Unsupported cathode type")

        for i in range(N):
            initial_settings = generator(cathode, V)
            initial_positions[i] = initial_settings[0]
            initial_velocities[i] = initial_settings[1]