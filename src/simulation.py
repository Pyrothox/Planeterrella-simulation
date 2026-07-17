import numpy as np 
from src.experiment import Experiment
from src.particles import Electrons
from src.geometry import Needle, Sphere
from src.monteCarlo import MonteCarloNeedle, MonteCarloSphere
from src.Boris import BorisPusher
class Simulation:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        simSettings = experiment.simSettings
        self.Nparticles = simSettings["Nparticles"]
        self.Nsteps = simSettings["Nsteps"]

    def run(self):
        print("Running simulation with the following settings:")
        print(f"Geometry: {self.experiment.geometry}")
        print(f"Gas: {self.experiment.gas}")
        print(f"Simulation Settings: {self.experiment.simSettings}")

        N = self.Nparticles

        # Generating initial velocities of electrons
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

        dt = 1e-5/N;    # initial time step before adaptive step size computation

        electrons = Electrons(initial_positions, initial_velocities, cathode, dt)   
        
        #running the simulation for Nsteps
        for step in range(self.Nsteps):

            BorisPusher(electrons, self.experiment.MagneticField)       #updating the positions and velocities of electrons using the Boris algorithm
            if self.experiment.collisions:
                pass   #handling collisions if enabled