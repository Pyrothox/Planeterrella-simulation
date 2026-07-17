import numpy as np
from src.particles import Particles
class TrajectoryRecorder:
    def __init__(self, trackedParticles: Particles):
        self.trajectories = []  # List to store trajectories of electrons
        self.trackedParticles = trackedParticles  # Store the tracked particles

    def record(self):
        """
        Record the positions of electrons at a given time step.
        """
        self.trajectories.append(self.trackedParticles.position.copy())  # Store a copy of the current positions

    def get_trajectories(self):
        """
        Get the recorded trajectories.
        
        :return: A list of numpy arrays, each representing the positions of electrons at a time step.
        """
        return self.trajectories
    


class Diagnostics:
    def __init__(self, trackedParticles : Particles):
        self.time = []
        self.trackedParticles = trackedParticles
        self.trajectoryRecorder = TrajectoryRecorder(trackedParticles)  # Initialize the trajectory recorder
        
    def recordStep(self, time):
        self.time.append(time)
        self.trajectoryRecorder.record()  # Record the positions of tracked particles at this time step