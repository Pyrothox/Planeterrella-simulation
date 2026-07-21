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
    
class CollisionRecorder:    
    collision_dtype = np.dtype([
        ("position", np.float32, (3,)),  # Position of the electron at the time of collision
        ("inelastic", np.bool_),  # Whether the collision was inelastic
        ("specie", np.bool_),  # True for N2, False for O2
    ])
    def __init__(self, size = 10_000_000):
        self.collisions = np.empty(size, dtype=self.collision_dtype)
        self.ncollisions = 0  # Counter for the number of recorded collisions

    def record(self, pos, inelastic, specie):
        self.collisions[self.ncollisions] = (pos, inelastic, specie)
        self.ncollisions += 1


class Diagnostics:
    def __init__(self, trackedParticles : Particles, collisionsEnabled: bool = False):
        self.time = []
        self.trackedParticles = trackedParticles
        self.trajectoryRecorder = TrajectoryRecorder(trackedParticles)  # Initialize the trajectory recorder
        self.collisionRecorder = CollisionRecorder() if collisionsEnabled else None  # Initialize the collision recorder

    def recordStep(self, time):
        self.time.append(time)
    
    def recordCollision(self, pos, inelastic, specie):
        self.collisionRecorder.record(pos, inelastic, specie)