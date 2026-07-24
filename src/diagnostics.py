import numpy as np
import h5py
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
    
    def save(self, g : h5py.Group):
        # Convert the list of trajectories to a 3D numpy array
        trajectories_array = np.array(self.trajectories)  # Shape: (Nsteps, Nelec, 3)
        dataset = g.create_dataset("trajectories", data=trajectories_array)
        dataset.attrs["dimensions"] = ["Save", "Electron", "xyz"]


class CollisionRecorder:    
    collision_dtype = np.dtype([
        ("position", np.float32, (3,)),  # Position of the electron at the time of collision
        ("inelastic", np.bool_),  # Whether the collision was inelastic
        ("specie", np.bool_),  # True for N2, False for O2
        ("color", np.uint32)  # color from hex to uint32
    ])
    def __init__(self, size = 10_000_000):
        self.collisions = np.empty(size, dtype=self.collision_dtype)
        self.ncollisions = 0  # Counter for the number of recorded collisions

    def record(self, pos, inelastic, specie, color = 0):
        self.collisions[self.ncollisions] = (pos, inelastic, specie, np.uint32(color))
        self.ncollisions += 1

    def save(self, g : h5py.Group):
        g.create_dataset("collisions", data=self.collisions[:self.ncollisions])  # Save only the recorded collisions

class Diagnostics:
    def __init__(self, trackedParticles : Particles, collisionsEnabled: bool = False):
        self.time = []
        self.trackedParticles = trackedParticles
        self.trajectoryRecorder = TrajectoryRecorder(trackedParticles)  # Initialize the trajectory recorder
        self.collisionRecorder = CollisionRecorder() if collisionsEnabled else None  # Initialize the collision recorder

    def recordStep(self, time):
        self.time.append(time)
        self.trajectoryRecorder.record()  # Record the positions of electrons at this time step

    def recordCollision(self, pos, inelastic, specie, color: np.uint32 = 0):
        self.collisionRecorder.record(pos, inelastic, specie, color)

    def save_to_hdf5(self, f: h5py.File):
        
        g = f.create_group("trajectories")
        self.trajectoryRecorder.save(g)

        if self.collisionRecorder is not None:
            g = f.create_group("collisions")
            self.collisionRecorder.save(g)