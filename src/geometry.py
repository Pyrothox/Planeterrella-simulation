from __future__ import annotations
import numpy as np
# Geometry of the Planeterrella

class Planeterrella:
    # the planeterrella physical setup. 
    def __init__(self, cathode: Sphere | Needle, anode: Sphere | Needle, dome: Dome):
        self.anode = anode
        self.cathode = cathode
        self.dome = dome
        self.dipoles = self.create_dipoles(self.anode, self.cathode)

    def create_dipoles(self, anode: Sphere | Needle, cathode: Sphere | Needle):
        """Create dipoles from the spheres magnets, used for the magnetic field calculation."""
        
        dipoles: list[Dipole] = []
        if isinstance(anode, Sphere):
            dipoles.append(Dipole(anode))
        if isinstance(cathode, Sphere):
            dipoles.append(Dipole(cathode))
        return dipoles

    def Out_of_Bounds(self, positions: np.ndarray) -> np.ndarray:
        """
        Check if particles at the given position are out of bounds of the Planeterrella.
        --------
        Inputs : 
        positions : np.ndarray (N,3)
            The positions of the particles to check for collision.
        --------
        Returns :
        np.ndarray (N,)
            A boolean array indicating whether each particle collides with any of the objects in the Planeterrella.
        """
        # Check collision with cathode
        alive = np.ones(positions.shape[0], dtype=bool)
        if isinstance(self.cathode, Sphere):
            distances = np.linalg.norm(positions - self.cathode.position, axis=1)
            alive[distances <= self.cathode.radius] = False
        elif isinstance(self.cathode, Needle):
            # Implement needle collision logic if needed
            pass

        # Check collision with anode
        if isinstance(self.anode, Sphere):
            distances = np.linalg.norm(positions - self.anode.position, axis=1)
            alive[distances <= self.anode.radius] = False
        elif isinstance(self.anode, Needle):
            # Implement needle collision logic if needed
            pass

        # Check collision with dome
        alive[positions[:,0]**2 + positions[:,1]**2 > self.dome.radius**2] = False
        alive[positions[:,2] > self.dome.height] = False
        alive[positions[:,2] < 0] = False

        return alive

class Sphere:
    # Spherical electrode with a magnet inside. 
    def __init__(self, radius, position, direction_vector, dipole_moment):
        self.position = np.array(position)
        self.radius = radius
        self.direction_vector = np.array(direction_vector)
        self.dipole_moment = dipole_moment      #aligned with the direction vector.


class Needle:
    # Needle electrode without magnet.
    def __init__(self, position, lc, lb, r, direction_vector):
        self.position = np.array(position)
        self.lc = lc    # length of the cylinder part 
        self.lb = lb    # length of the conical part
        self.r = r 
        self.direction_vector = np.array(direction_vector)


class Dome:
    # cylinder representing the vacuum chamber of the Planeterrella.
    def __init__(self, radius, height):
        self.radius = radius
        self.height = height

class Dipole:
    # Dipole to compute magnetic field.
    def __init__(self, sphere: Sphere):
        self.position = sphere.position
        self.direction = sphere.direction_vector
        self.moment = sphere.dipole_moment

