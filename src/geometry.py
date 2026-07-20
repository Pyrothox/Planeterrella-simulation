from __future__ import annotations
import numpy as np
# Geometry of the Planeterrella

class Sphere:
    def __init__(self, radius, position, direction_vector):
        self.position = np.array(position)
        self.radius = radius
        self.direction_vector = np.array(direction_vector)


class Needle:
    def __init__(self, position, lc, lb, r, direction_vector):
        self.position = np.array(position)
        self.lc = lc
        self.lb = lb
        self.r = r
        self.direction_vector = np.array(direction_vector)


class Dome:
    def __init__(self, radius, height):
        self.radius = radius
        self.height = height

class Dipole:
    def __init__(self, shere : Sphere):
        self.position = shere.position
        self.direction = shere.direction_vector
        self.moment = 250 #A/m2

class Planeterrella:
    def __init__(self, cathode: Sphere | Needle, anode: Sphere | Needle, dome: Dome):
        self.anode = anode
        self.cathode = cathode
        self.dome = dome
        self.dipoles = self.create_dipoles(self.anode, self.cathode)

    def create_dipoles(self, anode: Sphere | Needle, cathode: Sphere | Needle):
        dipoles: list[Dipole] = []
        if isinstance(anode, Sphere):
            dipoles.append(Dipole(anode))
        if isinstance(cathode, Sphere):
            dipoles.append(Dipole(cathode))
        return dipoles
    

    def OutofBounds(self, positions: np.ndarray) -> np.ndarray:
        """
        Check if a particle at the given position is out of bounds of the Planeterrella.
        """
        counter = 0
        alive = np.ones(positions.shape[0], dtype=bool)
        for i, pos in enumerate(positions):
            if pos[2] < 0 or pos[1]**2 + pos[0]**2 > self.dome.radius**2 or pos[2] > self.dome.height:
                alive[i] = False
                counter += 1
        return alive
            

    def check_collision(self, position: np.ndarray) -> bool:
        """
        Check if a particle at the given position collides with any of the objects in the Planeterrella.
        """
        # Check collision with cathode
        if isinstance(self.cathode, Sphere):
            if np.linalg.norm(position - self.cathode.position) <= self.cathode.radius:
                return True
        elif isinstance(self.cathode, Needle):
            # Implement needle collision logic if needed
            pass

        # Check collision with anode
        if isinstance(self.anode, Sphere):
            if np.linalg.norm(position - self.anode.position) <= self.anode.radius:
                return True
        elif isinstance(self.anode, Needle):
            # Implement needle collision logic if needed
            pass

        # Check collision with dome
        if np.linalg.norm(position) >= self.dome.radius:
            return True

        return False