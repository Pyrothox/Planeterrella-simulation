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
        self.moment = 50

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