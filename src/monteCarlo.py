from __future__ import annotations
from random import uniform
import numpy as np
from scipy.spatial.transform import Rotation as R
from src.geometry import Needle, Sphere

_EV_TO_JOULE = 1.602176634e-19   # elementary charge, C
_ME = 9.1093837015e-31  # electron mass, kg


#TODO : these methods generate electrons randomly at the surface of the electrodes, but the electric field should be considered, more electrons are generating between the electrodes. 

def MonteCarloNeedle(needle: Needle, emission_eV:float):
    """
    Gives a random inital velocity V and position X to the electron ejected from the cone of the needle.

    Parameters
    ----------
    needle : Needle
    emission_eV : float
        The energy of the electrons at emission, in electron volts.

    Returns
    -------
    global_pos : np.ndarray
        The initial position of the electron in the global frame.
    global_vel : np.ndarray
        The initial velocity of the electron in the global frame.
    """

    r = needle.r
    lb = needle.lb
    lc = needle.lc
    # lb : lenght of conical part of the needle, lc : lenght of the cylinder part of the needle

    #initial posistion
    z = lc + lb*uniform(0, 1)
    theta = uniform(0, 2*np.pi)
    x, y = r*( 1 - (z-lc)/lb )*np.cos(theta), r*( 1 - (z-lc)/lb )*np.sin(theta)

    local_pos = np.array([x, y, z])


    #initial velocity
    v = np.sqrt(2*_EV_TO_JOULE*emission_eV/_ME)  # speed of the electron in m/s

    psi = theta + np.pi/2 * uniform(-1, 1)
    phi = np.pi*uniform(0, 1) - np.arctan(r/lb)
    Vx = v*np.sin(phi)*np.cos(psi)
    Vy = v*np.sin(phi)*np.sin(psi)
    Vz = v*np.cos(phi)
    local_vel = np.array([Vx, Vy, Vz])

    #global position
    d = needle.direction_vector
    d = d/np.linalg.norm(d)

    temp = np.array([0, 0, 1])
    if np.allclose(d, temp):
        Rmat = np.eye(3)
    elif np.allclose(d, -temp):
        Rmat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        axis = np.cross(temp, d)
        angle = np.arccos(np.dot(temp, d))
        Rmat = R.from_rotvec(axis/np.linalg.norm(axis)*angle).as_matrix()
    global_pos = Rmat @ local_pos + needle.position
    global_vel = Rmat @ local_vel

    return global_pos, global_vel



def MonteCarloSphere(sphere: Sphere, emission_eV: float):
    """
    Gives a random initial velocity V and position X to the electron ejected from the surface of the sphere.

    Parameters
    ----------
    sphere : Sphere
    emission_eV : float
        The energy of the electrons at emission, in electron volts.

    Returns
    -------
    global_pos : np.ndarray
        The initial position of the electron in the global frame.
    global_vel : np.ndarray
        The initial velocity of the electron in the global frame.
    """
    
    r = sphere.radius

    # position
    z = uniform(-r, r)
    theta = uniform(0, 2*np.pi)
    rho = np.sqrt(r**2 - z**2)
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    local_pos = np.array([x, y, z])
    global_pos = local_pos + sphere.position

    #velocity
    e = 1.602176634e-19   # elementary charge, C
    me = 9.1093837015e-31  # electron mass, kg
    v = np.sqrt(2*_EV_TO_JOULE*emission_eV/_ME)  # speed of the electron in m/s

    # sample direction in a hemisphere around local z, then rotate to align with outward normal
    n = local_pos / r
    psi = uniform(0, 2*np.pi)
    u = uniform(0, 1)
    theta_prime = np.arcsin(np.sqrt(u))   # cosine-weighted (Lambertian) sampling
    elevation = np.pi/2 - theta_prime
    local_vel = np.array([
        v*np.cos(elevation)*np.cos(psi),
        v*np.cos(elevation)*np.sin(psi),
        v*np.sin(elevation)
    ])

    temp = np.array([0, 0, 1])
    if np.allclose(n, temp):
        Rmat = np.eye(3)
    elif np.allclose(n, -temp):
        Rmat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        axis = np.cross(temp, n)
        angle = np.arccos(np.dot(temp, n))
        Rmat = R.from_rotvec(axis/np.linalg.norm(axis)*angle).as_matrix()
    global_vel = Rmat @ local_vel


    return global_pos, global_vel
