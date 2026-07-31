from __future__ import annotations
from random import uniform
from matplotlib.pylab import normal
import numpy as np
from scipy.spatial.transform import Rotation as R
from src.geometry import Needle, Sphere
from fields import Fields
_EV_TO_JOULE = 1.602176634e-19   # elementary charge, C
_ME = 9.1093837015e-31  # electron mass, kg
_N_PROBE = 1000  # number of probes for Monte Carlo sampling
_EPS_FRAC = 1e-3 # fraction of the object size to avoid sampling too close to the edge (woud give dirichlet boundary conditions and not realistic emission)
_EXPONENT = 1  # exponent for the probability distribution of the electric field *
_NZ = 200
#TODO : these methods generate electrons randomly at the surface of the electrodes, but the electric field should be considered, more electrons are generating between the electrodes. 





def MonteCarloNeedle(needle: Needle, emission_eV:float, N:int):
    """
    Gives a random inital velocity V and position X to the electron ejected from the cone of the needle.

    Parameters
    ----------
    needle : Needle
    emission_eV : float
        The energy of the electrons at emission, in electron volts.
    N : int
        The number of electrons to generate.
    Returns
    -------
    global_pos : np.ndarray
        The initial position of the electron in the global frame.
    global_vel : np.ndarray
        The initial velocity of the electron in the global frame.
    """
    global_pos = np.tile(np.zeros((1, 3)), (N, 1))
    global_vel = np.zeros((N, 3))
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



def MonteCarloSphere(sphere: Sphere, emission_eV: float, N:int, Efield: Fields):
    """
    Gives a random initial velocity V and position X to the electron ejected from the surface of the sphere.

    Parameters
    ----------
    sphere : Sphere
    emission_eV : float
        The energy of the electrons at emission, in electron volts.
    N : int
        The number of electrons to generate.

    Returns
    -------
    global_pos : np.ndarray
        The initial position of the electron in the global frame.
    global_vel : np.ndarray
        The initial velocity of the electron in the global frame.
    """
    global_pos = np.tile(sphere.position, (N, 1))  
    global_vel = np.zeros((N, 3))
    r = sphere.radius

    def sample_uniform(n: int):
        """ samples uniformly on the sphere surface """
        z = uniform(-r, r, n)
        theta = uniform(0, 2*np.pi, n)
        rho = np.sqrt(r**2 - z**2)
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        local_pos = np.stack([x, y, z], axis=1)
        normal = local_pos / r
        return local_pos, normal

    if Efield is None:  # uniform sampling on the sphere surface
        local_pos, normal = sample_uniform(N)

    else: #sampling on the sphere surface with a probability proportional to the local electric field strength
        #probing
        probe_local, probe_normal = sample_uniform(_N_PROBE)
        mag_probe = np.linalg.norm(Efield.at(probe_local + sphere.position + probe_normal * _EPS_FRAC), axis=1)
        z_centers = -r + (np.arange(_NZ) + 0.5) * 2*r/_NZ
        phi_centers = (np.arange(_N_theta)+0.5) * 2*np.pi/_N_theta
        Z, PHI = np.meshgrid(z_centers, phi_centers, indexing='ij')
        Z, PHI = Z.unravel(), PHI.ravel()
        rho = np.sqrt(np.maximum(0, r**2 - Z**2))
        mesh_local = np.stack([rho * np.cos(PHI), rho * np.sin(PHI), Z], axis=1)
        normal = mesh_local / r
        area = np.ones_like(Z)
        
        global_probe = mesh_local + sphere.position + normal * _EPS_FRAC*r
        E = np.linalg.norm(Efield.at(global_probe), axis=1)

        log_w = 
    #velocity
    v = np.sqrt(2*_EV_TO_JOULE*emission_eV/_ME)  # speed of the electron in m/s

    # sample direction in a hemisphere around local z, then rotate to align with outward normal
    normal = local_pos / r
    psi = uniform(0, 2*np.pi, N)
    u = uniform(0, 1, N)
    theta_prime = np.arcsin(np.sqrt(u))   # cosine-weighted (Lambertian) sampling
    elevation = np.pi/2 - theta_prime
    local_vel = np.stack([
        v*np.cos(elevation)*np.cos(psi),
        v*np.cos(elevation)*np.sin(psi),
        v*np.sin(elevation)
    ], axis = 1)
    e_x, e_y = _tangent_basis(normal)
    global_vel = (local_vel[:, 0:1] * e_x + local_vel[:, 1:2] * e_y + local_vel[:, 2:3] * normal)

    return global_pos, global_vel



def _tangent_basis(e_z: np.ndarray):
    """Builds orthonormal basis given a row vectors e_z"""
    n = np.shape(e_z)[0]
    temp = np.tile(np.array([0, 0, 1]), (n, 1))
    allclose_mask = np.abs(e_z[:,2]) > 0.999
    temp[allclose_mask] = np.array([1, 0, 0])
    e_x = np.cross(e_z, temp)
    e_x /= np.linalg.norm(e_x, axis=1, keepdims=True)
    e_y = np.cross(e_z, e_x)
    return e_x, e_y