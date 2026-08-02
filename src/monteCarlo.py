from __future__ import annotations
from random import uniform
from matplotlib.pylab import normal
import numpy as np
from scipy.spatial.transform import Rotation as R
from src.geometry import Needle, Sphere
from src.fields import Fields
_EV_TO_JOULE = 1.602176634e-19   # elementary charge, C
_ME = 9.1093837015e-31  # electron mass, kg
_N_PROBE = 1000  # number of probes for Monte Carlo sampling
_EPS_FRAC = 1e-3 # fraction of the object size to avoid sampling too close to the edge (woud give dirichlet boundary conditions and not realistic emission)
_EXPONENT = 1  # exponent for the probability distribution of the electric field *
_NZ = 200
_N_theta = 200
#TODO : these methods generate electrons randomly at the surface of the electrodes, but the electric field should be considered, more electrons are generating between the electrodes. 





def MonteCarloNeedle(needle: Needle, emission_eV:float, N:int, Efield = None):
    #NOTE : considers the whole needle, not only conical part
    """
    Gives a random inital velocity V and position X to the electron ejected from the cone of the needle.

    Parameters
    ----------
    needle : Needle
    emission_eV : float
        The energy of the electrons at emission, in electron volts.
    N : int
        The number of electrons to generate.
    Efield : Fields, optional
        The electric field to consider for sampling.
    Returns
    -------
    global_pos : np.ndarray
        The initial position of the electron in the global frame.
    global_vel : np.ndarray
        The initial velocity of the electron in the global frame.
    """
    r = needle.r
    scale = (lb+lc) * _EPS_FRAC  # scale factor to avoid sampling too close to the edge
    lb = needle.lb  # lenght of conical part of the needle
    lc = needle.lc  # lenght of the cylinder part of the needle
     
    d = needle.direction_vector/ np.linalg.norm(needle.direction_vector)  # direction of the needle
    def local_geometry(z, theta):
        rho = np.where(z < lc, r, r*(1 - (z-lc)/lb))       # radial distance cylinder and cone
        x, y = rho * np.cos(theta), rho * np.sin(theta)
        local_pos = np.stack([x, y, z], axis=1)

        nz = np.where(z < lc, 0, r/lb)  # Z component of the normal vector in local coordinates
        normal = np.stack([np.cos(theta), np.sin(theta), nz], axis=1)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)

        area_weight = np.where(z<lc, r, rho*np.sqrt(1 + (r/lb)**2))  # area weight for sampling
        return local_pos, normal, area_weight
    
    #Rotation matrix to align the needle with the z-axis
    temp = np.array([0, 0, 1])
    if np.allclose(d, temp):
        Rmat = np.eye(3)
    elif np.allclose(d, -temp):
        Rmat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    else:
        axis = np.cross(d, temp)
        angle = np.arccos(np.dot(d, temp))
        Rmat = R.from_rotvec(axis/np.linalg.norm(axis)*angle).as_matrix()

    #initial posistion
    if Efield is None:  # uniform sampling on the needle surface
        z = (lc+lb)*np.random.uniform(0, 1, N)  #whole needle
        theta = np.random.uniform(0, 2*np.pi, N)
    else:
        z_centers = (np.arange(_NZ)+0.5)/_NZ * (lc+lb)  # whole needle
        theta_centers = (np.arange(_N_theta)+0.5)/_N_theta * 2*np.pi
        Z, THETA = np.meshgrid(z_centers, theta_centers, indexing='ij')
        Z, THETA = Z.ravel(), THETA.ravel()

        mesh_local, normal, area_weight = local_geometry(Z, THETA)

        global_query = mesh_local @ Rmat.T + needle.position + scale * (normal @ Rmat.T)      #probe points slightly outside to have better E variations

        #calculating weights for sampling
        E = np.linalg.norm(Efield.at(global_query), axis=1)
        log_w = np.log(area_weight) + _EXPONENT * np.log(E + 1e-12)   #all grid elements have the same area (archimedean projection of the needle onto a cylinder)
        w = np.exp(log_w - np.max(log_w))  # subtract max for numerical stability
        proba = w / np.sum(w)
        idx = np.random.choice(len(proba), size=N, replace=True, p=proba)

        #add some noise to the sampled positions to avoid clustering at the grid points
        dz, dtheta = (lc+lb)/_NZ, 2*np.pi/_N_theta
        z = Z[idx] + np.random.uniform(-dz/2, dz/2, N)
        theta = THETA[idx] + np.random.uniform(-dtheta/2, dtheta/2, N)

    local_pos, normal, _= local_geometry(z, theta)
    global_pos = local_pos @ Rmat.T + needle.position  # global position


    #initial velocity
    v = np.sqrt(2*_EV_TO_JOULE*emission_eV/_ME)  # speed of the electron in m/s

    psi = np.random.uniform(0, 2*np.pi, N)
    u = np.random.uniform(0, 1, N)
    theta_prime = np.arcsin(np.sqrt(u))   # cosine-weighted (Lambertian) sampling
    elevation = np.pi/2 - theta_prime
    local_vel = np.stack([
        v*np.cos(elevation)*np.cos(psi),
        v*np.cos(elevation)*np.sin(psi),
        v*np.sin(elevation)
    ], axis = 1)
    e_x, e_y = _tangent_basis(normal)
    local_vel = (local_vel[:, 0:1] * e_x + local_vel[:, 1:2] * e_y + local_vel[:, 2:3] * normal)
    global_vel = local_vel @ Rmat.T

    return global_pos, global_vel



def MonteCarloSphere(sphere: Sphere, emission_eV: float, N:int, Efield  = None):
    """
    Gives a random initial velocity V and position X to the electron ejected from the surface of the sphere.

    Parameters
    ----------
    sphere : Sphere
    emission_eV : float
        The energy of the electrons at emission, in electron volts.
    N : int
        The number of electrons to generate.
    Efield : Fields, optional
        The electric field to consider for sampling.
    Returns
    -------
    global_pos : np.ndarray
        The initial position of the electron in the global frame.
    global_vel : np.ndarray
        The initial velocity of the electron in the global frame.
    """
    r = sphere.radius
    scale = r * _EPS_FRAC  # scale factor to avoid sampling too close to the edge

    # INITIAL POSITION
    if Efield is None:  # uniform sampling on the sphere surface
        z = np.random.uniform(-r, r, N)
        phi = np.random.uniform(0, 2*np.pi, N)
    
    else: #probing the electric field on a grid on the sphere surface to sample probabilistically according to the field strength
        # create a grid on the sphere surface
        z_centers = -r + (np.arange(_NZ) + 0.5) * 2*r/_NZ
        phi_centers = (np.arange(_N_theta)+0.5) * 2*np.pi/_N_theta
        Z, PHI = np.meshgrid(z_centers, phi_centers, indexing='ij')
        Z, PHI = Z.ravel(), PHI.ravel()
        rho = np.sqrt(np.maximum(0, r**2 - Z**2))       # radial distance
        mesh_local = np.stack([rho * np.cos(PHI), rho * np.sin(PHI), Z], axis=1)    # XYZ coordinates
        normal = mesh_local / r
        global_query = mesh_local + sphere.position + scale * normal        #probe points slightly outside to have better E variations

        #calculating weights for sampling
        E = np.linalg.norm(Efield.at(global_query), axis=1)
        log_w = _EXPONENT * np.log(E + 1e-12)   #all grid elements have the same area (archimedean projection of the sphere onto a cylinder)
        w = np.exp(log_w - np.max(log_w))  # subtract max for numerical stability
        proba = w / np.sum(w)
        idx = np.random.choice(len(proba), size=N, replace=True, p=proba)

        #add some noise to the sampled positions to avoid clustering at the grid points
        dz, dphi = 2*r/_NZ, 2*np.pi/_N_theta
        z = Z[idx] + np.random.uniform(-dz/2, dz/2, N)
        phi = PHI[idx] + np.random.uniform(-dphi/2, dphi/2, N)
        print("sampled z range: ", z.min(), z.max())
    rho = np.sqrt(np.maximum(0, r**2 - z**2))       # radial distance
    x, y = rho * np.cos(phi), rho * np.sin(phi)
    local_pos = np.stack([x, y, z], axis=1)  
    global_pos = local_pos + sphere.position 

    # VELOCITY
    v = np.sqrt(2*_EV_TO_JOULE*emission_eV/_ME)  # speed of the electron in m/s

    # sample direction in a hemisphere around local z, then rotate to align with outward normal
    normal = local_pos / r
    psi = np.random.uniform(0, 2*np.pi, N)
    u = np.random.uniform(0, 1, N)
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