import numpy as np
from src.particles import Electrons
from src.fields import MagneticField, ElectricField

q = -1.602176634e-19  # Charge of an electron in Coulombs
m = 9.1093837015e-31  # Mass of an electron in kilograms
qm = q/m  # Charge-to-mass ratio

def cyclotron_dt(electrons:Electrons, magnetic_field: MagneticField) -> float:
    """
    Calculate the cyclotron time step for electrons in a magnetic field.
    """
    alive = electrons.alive
    pos = electrons.position[alive]

    # B field
    B0 = magnetic_field.at(pos)
    B_magnitude = np.linalg.norm(B0, axis=1)

    # Cyclotron frequency
    omega_c = np.abs(qm) * B_magnitude

    # Cyclotron period
    T_c = 2 * np.pi / omega_c

    # Return the minimum cyclotron period as the time step
    return np.min(T_c)

def BorisPusher(electrons:Electrons, magnetic_field : MagneticField, ElectricField : ElectricField):
    """
    Update the positions and velocities of electrons using the Boris algorithm.
    """
    
    alive = electrons.alive
    if alive.sum() == 0:
        return  # No alive electrons to update
    dt = 0.25*cyclotron_dt(electrons, magnetic_field)  # Calculate the cyclotron time step
    electrons.dt[alive] = dt  # Update the time step for alive electrons #TODO : each electron could get its own dt ?
    pos = electrons.position[alive]
    vel = electrons.velocity[alive]

    # B field
    B0 = magnetic_field.at(pos)
    
    if ElectricField is not None:
        E0 = ElectricField.at(pos)
        v_minus = vel + (0.5 * qm * E0 * dt)  # first half of electric acceleration
    else:
        v_minus = vel

    t = 0.5 * qm * B0 * dt # rotation vector
    t_mag2 = np.sum(t**2, axis=1)   # magnitude squared of t
    s = 2 * t / (1 + t_mag2)[:, None]   # Boris coefficient

    v_prime = v_minus + np.cross(v_minus, t)  # intermediate velocity
    v_plus = v_minus + np.cross(v_prime, s)    # final velocity

    if ElectricField is not None:
        v_new = v_plus + (0.5 * qm * E0 * dt)  # second half of electric acceleration
    else:
        v_new = v_plus

    x_new = pos + v_new * dt  # update position

    electrons.position[alive] = x_new
    electrons.velocity[alive] = v_new

    electrons.total_travel_distance[alive] += np.linalg.norm(v_plus * dt, axis=1)  # Update total travel distance
    electrons.cumdt[alive] += dt  # Update cumulative time for alive electrons