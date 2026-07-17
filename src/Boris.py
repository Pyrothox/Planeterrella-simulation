import numpy as np
from src.particles import Electrons
from src.fields import MagneticField


def BorisPusher(electrons:Electrons, magnetic_field : MagneticField):
    """
    Update the positions and velocities of electrons using the Boris algorithm.
    """
    q = -1.602176634e-19  # Charge of an electron in Coulombs
    m = 9.1093837015e-31  # Mass of an electron in kilograms
    qm = q/m  # Charge-to-mass ratio

    alive = electrons.alive
    dt = electrons.dtarray[alive]
    pos = electrons.position[alive]
    vel = electrons.velocity[alive]

    # B field
    B0 = magnetic_field.at(pos)
    
    #assuming no E field
    v_minus = vel

    t = 0.5 * qm * B0 * dt[:, None] # rotation vector
    t_mag2 = np.sum(t**2, axis=1)   # magnitude squared of t
    s = 2 * t / (1 + t_mag2)[:, None]   # Boris coefficient

    v_prime = v_minus + np.cross(v_minus, t)  # intermediate velocity
    v_plus = v_minus + np.cross(v_prime, s)    # final velocity

    # second half of electric acceleration goes there if E

    x_new = pos + v_plus * dt[:, None]  # update position

    electrons.position[alive] = x_new
    electrons.velocity[alive] = v_plus