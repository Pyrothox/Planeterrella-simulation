

#depreciated 
from src.particles import Electrons
from src.fields import MagneticField
import numpy as np 
def RungeKutta4(electrons: Electrons, magnetic_field: MagneticField):

    
    q = -1.602176634e-19  # Charge of an electron in Coulombs
    m = 9.1093837015e-31  # Mass of an electron in kilograms
    qm = q/m  # Charge-to-mass ratio
    
    alive = electrons.alive
    dt = electrons.dtarray[alive]
    pos = electrons.position[alive]
    vel = electrons.velocity[alive]

    # B field
    B0 = magnetic_field.at(pos)
    B1 = magnetic_field.at(pos + 0.5 * dt[:, None] * vel)
    B2 = magnetic_field.at(pos + 1.0 * dt[:, None] * vel)
    
    # Approx of the new positions:
    xk1 = vel
    xk2 = vel + 0.5 * qm * np.cross(vel, B0)

    