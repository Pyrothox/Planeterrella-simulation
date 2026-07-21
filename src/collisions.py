import numpy as np
from src.particles import Electrons
from cross_sections.cross_section_inelastic_N2 import cross_section_inelastic_N2
from cross_sections.cross_section_inelastic_O2 import cross_section_inelastic_O2
from cross_sections.cross_section_elastic import cross_section_elastic
_E_CHARGE = 1.602176634e-19     #eV to Joule conversion factor
_ME = 9.1093837015e-31
_R_GAS = 8.314
_NA = 6.022e23

class CollisionEngine:
    def __init__(self, gas, rng=None):
        self.gas = gas
        self.pressure = gas.P
        self.temperature = gas.T
        self.rng = rng if rng is not None else np.random.default_rng()  # Random number generator for collision probabilities
        self.col_n2 = 0.0  # Placeholder for N2 collision cross-section     #TODO put these in the diagnostic 
        self.col_o2 = 0.0  # Placeholder for O2 collision cross-section
        self.fracN2 = gas.fractions[gas.names.index("N2")]
        self.fracO2 = gas.fractions[gas.names.index("O2")]
        self.nN2 = self.fracN2 * self.pressure / (self.temperature * _R_GAS) * _NA  # Number density of N2
        self.nO2 = self.fracO2 * self.pressure / (self.temperature * _R_GAS) * _NA  # Number density of O2

    @staticmethod
    def _rotate_isotropic_dir(n, rng):
        """
        gives a random isotropic direction.

        Parameters
        ----------
        n : int
            Number of direction vectors to generate.
        rng : np.random.Generator
            Random number generator for generating random angles.

        Returns
        -------
        (N,3) ndarray
            Array of rotated direction vectors.
        """
        costheta = rng.uniform(-1.0, 1.0, n)
        sintheta = np.sqrt(1 - costheta**2)
        phi = rng.uniform(0, 2*np.pi, n)

        unit_dir = np.stack([sintheta*np.cos(phi), sintheta*np.sin(phi), costheta], axis=1)

        return unit_dir

    

    def collide(self, electrons: Electrons, diagnostics):
        alive = electrons.alive
        pos = electrons.position[alive]
        v = electrons.velocity[alive]
        dt = electrons.dt[alive]
        n = pos.shape[0]

        V = np.linalg.norm(v, axis=1)  # magnitude of velocity vector
        Ec = 0.5 * _ME * V**2  # kinetic energy in J
        eV = Ec / _E_CHARGE  # kinetic energy in eV

        nN2 = self.nN2
        nO2 = self.nO2

        # ineslatic cross sections and energy loss
        Qex_N2, Qex_N2p, Qex_N2i, loss_n2, loss_n2p, loss_n2i = cross_section_inelastic_N2(eV)
        Qex_O2, Qex_O2p, Qex_O2c, loss_o2, loss_o2p, loss_o2c = cross_section_inelastic_O2(eV)

        #elastic cross sections
        S_N2, S_O2 = cross_section_elastic(eV)
        
        nu_N2_inel = nN2 * V *(Qex_N2.sum(axis=0) + Qex_N2p.sum(axis=0))        #ignoring N2+ ion excitation for now (from matlab original code)
        nu_O2_inel = nO2 * V *(Qex_O2.sum(axis=0) + Qex_O2p.sum(axis=0))        #ignoring O2- capture for now (from matlab original code)
        nu_N2_el = nN2 * V * S_N2 
        nu_O2_el = nO2 * V * S_O2
        nu_N2 = nu_N2_inel + nu_N2_el
        nu_O2 = nu_O2_inel + nu_O2_el
        nu_tot = nu_N2 + nu_O2

        # collision probabilities
        P_coll = 1.0 - np.exp(-nu_tot * dt)
        
        collided = self.rng.random(n) < P_coll
        if not np.any(collided):
            return  # No collisions occurred
        
        idx = np.flatnonzero(collided)
        is_N2 = self.rng.random(idx.size) <= nu_N2[idx] / nu_tot[idx]        #collided with N2
        
        new_velocities = v.copy()  # Work on the full alive slice so absolute indices remain valid
        self._resolveSpecie(idx[is_N2], v, pos, Ec, new_velocities, nu_N2_el, nu_N2, Qex_N2, Qex_N2p, loss_n2, loss_n2p, diagnostics = None)        #processing collisions with N2
        self._resolveSpecie(idx[~is_N2], v, pos, Ec, new_velocities, nu_O2_el, nu_O2, Qex_O2, Qex_O2p, loss_o2, loss_o2p, diagnostics = None)      #processing collisions with O2

        electrons.velocity[alive] = new_velocities  # Update the velocities of collided electrons

    @staticmethod
    def _sampleChannel(weights, rng):
        """ 
        Inputs
        ----------
        weights : (n_ch, n) ndarray
            Array of weights for each channel and particle.
        rng : np.random.Generator

        Returns
        -------
        channel_idx : (n,) ndarray
            Array of sampled channel indices for each particle.
        """
        totals = weights.sum(axis=0)    # [s1+s2+..., s1'+s2'+..., ... ]
        totals[totals == 0] = 1.0  # Avoid division by zero for particles with no available channels
        cumfrac = np.cumsum(weights, axis=0) / totals       # [ [s1, s1', s1'',...], [s1 +s2, s1'+s2', s1''+s2'', ...], ...]  but with normalisation 
        nParticles = weights.shape[1]
        r = rng.random(nParticles)  # Random numbers for each particle             
        hit = cumfrac >= r[None, :]      # [ [False, True, True, True, always true], [false, false, true, true always true] ]]. TRANSPOSE
        channel_idx = hit.argmax(axis=0)  # Get the first True index for each particle
        channel_idx = np.where(totals > 0, channel_idx, -1)  # Set to -1 for particles with no available channels
        return channel_idx

    def _resolveSpecie(self, idx, vel, pos, Ec, new_vel, nu_el, nu_species, Q_ex, Q_ex_p, loss, loss_p, diagnostics):
        if idx.size == 0:
            return 
        
        #elastic collision ? 
        elastic = self.rng.random(idx.size) <= nu_el[idx] / nu_species[idx]
        el_idx = idx[elastic]
        if el_idx.size > 0:
            speed = np.linalg.norm(vel[el_idx], axis=1)
            new_vel[el_idx] = self._rotate_isotropic_dir(el_idx.size, self.rng) * speed[:, None]       #new travelling direction
            #TODO should we record elastic collision that don't create light ? 

        #inelastic collision ? 
        inel_idx = idx[~elastic]
        if inel_idx.size > 0:
            weight = np.concatenate([Q_ex[:, inel_idx], Q_ex_p[:, inel_idx]], axis=0)
            losses = np.concatenate([loss, loss_p], axis=0)
            channel_idx = self._sampleChannel(weight, self.rng) 
            valid = channel_idx >= 0 #filter out invalid collisions with channel = -1
            valid_inel_idx = inel_idx[valid]      
            if valid_inel_idx.size > 0:
                loss = losses[channel_idx[valid]]
                new_dir = self._rotate_isotropic_dir(valid_inel_idx.size, self.rng)
                new_E = np.maximum((Ec[valid_inel_idx] - loss) * _E_CHARGE, 0)  # Ensure non-negative energy
                new_speed = np.sqrt(2 * new_E / _ME)
                new_vel[valid_inel_idx] = new_dir * new_speed[:, None]
                if diagnostics is not None:
                    pass
                    #TODO create diagnostic for collisions you BAKA