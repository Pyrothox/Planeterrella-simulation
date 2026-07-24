import numpy as np
from src.particles import Electrons
from cross_sections.cross_section_inelastic_N2 import cross_section_inelastic_N2
from cross_sections.cross_section_inelastic_O2 import cross_section_inelastic_O2
from cross_sections.cross_section_elastic import cross_section_elastic
from src.diagnostics import Diagnostics, CollisionRecorder

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
    

    def collide(self, electrons: Electrons, diagnostics: Diagnostics = None, debug = False):
        """ 
        Handle collisions for a set of electrons with the gas molecules. Records collisions in the diagnostic object if provided.

        Parameters
        ----------
        electrons : Electrons
            The electrons to check for collisions. Only the alive electrons are considered.
        diagnostics : Diagnostics, optional
            The diagnostics object to record collision events.
        debug : bool, optional
            If True, prints debug information about the collision process.
        """
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

        # ineslatic cross sections, energy loss, and emitted colors for each channel, for each electron
        Qex_N2, Qex_N2p, Qex_N2i, loss_n2, loss_n2p, loss_n2i, col_n2, col_n2p, col_n2i = cross_section_inelastic_N2(eV)
        Qex_O2, Qex_O2p, Qex_O2c, loss_o2, loss_o2p, loss_o2c, col_o2, col_o2p, col_o2c = cross_section_inelastic_O2(eV)

        # elastic cross sections
        S_N2, S_O2 = cross_section_elastic(eV)

        # collision frequencies 
        nu_N2_inel = nN2 * V *(Qex_N2.sum(axis=0) + Qex_N2p.sum(axis=0))        #ignoring N2+ ion excitation for now (from matlab original code)
        nu_O2_inel = nO2 * V *(Qex_O2.sum(axis=0) + Qex_O2p.sum(axis=0))        #ignoring O2- capture for now (from matlab original code)
        nu_N2_el = nN2 * V * S_N2 
        nu_O2_el = nO2 * V * S_O2
        nu_N2 = nu_N2_inel + nu_N2_el
        nu_O2 = nu_O2_inel + nu_O2_el
        nu_tot = nu_N2 + nu_O2

        # collision probabilities
        P_coll = 1.0 - np.exp(-nu_tot * dt)
        
        collided = self.rng.random(n) < P_coll      # boolean mask of electrons that have collided

        if not np.any(collided):    # No collisions occurred
            if debug:
                print("No collisions occurred in this step.")
            return  
        
        idx = np.flatnonzero(collided)
        is_N2 = self.rng.random(idx.size) <= nu_N2[idx] / nu_tot[idx]        #collided with N2
        
        new_velocities = v.copy()  # Work on the full alive slice so absolute indices remain valid
        self._resolveSpecie(idx[is_N2], v, pos, eV, new_velocities, nu_N2_el, nu_N2, Qex_N2, Qex_N2p, loss_n2, loss_n2p, col_n2, col_n2p, "N2", diagnostics = diagnostics)        #processing collisions with N2
        self._resolveSpecie(idx[~is_N2], v, pos, eV, new_velocities, nu_O2_el, nu_O2, Qex_O2, Qex_O2p, loss_o2, loss_o2p, col_o2, col_o2p, "O2", diagnostics = diagnostics)      #processing collisions with O2

        electrons.velocity[alive] = new_velocities  # Update the velocities of collided electrons

        if debug:
            print("Ec range (eV):", eV.min(), eV.max(), "median:", np.median(eV))
            print("Fraction of electrons above N2 B3 threshold (7.35eV):", np.mean(eV > 7.35))
            print("Fraction above N2 C3 threshold (11.03eV):", np.mean(eV > 11.03))
            print("nu_N2_inel stats:", nu_N2_inel.min(), nu_N2_inel.max(), nu_N2_inel.mean())
            print("nu_N2_el stats:", nu_N2_el.min(), nu_N2_el.max(), nu_N2_el.mean())

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

    @staticmethod
    def _sampleChannel(weights, rng):
        """ 
        choose a chanel for each inelastic collision based on the weights of the channels.

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

    def _resolveSpecie(self, idx, vel, pos, eV, new_vel, nu_el, nu_species, Q_ex, Q_ex_p, loss, loss_p, col, col_p, specie, diagnostics : Diagnostics = None):
        """
        Determine whether each electron in idx undergoes an elastic or inelastic collision, chooses the appropriate channel, and updates its velocity accordingly.
        
        Inputs
        ----------
        idx : (n,) ndarray
            Indices of electrons that have collided with the specified specie.
        vel : (n, 3) ndarray
            Velocities of the electrons.
        pos : (n, 3) ndarray
            Positions of the electrons.
        eV : (n,) ndarray
            Energies of the electrons in eV.
        new_vel : (n, 3) ndarray
            Array to store the updated velocities of the electrons.
        nu_el : (n,) ndarray
            Elastic collision frequencies for each electron.
        nu_species : (n,) ndarray
            Total collision frequencies for each electron.
        Q_ex : (m, n) ndarray
            Excitation cross-sections for each channel and electron.
        Q_ex_p : (m, n) ndarray
            Excitation cross-sections for the product channels.
        loss : (m,) ndarray
            Energy losses for each excitation channel.
        loss_p : (m,) ndarray
            Energy losses for the product excitation channels.
        specie : str
            The specie with which the electrons have collided.
        diagnostics : Diagnostics, optional
            Diagnostic object to record collision events.
        """
        if idx.size == 0:       # break if no electrons to process
            return 
        
        specieBool = True if specie == "N2" else False          # for diagnostics :  True for N2, False for O2

        # elastic collision ? 
        elastic = self.rng.random(idx.size) <= nu_el[idx] / nu_species[idx]
        el_idx = idx[elastic]   #electrons that underwent elastic collisions
        if el_idx.size > 0:
            speed = np.linalg.norm(vel[el_idx], axis=1)
            new_vel[el_idx] = self._rotate_isotropic_dir(el_idx.size, self.rng) * speed[:, None]       # new travelling direction from isotropic distribution, same speed as before
            if diagnostics is not None:
                [diagnostics.recordCollision(pos[el_idx[i]], False, specieBool) for i in range(el_idx.size)]        # record elastic collision

        # inelastic collision ? 
        inel_idx = idx[~elastic]
        if inel_idx.size > 0:
            # compressing channels attributes into single arrays for sampling
            weight = np.concatenate([Q_ex[:, inel_idx], Q_ex_p[:, inel_idx]], axis=0)
            losses = np.concatenate([loss, loss_p], axis=0)
            colors = np.concatenate([col, col_p], axis=0)

            channel_idx = self._sampleChannel(weight, self.rng)         # determine which channel each electron undergoes based on the weights of the channels
            valid = channel_idx >= 0 #filter out invalid collisions with channel = -1, probably useless
            valid_inel_idx = inel_idx[valid]      

            if valid_inel_idx.size > 0:
                loss_ev = losses[channel_idx[valid]]  # appropriate energy loss for each electron
                colors_map = colors[channel_idx[valid]]  # attributed colors for the inelastic collisions
                new_dir = self._rotate_isotropic_dir(valid_inel_idx.size, self.rng)     #isotropic new direction for each electron
                new_E = np.maximum( (eV[valid_inel_idx] - loss_ev) * _E_CHARGE, 0.0)  # Ensure non-negative energy, in eV
                new_speed = np.sqrt(2 * new_E / _ME)
                new_vel[valid_inel_idx] = new_dir * new_speed[:, None]      # new speed and direction for each electron
                if diagnostics is not None:
                    [diagnostics.recordCollision(pos[valid_inel_idx[i]], True, specieBool, colors_map[i]) for i in range(valid_inel_idx.size)]  # Record inelastic collision events