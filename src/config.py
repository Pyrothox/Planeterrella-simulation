# Reads a TOML configuration and builds the objects needed for the simulation.
import tomllib
import numpy as np
from src.gas import Gas
from src.geometry import Planeterrella, Sphere, Needle, Dome
from src.experiment import Experiment


def load_experiment(file_path):
    """Load experiment parameters from a TOML file.
    
    Parameters
    --------
    file_path : str
        Path to the TOML configuration file.

    Returns
    -------
    Experiment
        An Experiment object containing the loaded parameters.
    """
    with open(file_path, "rb") as f:
        d = tomllib.load(f)
    
    collisions = d["collisions"]    # bool, are the collisions enabled or not ?

    # Read gas parameters
    gas_params = d["Gas"]
    species = gas_params["species"]
    names = [specie["name"] for specie in species]
    fractions = [specie["fraction"] for specie in species]
    assert np.isclose(np.sum(fractions), 1.0), "Fractions must sum to 1."

    gas = Gas(
        pressure=gas_params["pressure"],
        temperature=gas_params["temperature"],
        names=names,
        fractions=fractions
    )

    # Read geometry parameters
    mode = d["mode"]
    # 0 = The buse and one sphere : buse shooting electrons
    # 1 = The buse and two spheres
    # 2 = Two spheres, the smaller one ejecting the particles
    # 3 = Two spheres, the larger one ejecting the particles

    dome = Dome(
        radius=d["Dome"]["D"] / 2,
        height=d["Dome"]["height"]
    )

    if mode == 0:       # needle shoots e- (cathode) and small sphere receives (anode)
        cathode = Needle(
            position=np.array(d["Needle"]["position"]),
            lc=d["Needle"]["lc"],
            lb=d["Needle"]["lb"],
            r=d["Needle"]["r"],
            direction_vector=np.array(d["Needle"]["direction"])
        )
        anode = Sphere(
            radius=d["SmallSphere"]["R"],
            position=np.array(d["SmallSphere"]["position"]),
            direction_vector=np.array(d["SmallSphere"]["direction"]),
            dipole_moment = d["SmallSphere"]["dipole_moment"]
        )
    elif mode == 1:     # needs 3 objects  : not handled yet
        print("please don't")
    elif mode == 2:     # small sphere shoots e- (cathode) and large sphere receives (anode)
        cathode = Sphere(
            radius=d["SmallSphere"]["R"],
            position=np.array(d["SmallSphere"]["position"]),
            direction_vector=np.array(d["SmallSphere"]["direction"]),
            dipole_moment = d["SmallSphere"]["dipole_moment"]
        )
        anode = Sphere(
            radius=d["LargeSphere"]["R"],
            position=np.array(d["LargeSphere"]["position"]),
            direction_vector=np.array(d["LargeSphere"]["direction"]),
            dipole_moment = d["LargeSphere"]["dipole_moment"]
        )
    elif mode == 3:     # large sphere shoots e- (cathode) and small sphere receives (anode) : auroral observations 
        cathode = Sphere(
            radius=d["LargeSphere"]["R"],
            position=np.array(d["LargeSphere"]["position"]),
            direction_vector=np.array(d["LargeSphere"]["direction"]),
            dipole_moment = d["LargeSphere"]["dipole_moment"]
        )
        anode = Sphere(
            radius=d["SmallSphere"]["R"],
            position=np.array(d["SmallSphere"]["position"]),
            direction_vector=np.array(d["SmallSphere"]["direction"]),
            dipole_moment = d["SmallSphere"]["dipole_moment"]
        )

    simulation_settings = d["Simulation"]   # settings to transfer to simulation.py

    planeterrella = Planeterrella(cathode = cathode, anode=anode, dome=dome)        # physical setup of the experiment
    experiment = Experiment(planeterrella=planeterrella, gas=gas, simulationSettings=simulation_settings, collisions = collisions)

    # Checking if a special debug mod is enabled, overriding the normal simulation
    if d["Debug"]["Plot_B_field"]:
        experiment.debug = "B_field"
    if d["Debug"]["Plot_E_field"]:
        experiment.debug = "E_field"

    return experiment