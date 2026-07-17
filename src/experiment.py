import src.gas
import src.geometry
import src.fields

class Experiment:
    def __init__(self, planeterrella: src.geometry.Planeterrella, gas: src.gas.Gas, simulationSettings : dict, collisions : bool):
        self.planeterrella = planeterrella
        self.gas = gas
        self.simSettings = simulationSettings
        self.collisions = collisions
        self.MagneticField = src.fields.MagneticField(planeterrella)