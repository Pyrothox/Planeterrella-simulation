from src import gas
from src import geometry
from src import fields

class Experiment:
    def __init__(self, planeterrella: geometry.Planeterrella, gas: gas.Gas, simulationSettings : dict, collisions : bool):
        self.planeterrella = planeterrella
        self.gas = gas
        self.simSettings = simulationSettings
        self.collisions = collisions
        self.MagneticField = fields.MagneticField(planeterrella)