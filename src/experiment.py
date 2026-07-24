from src import gas
from src import geometry
from src import fields

class Experiment:
    # This class encapsulates the entire experimental setup, including geometry, gas properties, simulation settings, and fields.
    def __init__(self, planeterrella: geometry.Planeterrella, gas: gas.Gas, simulationSettings : dict, collisions : bool, debug = False):
        self.planeterrella = planeterrella
        self.gas = gas
        self.simSettings = simulationSettings
        self.collisions = collisions
        self.MagneticField = fields.MagneticField(planeterrella)
        self.ElectricField = fields.ElectricField(planeterrella, self.simSettings["voltage"]) if self.simSettings["E_field"] else None
        self.debug = debug