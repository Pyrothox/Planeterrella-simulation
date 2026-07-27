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
        match self.simSettings["E_field"]:
            case 0:
                self.ElectricField = None
            case 1:
                self.ElectricField = fields.ElectricField(planeterrella, self.simSettings["voltage"])
            case 2:
                self.ElectricField = fields.ElectricFieldv2(planeterrella, self.simSettings["voltage"])
            case _:
                raise ValueError(f"Invalid E_field value: {self.simSettings['E_field']}. Must be 0, 1, or 2.")
        self.debug = debug