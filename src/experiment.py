import src.gas
import src.geometry


class Experiment:
    def __init__(self, planeterrella: src.geometry.Planeterrella, gas: src.gas.Gas, simulationSettings : dict):
        self.planeterrella = planeterrella
        self.gas = gas
        self.simSettings = simulationSettings