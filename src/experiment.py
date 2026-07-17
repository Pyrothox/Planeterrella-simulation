import src.gas
import src.geometry


class Experiment:
    def __init__(self, geometry: src.geometry.Geometry, gas: src.gas.Gas, simulationSettings : dict):
        self.geometry = geometry
        self.gas = gas
        self.simSettings = simulationSettings