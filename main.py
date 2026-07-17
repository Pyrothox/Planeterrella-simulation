from src.config import load_experiment
from src.experiment import Experiment
from src.renderer import render_empty
from src.simulation import Simulation

def main():
    print("Loading experiment configuration...")
    experiment = load_experiment("config.toml")
    print("Experiment configuration loaded successfully.")
    render_empty(experiment)


    simulation = Simulation(experiment)
    print("Starting simulation...")
    simulation.run()


if __name__ == "__main__":
    main()
