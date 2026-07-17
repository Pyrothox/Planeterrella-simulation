from src.config import load_experiment
from src.experiment import Experiment
from src.renderer import render_empty, render_lines
from src.simulation import Simulation

def main():
    print("Loading experiment configuration...")
    experiment = load_experiment("config.toml")
    print("Experiment configuration loaded successfully.")
    print("Rendering experiment...")
    # render_empty(experiment)


    simulation = Simulation(experiment)
    print("Starting simulation...")
    data = simulation.run()
    data = data.trajectoryRecorder.get_trajectories()
    render_lines(data)

if __name__ == "__main__":
    main()
