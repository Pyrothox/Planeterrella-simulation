from src.config import load_experiment
from src.experiment import Experiment
from src.renderer import Renderer
from src.simulation import Simulation
def main():
    print("Loading experiment configuration...")
    experiment = load_experiment("config.toml")
    print("Experiment configuration loaded successfully.")
    print("Rendering experiment...")
    #renderer = Renderer(experiment)
    #renderer.render_empty()

    simulation = Simulation(experiment)
    print("Starting simulation...")
    diags = simulation.run()
    print("Simulation completed. Rendering trajectories...")
    data = diags.trajectoryRecorder.get_trajectories()
    #renderer.render_lines(data)
    print("Rendering completed. Locking the renderer to keep the window open. \n Close the window to exit the program before running a new simulation.")
    #renderer.lock()
    return diags

if __name__ == "__main__":
    main()
