from src.config import load_experiment
from src.experiment import Experiment
from src.renderer import Renderer
from src.simulation import Simulation
import h5py
def main():
    print("Loading experiment configuration...")
    experiment = load_experiment("config.toml")
    print("Experiment configuration loaded successfully.")
    print("Rendering experiment...")
    renderer = Renderer(experiment)
    renderer.PlotB()
    renderer.lock()
    renderer.render_empty()

    simulation = Simulation(experiment)
    print("Starting simulation...")
    diagnostics = simulation.run()

    print("Simulation completed. Saving diagnostics data...")
    name = "simulation_data.h5"
    with h5py.File(name, "w") as f:         #saving data in a hdf5 file
        diagnostics.save_to_hdf5(f)
    
    print(f"Diagnostics data saved to {name}. Rendering trajectories...")
    
    trajectories = diagnostics.trajectoryRecorder.trajectories
    renderer.render_lines(trajectories)
    print("Rendering completed. Locking the renderer to keep the window open. \n Close the window to exit the program before running a new simulation.")
    renderer.lock()

if __name__ == "__main__":
    main()
