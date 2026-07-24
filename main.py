from src.config import load_experiment
from src.experiment import Experiment
from src.renderer import Renderer, render_B_field, render_E_field
from src.simulation import Simulation
import h5py

def main():
    print("Loading experiment configuration...")
    experiment = load_experiment("config.toml")
    print("Experiment configuration loaded successfully.")

    if experiment.debug:            # special request that bypass the simulation
        if experiment.debug == "B_field":
            print("Debug mode: Plotting magnetic field lines only.")
            render_B_field(experiment)
        elif experiment.debug == "E_field":
            print("Debug mode: Plotting electric field lines only.")
            render_E_field(experiment)
        return

    print("Rendering experiment...")
    renderer = Renderer(experiment)
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
    collision_data = diagnostics.collisionRecorder.collisions[:diagnostics.collisionRecorder.ncollisions] if diagnostics.collisionRecorder else None
    if experiment.collisions :
        renderer.render_collisions(collision_data, point_size = 5.0)
    print("Rendering completed. Locking the renderer to keep the window open. \n Close the window to exit the program before running a new simulation.")
    renderer.lock()

if __name__ == "__main__":
    main()
