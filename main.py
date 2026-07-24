from src.config import load_experiment
from src.experiment import Experiment
from src.renderer import Renderer, render_B_field, render_E_field
from src.simulation import Simulation
import h5py

def main():
    print("Loading experiment configuration...")
    experiment = load_experiment("config.toml")     #creates an experiment object from the config file
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
    renderer.render_empty()          # 3d plot of the experiment setup

    simulation = Simulation(experiment)     #initialize the simulation with the experiment parameters
    print("Starting simulation...")
    diagnostics = simulation.run()          # running sim and retreiving diagnostics

    print("Simulation completed. Saving diagnostics data...")
    name = "simulation_data.h5"
    with h5py.File(name, "w") as f:         #saving diagnostics data in a hdf5 file
        diagnostics.save_to_hdf5(f)
    
    print(f"Diagnostics data saved to {name}. Rendering trajectories...")
    trajectories = diagnostics.trajectoryRecorder.trajectories
    renderer.render_lines(trajectories)             #rendering all trajectories (may be laggy if too many electrons)
    if experiment.collisions :
        collision_data = diagnostics.collisionRecorder.collisions[:diagnostics.collisionRecorder.ncollisions]       #shortening the collision data to only the recorded collisions
        renderer.render_collisions(collision_data, point_size = 5.0)        #rendering all collisions (many are invisible)

    print("Rendering completed. Locking the renderer to keep the window open. \n Close the window to exit the program before running a new simulation.")
    renderer.lock()     #lock the renderer to keep the window open until closed by the user

if __name__ == "__main__":
    main()
