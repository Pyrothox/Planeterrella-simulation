import numpy as np 
from src.experiment import Experiment
from src.particles import Electrons
from src.geometry import Needle, Sphere
from src.monteCarlo import MonteCarloNeedle, MonteCarloSphere
from src.Boris import BorisPusher
from src.diagnostics import Diagnostics
from src.collisions import CollisionEngine
from rich.progress import (Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn)

class Simulation:
    # Runs the physical simulation
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        simSettings = experiment.simSettings
        self.Nparticles = simSettings["Nparticles"]
        self.Nsteps = simSettings["Nsteps"]

    def run(self):
        print("Running simulation with the following settings:")
        print(f"Geometry: {self.experiment.planeterrella}")
        print(f"Gas: {self.experiment.gas}")
        print(f"Simulation Settings: {self.experiment.simSettings}")

        # creating required objects for the simulation
        N = self.Nparticles
        emission_eV = self.experiment.simSettings["emission_eV"]  #initial electron energy
        cathode = self.experiment.planeterrella.cathode
        dt = 1e-12/N;    # initial time step before adaptive step size computation

        electrons = Electrons(N, cathode, dt, emission_eV)
        diags = Diagnostics(electrons, collisionsEnabled=self.experiment.collisions)  # Initialize diagnostics with collision recording if enabled
        if self.experiment.collisions:
            collisionEngine = CollisionEngine(self.experiment.gas)
        else :
            collisionEngine = type('Dummy', (), {'collide': lambda *args, **kwargs: None})()        #dummy collision engine that does nothing if collisions are disabled#running the simulation for Nsteps
        
        # running simulation with a progress bar using rich library
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TextColumn("[yellow]Alive: {task.fields[alive]}%"),
            ) as progress:
            task = progress.add_task("[green]Simulating...", total=self.Nsteps, alive=100.0)
            
            for step in range(self.Nsteps):     # running the simulation for Nsteps

                BorisPusher(electrons, self.experiment.MagneticField, self.experiment.ElectricField)       #updating the positions and velocities of electrons using the Boris algorithm
                electrons.alive[electrons.alive] &= self.experiment.planeterrella.Out_of_Bounds(electrons.position[electrons.alive])       #electrons absorption check

                if step % (self.Nsteps // 10) == 0:
                #if False:
                    collisionEngine.collide(electrons, diags, debug=True)
                    print(f"Step {step}: dt = {electrons.dt[0]:.2e}, alive electrons: {electrons.alive.sum()}")
                else:
                    collisionEngine.collide(electrons, diags)       

                if step % 5 == 0:
                    if electrons.alive.sum() == 0:
                        print(f"All electrons are out of bounds at step {step}. Ending simulation.")
                        progress.update(task, advance=self.Nsteps - step)  # Update progress bar to complete
                        break
                    diags.recordStep() # record diagnostics every 25 steps
                

                progress.update(task, advance=1, alive= round(electrons.alive.sum()/N*100, 2))  # Update progress bar with alive percentage
            print("average travel distance: ", electrons.total_travel_distance.sum() / N)
            print("Cumulative time: ", electrons.cumdt.sum())
            if self.experiment.collisions:
                print("Total collisions: ", diags.collisionRecorder.ncollisions)
                print("Inelastic collisions: ", np.sum(diags.collisionRecorder.collisions["inelastic"][:diags.collisionRecorder.ncollisions]))
                print("mean free path: ", electrons.total_travel_distance.sum() / diags.collisionRecorder.ncollisions)
        return diags