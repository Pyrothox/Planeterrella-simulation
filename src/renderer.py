import __future__
from src.experiment import Experiment
from src.geometry import Planeterrella, Sphere, Needle, Dome
import pyvista as pv
import numpy as np
# Render the geometry of the experiment

class Renderer:
    def __init__(self, experiment: Experiment):
        self.plotter = pv.Plotter()
        self.plotter.show(interactive_update=True)
        self.experiment = experiment
        self.plotter.set_background("black")

    def lock(self):
        self.plotter.show(interactive_update=False)

    def render_empty(self):
        geo = self.experiment.planeterrella
        # Create a PyVista plotter
        plotter = self.plotter


        # Add the dome to the plotter
        dome_mesh = pv.Cylinder(center=[0, 0, geo.dome.height / 2], direction=[0, 0, 1], radius=geo.dome.radius, height=geo.dome.height)
        plotter.add_mesh(dome_mesh, color='lightblue', opacity=0.5)

        # Add the cathode and anode to the plotter
        if isinstance(geo.cathode, Needle):
            cathode_mesh = pv.Cylinder(center=geo.cathode.position,
                                        direction=geo.cathode.direction_vector,
                                        radius=0.01, height=geo.cathode.lc + geo.cathode.lb)
        elif isinstance(geo.cathode, Sphere):
            cathode_mesh = pv.Sphere(radius=geo.cathode.radius,
                                    center=geo.cathode.position) 
        plotter.add_mesh(cathode_mesh, color='silver')


        if isinstance(geo.anode, Sphere):
            anode_mesh = pv.Sphere(radius=geo.anode.radius,
                                center=geo.anode.position)
        elif isinstance(geo.anode, Needle):
            anode_mesh = pv.Cylinder(center=geo.anode.position,
                                    direction=geo.anode.direction_vector,
                                    radius=0.01, height=geo.anode.lc + geo.anode.lb)
        plotter.add_mesh(anode_mesh, color='silver')



    def render_lines(self, data : np.ndarray):
        # go from (Nstep, Nelec, 3) to (Nelec, Nstep, 3)
        trajectories = np.transpose(data, (1, 0, 2))
        points = trajectories.reshape(-1, 3)  # Flatten the array to (Nelec*Nstep, 3)
        lines =[]
        nElec = trajectories.shape[0]
        nsteps = trajectories.shape[1]  
        plotter = self.plotter
        for i in range(nElec):
            electron_points = trajectories[i]
            raw_segments = pv.lines_from_points(electron_points)
            plotter.add_mesh(raw_segments, color='red', line_width=1)
            
        

    def PlotB(self, n_points=10):
        geo = self.experiment.planeterrella
        # Create a grid of points in the simulation space
        x = np.linspace(-geo.dome.radius, geo.dome.radius, n_points)
        y = np.linspace(-geo.dome.radius, geo.dome.radius, n_points)
        z = np.linspace(0, geo.dome.height, n_points)
        X, Y, Z = np.meshgrid(x, y, z)
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        # Calculate the magnetic field at each point
        B = self.experiment.MagneticField.at(points)

        # Create a PyVista plotter
        plotter = self.plotter

        # Add the magnetic field vectors to the plotter
        plotter.add_arrows(points, B, mag=1.0, color='blue')

    def PlotB2(self, n=10):
        BOUNDS = (-2,2,-2,2,0,2)
        RESOLUTION = 15

        grid = pv.ImageData(
            dimensions=(RESOLUTION, RESOLUTION, RESOLUTION),
            spacing=(
                (BOUNDS[1] - BOUNDS[0]) / (RESOLUTION - 1),
                (BOUNDS[3] - BOUNDS[2]) / (RESOLUTION - 1),
                (BOUNDS[5] - BOUNDS[4]) / (RESOLUTION - 1),
            ),
            origin=(BOUNDS[0], BOUNDS[2], BOUNDS[4]),)
        
        points = grid.points
        B = self.experiment.MagneticField.at(points)
        B_magnitude = np.linalg.norm(B, axis=1)

        grid["B"] = B
        grid["B_magnitude"] = B_magnitude

        plotter = self.plotter

        glyphs = grid.glyph(orient="B", scale="B_magnitude", factor=0.3)
        plotter.add_mesh(glyphs, cmap="plasma", scalars="B_magnitude", show_scalar_bar=True)

        seed = pv.Sphere(radius=0.1, center=(0, -0.1, 0.2), theta_resolution=12, phi_resolution=12)
        streamlines = grid.streamlines_from_source(
            seed,
            vectors="B",
            max_length=50.0,
            integration_direction="both",
        )
        if streamlines.n_points > 0:
            plotter.add_mesh(streamlines.tube(radius=0.005), color="cyan")
        
        # --- Optional: translucent magnitude slice for spatial context ---
        slice_mesh = grid.slice(normal="z")
        plotter.add_mesh(slice_mesh, scalars="B_magnitude", cmap="coolwarm", opacity=0.4)
        
        plotter.add_axes()
        plotter.show_grid()
        plotter.add_title("Magnetic Field Visualization")
        plotter.show()


