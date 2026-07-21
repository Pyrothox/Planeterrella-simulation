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

    def PlotB1(self, n=10):
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



    def PlotB2(self):
        geo = self.experiment.planeterrella

        bounds = [-geo.dome.radius, geo.dome.radius, -geo.dome.radius, geo.dome.radius, -geo.dome.radius, geo.dome.radius]
        resolution = (100, 100, 100)

        grid = pv.ImageData(
        dimensions=resolution,
        spacing=(
            (bounds[1] - bounds[0]) / (resolution[0] - 1),
            (bounds[3] - bounds[2]) / (resolution[1] - 1),
            (bounds[5] - bounds[4]) / (resolution[2] - 1),
        ),
        origin=(bounds[0], bounds[2], bounds[4]),
        )

        grid["B"] = self.experiment.MagneticField.at(grid.points)
        r1 = geo.cathode.radius
        r2 = geo.anode.radius
        pos1 = geo.cathode.position
        pos2 = geo.anode.position
        seed1 = pv.Disc(
            center=pos1,
            inner=r1 * 1.05,
            outer=r1 * 3.0,
            r_res=2,  # Increased from 2 -> 8 (more radial concentric rings)
            c_res=12,  # Increased from 12 -> 36 (more angular points per ring)
        )

        seed2 = pv.Disc(
            center=pos2,
            inner=r2 * 1.05,
            outer=r2 * 3.0,
            r_res=2,
            c_res=12,
        )

        seeds  = seed1 + seed2

        # Trace streamlines in both directions
        # Generate Streamlines
        strl = grid.streamlines_from_source(
            seeds,
            vectors="B",
            max_length=180,
            initial_step_length=0.1,
            integration_direction="both",
        )


        # ==========================================
        # 4. Plotting (Your exact style)
        # ==========================================
        pl = pv.Plotter()

        # Field lines as tubes
        pl.add_mesh(
            strl.tube(radius=0.001),
            cmap="bwr",
            ambient=0.2,
            scalar_bar_args={"title": "|B| Field"},
        )

        # Add the 2 Spheres (Replacing your coil_block)
        s1_mesh = pv.Sphere(radius=r1, center=pos1)
        s2_mesh = pv.Sphere(radius=r2, center=pos2)

        pl.add_mesh(s1_mesh, color="w", opacity=0.9)
        pl.add_mesh(s2_mesh, color="w", opacity=0.9)

        pl.camera.zoom(1.8)
        pl.show()