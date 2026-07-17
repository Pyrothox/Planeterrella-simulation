from src.experiment import Experiment
from src.geometry import Geometry, Sphere, Needle, Dome
import pyvista as pv

# Render the geometry of the experiment


def render_empty(experiment: Experiment):
    geo = experiment.geometry
    # Create a PyVista plotter
    plotter = pv.Plotter()


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

    # Set camera position and show the plot
    plotter.show_axes()
    plotter.show()