import __future__
from src.experiment import Experiment
from src.geometry import Planeterrella, Sphere, Needle, Dome
import pyvista as pv
import numpy as np
from IPython import get_ipython

# Renders the geometry of the experiment

# Base colors for the different type of collisions. 
_COLOR_N2_ELASTIC   = np.array([ 40,  60, 200], dtype=np.uint8) # color = blue
_COLOR_N2_INELASTIC = np.array([200,  60,  40], dtype=np.uint8) # color = red
_COLOR_O2_ELASTIC   = np.array([100, 140, 255], dtype=np.uint8) # color = light blue
_COLOR_O2_INELASTIC = np.array([255, 140, 100], dtype=np.uint8) # color = light red


class Renderer:
    # creates a 3d rendering of the Planeterrella experiment using pyvista. Updated using class mthods. 
    def __init__(self, experiment: Experiment = None):
        self.plotter = pv.Plotter()     # pyvista plotter object 
        self.experiment = experiment
        self.plotter.set_background("black")

        if not (get_ipython() is not None):
            self.plotter.show(interactive_update=True)   # letting python code run while the plotter is opened if not in a notebook environment

    def lock(self):
        """ Locks the plotter to prevent further updates. Otherwise the render automatically closes when the main() function is done."""
        self.plotter.show(interactive_update=False)     # if the interactive_update is set to True, the code will automatically close the windows when main() is done

    def render_empty(self):
        """ Renders the geometry of the Planeterrella turned off."""
        geo = self.experiment.planeterrella
        plotter = self.plotter


        # Add the dome to the plotter
        dome_mesh = pv.Cylinder(center=[0, 0, geo.dome.height / 2], direction=[0, 0, 1], radius=geo.dome.radius, height=geo.dome.height)
        plotter.add_mesh(dome_mesh, color='lightblue', opacity=0.1, style='wireframe')

        # Add the cathode to the plotter
        if isinstance(geo.cathode, Needle):
            cathode_mesh = pv.Cylinder(center=geo.cathode.position,
                                        direction=geo.cathode.direction_vector,
                                        radius=0.01, height=geo.cathode.lc + geo.cathode.lb)
        elif isinstance(geo.cathode, Sphere):
            cathode_mesh = pv.Sphere(radius=geo.cathode.radius,
                                    center=geo.cathode.position) 
        plotter.add_mesh(cathode_mesh, color='silver')

        # add the anode to the plotter
        if isinstance(geo.anode, Sphere):
            anode_mesh = pv.Sphere(radius=geo.anode.radius,
                                center=geo.anode.position)
        elif isinstance(geo.anode, Needle):
            anode_mesh = pv.Cylinder(center=geo.anode.position,
                                    direction=geo.anode.direction_vector,
                                    radius=0.01, height=geo.anode.lc + geo.anode.lb)
        plotter.add_mesh(anode_mesh, color='silver')

    def render_lines(self, data : np.ndarray, color='red'):
        """ Renders the trajectories of given particles data
        Parameters
        ----------
        data : np.ndarray
            Array of shape (Nstep, Nparticles, 3) containing the particle positions over time.
        color : str, optional
            Color of the trajectories, by default 'red'
        """
        # go from (Nstep, Nparticles, 3) to (Nparticles, Nstep, 3)
        trajectories = np.transpose(data, (1, 0, 2))
        nElec = trajectories.shape[0]
        plotter = self.plotter
        for i in range(nElec):
            electron_points = trajectories[i]
            raw_segments = pv.lines_from_points(electron_points)
            plotter.add_mesh(raw_segments, color=color, line_width=1)
        print("rendered")

    def render_collisions(self, collision_data, point_size: float = 2.0, max_points: int = 100_000):
        """ Renders the collision events of the electrons.
        Parameters
        ----------
        collision_data : np.ndarray
            Array of shape (Ncollisions,) containing the collision events with fields: position, inelastic, specie, color.
        point_size : float, optional
            Size of the points representing collisions, by default 2.0
        max_points : int, optional
            Maximum number of collision points to render, by default 100_000. If the number of collisions exceeds this, a random subset will be rendered.
        """
        collision_data = np.asarray(collision_data)
        n = collision_data.shape[0]
        if n == 0:
            print("No collisions to render.")
            return

        if max_points is not None and n > max_points:       # too many collisions to render, randomly select a subset
            indices = np.random.choice(n, max_points, replace=False)
            collision_data = collision_data[indices]
            print(f"Rendering {max_points} random collisions out of {n} total collisions.")
        points = collision_data["position"]
        colors_uint = collision_data["color"]
        rgba_matrix = np.ascontiguousarray(colors_uint).view(np.uint8).reshape(-1, 4)[:, ::-1]  # reverse needed because of RAM storage order
        # lut = np.array([
        #     [_COLOR_O2_ELASTIC,  _COLOR_O2_INELASTIC],  # specie = False (O2)
        #     [_COLOR_N2_ELASTIC,  _COLOR_N2_INELASTIC],  # specie = True  (N2)
        # ], dtype=np.uint8)
        # colors = lut[collision_data["specie"].astype(np.uint8),
        #              collision_data["inelastic"].astype(np.uint8)]
        
        cloud = pv.PolyData(points)
        cloud["colors"] = rgba_matrix  
        self.plotter.add_points(cloud, scalars="colors", rgba=True, point_size=point_size, lighting=False)
        self.plotter.update()  # Update the plotter to reflect the new points



### standalone function to render the B and E field lines, can be used without creating a Renderer object
def _unit_field(F, positions):
    f = F(positions)
    norm = np.linalg.norm(f, axis=1, keepdims=True)
    norm[norm < 1e-15] = 1e-15
    return  f / norm, np.linalg.norm(f, axis=1)

def _trace_batch(F, seeds, spheres, step_size, max_steps, max_radius, sign):
    """ Trace lines using a simple Runge-Kutta 4th order integrator. Returns a list of points and field magnitudes for each seed point."""
    n = seeds.shape[0]
    pos = seeds.copy()
    active = np.ones(n, dtype=bool)
    points_out = [[] for _ in range(n)]
    mag_out = [[] for _ in range(n)]
    centers = np.array([s.position for s in spheres], dtype=float)
    radii = np.array([s.radius for s in spheres], dtype=float)

    for _ in range(max_steps):
        if not active.any():
            break
        idx = np.where(active)[0]
        p = pos[idx]

        k1, bm1 = _unit_field(F, p)
        k2, _ = _unit_field(F, p + 0.5 * step_size * k1)
        k3, _ = _unit_field(F, p + 0.5 * step_size * k2)
        k4, _ = _unit_field(F, p + step_size * k3)
        newp = p + sign * (step_size / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        pos[idx] = newp

        stop = np.linalg.norm(newp, axis=1) > max_radius
        for c, r in zip(centers, radii):
            stop |= np.linalg.norm(newp - c, axis=1) < 1.001 * r

        for j, i in enumerate(idx):
            points_out[i].append(newp[j].copy())
            mag_out[i].append(bm1[j])
        active[idx[stop]] = False

    return points_out, mag_out

def _orthonormal_basis(axis):
    """ Creates a local 3D orthonormal basis from one axis"""
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v

def seed_points_for_sphere(sphere : Sphere, n_azimuth=12, colatitudes_deg=(15, 30, 45),surface_offset=1.02):
    """Seed points just outside the sphere, in rings around both magnetic poles."""
    R = sphere.radius
    center = sphere.position
    axis = sphere.direction_vector
    axis = axis / np.linalg.norm(axis)
    u, v = _orthonormal_basis(axis)

    seeds = []
    azimuths = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
    for pole_sign in (+1, -1):
        pole_axis = pole_sign * axis
        for colat_deg in colatitudes_deg:
            colat = np.deg2rad(colat_deg)
            radial = np.cos(colat) * pole_axis
            for az in azimuths:
                tangential = np.sin(colat) * (np.cos(az) * u + np.sin(az) * v)
                d = radial + tangential
                d /= np.linalg.norm(d)
                seeds.append(center + R * surface_offset * d)
    return np.array(seeds)

def seed_points_uniform_sphere(sphere : Sphere, n_seeds=200, surface_offset=1.02):
    """
    Seed points uniformly distributed over an entire sphere's surface —
    appropriate for a charged sphere, where field lines emanate from
    (or converge onto) the whole surface, not just two poles.
    """
    R = sphere.radius
    center = np.asarray(sphere.position, dtype=float)

    # Fibonacci sphere: near-uniform point distribution
    i = np.arange(n_seeds)
    golden = (1 + 5 ** 0.5) / 2
    theta = np.arccos(1 - 2 * (i + 0.5) / n_seeds)   # colatitude
    phi = 2 * np.pi * i / golden                     # azimuth

    directions = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], axis=1)

    return center + R * surface_offset * directions

def trace_field_lines(F, spheres, seeds, step_size=0.002, max_steps=1500, max_radius=1.0, sign=1.0):
    """
    Trace magnetic field lines starting from specified colatitudes and azimuths around each sphere.
    """
    pts, mag = _trace_batch(F, seeds, spheres, step_size, max_steps, max_radius, sign)
    lines = []
    for i in range(seeds.shape[0]):
        full_points = [seeds[i]] + pts[i]
        full_mag = [np.linalg.norm(F(seeds[i][None, :])[0])] + mag[i]
        if len(full_points) >= 2:
            lines.append({"points": np.array(full_points), "mag": np.array(full_mag)})
    return lines

def render_B_field(experiment: Experiment):
    """
    Render the magnetic field lines of the experimental setup. Made to be used on its own.
    """
    B = experiment.MagneticField.at
    spheres = [experiment.planeterrella.cathode, experiment.planeterrella.anode]
    n_azimuth=12
    colatitudes_deg=(15, 30, 45)
    step_size=0.002
    max_steps=1500
    max_radius=1.0
    sphere_colors=None
    line_colormap="plasma"
    tube_radius=0.0015
    background="black"

    seeds = np.vstack([seed_points_for_sphere(s, n_azimuth, colatitudes_deg) for s in spheres])
    lines = trace_field_lines(B, spheres, seeds, step_size, max_steps, max_radius)
    plotter = pv.Plotter()
    plotter.background_color = background

    all_bmag = np.concatenate([l["mag"] for l in lines]) if lines else np.array([0, 1])
    clim = (np.percentile(all_bmag, 1), np.percentile(all_bmag, 99))

    for line in lines:
        pts = line["points"]
        if pts.shape[0] < 2:
            continue
        poly = pv.PolyData()
        poly.points = pts
        poly.lines = np.hstack([[pts.shape[0]], np.arange(pts.shape[0])])
        poly["Bmag"] = line["mag"]
        tube = poly.tube(radius=tube_radius)
        plotter.add_mesh(tube, scalars="Bmag", cmap=line_colormap,
                          clim=clim, show_scalar_bar=False)

    default_colors = ["#4da6ff", "#ff6b6b", "#7bed9f", "#feca57"]
    for i, sphere in enumerate(spheres):
        R = sphere.radius
        pos = sphere.position
        color = sphere_colors[i] if sphere_colors else default_colors[i % len(default_colors)]
        mesh = pv.Sphere(radius=R, center=pos,
                          theta_resolution=48, phi_resolution=48)
        plotter.add_mesh(mesh, color=color, smooth_shading=True)

    plotter.add_scalar_bar(title="|B| (T)", n_labels=4)
    plotter.show()

def render_E_field(experiment: Experiment):
    """
    Render the electric field lines of the experimental setup. Made to be used on its own.
    """
    E = experiment.ElectricField.at
    spheres = [experiment.planeterrella.cathode, experiment.planeterrella.anode]
    step_size=0.002
    max_steps=1500
    max_radius=1.0
    sphere_colors=None
    line_colormap="plasma"
    tube_radius=0.0015
    background="black"

    seeds = np.vstack([seed_points_uniform_sphere(s) for s in spheres])
    lines = trace_field_lines(E, spheres, seeds, step_size, max_steps, max_radius, sign=-1.0)
    lines += trace_field_lines(E, spheres, seeds, step_size, max_steps, max_radius, sign=1.0)
    plotter = pv.Plotter()
    plotter.background_color = background

    all_emag = np.concatenate([l["mag"] for l in lines]) if lines else np.array([0, 1])
    clim = (np.percentile(all_emag, 1), np.percentile(all_emag, 99))

    for line in lines:
        pts = line["points"]
        if pts.shape[0] < 2:
            continue
        poly = pv.PolyData()
        poly.points = pts
        poly.lines = np.hstack([[pts.shape[0]], np.arange(pts.shape[0])])
        poly["Emag"] = line["mag"]
        tube = poly.tube(radius=tube_radius)
        plotter.add_mesh(tube, scalars="Emag", cmap=line_colormap,
                          clim=clim, show_scalar_bar=False)

    default_colors = ["#4da6ff", "#ff6b6b", "#7bed9f", "#feca57"]
    for i, sphere in enumerate(spheres):
        R = sphere.radius
        pos = sphere.position
        color = sphere_colors[i] if sphere_colors else default_colors[i % len(default_colors)]
        mesh = pv.Sphere(radius=R, center=pos,
                          theta_resolution=48, phi_resolution=48)
        plotter.add_mesh(mesh, color=color, smooth_shading=True)

    plotter.add_scalar_bar(title="|E| (V/m)", n_labels=4, color="white")
    plotter.show()
    