import numpy as np
import pyvista as pv


# ---------------------------------------------------------------- seeding --
def _orthonormal_basis(axis):
    axis = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def seed_points_for_sphere(sphere, n_azimuth=12, colatitudes_deg=(15, 30, 45),
                            surface_offset=1.02):
    """Seed points just outside the sphere, in rings around both magnetic poles."""
    R = sphere["R"]
    center = np.asarray(sphere["position"], dtype=float)
    axis = np.asarray(sphere["direction"], dtype=float)
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


# ---------------------------------------------------------- RK4 integration --
def _unit_field(B, positions, sign):
    b = B(positions)
    norm = np.linalg.norm(b, axis=1, keepdims=True)
    norm[norm < 1e-15] = 1e-15
    return sign * b / norm, np.linalg.norm(b, axis=1)


def _trace_batch(B, seeds, sign, spheres, step_size, max_steps, max_radius):
    n = seeds.shape[0]
    pos = seeds.copy()
    active = np.ones(n, dtype=bool)
    points_out = [[] for _ in range(n)]
    bmag_out = [[] for _ in range(n)]
    centers = np.array([s["position"] for s in spheres], dtype=float)
    radii = np.array([s["R"] for s in spheres], dtype=float)

    for _ in range(max_steps):
        if not active.any():
            break
        idx = np.where(active)[0]
        p = pos[idx]

        k1, bm1 = _unit_field(B, p, sign)
        k2, _ = _unit_field(B, p + 0.5 * step_size * k1, sign)
        k3, _ = _unit_field(B, p + 0.5 * step_size * k2, sign)
        k4, _ = _unit_field(B, p + step_size * k3, sign)
        newp = p + (step_size / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        pos[idx] = newp

        stop = np.linalg.norm(newp, axis=1) > max_radius
        for c, r in zip(centers, radii):
            stop |= np.linalg.norm(newp - c, axis=1) < 1.001 * r

        for j, i in enumerate(idx):
            points_out[i].append(newp[j].copy())
            bmag_out[i].append(bm1[j])
        active[idx[stop]] = False

    return points_out, bmag_out


def trace_field_lines(B, spheres, n_azimuth=12, colatitudes_deg=(15, 30, 45),
                       step_size=0.002, max_steps=1500, max_radius=1.0):
    seeds = np.vstack([seed_points_for_sphere(s, n_azimuth, colatitudes_deg) for s in spheres])
    fwd_pts, fwd_b = _trace_batch(B, seeds, +1, spheres, step_size, max_steps, max_radius)
    bwd_pts, bwd_b = _trace_batch(B, seeds, -1, spheres, step_size, max_steps, max_radius)

    lines = []
    for i in range(seeds.shape[0]):
        full_points = list(reversed(bwd_pts[i])) + [seeds[i]] + fwd_pts[i]
        full_bmag = (list(reversed(bwd_b[i])) +
                     [np.linalg.norm(B(seeds[i][None, :])[0])] + fwd_b[i])
        if len(full_points) >= 2:
            lines.append({"points": np.array(full_points), "bmag": np.array(full_bmag)})
    return lines


# ------------------------------------------------------------------ plot --
def plot_field_lines(B, spheres, n_azimuth=12, colatitudes_deg=(15, 30, 45),
                      step_size=0.002, max_steps=1500, max_radius=1.0,
                      sphere_colors=None, line_colormap="plasma",
                      tube_radius=0.0015, background="black",
                      plotter=None, show=True):
    """
    Trace and plot planeterrella field lines with PyVista.

    B        : callable, B(positions (N,3)) -> (N,3)
    spheres  : list of {'R', 'position', 'direction'} dicts
    """
    lines = trace_field_lines(B, spheres, n_azimuth, colatitudes_deg,
                               step_size, max_steps, max_radius)

    if plotter is None:
        plotter = pv.Plotter()
    plotter.background_color = background

    all_bmag = np.concatenate([l["bmag"] for l in lines]) if lines else np.array([0, 1])
    clim = (np.percentile(all_bmag, 1), np.percentile(all_bmag, 99))

    for line in lines:
        pts = line["points"]
        if pts.shape[0] < 2:
            continue
        poly = pv.PolyData()
        poly.points = pts
        poly.lines = np.hstack([[pts.shape[0]], np.arange(pts.shape[0])])
        poly["Bmag"] = line["bmag"]
        tube = poly.tube(radius=tube_radius)
        plotter.add_mesh(tube, scalars="Bmag", cmap=line_colormap,
                          clim=clim, show_scalar_bar=False)

    default_colors = ["#4da6ff", "#ff6b6b", "#7bed9f", "#feca57"]
    for i, sphere in enumerate(spheres):
        color = sphere_colors[i] if sphere_colors else default_colors[i % len(default_colors)]
        mesh = pv.Sphere(radius=sphere["R"], center=sphere["position"],
                          theta_resolution=48, phi_resolution=48)
        plotter.add_mesh(mesh, color=color, smooth_shading=True, opacity=0.5)

    plotter.add_scalar_bar(title="|B| (T)", n_labels=4)
    if show:
        plotter.show()
    return plotter


# --------------------------------------------------------------- example --
if __name__ == "__main__":
    spheres = [
        {"R": 0.05, "position": [0, 0.12, 0.2], "direction": [0, 0, 1]},
        {"R": 0.1, "position": [0, -0.1, 0.2],
         "direction": [0, 0.34202014332, 0.93969262078]},
    ]
    from src.config import load_experiment
    experiment = load_experiment("config.toml")
    B = experiment.MagneticField.at
    # plug in your real B(positions) here
    plot_field_lines(B, spheres)