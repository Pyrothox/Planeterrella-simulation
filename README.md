# Planeterrella Simulation

A Python particle simulation of a **Planeterrella** — a tabletop device that recreates aurora-like glow discharges by launching electrons from a cathode into a dipole magnetic field, in a low-pressure N₂/O₂ atmosphere. This is a Python port and extension of the original MATLAB simulation by **Mathieu Barthélémy**.

The simulation integrates individual electron trajectories under combined electric and magnetic fields (Boris pusher), and models electron–neutral collisions (elastic and inelastic) with a Monte Carlo collision engine using real N₂ and O₂ cross-section data.

## Physics model

- **Fields**: a magnetic dipole field from the small/large sphere(s), and an electrostatic field between cathode and anode.
- **Particle pusher**: the Boris algorithm for charged-particle motion in combined E/B fields, with adaptive time-stepping based on the local cyclotron period.
- **Electron emission**: electrons are launched from the cathode surface (needle or sphere) via a Monte Carlo model with a cosine-weighted (Lambertian) angular distribution.
- **Collisions**: electron–N₂ and electron–O₂ collisions (elastic scattering, electronic excitation, and ionization) using tabulated cross-sections, sampled via a null-collision Monte Carlo scheme.
- **Geometry**: configurable dome, cathode/anode spheres or needle, all defined in `config.toml`.

## Project structure

```
.
├── main.py                    # Entry point — loads config, runs simulation, renders results
├── config.toml                 # All experiment parameters (geometry, gas, voltage, particle counts, etc.)
├── src/
│   ├── config.py                # Loads config.toml into an Experiment object
│   ├── experiment.py             # Experiment container (geometry + gas + fields + settings)
│   ├── geometry.py                # Planeterrella geometry: Dome, Sphere, Needle, Dipole
│   ├── fields.py                  # Magnetic and electric field models
│   ├── particles.py                # Electron population state (position, velocity, alive mask)
│   ├── Boris.py                     # Boris pusher for particle integration
│   ├── monteCarlo.py                 # Electron emission sampling (cathode surface)
│   ├── collisions.py                  # Monte Carlo collision engine (elastic + inelastic)
│   ├── gas.py                          # Background gas (species, pressure, temperature)
│   ├── diagnostics.py                   # Trajectory & collision recording, HDF5 export
│   └── renderer.py                       # 3D rendering (PyVista): geometry, fields, trajectories
├── cross_sections/
│   ├── cross_section_elastic.py           # Elastic e-N2/O2 cross-sections
│   ├── cross_section_inelastic_N2.py       # N2 excitation/ionization channels
│   └── cross_section_inelastic_O2.py        # O2 excitation/ionization channels
└── test.ipynb                              # Scratch notebook for experimentation
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management (a `uv.lock` is included).

```bash
git clone <your-repo-url>
cd Planeterrella-simulation
uv sync
```

Alternatively, with plain `pip`:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -e .
```

Requires **Python ≥ 3.12**.

## Usage

Edit `config.toml` to set up your experiment (geometry, gas, voltage, particle count, number of steps), then run:

```bash
uv run main.py
```

This will:
1. Load the experiment configuration.
2. Open a 3D PyVista window showing the empty geometry.
3. Run the particle simulation with a progress bar.
4. Save diagnostics (trajectories + collisions) to `simulation_data.h5`.
5. Render electron trajectories and collision points in 3D.

### Debug modes

Set `Plot_B_field = true` or `Plot_E_field = true` under `[Debug]` in `config.toml` to visualize the field geometry only, without running a full particle simulation.

### Key configuration options (`config.toml`)

| Section | Parameter | Description |
|---|---|---|
| top-level | `mode` | Planeterrella configuration: needle+sphere, or two spheres (see comments in `config.toml`) |
| top-level | `collisions` | Enable/disable electron–neutral collisions |
| `[Simulation]` | `Nparticles`, `Nsteps` | Number of electrons and integration steps |
| `[Simulation]` | `voltage` | Cathode–anode discharge voltage (V) |
| `[Gas]` | `pressure`, `temperature`, `species` | Background N₂/O₂ atmosphere |
| `[SmallSphere]` / `[LargeSphere]` | `R`, `position`, `direction` | Electrode geometry |
| `[Dome]` | `D`, `height` | Simulation domain / chamber boundary |

## Output

- `simulation_data.h5`: HDF5 file with recorded electron trajectories and collision events (species, type, energy loss).
- Interactive 3D render of trajectories and collision locations, color-coded by type.

## Known limitations / roadmap

This is an active work in progress. Known simplifications currently being addressed:

- Electron emission energy is being decoupled from the discharge voltage (secondary-emission electrons start near-thermal, not pre-accelerated).
- Elastic scattering is currently isotropic in the lab frame rather than forward-peaked.
- No space-charge feedback (test-particle model in a fixed background field).
- No secondary electron emission cascade from collisions with the cathode.

## Credits

- Original MATLAB Planeterrella simulation: **Mathieu Barthélémy**.