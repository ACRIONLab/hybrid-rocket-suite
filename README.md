# FIRELINE: Hybrid Rocket Simulator

  <img width="800" alt="FIRELINE Hybrid Rocket Simulator" src="https://github.com/user-attachments/assets/497d0b11-8db7-4a6b-8fa4-99d5b956ad76" />


[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fireline-hybrid-rocket.streamlit.app)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**🔥 Live demo: https://fireline-hybrid-rocket.streamlit.app**
*(Free hosting sleeps when idle, so the first load may take a few seconds.)*

**Hybrid Rocket Suite** is a collection of tools for hybrid rocket engine
calculations, performance analysis, design optimization, and interactive web
visualization.

## Features

- Live 1D internal-ballistics simulation: port regression, choked chamber
  pressure, and nozzle thrust
- Interactive input manifold for oxidizer flow rate, port diameter, grain
  length, and burn time
- Altitude and nozzle optimization with a robust ascent design search
- Verified against analytical solutions (see [Verification](#verification))

## Example output (default inputs: 1.5 kg/s, 40 mm port, 0.5 m grain, 5 s burn)

| Mean thrust | Peak chamber pressure | Total impulse | Mean O/F |
|---|---|---|---|
| 4.29 kN | 57.9 bar | 21.5 kN·s | 5.51 |

## Prerequisites

- Python 3.10 or newer
- `pip`
- A terminal or command prompt

## Installation

Clone or download the repository, then enter the project directory and create
an isolated Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

With the virtual environment active, launch the application from the
repository root:

```bash
streamlit run app.py
```

Streamlit will start the application and provide a local browser address. Stop
the server with `Ctrl+C`.

## Architecture

```text
hybrid_rocket_suite/
├── app.py                 # Streamlit backend and application entry point
├── index.html             # Frontend interface and visualization components
├── core/                  # Hybrid rocket simulation and engine calculations
├── optimization/          # Performance and design-optimization routines
├── data/                  # Generated or reference design data
├── web/                   # Web application source
├── deploy/                # Deployable frontend assets
└── requirements.txt       # Python dependency definitions
```

`app.py` connects the calculation and optimization modules to the interactive
web UI. `index.html` contains the frontend components used to present controls
and results. Python packages and version constraints are managed through
`requirements.txt`, while the domain logic is separated into `core` and
`optimization` modules.

## Verification

The simulator is checked against exact analytical solutions: port regression, nozzle
relations, mass conservation and choked-flow consistency. The port-radius error at the
default time step is 0.05 % and falls in proportion to the step size. This is code
verification, not validation against real motor tests. See
[docs/VERIFICATION.md](docs/VERIFICATION.md) for results and limitations.

Run the tests from the repository root:

```bash
python tests/test_engine_verification.py
```
## Model limitations

- Quasi-steady model: no ignition transient, chamber filling or feed-system dynamics.
- Characteristic velocity (c*) and discharge coefficient are fixed inputs; combustion
  chemistry is not computed.
- A single power-law regression model; erosive and radiation effects are ignored.
- The optimizer searches grain length 0.25 to 1.20 m, throat area 280 to 1100 mm², and
  expansion ratio 2 to 14. The reported optimum grain length (1.20 m) sits at the upper
  search limit, so it is constrained by the bound and is not a true interior optimum.

## License

This project is licensed under the [MIT License](LICENSE).
