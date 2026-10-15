# Hybrid Rocket Suite

Hybrid Rocket Suite is a collection of tools for hybrid rocket engine
calculations, performance analysis, design optimization, and interactive web
visualization. It combines the simulation backend with a browser-based user
interface for exploring engine inputs and results.

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

## License

This project is licensed under the [MIT License](LICENSE).
