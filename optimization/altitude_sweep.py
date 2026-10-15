"""Robust multi-altitude grain and nozzle optimization.

SciPy differential evolution tunes grain length, throat area, and expansion
ratio.  A weighted robust objective rewards both mean and worst-case total
impulse over a 0--30 km International Standard Atmosphere pressure sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import differential_evolution

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.engine import HybridRocketSimulator


DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "optimized_design.json"


def isa_pressure(altitude_m: float) -> float:
    """Return ISA static pressure from sea level through 30 km."""
    altitude = float(np.clip(altitude_m, 0.0, 30_000.0))
    gravity = 9.80665
    gas_constant = 287.05287

    if altitude <= 11_000.0:
        temperature = 288.15 - 0.0065 * altitude
        return 101_325.0 * (temperature / 288.15) ** (
            -gravity / (gas_constant * -0.0065)
        )

    temperature_11 = 216.65
    pressure_11 = 22_632.06
    if altitude <= 20_000.0:
        return pressure_11 * np.exp(
            -gravity * (altitude - 11_000.0) / (gas_constant * temperature_11)
        )

    pressure_20 = 5_474.89
    temperature = temperature_11 + 0.001 * (altitude - 20_000.0)
    return pressure_20 * (temperature / temperature_11) ** (
        -gravity / (gas_constant * 0.001)
    )


def evaluate_design(
    design: Iterable[float],
    altitudes_m: np.ndarray,
    *,
    oxidizer_flow_rate: float,
    initial_port_diameter: float,
    burn_time: float,
    time_step: float = 0.05,
) -> list[dict[str, float]]:
    """Evaluate a design at every supplied altitude."""
    grain_length, throat_area, expansion_ratio = map(float, design)
    throat_diameter = np.sqrt(4.0 * throat_area / np.pi)
    performance: list[dict[str, float]] = []

    for altitude in altitudes_m:
        simulator = HybridRocketSimulator(
            oxidizer_flow_rate=oxidizer_flow_rate,
            initial_port_diameter=initial_port_diameter,
            grain_length=grain_length,
            burn_time=burn_time,
            throat_diameter=throat_diameter,
            expansion_ratio=expansion_ratio,
            ambient_pressure=isa_pressure(float(altitude)),
            time_step=time_step,
        )
        history = simulator.simulate()
        performance.append(
            {
                "altitude_m": float(altitude),
                "ambient_pressure_pa": float(simulator.ambient_pressure),
                "total_impulse_ns": float(
                    trapezoid(history["thrust_n"], history["time_s"])
                ),
                "mean_thrust_n": float(history["thrust_n"].mean()),
                "peak_thrust_n": float(history["thrust_n"].max()),
                "peak_chamber_pressure_bar": float(
                    history["chamber_pressure_bar"].max()
                ),
            }
        )
    return performance


def bell_nozzle_geometry(
    throat_area_m2: float,
    expansion_ratio: float,
    points: int = 120,
) -> dict[str, list[float]]:
    """Generate an 80%-length bell contour using a cubic Hermite profile."""
    throat_radius = np.sqrt(throat_area_m2 / np.pi)
    exit_radius = throat_radius * np.sqrt(expansion_ratio)
    bell_length = 0.8 * (exit_radius - throat_radius) / np.tan(np.deg2rad(15.0))

    x_diverging = np.linspace(0.0, bell_length, points)
    u = x_diverging / bell_length
    slope_throat = np.tan(np.deg2rad(30.0))
    slope_exit = np.tan(np.deg2rad(9.0))
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    radius_diverging = (
        h00 * throat_radius
        + h10 * bell_length * slope_throat
        + h01 * exit_radius
        + h11 * bell_length * slope_exit
    )

    chamber_radius = max(2.2 * throat_radius, 0.035)
    converging_length = 1.35 * (chamber_radius - throat_radius)
    x_converging = np.linspace(-converging_length, 0.0, max(points // 3, 20))
    v = (x_converging + converging_length) / converging_length
    radius_converging = throat_radius + (chamber_radius - throat_radius) * (
        1.0 - (3.0 * v**2 - 2.0 * v**3)
    )

    x = np.concatenate((x_converging[:-1], x_diverging))
    radius = np.concatenate((radius_converging[:-1], radius_diverging))
    return {
        "x_m": x.tolist(),
        "radius_m": radius.tolist(),
        "throat_radius_m": float(throat_radius),
        "exit_radius_m": float(exit_radius),
        "bell_length_m": float(bell_length),
    }


def optimize_altitude_design(
    *,
    oxidizer_flow_rate: float = 1.5,
    initial_port_diameter: float = 0.040,
    initial_grain_length: float | None = None,
    burn_time: float = 5.0,
    altitudes_m: Iterable[float] | None = None,
    max_iterations: int = 16,
    population_size: int = 7,
    seed: int = 42,
    output_path: str | Path | None = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Run robust multi-objective optimization and optionally save JSON output.

    The scalarized objective assigns 70% weight to mean altitude-sweep impulse
    and 30% to worst-altitude impulse. Pressure penalties enforce a practical
    15--85 bar operating envelope.
    """
    altitudes = np.asarray(
        list(altitudes_m) if altitudes_m is not None else np.linspace(0, 30_000, 7),
        dtype=float,
    )
    if altitudes.ndim != 1 or len(altitudes) < 2:
        raise ValueError("altitudes_m must contain at least two altitudes")
    if np.any((altitudes < 0.0) | (altitudes > 30_000.0)):
        raise ValueError("altitudes must be within 0 to 30,000 m")

    bounds = [
        (0.25, 1.20),       # grain length, m
        (2.8e-4, 1.10e-3), # throat area, m^2
        (2.0, 14.0),        # expansion ratio
    ]
    raw_seed_grain = 0.50 if initial_grain_length is None else initial_grain_length
    seed_grain_length = float(
        np.clip(
            np.nan_to_num(raw_seed_grain, nan=0.50, posinf=1.20, neginf=0.25),
            bounds[0][0], bounds[0][1],
        )
    )
    initial_candidate = [
        seed_grain_length,
        (bounds[1][0] + bounds[1][1]) / 2.0,
        (bounds[2][0] + bounds[2][1]) / 2.0,
    ]

    def objective(design: np.ndarray) -> float:
        performance = evaluate_design(
            design,
            altitudes,
            oxidizer_flow_rate=oxidizer_flow_rate,
            initial_port_diameter=initial_port_diameter,
            burn_time=burn_time,
        )
        impulses = np.asarray([item["total_impulse_ns"] for item in performance])
        pressures = np.asarray(
            [item["peak_chamber_pressure_bar"] for item in performance]
        )
        robust_impulse = 0.70 * impulses.mean() + 0.30 * impulses.min()
        high_pressure_penalty = 800.0 * np.square(
            np.maximum(pressures - 85.0, 0.0)
        ).mean()
        low_pressure_penalty = 150.0 * np.square(
            np.maximum(15.0 - pressures, 0.0)
        ).mean()
        return float(-robust_impulse + high_pressure_penalty + low_pressure_penalty)

    solution = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=max_iterations,
        popsize=population_size,
        tol=2.0e-3,
        mutation=(0.5, 1.0),
        recombination=0.75,
        polish=True,
        seed=seed,
        workers=1,
        updating="immediate",
        x0=initial_candidate,
    )

    grain_length, throat_area, expansion_ratio = map(float, solution.x)
    performance = evaluate_design(
        solution.x,
        altitudes,
        oxidizer_flow_rate=oxidizer_flow_rate,
        initial_port_diameter=initial_port_diameter,
        burn_time=burn_time,
        time_step=0.02,
    )
    impulses = np.asarray([item["total_impulse_ns"] for item in performance])
    result: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "oxidizer_flow_rate_kg_s": float(oxidizer_flow_rate),
            "initial_port_diameter_m": float(initial_port_diameter),
            "initial_grain_length_m": seed_grain_length,
            "burn_time_s": float(burn_time),
            "altitudes_m": altitudes.tolist(),
        },
        "objective": {
            "description": "70% mean impulse + 30% worst-altitude impulse",
            "robust_total_impulse_ns": float(
                0.70 * impulses.mean() + 0.30 * impulses.min()
            ),
            "mean_total_impulse_ns": float(impulses.mean()),
            "worst_total_impulse_ns": float(impulses.min()),
        },
        "optimized_design": {
            "grain_length_m": grain_length,
            "throat_area_m2": throat_area,
            "throat_diameter_mm": float(np.sqrt(4.0 * throat_area / np.pi) * 1_000.0),
            "expansion_ratio": expansion_ratio,
        },
        "solver": {
            "method": "scipy.optimize.differential_evolution",
            "success": bool(solution.success),
            "message": str(solution.message),
            "iterations": int(solution.nit),
            "function_evaluations": int(solution.nfev),
            "seed": seed,
        },
        "performance_by_altitude": performance,
        "bell_nozzle_geometry": bell_nozzle_geometry(throat_area, expansion_ratio),
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oxidizer-flow", type=float, default=1.5)
    parser.add_argument("--port-diameter-mm", type=float, default=40.0)
    parser.add_argument("--burn-time", type=float, default=5.0)
    parser.add_argument("--max-iterations", type=int, default=16)
    parser.add_argument("--population-size", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    result = optimize_altitude_design(
        oxidizer_flow_rate=arguments.oxidizer_flow,
        initial_port_diameter=arguments.port_diameter_mm / 1_000.0,
        burn_time=arguments.burn_time,
        max_iterations=arguments.max_iterations,
        population_size=arguments.population_size,
        output_path=arguments.output,
    )
    design = result["optimized_design"]
    print(
        "Optimized design: "
        f"L={design['grain_length_m']:.3f} m, "
        f"At={design['throat_area_m2'] * 1e6:.1f} mm^2, "
        f"epsilon={design['expansion_ratio']:.2f}"
    )
    print(f"Saved results to {arguments.output}")


if __name__ == "__main__":
    main()
