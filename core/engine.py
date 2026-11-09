"""One-dimensional, quasi-steady hybrid rocket internal-ballistics model.

The model couples a Marxman-style fuel regression law to a choked nozzle.
It is intended for conceptual design and educational trade studies, not for
flight qualification or safety-critical hardware decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import brentq


_FLOAT_EPSILON = np.finfo(float).eps


def _finite_scalar(value: object, default: float = 0.0) -> float:
    """Convert a scalar-like value to a finite float with a strict fallback."""
    try:
        array = np.asarray(value, dtype=float)
        if array.size != 1:
            raise ValueError("expected a scalar")
        scalar = array.reshape(-1)[0]
    except (TypeError, ValueError, OverflowError):
        scalar = default
    return float(
        np.nan_to_num(scalar, nan=default, posinf=default, neginf=default)
    )


def _safe_denominator(value: object) -> float:
    """Return a finite, non-zero denominator."""
    return max(abs(_finite_scalar(value, _FLOAT_EPSILON)), _FLOAT_EPSILON)


@lru_cache(maxsize=256)
def _solve_supersonic_exit_mach(gamma: float, expansion_ratio: float) -> float:
    """Solve the isentropic area-Mach relation and cache repeated designs."""

    def area_residual(mach: float) -> float:
        area_ratio = (
            (1.0 / mach)
            * (
                (2.0 / (gamma + 1.0))
                * (1.0 + (gamma - 1.0) * mach**2 / 2.0)
            )
            ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        )
        return area_ratio - expansion_ratio

    return float(brentq(area_residual, 1.0001, 20.0))


@dataclass(slots=True)
class HybridRocketSimulator:
    """Simulate a cylindrical single-port hybrid rocket grain.

    All inputs use SI units.  The Marxman law is
    ``regression_rate = a * oxidizer_mass_flux ** n``.
    """

    oxidizer_flow_rate: float = 1.50
    initial_port_diameter: float = 0.040
    grain_length: float = 0.50
    burn_time: float = 5.0
    outer_grain_diameter: float = 0.120
    fuel_density: float = 930.0
    regression_coefficient: float = 6.30e-5
    regression_exponent: float = 0.62
    throat_diameter: float = 0.025
    expansion_ratio: float = 4.0
    characteristic_velocity: float = 1550.0
    specific_heat_ratio: float = 1.22
    nozzle_discharge_coefficient: float = 0.98
    ambient_pressure: float = 101_325.0
    time_step: float = 0.02

    def __post_init__(self) -> None:
        defaults = {
            "oxidizer_flow_rate": 1.50,
            "initial_port_diameter": 0.040,
            "grain_length": 0.50,
            "burn_time": 5.0,
            "outer_grain_diameter": 0.120,
            "fuel_density": 930.0,
            "regression_coefficient": 6.30e-5,
            "regression_exponent": 0.62,
            "throat_diameter": 0.025,
            "expansion_ratio": 4.0,
            "characteristic_velocity": 1550.0,
            "specific_heat_ratio": 1.22,
            "nozzle_discharge_coefficient": 0.98,
            "ambient_pressure": 101_325.0,
            "time_step": 0.02,
        }
        for name, default in defaults.items():
            setattr(self, name, _finite_scalar(getattr(self, name), default))
        self.ambient_pressure = max(self.ambient_pressure, 0.0)

        positive = {
            "oxidizer_flow_rate": self.oxidizer_flow_rate,
            "initial_port_diameter": self.initial_port_diameter,
            "grain_length": self.grain_length,
            "burn_time": self.burn_time,
            "outer_grain_diameter": self.outer_grain_diameter,
            "fuel_density": self.fuel_density,
            "regression_coefficient": self.regression_coefficient,
            "throat_diameter": self.throat_diameter,
            "expansion_ratio": self.expansion_ratio,
            "characteristic_velocity": self.characteristic_velocity,
            "time_step": self.time_step,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.initial_port_diameter >= self.outer_grain_diameter:
            raise ValueError("initial port diameter must be smaller than the grain diameter")
        if not 0 < self.regression_exponent < 1:
            raise ValueError("regression_exponent must be between zero and one")
        if not 1 < self.specific_heat_ratio < 2:
            raise ValueError("specific_heat_ratio must be between one and two")
        if self.expansion_ratio <= 1:
            raise ValueError("expansion_ratio must be greater than one")

    @property
    def throat_area(self) -> float:
        """Nozzle throat area in square metres."""
        area = np.pi * self.throat_diameter**2 / 4.0
        return max(_finite_scalar(area, _FLOAT_EPSILON), _FLOAT_EPSILON)

    def oxidizer_mass_flux(self, port_radius: float) -> float:
        """Return axial oxidizer mass flux at the port inlet."""
        radius = max(
            _finite_scalar(port_radius, self.initial_port_diameter / 2.0),
            _FLOAT_EPSILON,
        )
        port_area = _safe_denominator(np.pi * radius**2)
        flux = np.divide(self.oxidizer_flow_rate, port_area)
        return max(_finite_scalar(flux, 0.0), 0.0)

    def regression_rate(self, port_radius: float) -> float:
        """Return instantaneous Marxman radial regression rate in m/s."""
        flux = max(self.oxidizer_mass_flux(port_radius), 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            rate = self.regression_coefficient * np.power(
                flux, self.regression_exponent
            )
        return max(_finite_scalar(rate, 0.0), 0.0)

    def fuel_mass_flow(self, port_radius: float) -> float:
        """Return fuel mass generation rate from the burning port surface."""
        radius = max(
            _finite_scalar(port_radius, self.initial_port_diameter / 2.0), 0.0
        )
        burning_area = max(
            _finite_scalar(2.0 * np.pi * radius * self.grain_length, 0.0), 0.0
        )
        fuel_flow = self.fuel_density * burning_area * self.regression_rate(radius)
        return max(_finite_scalar(fuel_flow, 0.0), 0.0)

    def chamber_pressure(self, total_mass_flow: float) -> float:
        """Estimate quasi-steady chamber pressure from choked nozzle flow."""
        mass_flow = max(_finite_scalar(total_mass_flow, 0.0), 0.0)
        denominator = _safe_denominator(
            self.nozzle_discharge_coefficient * self.throat_area
        )
        pressure = np.divide(mass_flow * self.characteristic_velocity, denominator)
        return max(_finite_scalar(pressure, 0.0), 0.0)

    def _exit_mach(self) -> float:
        return _solve_supersonic_exit_mach(
            self.specific_heat_ratio, self.expansion_ratio
        )

    def thrust_coefficient(self, chamber_pressure: float) -> float:
        """Return ideal nozzle thrust coefficient with ambient correction."""
        gamma = self.specific_heat_ratio
        exit_mach = self._exit_mach()
        safe_pressure = _safe_denominator(chamber_pressure)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            pressure_ratio = (1.0 + (gamma - 1.0) * exit_mach**2 / 2.0) ** (
                -gamma / (gamma - 1.0)
            )
            momentum_term = np.sqrt(
                (2.0 * gamma**2 / (gamma - 1.0))
                * (2.0 / (gamma + 1.0))
                ** ((gamma + 1.0) / (gamma - 1.0))
                * (1.0 - pressure_ratio ** ((gamma - 1.0) / gamma))
            )
        pressure_term = self.expansion_ratio * (
            pressure_ratio - np.divide(self.ambient_pressure, safe_pressure)
        )
        coefficient = np.nan_to_num(
            momentum_term + pressure_term, nan=0.0, posinf=0.0, neginf=0.0
        )
        return max(float(coefficient), 0.0)

    def simulate(self) -> pd.DataFrame:
        """Integrate port regression and return the full time history."""
        rows: list[dict[str, float]] = []
        radius = self.initial_port_diameter / 2.0
        outer_radius = self.outer_grain_diameter / 2.0
        time = 0.0

        while time <= self.burn_time + 1e-12 and radius < outer_radius:
            flux = self.oxidizer_mass_flux(radius)
            rate = self.regression_rate(radius)
            fuel_flow = self.fuel_mass_flow(radius)
            total_flow = max(
                _finite_scalar(self.oxidizer_flow_rate + fuel_flow, 0.0), 0.0
            )
            of_ratio = _finite_scalar(
                np.divide(self.oxidizer_flow_rate, _safe_denominator(fuel_flow)),
                0.0,
            )
            pressure = self.chamber_pressure(total_flow)
            coefficient = self.thrust_coefficient(pressure)
            thrust = max(
                _finite_scalar(coefficient * pressure * self.throat_area, 0.0),
                0.0,
            )
            specific_impulse = _finite_scalar(
                np.divide(thrust, _safe_denominator(total_flow * 9.80665)), 0.0
            )

            rows.append(
                {
                    "time_s": time,
                    "port_diameter_mm": radius * 2.0 * 1_000.0,
                    "oxidizer_flux_kg_m2_s": flux,
                    "regression_rate_mm_s": rate * 1_000.0,
                    "fuel_flow_kg_s": fuel_flow,
                    "of_ratio": of_ratio,
                    "chamber_pressure_bar": pressure / 100_000.0,
                    "thrust_n": thrust,
                    "specific_impulse_s": specific_impulse,
                }
            )

            step = min(self.time_step, self.burn_time - time)
            if step <= 1e-9:
                break
            radius = min(
                _finite_scalar(radius + rate * step, outer_radius), outer_radius
            )
            time = _finite_scalar(time + step, self.burn_time)

        result = pd.DataFrame.from_records(rows)
        if not result.empty:
            result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return result
