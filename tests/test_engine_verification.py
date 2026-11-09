"""Verification tests for core/engine.py.

These check that the code solves its own equations correctly, using exact
analytical solutions and independent re-derivations. They do NOT check the
model against real motor data (see docs/VERIFICATION.md, "Scope").

Run from the repository root:
    python tests/test_engine_verification.py        # prints a report
    pytest tests/                                   # if pytest is installed
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.engine import HybridRocketSimulator  # noqa: E402


def exact_port_radius(sim, t):
    """Exact solution of dr/dt = a (mdot_ox / (pi r^2))^n.

    Separating variables gives
        r^(2n+1) = r0^(2n+1) + (2n+1) a (mdot_ox/pi)^n t
    """
    n, a, m = sim.regression_exponent, sim.regression_coefficient, sim.oxidizer_flow_rate
    r0 = sim.initial_port_diameter / 2.0
    return (r0 ** (2 * n + 1) + (2 * n + 1) * a * (m / np.pi) ** n * t) ** (1.0 / (2 * n + 1))


def port_radius_error_percent(time_step):
    sim = HybridRocketSimulator(time_step=time_step)
    df = sim.simulate()
    r_num = df["port_diameter_mm"].to_numpy() / 2.0 / 1000.0
    r_ex = exact_port_radius(sim, df["time_s"].to_numpy())
    return float(np.max(np.abs(r_num - r_ex) / r_ex) * 100.0)


def test_port_radius_matches_exact_solution():
    """At the default time step the port radius is within 0.1 % of exact."""
    assert port_radius_error_percent(0.02) < 0.1


def test_port_radius_converges_first_order():
    """Halving the time step should roughly halve the error (forward Euler)."""
    e1 = port_radius_error_percent(0.02)
    e2 = port_radius_error_percent(0.01)
    assert 1.8 < e1 / e2 < 2.2


def test_exit_mach_matches_textbook():
    """gamma = 1.4, A/A* = 1.6875 must give Mach 2.000."""
    sim = HybridRocketSimulator(specific_heat_ratio=1.4, expansion_ratio=1.6875)
    assert abs(sim._exit_mach() - 2.0) < 1e-4


def test_thrust_coefficient_matches_momentum_balance():
    """CF from the code equals CF from an independent momentum balance.

    F = mdot*ue + (pe - pa)*Ae with mdot = pc*At/c*, and ue from the energy
    equation, so CF = F/(pc*At). Different algebra from the code's formula.
    """
    g, eps, pa, pc = 1.22, 4.0, 101_325.0, 5.0e6
    sim = HybridRocketSimulator(specific_heat_ratio=g, expansion_ratio=eps, ambient_pressure=pa)
    me = sim._exit_mach()
    pe_pc = (1 + 0.5 * (g - 1) * me**2) ** (-g / (g - 1))
    gamma_fn = np.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))   # mdot/(pc*At) = Gamma/sqrt(R Tc)
    ue_over_sqrtRTc = np.sqrt(2 * g / (g - 1) * (1 - pe_pc ** ((g - 1) / g)))
    cf_independent = gamma_fn * ue_over_sqrtRTc + (pe_pc - pa / pc) * eps
    assert abs(sim.thrust_coefficient(pc) - cf_independent) / cf_independent < 1e-9


def test_mass_conservation():
    """Integrated fuel flow equals the grain mass actually lost."""
    sim = HybridRocketSimulator(time_step=0.01)
    df = sim.simulate()
    burned = trapezoid(df["fuel_flow_kg_s"].to_numpy(), df["time_s"].to_numpy())
    r0 = sim.initial_port_diameter / 2.0
    rf = df["port_diameter_mm"].iloc[-1] / 2.0 / 1000.0
    lost = sim.fuel_density * np.pi * (rf**2 - r0**2) * sim.grain_length
    assert abs(burned - lost) / lost < 0.005


def test_choked_flow_consistency():
    """Pc * Cd * At / c* reproduces the total mass flow at every step."""
    sim = HybridRocketSimulator()
    df = sim.simulate()
    mdot = sim.oxidizer_flow_rate + df["fuel_flow_kg_s"].to_numpy()
    mdot_back = (
        df["chamber_pressure_bar"].to_numpy() * 1e5
        * sim.nozzle_discharge_coefficient * sim.throat_area / sim.characteristic_velocity
    )
    assert np.max(np.abs(mdot_back - mdot) / mdot) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError:
            failed += 1
            print(f"FAIL  {t.__name__}")
    print("\nPort-radius error vs exact solution:")
    for dt in [0.1, 0.05, 0.02, 0.01, 0.005, 0.0025]:
        print(f"  dt = {dt:<7} -> {port_radius_error_percent(dt):.5f} %")
    sys.exit(1 if failed else 0)
