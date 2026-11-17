"""Tests for the optional injector-coupled oxidizer flow in core/engine.py.

Run from the repository root:  python tests/test_injector_coupling.py   (or: pytest tests/)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.engine import HybridRocketSimulator  # noqa: E402

CD_INJ, RHO_OX, P_TANK = 0.70, 800.0, 7.6e6


def design_injector_area(target_flow, tank_pressure=P_TANK, **kw):
    """Injector area that passes `target_flow` at the constant-flow design chamber pressure."""
    base = HybridRocketSimulator(oxidizer_flow_rate=target_flow, **kw)
    pc0 = base.simulate()["chamber_pressure_bar"].iloc[0] * 1e5
    return target_flow / (CD_INJ * np.sqrt(2 * RHO_OX * (tank_pressure - pc0)))


def coupled_sim(**kw):
    area = design_injector_area(1.5)
    return HybridRocketSimulator(tank_pressure=P_TANK, injector_area=area,
                                 injector_discharge_coefficient=CD_INJ,
                                 oxidizer_density=RHO_OX, **kw)


def test_feature_is_off_by_default_and_changes_nothing():
    df = HybridRocketSimulator().simulate()
    assert "oxidizer_flow_kg_s" not in df.columns
    off = HybridRocketSimulator(tank_pressure=P_TANK, injector_area=0.0).simulate()
    assert np.array_equal(df.to_numpy(), off.to_numpy())


def test_injector_equation_is_satisfied_at_every_step():
    sim = coupled_sim()
    df = sim.simulate()
    pc = df["chamber_pressure_bar"].to_numpy() * 1e5
    expected = CD_INJ * sim.injector_area * np.sqrt(2 * RHO_OX * (P_TANK - pc))
    got = df["oxidizer_flow_kg_s"].to_numpy()
    assert np.max(np.abs(got - expected) / expected) < 1e-9


def test_matches_an_independent_fixed_point_solution_at_t0():
    """Different algorithm: damped fixed-point iteration on the same two equations."""
    sim = coupled_sim()
    r0 = sim.initial_port_diameter / 2
    m = 1.0
    for _ in range(2000):
        fuel = sim.fuel_density * 2 * np.pi * r0 * sim.grain_length * sim.regression_coefficient * (
            m / (np.pi * r0**2)) ** sim.regression_exponent
        pc = (m + fuel) * sim.characteristic_velocity / (sim.nozzle_discharge_coefficient * sim.throat_area)
        m_new = CD_INJ * sim.injector_area * np.sqrt(2 * RHO_OX * (P_TANK - pc))
        m = 0.5 * m + 0.5 * m_new
    assert abs(sim.simulate()["oxidizer_flow_kg_s"].iloc[0] - m) / m < 1e-9


def test_recovers_constant_flow_when_tank_pressure_is_very_high():
    pt = 1.0e10
    area = 1.5 / (CD_INJ * np.sqrt(2 * RHO_OX * pt))
    df = HybridRocketSimulator(tank_pressure=pt, injector_area=area,
                               injector_discharge_coefficient=CD_INJ,
                               oxidizer_density=RHO_OX).simulate()
    flow = df["oxidizer_flow_kg_s"].to_numpy()
    assert np.max(np.abs(flow - 1.5) / 1.5) < 0.005


def test_flow_rises_as_the_port_opens_and_pressure_falls():
    df = coupled_sim().simulate()
    assert np.all(np.diff(df["oxidizer_flow_kg_s"]) > 0)
    assert np.all(np.diff(df["chamber_pressure_bar"]) < 0)
    assert df["chamber_pressure_bar"].max() * 1e5 < P_TANK


def test_fuel_mass_balance_still_closes():
    sim = coupled_sim(time_step=0.01)
    df = sim.simulate()
    burned = trapezoid(df["fuel_flow_kg_s"], df["time_s"])
    r0, rf = sim.initial_port_diameter / 2, df["port_diameter_mm"].iloc[-1] / 2000
    lost = sim.fuel_density * np.pi * (rf**2 - r0**2) * sim.grain_length
    assert abs(burned - lost) / lost < 0.01


def test_converges_with_time_step():
    d = [coupled_sim(time_step=dt).simulate()["port_diameter_mm"].iloc[-1] for dt in (0.04, 0.02, 0.01, 0.005)]
    e = [abs(d[i] - d[-1]) for i in range(3)]
    assert e[0] > e[1] > e[2]


def test_users_oxidizer_input_is_not_modified_by_a_run():
    sim = coupled_sim()
    sim.simulate()
    assert sim.oxidizer_flow_rate == 1.50


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print(f"PASS  {name}")
            except AssertionError:
                fails += 1; print(f"FAIL  {name}")
    sys.exit(1 if fails else 0)
