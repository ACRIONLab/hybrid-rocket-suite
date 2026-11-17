"""Constant oxidizer flow vs injector-coupled oxidizer flow (illustrative design).

Run from the repository root:  python examples/injector_coupling_demo.py
The injector is sized to pass 1.5 kg/s at the start of the burn, so both runs begin at
the same operating point. Tank pressure, injector discharge coefficient and oxidizer
density are illustrative values, not a specific hardware design.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.engine import HybridRocketSimulator  # noqa: E402

P_TANK, CD_INJ, RHO_OX, M0 = 7.6e6, 0.70, 800.0, 1.5

const = HybridRocketSimulator(oxidizer_flow_rate=M0).simulate()
pc0 = const["chamber_pressure_bar"].iloc[0] * 1e5
area = M0 / (CD_INJ * np.sqrt(2 * RHO_OX * (P_TANK - pc0)))
coupled = HybridRocketSimulator(tank_pressure=P_TANK, injector_area=area,
                                injector_discharge_coefficient=CD_INJ,
                                oxidizer_density=RHO_OX).simulate()

print(f"injector area {area*1e6:.1f} mm^2 (tank {P_TANK/1e5:.0f} bar, Cd {CD_INJ}, rho {RHO_OX:.0f} kg/m3)")
for name, df in (("constant flow", const), ("injector-coupled", coupled)):
    ox = df["oxidizer_flow_kg_s"] if "oxidizer_flow_kg_s" in df else None
    print(f"{name:17s} Pc {df.chamber_pressure_bar.iloc[0]:.1f} -> {df.chamber_pressure_bar.iloc[-1]:.1f} bar | "
          f"thrust {df.thrust_n.iloc[0]/1000:.2f} -> {df.thrust_n.iloc[-1]/1000:.2f} kN | "
          f"O/F {df.of_ratio.iloc[0]:.2f} -> {df.of_ratio.iloc[-1]:.2f}"
          + (f" | mox {ox.iloc[0]:.3f} -> {ox.iloc[-1]:.3f} kg/s" if ox is not None else ""))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def pct(series):
        return 100.0 * (series / series.iloc[0] - 1.0)

    const_flow = const["of_ratio"] * 0 + M0                  # constant-flow run: flow never changes
    panels = [("oxidizer flow", const_flow, coupled["oxidizer_flow_kg_s"]),
              ("chamber pressure", const["chamber_pressure_bar"], coupled["chamber_pressure_bar"]),
              ("thrust", const["thrust_n"], coupled["thrust_n"]),
              ("O/F ratio", const["of_ratio"], coupled["of_ratio"])]
    fig, ax = plt.subplots(2, 2, figsize=(9, 6), dpi=140, sharex=True)
    for a, (name, c_series, k_series) in zip(ax.ravel(), panels):
        a.plot(const["time_s"], pct(c_series), label="constant flow")
        a.plot(coupled["time_s"], pct(k_series), label="injector-coupled")
        a.set_ylabel(f"{name}: change from t=0 [%]")
        a.grid(True, alpha=0.3)
    for a in ax[1]:
        a.set_xlabel("time [s]")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Constant vs injector-coupled oxidizer flow: change from start of burn (illustrative design)")
    fig.tight_layout()
    out = ROOT / "docs" / "injector_coupling.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out)
    print("Saved", out)
except ImportError:
    print("(matplotlib not installed: plot skipped)")
