"""Compare FIRELINE's fuel-mass prediction with six published N2O/HDPE firings.

Data: Kamps, Sakurai, Saito and Nagata, "Comprehensive Data Reduction for N2O/HDPE
Hybrid Rocket Motor Performance Evaluation", Aerospace 6(4), 45 (2019),
doi:10.3390/aerospace6040045 (CC BY 4.0). Test data from Table 1, geometry from
Figure 5c, regression constants from Figure 9 / Section 3.2.1 of the paper.

Run from the repository root:
    python validation/validate_kamps2019.py
Writes docs/validation_fuel_mass.png if matplotlib is installed.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.engine import HybridRocketSimulator  # noqa: E402

RHO = 955.0           # polyethylene density [kg/m3] (paper)
OD = 0.0467           # fuel grain outer diameter [m], read from Fig. 5c
THROAT, EPS = 0.004, 4.0
# (label, nominal burn time [s], time-averaged oxidizer flow [g/s], fuel mass used [g]) - Table 1
TESTS = [("3s", 3, 41, 30), ("4s", 4, 41, 35), ("10s-I", 10, 39, 74),
         ("10s-II", 10, 38, 73), ("10s-III", 10, 39, 74), ("15s", 15, 37, 110)]
# Stepped port from Fig. 5c: (port diameter [m], length [m]). Each section is run as its own
# single-port grain with the full oxidizer flow, which is what an oxidizer-flux-only law implies.
SECTIONS = [(0.020, 0.072), (0.030, 0.092)]          # 92 mm includes the 10 mm head pocket
SECTIONS_NO_POCKET = [(0.020, 0.072), (0.030, 0.082)]


def fuel_mass_g(a, n, burn, mox_gs, sections):
    total = 0.0
    for d, length in sections:
        sim = HybridRocketSimulator(
            oxidizer_flow_rate=mox_gs / 1000, initial_port_diameter=d, grain_length=length,
            burn_time=float(burn), outer_grain_diameter=OD, fuel_density=RHO,
            regression_coefficient=a, regression_exponent=n,
            throat_diameter=THROAT, expansion_ratio=EPS)
        df = sim.simulate()
        total += trapezoid(df["fuel_flow_kg_s"], df["time_s"])
    return total * 1000.0


def table(a, n, sections=SECTIONS, idx=None):
    rows = []
    for i, (lab, tb, mox, meas) in enumerate(TESTS):
        if idx is not None and i not in idx:
            continue
        pred = fuel_mass_g(a, n, tb, mox, sections)
        rows.append((lab, tb, mox, meas, pred, 100.0 * (pred - meas) / meas))
    return rows


def show(title, rows):
    print(f"\n{title}")
    print(f"{'test':8s}{'burn s':>7s}{'mox g/s':>9s}{'measured g':>12s}{'model g':>10s}{'error %':>9s}")
    for lab, tb, mox, meas, pred, e in rows:
        print(f"{lab:8s}{tb:7d}{mox:9d}{meas:12d}{pred:10.1f}{e:9.1f}")
    errs = [r[5] for r in rows]
    print(f"mean |error| = {np.mean(np.abs(errs)):.1f} %   max |error| = {np.max(np.abs(errs)):.1f} %")


if __name__ == "__main__":
    vol = np.pi / 4 * (OD**2 * 0.164 - sum(d**2 * L for d, L in SECTIONS))
    print(f"geometry check: initial fuel mass = {vol * RHO * 1000:.0f} g")

    runs = {
        "A) paper endpoint correlation (a=4.36e-5, n=0.62)": table(4.36e-5, 0.62),
        "B) paper history correlation (a=2.52e-5, n=0.76)": table(2.52e-5, 0.76),
        "C) FIRELINE default constants (a=6.30e-5, n=0.62)": table(6.30e-5, 0.62),
    }
    for title, rows in runs.items():
        show(title, rows)
    show("A') as A, 10 mm head pocket removed", table(4.36e-5, 0.62, SECTIONS_NO_POCKET))

    # Hold-out test: fit one constant (n = 0.62) on 3 s, 4 s, 15 s; predict the three 10 s tests.
    cal, hold = [0, 1, 5], [2, 3, 4]
    fit = minimize_scalar(
        lambda a: float(sum(r[5] ** 2 for r in table(a, 0.62, idx=cal))),
        bounds=(1e-5, 1.2e-4), method="bounded")
    print(f"\nD) hold-out: a fitted on 3 s, 4 s, 15 s (n = 0.62): a = {fit.x:.3e}")
    show("   calibration tests", table(fit.x, 0.62, idx=cal))
    held = table(fit.x, 0.62, idx=hold)
    show("   held-out predictions (10 s tests)", held)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.6), dpi=150)
        meas = np.array([t[3] for t in TESTS])
        styles = [("A) paper endpoint a, n", runs[list(runs)[0]], "o"),
                  ("B) paper history a, n", runs[list(runs)[1]], "s"),
                  ("C) FIRELINE defaults", runs[list(runs)[2]], "^")]
        for label, rows, mk in styles:
            ax.scatter(meas, [r[4] for r in rows], marker=mk, label=label)
        ax.plot([20, 125], [20, 125], "k--", lw=1, label="perfect agreement")
        ax.set_xlabel("Measured fuel mass used [g]  (Kamps et al. 2019, Table 1)")
        ax.set_ylabel("FIRELINE predicted fuel mass [g]")
        ax.set_title("Fuel mass consumed: model vs six N2O/HDPE firings")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = ROOT / "docs" / "validation_fuel_mass.png"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out)
        print(f"\nSaved {out}")
    except ImportError:
        print("\n(matplotlib not installed: plot skipped)")
