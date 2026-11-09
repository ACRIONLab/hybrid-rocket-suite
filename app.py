"""Interactive Streamlit dashboard for the hybrid rocket simulator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.integrate import trapezoid

PROJECT_ROOT = Path(__file__).resolve().parents[0]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.engine import HybridRocketSimulator
from optimization.altitude_sweep import (
    DEFAULT_OUTPUT,
    isa_pressure,
    optimize_altitude_design,
)


def _safe_float(
    value: object,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """Return a finite, range-limited float for widget and URL state values."""
    try:
        scalar = np.asarray(value, dtype=float)
        if scalar.size != 1:
            raise ValueError("expected scalar input")
        scalar_value = scalar.reshape(-1)[0]
    except (TypeError, ValueError, OverflowError):
        scalar_value = default
    finite = float(
        np.nan_to_num(
            scalar_value,
            nan=default,
            posinf=default,
            neginf=default,
        )
    )
    return float(np.clip(finite, minimum, maximum))


INPUT_SPECS = {
    "oxidizer_flow_rate": {
        "widget_key": "oxidizer_flow_slider", "default": 1.50,
        "minimum": 0.20, "maximum": 5.00,
    },
    "port_diameter_mm": {
        "widget_key": "port_diameter_slider", "default": 40.0,
        "minimum": 15.0, "maximum": 95.0,
    },
    "grain_length": {
        "widget_key": "grain_len_slider", "default": 0.50,
        "minimum": 0.10, "maximum": 1.50,
    },
    "burn_duration": {
        "widget_key": "burn_dur_slider", "default": 5.0,
        "minimum": 0.50, "maximum": 20.0,
    },
}


def _sanitized_inputs(source: dict[str, object]) -> dict[str, float]:
    """Normalize a manifold state mapping to finite floats."""
    if not isinstance(source, dict):
        source = {}
    return {
        name: _safe_float(
            source.get(name, spec["default"]),
            spec["default"], spec["minimum"], spec["maximum"],
        )
        for name, spec in INPUT_SPECS.items()
    }


def _apply_input_manifold() -> None:
    """Commit form widget values as the active simulation manifold."""
    pending = {
        name: st.session_state.get(spec["widget_key"], spec["default"])
        for name, spec in INPUT_SPECS.items()
    }
    st.session_state["applied_inputs"] = _sanitized_inputs(pending)


def _reset_input_manifold() -> None:
    """Reset both pending form controls and active inputs to baseline."""
    defaults = {
        name: float(spec["default"]) for name, spec in INPUT_SPECS.items()
    }
    for name, spec in INPUT_SPECS.items():
        st.session_state[spec["widget_key"]] = defaults[name]
    st.session_state["applied_inputs"] = defaults


@st.cache_data(show_spinner=False)
def _simulate_cached(
    oxidizer_flow_rate: float,
    port_diameter_mm: float,
    grain_length: float,
    burn_duration: float,
    throat_diameter: float = 0.025,
    expansion_ratio: float = 4.0,
    ambient_pressure: float = 101_325.0,
):
    """Cache time histories until applied inputs or nozzle settings change."""
    return HybridRocketSimulator(
        oxidizer_flow_rate=float(oxidizer_flow_rate),
        initial_port_diameter=float(port_diameter_mm) / 1_000.0,
        grain_length=float(grain_length),
        burn_time=float(burn_duration),
        throat_diameter=float(throat_diameter),
        expansion_ratio=float(expansion_ratio),
        ambient_pressure=float(ambient_pressure),
    ).simulate()


st.set_page_config(
    page_title="Hybrid Rocket // Test Console",
    page_icon="HR",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Rajdhani:wght@500;600;700&display=swap');
    :root { --ink:#e8e5dc; --muted:#91968f; --orange:#ff6b2c; --cyan:#49d6cf; --panel:#111714; }
    .stApp {
        background:
          linear-gradient(rgba(73,214,207,.025) 1px, transparent 1px),
          linear-gradient(90deg, rgba(73,214,207,.025) 1px, transparent 1px),
          radial-gradient(circle at 78% 7%, rgba(255,107,44,.12), transparent 25%), #080c0a;
        background-size: 32px 32px, 32px 32px, auto;
        color: var(--ink);
    }
    html, body, [class*="css"] { font-family: "IBM Plex Mono", monospace; }
    h1, h2, h3 { font-family: "Rajdhani", sans-serif !important; letter-spacing:.04em; }
    h1 { font-size: clamp(2.6rem, 5vw, 5rem) !important; line-height:.85 !important; text-transform:uppercase; }
    [data-testid="stSidebar"] { background:#0c110f; border-right:1px solid #27312c; }
    [data-testid="stMetric"] {
        background:linear-gradient(145deg, rgba(20,28,24,.96), rgba(11,16,13,.96));
        border:1px solid #2b3731; border-top:2px solid var(--orange); padding:1rem 1.1rem;
        box-shadow:0 12px 30px rgba(0,0,0,.22);
    }
    [data-testid="stMetricLabel"] { color:var(--muted); text-transform:uppercase; letter-spacing:.09em; }
    [data-testid="stMetricValue"] { font-family:"Rajdhani", sans-serif; }
    .eyebrow { color:var(--orange); font-weight:600; letter-spacing:.18em; text-transform:uppercase; }
    .deck { max-width:740px; color:#aeb5af; font-size:.98rem; line-height:1.7; }
    .status { display:inline-flex; align-items:center; gap:.55rem; border:1px solid #285149;
      color:var(--cyan); padding:.35rem .65rem; font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; }
    .status:before { content:""; width:7px; height:7px; border-radius:50%; background:var(--cyan); box-shadow:0 0 10px var(--cyan); }
    .rule { height:1px; background:linear-gradient(90deg,var(--orange),#333c37 40%,transparent); margin:1rem 0 1.7rem; }
    .section-tag { color:var(--cyan); font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; margin-top:1rem; }
    .note { border-left:2px solid var(--orange); padding:.7rem 1rem; color:#9ca39d; background:rgba(255,107,44,.04); font-size:.78rem; }
    div[data-testid="stPlotlyChart"] { border:1px solid #222d27; background:rgba(10,15,12,.72); }
    .stSlider [data-baseweb="slider"] > div > div { background-color:var(--orange) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


if "applied_inputs" not in st.session_state:
    st.session_state["applied_inputs"] = _sanitized_inputs({})
else:
    st.session_state["applied_inputs"] = _sanitized_inputs(
        st.session_state["applied_inputs"]
    )
for input_name, input_spec in INPUT_SPECS.items():
    widget_key = input_spec["widget_key"]
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state["applied_inputs"][input_name]


with st.sidebar:
    st.markdown("### INPUT MANIFOLD")
    st.caption("SI UNITS · SINGLE CYLINDRICAL PORT")
    with st.form(key="input_manifold_form"):
        st.slider(
            "Oxidizer flow rate · kg/s", 0.20, 5.00, step=0.05,
            key="oxidizer_flow_slider",
        )
        st.slider(
            "Initial port diameter · mm", 15.0, 95.0, step=1.0,
            key="port_diameter_slider",
        )
        st.slider(
            "Grain length · m", 0.10, 1.50, step=0.05,
            key="grain_len_slider",
        )
        st.slider(
            "Commanded burn time · s", 0.50, 20.00, step=0.25,
            key="burn_dur_slider",
        )
        apply_col, default_col = st.columns(2)
        apply_col.form_submit_button(
            "Set Changes", type="primary", width="stretch",
            on_click=_apply_input_manifold,
        )
        default_col.form_submit_button(
            "Set Default", type="secondary", width="stretch",
            on_click=_reset_input_manifold,
        )
        st.caption("Plots update only after the manifold is applied.")
    st.divider()
    with st.expander("Model constants"):
        st.markdown(
            """
            **Fuel:** HTPB-equivalent  
            **Density:** 930 kg/m³  
            **Marxman:** a = 6.30×10⁻⁵, n = 0.62  
            **Grain OD:** 120 mm  
            **Throat:** 25 mm  
            **Expansion ratio:** 4.0  
            **c-star:** 1550 m/s
            """
        )
    st.markdown(
        '<div class="note">Conceptual quasi-steady model. Validate material, combustion, thermal, and structural assumptions before hardware use.</div>',
        unsafe_allow_html=True,
    )


active_inputs = _sanitized_inputs(st.session_state["applied_inputs"])
oxidizer_flow = float(active_inputs["oxidizer_flow_rate"])
port_diameter_mm = float(active_inputs["port_diameter_mm"])
grain_length = float(active_inputs["grain_length"])
burn_duration = float(active_inputs["burn_duration"])

results = _simulate_cached(
    oxidizer_flow, port_diameter_mm, grain_length, burn_duration
)
simulator_time_step = 0.02

average_thrust = results["thrust_n"].mean()
peak_pressure = results["chamber_pressure_bar"].max()
total_impulse = trapezoid(results["thrust_n"], results["time_s"])
average_of = results["of_ratio"].mean()
burnout_early = (
    results["time_s"].iloc[-1] < burn_duration - simulator_time_step
)

st.markdown('<div class="eyebrow">Propulsion development unit · HR-01</div>', unsafe_allow_html=True)
title_col, status_col = st.columns([4, 1])
with title_col:
    st.title("FIRELINE / SIMULATOR")
    st.markdown(
        '<div class="deck">A live one-dimensional internal-ballistics console coupling Marxman port regression, mixture evolution, choked chamber pressure, and nozzle thrust.</div>',
        unsafe_allow_html=True,
    )
with status_col:
    st.markdown('<div class="status">Simulation nominal</div>', unsafe_allow_html=True)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

if burnout_early:
    st.warning("Fuel web is fully regressed before the commanded burn ends; plots stop at web burnout.")

metric_cols = st.columns(4)
metric_cols[0].metric("Mean thrust", f"{average_thrust / 1_000:.2f} kN")
metric_cols[1].metric("Peak chamber P", f"{peak_pressure:.1f} bar")
metric_cols[2].metric("Total impulse", f"{total_impulse / 1_000:.1f} kN·s")
metric_cols[3].metric("Mean O/F", f"{average_of:.2f}")

overview_tab, optimization_tab = st.tabs(
    ["LIVE BURN ANALYSIS", "ALTITUDE & NOZZLE OPTIMIZATION"]
)

overview_tab.markdown('<div class="section-tag">01 // Combustion timeline</div>', unsafe_allow_html=True)

chart_bg = "rgba(0,0,0,0)"
grid = "rgba(117,132,123,.15)"
text = "#aab2ac"

primary = make_subplots(specs=[[{"secondary_y": True}]])
primary.add_trace(
    go.Scatter(
        x=results["time_s"], y=results["thrust_n"] / 1_000,
        name="Thrust", line=dict(color="#ff6b2c", width=3),
        fill="tozeroy", fillcolor="rgba(255,107,44,.09)",
        hovertemplate="%{x:.2f} s · %{y:.2f} kN<extra>Thrust</extra>",
    ),
    secondary_y=False,
)
primary.add_trace(
    go.Scatter(
        x=results["time_s"], y=results["chamber_pressure_bar"],
        name="Chamber pressure", line=dict(color="#49d6cf", width=2, dash="dot"),
        hovertemplate="%{x:.2f} s · %{y:.1f} bar<extra>Pressure</extra>",
    ),
    secondary_y=True,
)
primary.update_yaxes(title_text="THRUST · kN", secondary_y=False, gridcolor=grid, rangemode="tozero")
primary.update_yaxes(title_text="CHAMBER PRESSURE · bar", secondary_y=True, gridcolor="rgba(0,0,0,0)", rangemode="tozero")
primary.update_xaxes(title_text="MISSION ELAPSED TIME · s", gridcolor=grid)
primary.update_layout(
    height=430, margin=dict(l=30, r=30, t=55, b=30), paper_bgcolor=chart_bg,
    plot_bgcolor=chart_bg, font=dict(family="IBM Plex Mono", color=text, size=11),
    legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified",
)
overview_tab.plotly_chart(primary, width="stretch", config={"displayModeBar": False})

overview_tab.markdown('<div class="section-tag">02 // Grain & mixture state</div>', unsafe_allow_html=True)
left, right = overview_tab.columns(2)

mixture = make_subplots(specs=[[{"secondary_y": True}]])
mixture.add_trace(
    go.Scatter(x=results["time_s"], y=results["of_ratio"], name="O/F",
               line=dict(color="#e8e5dc", width=2.5)), secondary_y=False,
)
mixture.add_trace(
    go.Scatter(x=results["time_s"], y=results["regression_rate_mm_s"], name="Regression",
               line=dict(color="#ff6b2c", width=2)), secondary_y=True,
)
mixture.update_yaxes(title_text="O/F RATIO", secondary_y=False, gridcolor=grid)
mixture.update_yaxes(title_text="REGRESSION · mm/s", secondary_y=True, gridcolor="rgba(0,0,0,0)")
mixture.update_xaxes(title_text="TIME · s", gridcolor=grid)
mixture.update_layout(title="MIXTURE EVOLUTION", height=350, paper_bgcolor=chart_bg,
                      plot_bgcolor=chart_bg, font=dict(family="IBM Plex Mono", color=text, size=10),
                      margin=dict(l=20, r=20, t=60, b=25), legend=dict(orientation="h", y=1.13))
left.plotly_chart(mixture, width="stretch", config={"displayModeBar": False})

geometry = make_subplots(specs=[[{"secondary_y": True}]])
geometry.add_trace(
    go.Scatter(x=results["time_s"], y=results["port_diameter_mm"], name="Port Ø",
               line=dict(color="#49d6cf", width=2.5), fill="tozeroy", fillcolor="rgba(73,214,207,.06)"),
    secondary_y=False,
)
geometry.add_trace(
    go.Scatter(x=results["time_s"], y=results["oxidizer_flux_kg_m2_s"], name="Gox",
               line=dict(color="#d2a84a", width=2)), secondary_y=True,
)
geometry.update_yaxes(title_text="PORT Ø · mm", secondary_y=False, gridcolor=grid)
geometry.update_yaxes(title_text="GOX · kg/m²/s", secondary_y=True, gridcolor="rgba(0,0,0,0)")
geometry.update_xaxes(title_text="TIME · s", gridcolor=grid)
geometry.update_layout(title="PORT GEOMETRY", height=350, paper_bgcolor=chart_bg,
                       plot_bgcolor=chart_bg, font=dict(family="IBM Plex Mono", color=text, size=10),
                       margin=dict(l=20, r=20, t=60, b=25), legend=dict(orientation="h", y=1.13))
right.plotly_chart(geometry, width="stretch", config={"displayModeBar": False})

with overview_tab.expander("Open numerical telemetry"):
    st.dataframe(
        results.rename(
            columns={
                "time_s": "Time (s)", "port_diameter_mm": "Port Ø (mm)",
                "oxidizer_flux_kg_m2_s": "Gox (kg/m²/s)",
                "regression_rate_mm_s": "ṙ (mm/s)", "fuel_flow_kg_s": "Fuel flow (kg/s)",
                "of_ratio": "O/F", "chamber_pressure_bar": "Pc (bar)",
                "thrust_n": "Thrust (N)", "specific_impulse_s": "Isp (s)",
            }
        ),
        width="stretch",
        hide_index=True,
    )


with optimization_tab:
    st.markdown('<div class="section-tag">03 // Robust ascent design search</div>', unsafe_allow_html=True)
    st.subheader("ALTITUDE & NOZZLE OPTIMIZATION")
    st.markdown(
        '<div class="deck">Differential evolution searches grain length, throat area, and bell expansion ratio against a pressure sweep from sea level to 30 km. The robust score balances mean impulse with worst-altitude impulse.</div>',
        unsafe_allow_html=True,
    )

    setup_a, setup_b, setup_c = st.columns([1, 1, 1.2])
    altitude_nodes = setup_a.slider(
        "Altitude pressure nodes", 4, 10, 7, 1, key="optimization_altitude_nodes"
    )
    optimizer_iterations = setup_b.slider(
        "Optimizer generations", 6, 24, 12, 2, key="optimization_generations"
    )
    run_optimizer = setup_c.button(
        "REFRESH COMPARISON", type="primary", width="stretch",
        key="refresh_optimization",
    )

    if run_optimizer:
        with st.spinner("Evolving grain and nozzle candidates across the ascent envelope…"):
            st.session_state["optimization_result"] = optimize_altitude_design(
                oxidizer_flow_rate=oxidizer_flow,
                initial_port_diameter=port_diameter_mm / 1_000.0,
                initial_grain_length=grain_length,
                burn_time=burn_duration,
                altitudes_m=np.linspace(0.0, 30_000.0, altitude_nodes),
                max_iterations=optimizer_iterations,
                population_size=6,
                output_path=DEFAULT_OUTPUT,
            )
            st.session_state["optimization_input_signature"] = tuple(
                active_inputs[name] for name in INPUT_SPECS
            )

    optimization_result = st.session_state.get("optimization_result")
    if optimization_result is None and DEFAULT_OUTPUT.exists():
        try:
            optimization_result = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            optimization_result = None

    if optimization_result is None:
        st.info("Run the optimizer to generate a robust design and comparative plots.")
    else:
        current_signature = tuple(active_inputs[name] for name in INPUT_SPECS)
        optimized_signature = st.session_state.get("optimization_input_signature")
        if optimized_signature is not None and optimized_signature != current_signature:
            st.info(
                "The baseline reflects the applied manifold. Select "
                "REFRESH COMPARISON to optimize the nozzle for these inputs."
            )
        design = optimization_result["optimized_design"]
        objective = optimization_result["objective"]
        opt_metrics = st.columns(4)
        opt_metrics[0].metric("Optimized grain", f"{design['grain_length_m']:.3f} m")
        opt_metrics[1].metric("Throat area", f"{design['throat_area_m2'] * 1e6:.0f} mm²")
        opt_metrics[2].metric("Expansion ratio", f"{design['expansion_ratio']:.2f}")
        opt_metrics[3].metric(
            "Robust impulse", f"{objective['robust_total_impulse_ns'] / 1_000:.2f} kN·s"
        )

        comparison_altitudes = [0.0, 15_000.0, 30_000.0]
        colors = ["#ff6b2c", "#d2a84a", "#49d6cf"]
        comparison = go.Figure()
        optimized_throat_diameter = design["throat_diameter_mm"] / 1_000.0

        for altitude, color in zip(comparison_altitudes, colors):
            pressure = isa_pressure(altitude)
            baseline_history = _simulate_cached(
                oxidizer_flow, port_diameter_mm, grain_length, burn_duration,
                ambient_pressure=pressure,
            )
            optimized_history = _simulate_cached(
                oxidizer_flow, port_diameter_mm, design["grain_length_m"],
                burn_duration, optimized_throat_diameter,
                design["expansion_ratio"], pressure,
            )
            altitude_label = f"{altitude / 1_000:.0f} km"
            comparison.add_trace(
                go.Scatter(
                    x=baseline_history["time_s"],
                    y=baseline_history["thrust_n"] / 1_000.0,
                    name=f"Baseline · {altitude_label}",
                    line=dict(color=color, width=1.4, dash="dot"),
                    legendgroup=altitude_label,
                )
            )
            comparison.add_trace(
                go.Scatter(
                    x=optimized_history["time_s"],
                    y=optimized_history["thrust_n"] / 1_000.0,
                    name=f"Optimized · {altitude_label}",
                    line=dict(color=color, width=3),
                    legendgroup=altitude_label,
                )
            )

        comparison.update_layout(
            title=dict(
                text="BASELINE VS OPTIMIZED THRUST ENVELOPE",
                x=0, xanchor="left", y=0.98, yanchor="top",
            ),
            height=500,
            paper_bgcolor=chart_bg,
            plot_bgcolor=chart_bg,
            font=dict(family="IBM Plex Mono", color=text, size=10),
            margin=dict(t=135, b=50, l=50, r=50),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0,
            ),
            hovermode="x unified",
            xaxis=dict(title="TIME · s", gridcolor=grid),
            yaxis=dict(title="THRUST · kN", gridcolor=grid),
        )
        st.plotly_chart(
            comparison, width="stretch", config={"displayModeBar": False}
        )

        plot_left, plot_right = st.columns([1.2, 1])
        sweep = optimization_result["performance_by_altitude"]
        impulse_figure = go.Figure(
            go.Scatter(
                x=[row["altitude_m"] / 1_000.0 for row in sweep],
                y=[row["total_impulse_ns"] / 1_000.0 for row in sweep],
                mode="lines+markers",
                line=dict(color="#49d6cf", width=3),
                marker=dict(color="#080c0a", line=dict(color="#49d6cf", width=2), size=8),
                fill="tozeroy",
                fillcolor="rgba(73,214,207,.07)",
                hovertemplate="%{x:.0f} km · %{y:.2f} kN·s<extra></extra>",
            )
        )
        impulse_figure.update_layout(
            title="OPTIMIZED IMPULSE BY ALTITUDE",
            height=390, paper_bgcolor=chart_bg, plot_bgcolor=chart_bg,
            font=dict(family="IBM Plex Mono", color=text, size=10),
            margin=dict(l=25, r=25, t=60, b=30),
            xaxis=dict(title="ALTITUDE · km", gridcolor=grid),
            yaxis=dict(title="TOTAL IMPULSE · kN·s", gridcolor=grid),
        )
        plot_left.plotly_chart(
            impulse_figure, width="stretch", config={"displayModeBar": False}
        )

        nozzle = optimization_result["bell_nozzle_geometry"]
        nozzle_x = np.asarray(nozzle["x_m"]) * 1_000.0
        nozzle_r = np.asarray(nozzle["radius_m"]) * 1_000.0
        contour = go.Figure()
        contour.add_trace(
            go.Scatter(
                x=np.concatenate((nozzle_x, nozzle_x[::-1])),
                y=np.concatenate((nozzle_r, -nozzle_r[::-1])),
                fill="toself", fillcolor="rgba(255,107,44,.12)",
                line=dict(color="#ff6b2c", width=2.5), name="Bell wall",
                hovertemplate="x %{x:.1f} mm · r %{y:.1f} mm<extra></extra>",
            )
        )
        contour.add_hline(y=0, line_color="rgba(232,229,220,.25)", line_dash="dot")
        contour.update_layout(
            title="OPTIMIZED 80% BELL CONTOUR",
            height=390, paper_bgcolor=chart_bg, plot_bgcolor=chart_bg,
            font=dict(family="IBM Plex Mono", color=text, size=10),
            margin=dict(l=25, r=25, t=60, b=30), showlegend=False,
            xaxis=dict(title="AXIAL STATION · mm", gridcolor=grid),
            yaxis=dict(title="RADIUS · mm", gridcolor=grid, scaleanchor="x", scaleratio=1),
        )
        plot_right.plotly_chart(
            contour, width="stretch", config={"displayModeBar": False}
        )

        with st.expander("Open optimized design record"):
            st.json(optimization_result, expanded=False)
