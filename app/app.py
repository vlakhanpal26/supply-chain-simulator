# app/app.py
# Supply Chain Simulator — Executive Dashboard (Apple-style)

import sys, math, io
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---- allow importing engine.py from repo root ----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from engine import run_scenario, compare_scenarios  # Phase-2 engine (with uniform/empirical/fixed support)

# ---------------- Page + Style ----------------
st.set_page_config(page_title="Supply Chain Simulator", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root{
  --bg:#F5F7FA; --card:#FFFFFF; --ink:#0B1220; --sub:#4B5563; --line:#E5E7EB; --accent:#0A84FF;
}
.stApp { background:var(--bg); }
* { font-family: Inter, -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
.hero{
  background: linear-gradient(180deg, #FFFFFF 0%, #F7F9FC 100%);
  border:1px solid var(--line); border-radius:24px; padding:18px 22px; margin-bottom:12px;
  box-shadow:0 6px 18px rgba(18,38,63,.06);
}
.hero h1{ margin:0; font-weight:700; color:var(--ink); }
.hero p{ margin:.25rem 0 0; color:var(--sub); }
.kpi{
  border:1px solid var(--line); border-radius:18px; background:var(--card);
  padding:14px 16px; box-shadow:0 2px 10px rgba(15,23,42,.04);
}
.kpi .label{ color:var(--sub); font-size:12px; letter-spacing:.02em; }
.kpi .value{ color:var(--ink); font-size:24px; font-weight:700; }
.stButton>button { border-radius:12px; padding:8px 16px; border:1px solid var(--line); }
.stTabs [data-baseweb="tab-list"]{ gap: 6px; }
.stTabs [data-baseweb="tab"]{ background:var(--card); border:1px solid var(--line); padding:10px 14px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers ----------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def kpi_strip(kpis: dict):
    c1, c2, c3, c4 = st.columns(4)
    rows = [
        (c1, "Fill Rate (FR)", f"{kpis['FR']:.3f}"),
        (c2, "Service Level (CSL)", f"{kpis['CSL']:.3f}"),
        (c3, "Avg Inventory", f"{kpis['AvgInventory']:.1f}"),
        (c4, "Total Cost", f"{kpis['TotalCost']:.0f}")
    ]
    for col, label, value in rows:
        with col:
            st.markdown(f"""
            <div class="kpi">
              <div class="label">{label}</div>
              <div class="value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

def plot_inventory(df: pd.DataFrame, title: str, day_range=None, overlay=None):
    data = df.copy()
    if day_range: data = data[(data["Day"]>=day_range[0]) & (data["Day"]<=day_range[1])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Day"], y=data["OnHandEnd"], mode="lines", name="On-hand", line=dict(width=3)))
    # Markers: Orders and Arrivals
    o = data.query("OrderPlaced>0"); a = data.query("Arrivals>0")
    if not o.empty:
        fig.add_trace(go.Scatter(x=o["Day"], y=o["OnHandEnd"], mode="markers", name="Order placed",
                                 marker=dict(symbol="triangle-up", size=10)))
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Day"], y=a["OnHandEnd"], mode="markers", name="Delivery arrived",
                                 marker=dict(symbol="diamond", size=9)))
    # Overlay scenario lines {pretty_name: df}
    if overlay:
        for pretty, df_sc in overlay.items():
            dd = df_sc.copy()
            if day_range: dd = dd[(dd["Day"]>=day_range[0]) & (dd["Day"]<=day_range[1])]
            fig.add_trace(go.Scatter(x=dd["Day"], y=dd["OnHandEnd"], mode="lines",
                                     name=f"{pretty} (On-hand)", line=dict(width=1.8, dash="dot")))
    fig.update_layout(title=title, height=420, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def plot_demand_served(df: pd.DataFrame, title: str, day_range=None):
    data = df.copy()
    if day_range: data = data[(data["Day"]>=day_range[0]) & (data["Day"]<=day_range[1])]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data["Day"], y=data["Demand"], name="Demand", opacity=0.5))
    fig.add_trace(go.Scatter(x=data["Day"], y=data["ServedToday"], mode="lines", name="Served", line=dict(width=3)))
    fig.update_layout(title=title, barmode="overlay", height=380, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def plot_backorder(df: pd.DataFrame, title: str, day_range=None):
    data = df.copy()
    if day_range: data = data[(data["Day"]>=day_range[0]) & (data["Day"]<=day_range[1])]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Day"], y=data["BackorderEnd"], fill="tozeroy", mode="lines",
                             name="Backorder", line=dict(width=2)))
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def plot_cum_cost(df: pd.DataFrame, title: str, day_range=None):
    data = df.copy()
    if day_range: data = data[(data["Day"]>=day_range[0]) & (data["Day"]<=day_range[1])]
    cum_hold = data["HoldingCost"].cumsum()
    cum_oos  = data["StockoutCost"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Day"], y=cum_hold, mode="lines", name="Cum Holding", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=data["Day"], y=cum_oos,  mode="lines", name="Cum Stockout", line=dict(width=3, dash="dot")))
    fig.update_layout(title=title, height=360, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def plot_rolling_fill(df: pd.DataFrame, window: int, title: str, day_range=None):
    data = df.copy()
    if day_range: data = data[(data["Day"]>=day_range[0]) & (data["Day"]<=day_range[1])]
    served = data["ServedToday"].rolling(window).sum()
    demand = data["Demand"].rolling(window).sum().replace(0, np.nan)
    fr = served / demand
    fig = px.line(x=data["Day"], y=fr)
    fig.update_traces(line=dict(width=3), name=f"Fill (w={window})")
    fig.update_yaxes(range=[0,1.05], title="Fill (0–1)")
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---------------- Load YAML ----------------
CONFIG_PATH = ROOT / "config.yaml"
SCENARIO_PATH = ROOT / "scenarios.yaml"
DEFAULTS = yaml.safe_load(open(CONFIG_PATH, "r"))
SCENARIOS = yaml.safe_load(open(SCENARIO_PATH, "r"))

# map pretty display -> internal key, so UI shows formal names
SCEN_DISPLAY = { (sc.get("display") or sc["name"]): sc["name"] for sc in SCENARIOS if sc["name"] != "baseline" }
DISPLAY_LIST = list(SCEN_DISPLAY.keys())

# ---------------- HERO ----------------
st.markdown(
    '<div class="hero"><h1>Supply Chain Simulator — Executive Dashboard</h1>'
    '<p>Adjust inputs, run baseline, stress with scenarios, and read KPIs the board cares about.</p></div>',
    unsafe_allow_html=True
)

# ---------------- Sidebar (friendly labels) ----------------
with st.sidebar:
    st.header("Baseline Inputs")
    # Demand
    d1, d2 = st.columns(2)
    mu = d1.number_input("Demand mean (μ)", 0.0, 100000.0, float(DEFAULTS["Demand"]["mu"]), 1.0)
    sigma = d2.number_input("Demand std (σ)", 0.0, 100000.0, float(DEFAULTS["Demand"]["sigma"]), 1.0)
    dist_label = st.selectbox("Demand distribution", ["Normal", "Poisson", "Uniform", "Empirical (CSV)"],
                              index=0 if DEFAULTS["Demand"]["distribution"]=="normal" else
                                    (1 if DEFAULTS["Demand"]["distribution"]=="poisson" else
                                     2 if DEFAULTS["Demand"].get("distribution")=="uniform" else 0))
    dist = dist_label.split()[0].lower()
    # distribution-specific controls
    uniform_low = uniform_high = None
    emp_values = None
    if dist == "uniform":
        uniform_low  = st.number_input("Uniform low", 0.0, 1e9, float(DEFAULTS["Demand"].get("low", max(0.0, mu - sigma))), 1.0)
        uniform_high = st.number_input("Uniform high", 0.0, 1e9, float(DEFAULTS["Demand"].get("high", mu + sigma)), 1.0)
        if uniform_high < uniform_low: st.warning("Uniform high should be ≥ low.")
    elif dist.startswith("empirical"):
        uploaded = st.file_uploader("Upload CSV with a 'demand' column", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            if "demand" in df_up.columns:
                emp_values = df_up["demand"].dropna().astype(float).clip(lower=0).tolist()
                st.success(f"Loaded {len(emp_values)} demand values.")
            else:
                st.error("CSV must contain a 'demand' column.")
        dist = "empirical"

    # Delivery time (lead time)
    lt_label = st.selectbox("Delivery time model", ["Fixed (same every order)", "Variable (random around a mean)", "From Table (advanced)"],
                            index=0 if DEFAULTS["LeadTime"]["type"]=="fixed" else
                                  (2 if DEFAULTS["LeadTime"]["type"]=="discrete" else 1))
    lt_type = "fixed" if lt_label.startswith("Fixed") else ("discrete" if lt_label.startswith("From Table") else "lognormal")
    lt_fixed_days = int(DEFAULTS["LeadTime"].get("fixed_days", 5))
    lt_var_mean   = float(DEFAULTS["LeadTime"].get("variable_mean_days", 5))
    lt_var_cv     = float(DEFAULTS["LeadTime"].get("variable_cv", 0.3))
    if lt_type == "fixed":
        lt_fixed_days = st.number_input("Fixed delivery time (days)", 1, 365, lt_fixed_days, 1)
    elif lt_type == "lognormal":
        lt_var_mean = st.number_input("Average delivery time (days)", 1.0, 365.0, lt_var_mean, 0.5)
        lt_var_cv   = st.number_input("Variability (CV = std/mean)", 0.0, 2.0, lt_var_cv, 0.05)
        # convert mean+cv -> lognormal params
        sigma_log = math.sqrt(math.log(1 + lt_var_cv*lt_var_cv)) if lt_var_cv > 0 else 1e-6
        mu_log = math.log(lt_var_mean) - 0.5 * sigma_log * sigma_log
    # Policy & costs
    s_val = st.number_input("Reorder point (s)", 0, 100000, int(DEFAULTS["Policy"]["s"]), 10)
    S_val = st.number_input("Order-up-to (S)", 0, 200000, int(DEFAULTS["Policy"]["S"]), 10)
    hold  = st.number_input("Holding cost ($/u/day)", 0.0, 10000.0, float(DEFAULTS["Costs"]["holding_cost"]), 0.1)
    oos   = st.number_input("Stockout penalty ($/u)", 0.0, 10000.0, float(DEFAULTS["Costs"]["stockout_penalty"]), 1.0)
    days  = st.slider("Horizon (days)", 7, 365, int(DEFAULTS["N_DAYS"]), 1)
    seed  = st.number_input("Random seed", 0, 10_000_000, int(DEFAULTS.get("seed", 42)), 1)

    # Suggest s & S
    if st.button("Suggest s & S (95% target)"):
        if lt_type == "fixed":
            lt_mean = lt_fixed_days
        elif lt_type == "lognormal":
            lt_mean = lt_var_mean
        else:
            pmf = DEFAULTS["LeadTime"].get("pmf", {})
            lt_mean = sum(int(k)*float(v) for k,v in pmf.items())
        z = 1.65
        s_suggest = int(round(mu*lt_mean + z*sigma*math.sqrt(lt_mean)))
        S_suggest = s_suggest + 300
        st.info(f"Suggested **s ≈ {s_suggest}**, **S ≈ {S_suggest}** (ensure S ≥ s).")

    st.markdown("---")
    st.subheader("Scenarios")
    # Use pretty labels (no underscores)
    preset_display = st.multiselect("Select scenarios to compare", DISPLAY_LIST)
    run_base = st.button("Run Baseline")
    run_scen = st.button("Run Selected Scenarios")

if S_val < s_val:
    st.error("Order-up-to level **S** must be ≥ **s**.")

# ---------------- Build baseline config ----------------
baseline_cfg = dict(DEFAULTS)
baseline_cfg["N_DAYS"] = int(days); baseline_cfg["seed"] = int(seed)
baseline_cfg["Demand"] = {"distribution": dist, "mu": float(mu), "sigma": float(sigma)}
if dist == "uniform":
    baseline_cfg["Demand"]["low"]  = float(uniform_low)
    baseline_cfg["Demand"]["high"] = float(uniform_high)
elif dist == "empirical":
    baseline_cfg["Demand"]["values"] = emp_values or []

if lt_type == "fixed":
    baseline_cfg["LeadTime"] = {"type": "fixed", "days": int(lt_fixed_days)}
elif lt_type == "lognormal":
    baseline_cfg["LeadTime"] = {"type":"lognormal","mu":mu_log,"sigma":sigma_log,"Lmax":60}
else:
    baseline_cfg["LeadTime"] = dict(DEFAULTS["LeadTime"]); baseline_cfg["LeadTime"]["type"] = "discrete"
baseline_cfg["Policy"] = {"s":int(s_val), "S":int(S_val)}
baseline_cfg["Costs"]  = {"holding_cost":float(hold), "stockout_penalty":float(oos)}
baseline_cfg["Initial"] = dict(DEFAULTS["Initial"])

# ---------------- Main ----------------
df_base, kpi_base = None, None
st.write("")  # small spacing

# Zoom controls for charts
with st.expander("Chart Controls", expanded=False):
    day_zoom = st.slider("Zoom day range", 1, int(days), (1, int(days)), 1)
    roll_w   = st.slider("Rolling Fill window", 3, 30, 7, 1)

# --- Baseline run ---
if run_base:
    df_base, kpi_base = run_scenario("baseline", baseline_cfg, overrides={})
    st.success("Baseline run completed.")
    kpi_strip(kpi_base)

    tab_overview, tab_perf, tab_log = st.tabs(["Overview", "Performance", "Daily Log"])

    with tab_overview:
        left, right = st.columns([2,1])
        with left:
            # ★ Exec-quality titles
            plot_inventory(df_base, "Inventory Position & Events — Baseline", day_range=day_zoom)
            plot_demand_served(df_base, "Customer Demand vs Orders Fulfilled — Baseline", day_range=day_zoom)
        with right:
            plot_cum_cost(df_base, "Cumulative Cost Breakdown — Baseline", day_range=day_zoom)
            plot_backorder(df_base, "Backorder Exposure Over Time", day_range=day_zoom)

    with tab_perf:
        plot_rolling_fill(df_base, roll_w, f"Rolling Fill Rate (window={roll_w} days)", day_range=day_zoom)

    with tab_log:
        st.dataframe(df_base, use_container_width=True, height=420)
        st.download_button("Download Baseline Log (CSV)", data=df_to_csv_bytes(df_base),
                           file_name="baseline_log.csv", mime="text/csv")

# --- Scenarios ---
if run_scen:
    if df_base is None or kpi_base is None:
        df_base, kpi_base = run_scenario("baseline", baseline_cfg, overrides={})

    if not preset_display:
        st.info("Select at least one scenario, then click **Run Selected Scenarios**.")
    else:
        all_kpis = [dict(kpi_base, Scenario="baseline")]
        overlays = {}  # for inventory overlay plot
        st.markdown("### Scenario Runs")

        for disp in preset_display:
            name = SCEN_DISPLAY[disp]   # internal key
            sc = next(sc for sc in SCENARIOS if sc["name"] == name)
            df_sc, kpi_sc = run_scenario(name, baseline_cfg, sc.get("overrides", {}))
            overlays[disp] = df_sc      # keep pretty label in legend

            # mini KPI strip
            c1, c2, c3, c4 = st.columns(4)
            for col, label, val in [
                (c1, "Fill Rate", f"{kpi_sc['FR']:.3f}"),
                (c2, "CSL", f"{kpi_sc['CSL']:.3f}"),
                (c3, "Avg Inv", f"{kpi_sc['AvgInventory']:.1f}"),
                (c4, "Total Cost", f"{kpi_sc['TotalCost']:.0f}")
            ]:
                with col:
                    st.markdown(f"""
                    <div class="kpi">
                      <div class="label">{label}</div>
                      <div class="value">{val}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ★ Exec-quality titles for scenario charts
            with st.expander(f"Charts — {disp}", expanded=False):
                plot_inventory(df_sc, f"Inventory Position & Events — {disp}", day_range=day_zoom)
                plot_demand_served(df_sc, f"Customer Demand vs Orders Fulfilled — {disp}", day_range=day_zoom)
                plot_cum_cost(df_sc, f"Cumulative Cost Breakdown — {disp}", day_range=day_zoom)

            st.download_button(f"Download Log (CSV) — {disp}",
                               data=df_to_csv_bytes(df_sc),
                               file_name=f"{name}_log.csv", mime="text/csv")

            all_kpis.append(dict(kpi_sc, Scenario=disp))  # use pretty label in compare table

        # Overlay inventory comparison (pretty labels in legend)
        st.subheader("Inventory Overlay — Baseline vs Scenarios")
        plot_inventory(df_base, "Inventory Overlay — Baseline vs Scenarios", day_range=day_zoom, overlay=overlays)

        # KPI comparison table (clean, rounded)
        comp = compare_scenarios(all_kpis, baseline_name="baseline").copy()
        keep = ["Scenario","FR","CSL","AvgInventory","TotalCost","StockoutDays","OrdersPlacedCount",
                "FR_delta","CSL_delta","TotalCost_delta","StockoutDays_delta"]
        comp = comp[keep]
        for c in comp.columns:
            if c != "Scenario": comp[c] = pd.to_numeric(comp[c]).round(3)
        st.subheader("Scenario Comparison vs Baseline")
        st.dataframe(comp, use_container_width=True)
        st.download_button("Download KPI Comparison (CSV)", data=df_to_csv_bytes(comp),
                           file_name="kpi_compare.csv", mime="text/csv")

st.caption("Tip: Use **Suggest s & S** for a strong starting policy. Raise s/S to boost service (watch holding cost). Zoom/roll sliders refine the view.")
