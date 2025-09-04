# app/pages/02_Network_Simulator.py
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.network import run_network_once
from engine.mc import monte_carlo

st.set_page_config(page_title="Network Simulator", layout="wide")

st.title("Network Simulator (Multi-Echelon)")

# --- pick folder with CSVs (defaults to examples/network_small)
default_dir = ROOT / "examples" / "network_small"
csv_dir = st.text_input("Folder with network CSVs", str(default_dir))

def load_tables(folder: Path):
    folder = Path(folder)
    nodes = pd.read_csv(folder/"nodes.csv")
    arcs = pd.read_csv(folder/"arcs.csv")
    policies = pd.read_csv(folder/"policies.csv")
    items = pd.read_csv(folder/"items.csv")
    demand = pd.read_csv(folder/"demand.csv")
    return nodes, arcs, policies, items, demand

nodes, arcs, policies, items, demand = load_tables(csv_dir)

col1, col2, col3, col4 = st.columns(4)
with col1:
    days = st.number_input("Horizon (days)", 7, 365, 60, 1)
with col2:
    mu = st.number_input("Demand mean (μ) for random fill", 0.0, 1e5, 100.0, 1.0)
with col3:
    sigma = st.number_input("Demand std (σ) for random fill", 0.0, 1e5, 20.0, 1.0)
with col4:
    seed = st.number_input("Random seed", 0, 10_000_000, 42, 1)

config = {
    "N_DAYS": int(days),
    "Demand": {"distribution": "normal", "mu": float(mu), "sigma": float(sigma)},
    "seed": int(seed),
    "target_FR": 0.95,
}

st.markdown("---")
c1, c2 = st.columns(2)
run_once_btn = c1.button("Run single simulation")
run_mc_btn = c2.button("Run Monte Carlo (R=200)")

if run_once_btn:
    df, k = run_network_once(config, nodes, arcs, items, policies, demand)
    st.success("Single run complete.")
    kcols = st.columns(6)
    kshow = [
        ("FR", k["FR"]), ("CSL", k["CSL"]), ("AvgInv", k["AvgInventory"]),
        ("TotalCost", k["TotalCost"]), ("StockoutDays", k["StockoutDays"]),
        ("Orders", k["OrdersPlacedCount"])
    ]
    for (label, val), col in zip(kshow, kcols):
        col.metric(label, f"{val:.3f}" if isinstance(val, float) else f"{val}")

    with st.expander("Daily log", expanded=False):
        st.dataframe(df, use_container_width=True, height=420)
        st.download_button("Download log (CSV)", df.to_csv(index=False), "network_log.csv", "text/csv")

    fig_inv = px.line(df, x="Day", y="OnHandEnd", title="Inventory at DC (End-of-Day)")
    st.plotly_chart(fig_inv, use_container_width=True)
    fig_dem = px.bar(df, x="Day", y="Demand", title="Customer Demand (C1)")
    st.plotly_chart(fig_dem, use_container_width=True)

if run_mc_btn:
    with st.spinner("Running 200 replications…"):
        runs, summary = monte_carlo(config, nodes, arcs, items, policies, demand, R=200)
    st.success("Monte Carlo complete.")
    st.dataframe(summary, use_container_width=True)

    fig_fr = px.histogram(runs, x="FR", nbins=30, title="Distribution of Fill Rate")
    st.plotly_chart(fig_fr, use_container_width=True)
    fig_cost = px.histogram(runs, x="TotalCost", nbins=30, title="Distribution of Total Cost")
    st.plotly_chart(fig_cost, use_container_width=True)
