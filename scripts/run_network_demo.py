# scripts/run_network_demo.py
# Quick smoke test for Phase 4: network simulator + Monte Carlo

from pathlib import Path
import pandas as pd

# Import from our engine package
from engine.network import run_network_once
from engine.mc import monte_carlo

# Paths to the example CSVs
ROOT = Path(__file__).resolve().parents[1]
ex = ROOT / "examples" / "network_small"

# Load input tables
nodes   = pd.read_csv(ex / "nodes.csv")
arcs    = pd.read_csv(ex / "arcs.csv")
pol     = pd.read_csv(ex / "policies.csv")
items   = pd.read_csv(ex / "items.csv")
demand  = pd.read_csv(ex / "demand.csv")  # set to None to use random demand

# Base config
config = {
    "N_DAYS": 60,
    "Demand": {"distribution": "normal", "mu": 100, "sigma": 20},
    "seed": 42,
    "target_FR": 0.95
}

# -------- One run (sanity) --------
df, kpis = run_network_once(config, nodes, arcs, items, pol, demand)
print("One run KPIs:", kpis)
print(df.head(), "\n")

# -------- Monte Carlo (uncertainty) --------
kpi_runs, summary = monte_carlo(config, nodes, arcs, items, pol, demand, R=50)
print("First 5 MC KPI rows:\n", kpi_runs.head(), "\n")
print("Monte Carlo summary:\n", summary)
