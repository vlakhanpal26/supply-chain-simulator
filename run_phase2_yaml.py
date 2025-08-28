# run_phase2_yaml.py
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from engine import run_scenario, compare_scenarios, plot_inventory, plot_demand_served, plot_cum_cost

# --- load YAML ---
with open("config.yaml", "r") as f:
    DEFAULTS = yaml.safe_load(f)

with open("scenarios.yaml", "r") as f:
    SCENARIOS = yaml.safe_load(f)

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

all_kpis = []
logs_saved = []

for sc in SCENARIOS:
    name = sc["name"]
    overrides = sc.get("overrides", {})
    df, kpis = run_scenario(name, DEFAULTS, overrides)
    all_kpis.append(kpis)

    # save per-day log
    csv_path = OUT / f"{name}_log.csv"
    df.to_csv(csv_path, index=False)
    logs_saved.append(str(csv_path))

    # plots
    plot_inventory(df, title=f"Inventory vs Day — {name}")
    plot_demand_served(df, title=f"Demand vs Served — {name}")
    plot_cum_cost(df, title=f"Cumulative Cost — {name}")

# KPI summary + comparison
kpi_df = pd.DataFrame(all_kpis)
kpi_df.to_csv(OUT / "kpi_summary.csv", index=False)

comp = compare_scenarios(all_kpis, baseline_name="baseline")
comp.to_csv(OUT / "kpi_compare.csv", index=False)

print("Saved logs:")
for p in logs_saved:
    print(" -", p)
print("Saved:", OUT / "kpi_summary.csv")
print("Saved:", OUT / "kpi_compare.csv")

plt.show()
