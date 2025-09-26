# run_phase2.py
from __future__ import annotations
import os
import pandas as pd
from engine2 import (
    compare_scenarios,
    ensure_dir,
    generate_narrative,
    grid_policy_search,
    run_monte_carlo,
    run_scenario,
    save_cost_plot,
    save_inventory_plot,
)
from engine import optimize_policy_spsa, counterfactual_policy_search, summarize_kpis

# ---- Baseline configuration ----
BASE_CFG = {
    "N_DAYS": 60,
    "Demand": {"distribution": "normal", "mu": 100, "sigma": 20},
    "LeadTime": {"type": "discrete", "pmf": {"3":0.2, "5":0.6, "7":0.2}},
    "Policy": {"s": 575, "S": 1200},   # from Phase-1 tuning (95% target idea)
    "Initial": {"on_hand": 500, "backorder": 0},
    "Costs": {"holding_cost": 1.0, "stockout_penalty": 10.0},
    "seed": 42
}

# ---- Scenarios ----
SCENARIOS = [
    {"name": "baseline", "overrides": {}},
    {"name": "supplier_delay_plus2", "overrides":
        {"LeadTime": {"type": "discrete", "pmf": {"5":0.4,"7":0.5,"9":0.1}}}},
    {"name": "demand_spike_+15pct_days_8_14", "overrides":
        {"Demand": {"spike": {"start": 8, "end": 14, "mult": 1.15}}}},
    {"name": "raise_policy_sS", "overrides":
        {"Policy": {"s": 700, "S": 1400}}}
]

OUT = "outputs"
ensure_dir(OUT)

all_kpis = []
logs_saved = []
mc_stats = []
MC_RUNS = 300

for idx, sc in enumerate(SCENARIOS):
    name = sc["name"]
    overrides = sc.get("overrides")
    df, kpi = run_scenario(name, BASE_CFG, overrides)
    # save logs
    p_csv = os.path.join(OUT, f"{name}_log.csv")
    df.to_csv(p_csv, index=False)
    logs_saved.append(p_csv)
    # save plots
    save_inventory_plot(df, f"Inventory — {name}", os.path.join(OUT, f"{name}_inventory.png"))
    save_cost_plot(df, f"Cumulative Cost — {name}", os.path.join(OUT, f"{name}_cost.png"))
    # kpi
    all_kpis.append(kpi)

    # Monte Carlo ensemble per scenario
    seed_offset = BASE_CFG.get("seed", 0) + idx * 1000
    runs_df, stats_df = run_monte_carlo(
        BASE_CFG,
        overrides,
        n_runs=MC_RUNS,
        base_seed=seed_offset,
        scenario_name=name,
    )
    runs_df.to_csv(os.path.join(OUT, f"{name}_mc_runs.csv"), index=False)
    stats_path = os.path.join(OUT, f"{name}_mc_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    mc_stats.append(stats_df)

# KPI summary & comparison
kpi_df = pd.DataFrame(all_kpis)
kpi_df.to_csv(os.path.join(OUT, "kpi_summary.csv"), index=False)

comp = compare_scenarios(all_kpis, baseline_name="baseline")
comp.to_csv(os.path.join(OUT, "kpi_compare.csv"), index=False)

if mc_stats:
    mc_summary = pd.concat(mc_stats, ignore_index=True)
    mc_summary.to_csv(os.path.join(OUT, "monte_carlo_summary.csv"), index=False)
    mean_stats = mc_summary[mc_summary["stat"] == "mean"].copy()
    tornado_path = os.path.join(OUT, "monte_carlo_tornado.csv")
    if not mean_stats.empty and "Scenario" in mean_stats.columns and "TotalCost" in mean_stats.columns:
        base_mask = mean_stats["Scenario"].str.lower() == "baseline"
        if base_mask.any():
            base_cost = float(mean_stats[base_mask].iloc[0]["TotalCost"])
            mean_stats["CostDelta"] = mean_stats["TotalCost"] - base_cost
            mean_stats.to_csv(tornado_path, index=False)
        else:
            tornado_path = None
    else:
        tornado_path = None
else:
    mc_summary = None
    tornado_path = None

# Advisor search for improved (s,S)
s0 = BASE_CFG["Policy"]["s"]
S0 = BASE_CFG["Policy"]["S"]
span = 200
step = 50
s_candidates = list(range(max(0, s0 - span), s0 + span + step, step))
S_candidates = list(range(max(s0, S0 - span), S0 + span + step, step))
advisor_df = grid_policy_search(
    BASE_CFG,
    s_candidates,
    S_candidates,
    service_target=0.95,
    objective="total_cost",
    n_runs=150,
    base_seed=BASE_CFG.get("seed", 0),
)
advisor_path = os.path.join(OUT, "advisor_results.csv")
advisor_df.to_csv(advisor_path, index=False)

baseline_kpi = next((k for k in all_kpis if k.get("Scenario") == "baseline"), None)
if baseline_kpi and not advisor_df.empty:
    best = advisor_df.iloc[0].to_dict()
    rec_kpi = {
        "Scenario": "advisor",
        "FR": best.get("FR"),
        "CSL": best.get("CSL"),
        "TotalCost": best.get("TotalCost"),
        "AvgInventory": best.get("AvgInventory"),
        "AvgBackorder": best.get("AvgBackorder"),
        "StockoutDays": best.get("StockoutDays", 0),
    }
    story = generate_narrative(baseline_kpi, rec_kpi, service_target=0.95)
    narrative_path = os.path.join(OUT, "advisor_summary.txt")
    with open(narrative_path, "w", encoding="utf-8") as fh:
        fh.write(f"Recommended policy: s={int(best['s'])}, S={int(best['S'])}\n")
        fh.write(story["headline"] + "\n")
        for bullet in story["details"]:
            fh.write(f"- {bullet}\n")
else:
    narrative_path = None

# SPSA optimiser and counterfactual analysis
spsa_out = optimize_policy_spsa(
    BASE_CFG,
    objective="total_cost",
    service_target=0.95,
    n_runs=40,
    iterations=50,
    seed=BASE_CFG.get("seed", 0),
)
spsa_history = spsa_out.get("history")
advisor_spsa_path = os.path.join(OUT, "advisor_spsa_history.csv")
if isinstance(spsa_history, pd.DataFrame) and not spsa_history.empty:
    spsa_history.to_csv(advisor_spsa_path, index=False)
else:
    advisor_spsa_path = None

counter_out = counterfactual_policy_search(
    BASE_CFG,
    target_fr=0.95,
    step=25,
    max_iter=20,
    n_runs=10,
)
counter_hist = pd.DataFrame(counter_out.get("iterations", []))
counter_path = os.path.join(OUT, "advisor_counterfactual.csv")
if not counter_hist.empty:
    counter_hist.to_csv(counter_path, index=False)
else:
    counter_path = None

print("Saved:")
for p in logs_saved:
    print(" -", p)
print(" -", os.path.join(OUT, "kpi_summary.csv"))
print(" -", os.path.join(OUT, "kpi_compare.csv"))
print(" -", os.path.join(OUT, "baseline_inventory.png"), "(and other PNGs)")
print(" -", advisor_path)
if narrative_path:
    print(" -", narrative_path)
if tornado_path:
    print(" -", tornado_path)
if advisor_spsa_path:
    print(" -", advisor_spsa_path)
if counter_path:
    print(" -", counter_path)
from report import make_pdf
make_pdf(
    os.path.join(OUT, "kpi_compare.csv"),
    os.path.join(OUT, "phase2_report.pdf"),
    OUT,
    risk_csv=os.path.join(OUT, "monte_carlo_summary.csv") if mc_stats else None,
    advisor_txt=narrative_path,
)
