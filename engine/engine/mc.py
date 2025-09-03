# engine/mc.py
from typing import List, Tuple
import numpy as np
import pandas as pd
from .network import run_network_once

def monte_carlo(config: dict, nodes_df: pd.DataFrame, arcs_df: pd.DataFrame,
                items_df: pd.DataFrame, policies_df: pd.DataFrame,
                demand_df: pd.DataFrame | None, R: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs R replications, returns:
      - kpi_runs: DataFrame of KPIs per run
      - summary : aggregated stats (mean/p10/p50/p90 and P(FR>=target))
    """
    rng = np.random.default_rng(int(config.get("seed", 42)))
    kpi_rows = []

    for r in range(R):
        cfg = dict(config)
        cfg["seed"] = int(rng.integers(0, 1_000_000_000))
        df, k = run_network_once(cfg, nodes_df, arcs_df, items_df, policies_df, demand_df)
        kpi_rows.append(dict(k, run=r))

    kpi_runs = pd.DataFrame(kpi_rows)

    def pct(col, p): return float(np.percentile(kpi_runs[col], p))
    target = float(config.get("target_FR", 0.95))
    prob_meet = float((kpi_runs["FR"] >= target).mean())

    summary = pd.DataFrame([{
        "FR_mean": float(kpi_runs["FR"].mean()),
        "FR_p10": pct("FR", 10),
        "FR_p50": pct("FR", 50),
        "FR_p90": pct("FR", 90),
        "TotalCost_mean": float(kpi_runs["TotalCost"].mean()),
        "TotalCost_p10": pct("TotalCost", 10),
        "TotalCost_p50": pct("TotalCost", 50),
        "TotalCost_p90": pct("TotalCost", 90),
        "P_FR_ge_target": prob_meet,
        "target_FR": target,
        "R": R
    }])
    return kpi_runs, summary
