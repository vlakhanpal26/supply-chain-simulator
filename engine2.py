from __future__ import annotations
"""Compatibility wrapper for Phase 2 scripts.

Re-exports the unified engine while keeping the historical signature used by
``run_phase2.py`` and notebooks (scenario name first).
"""

from typing import Dict, Optional, Tuple

import pandas as pd

from engine_core import (
    compare_scenarios,
    compute_kpis,
    ensure_dir,
    generate_narrative,
    grid_policy_search,
    run_two_echelon,
    compute_two_echelon_kpis,
    run_network,
    compute_network_kpis,
    run_monte_carlo,
    run_once,
    run_scenario as _run_scenario,
    save_cost_plot,
    save_inventory_plot,
    summarize_kpis,
)


def run_scenario(name: str, base_cfg: Dict, overrides: Optional[Dict]) -> Tuple[pd.DataFrame, Dict]:
    df, kpi = _run_scenario(base_cfg, overrides, scenario_name=name)
    return df, kpi


__all__ = [
    "run_once",
    "run_scenario",
    "compute_kpis",
    "compare_scenarios",
    "ensure_dir",
    "save_inventory_plot",
    "save_cost_plot",
    "run_monte_carlo",
    "summarize_kpis",
    "grid_policy_search",
    "generate_narrative",
    "run_two_echelon",
    "compute_two_echelon_kpis",
    "run_network",
    "compute_network_kpis",
]
