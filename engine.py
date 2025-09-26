from __future__ import annotations
"""Unified simulation engine wrapper (Phase 1 – core consolidation).

This module re-exports the core API from ``engine_core`` while keeping
backwards-compatible helper signatures used across the codebase.
"""

from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

from engine_core import (
    SCENARIO_REGISTRY,
    apply_preset,
    compare_scenarios as _compare_scenarios,
    compute_kpis,
    draw_demand,
    draw_lead_time,
    ensure_dir,
    generate_narrative as _generate_narrative,
    grid_policy_search as _grid_policy_search,
    run_once as _run_once,
    run_scenario as _run_scenario,
    run_monte_carlo as _run_monte_carlo,
    optimize_policy_spsa as _optimize_policy_spsa,
    counterfactual_policy_search as _counterfactual_policy_search,
    run_two_echelon,
    run_network,
    save_cost_plot,
    save_inventory_plot,
    summarize_kpis,
    compute_two_echelon_kpis,
    compute_network_kpis,
)


def run_once(cfg: Dict) -> pd.DataFrame:
    """Execute a single seeded simulation run and return the daily log."""
    return _run_once(cfg)


def run_scenario(*args, **kwargs) -> Tuple[pd.DataFrame, Dict]:
    """Backward-compatible scenario runner supporting legacy signatures."""
    scenario_name_kw = kwargs.pop("scenario_name", None)
    overrides_kw = kwargs.pop("overrides", None)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

    def _call(base_cfg, overrides, scenario_name):
        df, kpis = _run_scenario(base_cfg, overrides, scenario_name=scenario_name)
        return df, kpis

    if len(args) == 1 and isinstance(args[0], dict):
        base_cfg = args[0]
        return _call(base_cfg, overrides_kw, scenario_name_kw)

    if len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
        base_cfg, overrides_pos = args
        overrides = overrides_kw if overrides_kw is not None else overrides_pos
        return _call(base_cfg, overrides, scenario_name_kw)

    if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
        name = args[0]
        base_cfg = args[1]
        overrides = overrides_kw
        if len(args) >= 3:
            overrides = args[2]
        scenario = name if name else scenario_name_kw
        if scenario_name_kw and name and scenario_name_kw != name:
            raise ValueError("Conflicting scenario names provided")
        return _call(base_cfg, overrides, scenario)

    raise TypeError(
        "run_scenario expects (base_cfg[, overrides]) or (name, base_cfg[, overrides])"
    )



def compare_scenarios(kpis: Iterable[Dict], baseline_name: str = "baseline") -> pd.DataFrame:
    return _compare_scenarios(kpis, baseline_name=baseline_name)


def run_monte_carlo(*args, **kwargs):
    return _run_monte_carlo(*args, **kwargs)


def optimize_policy_spsa(*args, **kwargs):
    return _optimize_policy_spsa(*args, **kwargs)


def counterfactual_policy_search(*args, **kwargs):
    return _counterfactual_policy_search(*args, **kwargs)


def grid_policy_search(*args, **kwargs):
    return _grid_policy_search(*args, **kwargs)


def generate_narrative(*args, **kwargs):
    return _generate_narrative(*args, **kwargs)


def compare_kpis(baseline: Dict, other: Dict, baseline_name: str = "baseline", other_name: str = "scenario") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy two-row comparison used in early notebooks."""
    cols = [
        "FR",
        "CSL",
        "AvgInventory",
        "AvgBackorder",
        "TotalHoldingCost",
        "TotalStockoutCost",
        "TotalCost",
        "StockoutDays",
        "OrdersPlacedCount",
    ]
    row_b = {"Scenario": baseline_name, **{c: baseline.get(c) for c in cols}}
    row_o = {"Scenario": other_name, **{c: other.get(c) for c in cols}}
    df = pd.DataFrame([row_b, row_o])
    delta = {"Scenario": "Δ (scenario - base)"}
    for c in cols:
        try:
            delta[c] = float(other.get(c, 0.0)) - float(baseline.get(c, 0.0))
        except (TypeError, ValueError):
            delta[c] = None
    return df, pd.DataFrame([delta])


__all__ = [
    "SCENARIO_REGISTRY",
    "run_once",
    "run_scenario",
    "compute_kpis",
    "compare_scenarios",
    "compare_kpis",
    "draw_demand",
    "draw_lead_time",
    "ensure_dir",
    "save_inventory_plot",
    "save_cost_plot",
    "apply_preset",
    "run_monte_carlo",
    "summarize_kpis",
    "optimize_policy_spsa",
    "counterfactual_policy_search",
    "grid_policy_search",
    "generate_narrative",
    "run_two_echelon",
    "compute_two_echelon_kpis",
    "run_network",
    "compute_network_kpis",
]
