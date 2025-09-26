import math

import numpy as np
import pandas as pd
import pytest

import engine
from engine_core import (
    draw_demand,
    generate_narrative,
    grid_policy_search,
    run_once,
    run_monte_carlo,
    run_two_echelon,
    run_network,
    compute_two_echelon_kpis,
    compute_network_kpis,
    optimize_policy_spsa,
    counterfactual_policy_search,
    summarize_kpis,
    validate_config,
)


def _base_cfg(**kwargs):
    cfg = {
        "N_DAYS": 5,
        "seed": 123,
        "Demand": {"distribution": "normal", "mu": 0, "sigma": 0},
        "LeadTime": {"type": "fixed", "days": 2},
        "Policy": {"s": 10, "S": 20},
        "Initial": {"on_hand": 0, "backorder": 0},
        "Costs": {"holding_cost": 0.0, "stockout_penalty": 0.0},
    }
    cfg.update(kwargs)
    return cfg


def test_backlog_triggers_full_order():
    cfg = _base_cfg(
        N_DAYS=1,
        Policy={"s": 50, "S": 100},
        Initial={"on_hand": 0, "backorder": 10},
    )
    df = run_once(cfg)
    order_qty = int(df.loc[0, "OrderPlaced"])
    assert order_qty == 110


def test_discrete_lead_time_arrival_honored():
    cfg = _base_cfg(
        N_DAYS=5,
        LeadTime={"type": "discrete", "pmf": {"3": 1.0}},
        Policy={"s": 0, "S": 50},
        Initial={"on_hand": 0, "backorder": 0},
    )
    df = run_once(cfg)
    # Order on day 1, arrival on day 4 (1 + 3)
    assert int(df.loc[0, "OrderPlaced"]) == 50
    assert int(df.loc[3, "Arrivals"]) == 50


def test_uniform_demand_with_zero_sigma_is_deterministic():
    rng = np.random.default_rng(0)
    cfg = {
        "Demand": {"distribution": "uniform", "mu": 20, "sigma": 0},
    }
    val = draw_demand(cfg, day=1, rng=rng)
    assert val == 20


def test_poisson_demand_non_negative():
    rng = np.random.default_rng(42)
    cfg = {"Demand": {"distribution": "poisson", "mu": 5}}
    draws = [draw_demand(cfg, day=1, rng=rng) for _ in range(50)]
    assert all(isinstance(x, int) and x >= 0 for x in draws)


def test_kpi_counts_backlog_served():
    cfg = _base_cfg(
        N_DAYS=2,
        Policy={"s": 0, "S": 0},
        Initial={"on_hand": 10, "backorder": 5},
        Demand={"distribution": "normal", "mu": 8, "sigma": 0},
    )
    df = run_once(cfg)
    kpis = engine.compute_kpis(df)
    # First day serves backlog + demand until inventory exhausted
    assert math.isclose(kpis["FR"], (df["ServedBacklog"].sum() + df["ServedToday"].sum()) / df["Demand"].sum())
    assert "AvgBackorder" in kpis


def test_run_scenario_signature_compatibility():
    base_cfg = _base_cfg()
    df1, kpi1 = engine.run_scenario(base_cfg)
    df2, kpi2 = engine.run_scenario("baseline", base_cfg, {})
    assert df1.equals(df2)
    assert kpi1 == {k: v for k, v in kpi2.items() if k != "Scenario"}


def test_run_monte_carlo_reproducible():
    base_cfg = _base_cfg()
    run_df, stats_df = run_monte_carlo(base_cfg, n_runs=5, base_seed=123, batch_size=2)
    assert len(run_df) == 5
    assert "stat" in stats_df.columns
    assert "TotalCost_cvar95" in stats_df.columns
    # Running again with same seed matches
    run_df2, stats_df2 = run_monte_carlo(base_cfg, n_runs=5, base_seed=123, batch_size=2)
    pd.testing.assert_frame_equal(run_df, run_df2)
    pd.testing.assert_frame_equal(stats_df, stats_df2)


def test_summarize_kpis_handles_empty():
    empty = pd.DataFrame()
    out = summarize_kpis(empty)
    assert out.empty


def test_run_monte_carlo_adds_scenario_column():
    base_cfg = _base_cfg()
    run_df, stats_df = run_monte_carlo(base_cfg, n_runs=3, base_seed=42, scenario_name="Test", batch_size=1)
    assert "Scenario" in run_df.columns
    assert run_df["Scenario"].nunique() == 1
    assert stats_df["Scenario"].nunique() == 1


def test_grid_policy_search_orders_by_objective():
    base_cfg = _base_cfg()
    s_vals = [0, 10]
    S_vals = [10, 20]
    df = grid_policy_search(base_cfg, s_vals, S_vals, service_target=0.0, objective="total_cost")
    assert not df.empty
    assert "objective_value" in df.columns
    assert df.iloc[0]["objective_value"] <= df.iloc[-1]["objective_value"]


def test_generate_narrative_structure():
    base_kpi = {"Scenario": "base", "FR": 0.9, "CSL": 0.92, "TotalCost": 1000, "AvgInventory": 50}
    scen_kpi = {"Scenario": "opt", "FR": 0.95, "CSL": 0.96, "TotalCost": 900, "AvgInventory": 60}
    story = generate_narrative(base_kpi, scen_kpi, service_target=0.95)
    assert "headline" in story and "details" in story
    assert isinstance(story["details"], list)


def test_run_two_echelon_basic_flow():
    cfg = {
        "N_DAYS": 5,
        "seed": 7,
        "Demand": {"distribution": "normal", "mu": 5, "sigma": 0},
        "Store": {
            "Policy": {"s": 10, "S": 20},
            "LeadTime": {"type": "fixed", "days": 1},
            "Initial": {"on_hand": 25, "backorder": 0},
            "Costs": {"holding_cost": 0.0, "stockout_penalty": 0.0},
        },
        "DC": {
            "Policy": {"s": 40, "S": 80},
            "LeadTime": {"type": "fixed", "days": 2},
            "Initial": {"on_hand": 60, "backorder": 0},
            "Costs": {"holding_cost": 0.0, "stockout_penalty": 0.0},
        },
    }
    df = run_two_echelon(cfg)
    assert not df.empty
    kpis = compute_two_echelon_kpis(df)
    assert "Store_FR" in kpis
    assert 0 <= kpis["Store_FR"] <= 1


def test_run_monte_carlo_progress_callback():
    base_cfg = _base_cfg()
    ticks: list[int] = []

    def cb(done: int, total: int) -> None:
        ticks.append(done)

    run_monte_carlo(base_cfg, n_runs=4, base_seed=1, batch_size=2, progress_cb=cb)
    assert ticks[0] == 0
    assert ticks[-1] == 4


def test_optimize_policy_spsa_runs():
    base_cfg = _base_cfg()
    out = optimize_policy_spsa(
        base_cfg,
        iterations=5,
        n_runs=1,
        seed=0,
    )
    assert "best" in out and "history" in out
    assert not out["history"].empty


def test_counterfactual_policy_search_hits_or_progresses():
    base_cfg = _base_cfg()
    out = counterfactual_policy_search(base_cfg, target_fr=0.8, step=5, max_iter=5, n_runs=1)
    assert "iterations" in out
    assert len(out["iterations"]) >= 1


def test_run_network_basic_chain():
    cfg = {
        "N_DAYS": 3,
        "seed": 123,
        "Demand": {"distribution": "normal", "mu": 5, "sigma": 0},
        "Nodes": [
            {"name": "Supplier", "type": "supplier", "Policy": {"s": 0, "S": 20}, "Initial": {"on_hand": 100}},
            {"name": "DC", "type": "dc", "Policy": {"s": 5, "S": 15}, "Initial": {"on_hand": 20}},
            {"name": "Store", "type": "store", "Policy": {"s": 5, "S": 15}, "Initial": {"on_hand": 15}},
        ],
        "Lanes": [
            {"from": "Supplier", "to": "DC", "LeadTime": {"type": "fixed", "days": 1}},
            {"from": "DC", "to": "Store", "LeadTime": {"type": "fixed", "days": 1}},
        ],
    }
    df = run_network(cfg)
    assert not df.empty
    kpis = compute_network_kpis(df)
    assert not kpis.empty


def test_validate_config_missing_section():
    with pytest.raises(KeyError):
        validate_config({}, required={"Policy": ("s",)})
