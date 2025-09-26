from __future__ import annotations
import copy
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:  # Matplotlib is optional for non-plotting contexts
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore


# ---------------------------------------------------------------------------
# Config + scenario helpers
# ---------------------------------------------------------------------------
SCENARIO_REGISTRY: Dict[str, Dict] = {
    "baseline": {},
    "supplier_delay_plus2": {
        "LeadTime": {"type": "discrete", "pmf": {"5": 0.4, "7": 0.5, "9": 0.1}},
    },
    "demand_spike_15pct": {
        "Demand": {"spike": {"start": 8, "end": 14, "mult": 1.15}},
    },
    "policy_raise_sS": {
        "Policy": {"s": 100, "S": 100},  # interpreted as deltas later
    },
}


def deep_merge(base: Dict, overrides: Optional[Dict]) -> Dict:
    """Recursive copy + merge."""
    if not overrides:
        return copy.deepcopy(base)
    out = copy.deepcopy(base)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


# ---------------------------------------------------------------------------
# Demand & lead-time draws
# ---------------------------------------------------------------------------
@dataclass
class DemandParams:
    distribution: str
    mu: float
    sigma: float
    spike: Optional[Dict[str, float]] = None


@dataclass
class SimulationState:
    on_hand: int
    backorder: int
    on_order: int


DEFAULT_COLUMNS = [
    "Day",
    "Demand",
    "ServedBacklog",
    "ServedToday",
    "UnfilledToday",
    "OnHandEnd",
    "BackorderEnd",
    "Arrivals",
    "OrderPlaced",
    "HoldingCost",
    "StockoutCost",
]


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))


def _demand_multiplier(spike_cfg: Optional[Dict[str, float]], day: int) -> float:
    if not spike_cfg:
        return 1.0
    start = int(spike_cfg.get("start", -1))
    end = int(spike_cfg.get("end", -1))
    if start <= day <= end:
        return float(spike_cfg.get("mult", 1.0))
    return 1.0


def _cvar(series: pd.Series | Iterable[float], alpha: float, tail: str = "upper") -> float:
    data = pd.Series(series).dropna()
    if data.empty:
        return float("nan")
    alpha = min(max(alpha, 0.0), 1.0)
    if tail == "upper":
        cutoff = data.quantile(alpha)
        tail_vals = data[data >= cutoff]
    else:
        cutoff = data.quantile(alpha)
        tail_vals = data[data <= cutoff]
    if tail_vals.empty:
        return float(cutoff)
    return float(tail_vals.mean())


def validate_config(cfg: Dict, required: Optional[Dict[str, Iterable[str]]] = None) -> None:
    """Shallow validation for configuration dictionaries.

    Parameters
    ----------
    cfg : Dict
        Configuration to validate.
    required : Dict[str, Iterable[str]]
        Mapping of section name → required keys.
    """

    if not isinstance(cfg, dict):
        raise TypeError("Configuration must be a dictionary")
    required = required or {
        "Demand": ("mu", "sigma"),
        "Policy": ("s", "S"),
        "Costs": ("holding_cost", "stockout_penalty"),
    }
    for section, keys in required.items():
        if section not in cfg:
            raise KeyError(f"Missing configuration section '{section}'")
        for key in keys:
            if key not in cfg[section]:
                raise KeyError(f"Missing key '{section}.{key}'")


def profile_run(func: Callable[[], Tuple[pd.DataFrame, Dict[str, float]]]) -> Dict[str, float]:
    """Utility to time a callable returning (DataFrame, KPIs)."""

    start = pd.Timestamp.utcnow()
    result = func()
    end = pd.Timestamp.utcnow()
    duration = (end - start).total_seconds()
    if isinstance(result, tuple) and len(result) == 2:
        _, kpis = result
    else:
        kpis = {}
    return {"duration_s": duration, "kpis": kpis}


def draw_demand(cfg: Dict, day: int, rng: np.random.Generator) -> int:
    dcfg = cfg["Demand"]
    params = DemandParams(
        distribution=str(dcfg.get("distribution", "normal")).lower(),
        mu=float(dcfg.get("mu", 100.0)),
        sigma=float(dcfg.get("sigma", 0.0)),
        spike=dcfg.get("spike"),
    )
    mult = _demand_multiplier(params.spike, day)
    if params.distribution == "normal":
        val = rng.normal(params.mu * mult, params.sigma * mult)
        return max(0, int(round(val)))
    if params.distribution == "poisson":
        lam = max(0.0, params.mu * mult)
        return int(rng.poisson(lam))
    if params.distribution == "uniform":
        # Derive bounds from mu/sigma. When sigma==0 fall back to deterministic mu.
        if params.sigma <= 0:
            return max(0, int(round(params.mu * mult)))
        half_width = math.sqrt(12) * params.sigma * mult / 2
        low = max(0.0, params.mu * mult - half_width)
        high = max(low, params.mu * mult + half_width)
        return int(round(rng.uniform(low, high)))
    raise ValueError(f"Unsupported demand distribution: {params.distribution}")


def draw_lead_time(cfg: Dict, rng: np.random.Generator) -> int:
    ltcfg = cfg["LeadTime"]
    kind = str(ltcfg.get("type", "fixed")).lower()
    if kind == "fixed":
        return max(1, int(ltcfg.get("days", 1)))
    if kind == "lognormal":
        mu = float(ltcfg.get("mu", math.log(max(1.0, float(ltcfg.get("mean", 5.0))))))
        sigma = float(ltcfg.get("sigma", 0.25))
        Lmax = int(ltcfg.get("Lmax", 60))
        val = int(round(rng.lognormal(mean=mu, sigma=sigma)))
        return max(1, min(Lmax, val))
    if kind == "discrete":
        pmf = ltcfg.get("pmf")
        if not pmf:
            raise ValueError("Discrete lead-time requires 'pmf' dict")
        days = np.array([int(k) for k in pmf.keys()], dtype=int)
        probs = np.array([float(v) for v in pmf.values()], dtype=float)
        probs = probs / probs.sum()
        return int(rng.choice(days, p=probs))
    raise ValueError(f"Unsupported lead-time model: {kind}")


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def run_once(cfg: Dict) -> pd.DataFrame:
    days = int(cfg.get("N_DAYS", 30))
    if days <= 0:
        raise ValueError("N_DAYS must be positive")
    validate_config(cfg)

    policy = cfg.get("Policy", {})
    s = int(policy.get("s", 0))
    S = int(policy.get("S", s))
    if S < s:
        raise ValueError("Policy invalid: S must be >= s")

    costs = cfg.get("Costs", {})
    hold_cost_rate = float(costs.get("holding_cost", 0.0))
    stockout_penalty = float(costs.get("stockout_penalty", 0.0))

    init = cfg.get("Initial", {})
    state = SimulationState(
        on_hand=int(init.get("on_hand", 0)),
        backorder=int(init.get("backorder", 0)),
        on_order=int(init.get("on_order", 0)),
    )

    rng = _rng(cfg.get("seed"))
    outstanding: List[Tuple[int, int]] = []  # (arrival_day, qty)

    rows: List[Dict] = []
    for day in range(1, days + 1):
        arrivals_today = 0
        remaining_pos: List[Tuple[int, int]] = []
        for arrive_day, qty in outstanding:
            if arrive_day == day:
                state.on_hand += qty
                state.on_order -= qty
                arrivals_today += qty
            else:
                remaining_pos.append((arrive_day, qty))
        outstanding = remaining_pos

        demand_units = draw_demand(cfg, day, rng)

        served_backlog = min(state.on_hand, state.backorder)
        state.on_hand -= served_backlog
        state.backorder -= served_backlog

        served_today = min(state.on_hand, demand_units)
        state.on_hand -= served_today
        unfilled_today = demand_units - served_today
        state.backorder += unfilled_today

        stock_position = state.on_hand + state.on_order - state.backorder
        order_qty = 0
        if stock_position <= s:
            order_qty = max(0, int(S - stock_position))
            if order_qty > 0:
                lead = draw_lead_time(cfg, rng)
                arrival_day = day + max(1, int(lead))
                outstanding.append((arrival_day, order_qty))
                state.on_order += order_qty

        holding_cost = state.on_hand * hold_cost_rate
        stockout_cost = unfilled_today * stockout_penalty

        rows.append({
            "Day": day,
            "Demand": demand_units,
            "ServedBacklog": served_backlog,
            "ServedToday": served_today,
            "UnfilledToday": unfilled_today,
            "OnHandEnd": state.on_hand,
            "BackorderEnd": state.backorder,
            "Arrivals": arrivals_today,
            "OrderPlaced": order_qty,
            "HoldingCost": holding_cost,
            "StockoutCost": stockout_cost,
            "StockPositionEnd": stock_position,
        })

    return pd.DataFrame(rows, columns=DEFAULT_COLUMNS + ["StockPositionEnd"])


# ---------------------------------------------------------------------------
# KPIs & scenario utilities
# ---------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "FR": 1.0,
            "CSL": 1.0,
            "AvgInventory": 0.0,
            "AvgBackorder": 0.0,
            "TotalHoldingCost": 0.0,
            "TotalStockoutCost": 0.0,
            "TotalCost": 0.0,
            "StockoutDays": 0,
            "OrdersPlacedCount": 0,
        }
    demand_total = float(df["Demand"].sum())
    served_total = float(df["ServedBacklog"].sum() + df["ServedToday"].sum())
    fill_rate = served_total / demand_total if demand_total > 0 else 1.0

    days_stockout = int(((df["UnfilledToday"] > 0) | (df["BackorderEnd"] > 0)).sum())
    cycle_service_level = 1.0 - days_stockout / len(df)

    holding_cost_sum = float(df["HoldingCost"].sum())
    stockout_cost_sum = float(df["StockoutCost"].sum())

    return {
        "FR": fill_rate,
        "CSL": cycle_service_level,
        "AvgInventory": float(df["OnHandEnd"].mean()),
        "AvgBackorder": float(df["BackorderEnd"].mean()),
        "TotalHoldingCost": holding_cost_sum,
        "TotalStockoutCost": stockout_cost_sum,
        "TotalCost": holding_cost_sum + stockout_cost_sum,
        "StockoutDays": days_stockout,
        "OrdersPlacedCount": int((df["OrderPlaced"] > 0).sum()),
    }


def run_scenario(
    base_cfg: Dict,
    overrides: Optional[Dict] = None,
    *,
    scenario_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    cfg = deep_merge(base_cfg, overrides)
    validate_config(cfg)
    df = run_once(cfg)
    kpis = compute_kpis(df)
    if scenario_name:
        kpis = {**kpis, "Scenario": scenario_name}
    return df, kpis


def apply_preset(base_cfg: Dict, preset_name: str) -> Dict:
    overrides = SCENARIO_REGISTRY.get(preset_name)
    if overrides is None:
        raise KeyError(f"Scenario preset '{preset_name}' not found")
    cfg = deep_merge(base_cfg, overrides)
    if preset_name == "policy_raise_sS":
        cfg = copy.deepcopy(cfg)
        cfg.setdefault("Policy", {})
        cfg["Policy"]["s"] = int(cfg["Policy"].get("s", 0)) + 100
        cfg["Policy"]["S"] = int(cfg["Policy"].get("S", 0)) + 100
    return cfg


def compare_scenarios(kpis: Iterable[Dict[str, float]], baseline_name: str = "baseline") -> pd.DataFrame:
    df = pd.DataFrame(list(kpis))
    if df.empty or "Scenario" not in df.columns:
        return df
    mask_base = df["Scenario"].str.lower() == baseline_name.lower()
    if not mask_base.any():
        baseline = df.iloc[0]
    else:
        baseline = df[mask_base].iloc[0]
    base = baseline.to_dict()

    def delta(col: str) -> pd.Series:
        if col not in df.columns or pd.isna(base.get(col)):
            return pd.Series([np.nan] * len(df))
        return df[col] - base[col]

    def pct(col: str) -> pd.Series:
        if col not in df.columns or not base.get(col):
            return pd.Series([np.nan] * len(df))
        return (df[col] - base[col]) / base[col]

    numeric_cols = [
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
    for col in numeric_cols:
        if col in df.columns:
            df[f"{col}_delta"] = delta(col)
            df[f"{col}_delta_pct"] = pct(col)
    return df


def _draw_seeds(rng: np.random.Generator, n_runs: int) -> np.ndarray:
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    return rng.integers(low=0, high=2**32 - 1, size=n_runs, dtype=np.uint32)


def run_monte_carlo(
    base_cfg: Dict,
    overrides: Optional[Dict] = None,
    *,
    n_runs: int = 100,
    base_seed: Optional[int] = None,
    scenario_name: Optional[str] = None,
    batch_size: int = 32,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute repeated simulations and return per-run KPIs + summary stats.

    Parameters
    ----------
    base_cfg : dict
        Baseline configuration shared across runs.
    overrides : dict | None
        Scenario-specific overrides applied before each run.
    n_runs : int, optional
        Number of Monte Carlo samples to execute (default 100).
    base_seed : int | None, optional
        Seed for drawing independent seeds per replication. ``None`` leaves RNG
        non-deterministic.
    scenario_name : str | None, optional
        Name used in the KPI outputs; if omitted, defaults to ``overrides`` name.

    Returns
    -------
    tuple(DataFrame, DataFrame)
        The first DataFrame contains per-run KPIs. The second contains
        aggregated statistics (mean, std, quantiles) for each KPI.
    """

    rng = _rng(base_seed)
    seeds = _draw_seeds(rng, n_runs)
    rows: List[Dict[str, float]] = []

    total = len(seeds)
    batch_size = max(1, int(batch_size))
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        if progress_cb:
            try:
                progress_cb(start, total)
            except Exception:
                pass
        for idx, seed in enumerate(seeds[start:end], start=start + 1):
            cfg = deep_merge(base_cfg, overrides)
            cfg["seed"] = int(seed)
            df = run_once(cfg)
            kpis = compute_kpis(df)
            kpis.update({
                "run": idx,
                "seed": int(seed),
            })
            if scenario_name:
                kpis.setdefault("Scenario", scenario_name)
            rows.append(kpis)

    if progress_cb:
        try:
            progress_cb(total, total)
        except Exception:
            pass

    run_df = pd.DataFrame(rows)
    stats_df = summarize_kpis(run_df)
    if scenario_name:
        run_df["Scenario"] = scenario_name
        stats_df["Scenario"] = scenario_name
    return run_df, stats_df


def summarize_kpis(kpis_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate KPI runs into mean/std/quantiles and risk indicators."""

    if kpis_df.empty:
        return pd.DataFrame()

    numeric_cols = [c for c in kpis_df.columns if pd.api.types.is_numeric_dtype(kpis_df[c])]
    agg = kpis_df[numeric_cols].agg(["mean", "std", "min", "max"])
    quantiles = kpis_df[numeric_cols].quantile([0.05, 0.5, 0.95]).rename(index={0.05: "q05", 0.5: "q50", 0.95: "q95"})
    summary = pd.concat([agg, quantiles])

    # Risk-specific helpers
    extras = {}
    if "FR" in kpis_df.columns:
        extras["FR_prob_below_90"] = (kpis_df["FR"] < 0.90).mean()
        extras["FR_cvar05"] = _cvar(kpis_df["FR"], alpha=0.05, tail="lower")
    if "CSL" in kpis_df.columns:
        extras["CSL_prob_below_95"] = (kpis_df["CSL"] < 0.95).mean()
        extras["CSL_cvar05"] = _cvar(kpis_df["CSL"], alpha=0.05, tail="lower")
    if "TotalCost" in kpis_df.columns:
        extras["TotalCost_var95"] = kpis_df["TotalCost"].quantile(0.95)
        extras["TotalCost_cvar95"] = _cvar(kpis_df["TotalCost"], alpha=0.95, tail="upper")

    if extras:
        extras_df = pd.DataFrame(extras, index=["risk"])
        summary = pd.concat([summary, extras_df])

    # Move index to column for readability
    summary = summary.reset_index().rename(columns={"index": "stat"})
    return summary


def _kpi_from_stats(stats_df: pd.DataFrame, stat: str = "mean") -> Dict[str, float]:
    subset = stats_df[stats_df["stat"] == stat]
    if subset.empty:
        return {}
    row = subset.iloc[0].to_dict()
    row.pop("stat", None)
    return row


def _objective_value(kpis: Dict[str, float], objective: str, service_target: Optional[float]) -> float:
    obj = objective.lower()
    total_cost = float(kpis.get("TotalCost", 0.0))
    fr = float(kpis.get("FR", 0.0))
    csl = float(kpis.get("CSL", 0.0))

    penalty = 0.0
    if service_target is not None and fr < service_target:
        penalty += (service_target - fr) * 1e6
    if service_target is not None and csl < service_target:
        penalty += (service_target - csl) * 5e5

    if obj in {"total_cost", "cost"}:
        return total_cost + penalty
    if obj in {"inventory"}:
        return float(kpis.get("AvgInventory", 0.0)) + penalty
    if obj in {"stockout"}:
        return float(kpis.get("TotalStockoutCost", 0.0)) + penalty
    if obj in {"fill_rate", "fr"}:
        return -fr + penalty
    # default fallback to total cost
    return total_cost + penalty


def grid_policy_search(
    base_cfg: Dict,
    s_values: Sequence[int],
    S_values: Sequence[int],
    *,
    service_target: Optional[float] = None,
    objective: str = "total_cost",
    n_runs: int | None = None,
    base_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Evaluate a grid of (s,S) policies and rank by objective.

    Parameters
    ----------
    s_values, S_values : iterable of ints
        Candidate reorder and order-up-to levels.
    service_target : float, optional
        Minimum FR/CSL required; violations receive a heavy penalty.
    objective : str
        One of ``total_cost`` (default), ``inventory``, ``stockout``, ``fill_rate``.
    n_runs : int, optional
        If provided (>1), Monte Carlo repetitions per candidate; otherwise single run.
    base_seed : int, optional
        Seed for Monte Carlo sampling.
    """

    records: List[Dict[str, float]] = []
    mc = n_runs is not None and n_runs > 1

    for s in s_values:
        for S in S_values:
            if S < s:
                continue
            overrides = {"Policy": {"s": int(s), "S": int(S)}}
            if mc:
                runs_df, stats_df = run_monte_carlo(
                    base_cfg,
                    overrides,
                    n_runs=int(n_runs),
                    base_seed=base_seed,
                )
                mean_kpis = _kpi_from_stats(stats_df, "mean")
                q05 = _kpi_from_stats(stats_df, "q05")
                q95 = _kpi_from_stats(stats_df, "q95")
                row = {**mean_kpis}
                row.update({f"{k}_q05": v for k, v in q05.items()})
                row.update({f"{k}_q95": v for k, v in q95.items()})
                row["run_count"] = len(runs_df)
            else:
                _, kpis = run_scenario(base_cfg, overrides)
                row = dict(kpis)
                row["run_count"] = 1

            row["s"] = int(s)
            row["S"] = int(S)
            row["objective_value"] = _objective_value(row, objective, service_target)
            row["ServiceTargetMet"] = (
                (service_target is None)
                or (row.get("FR", 0.0) >= service_target and row.get("CSL", 0.0) >= service_target)
            )
            records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df.sort_values(by="objective_value", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def generate_narrative(
    baseline: Dict[str, float],
    scenario: Dict[str, float],
    *,
    service_target: Optional[float] = None,
) -> Dict[str, str | List[str]]:
    """Produce a lightweight textual summary comparing two KPI dictionaries."""

    base_name = baseline.get("Scenario", "baseline")
    scen_name = scenario.get("Scenario", "scenario")

    deltas = {}
    for key in ["FR", "CSL", "TotalCost", "AvgInventory", "AvgBackorder", "StockoutDays"]:
        if key in baseline and key in scenario:
            deltas[key] = scenario[key] - baseline[key]

    fr_delta = deltas.get("FR", 0.0)
    cost_delta = deltas.get("TotalCost", 0.0)
    headline_parts = []

    if abs(fr_delta) >= 0.005:
        headline_parts.append(
            f"fill rate {('improves' if fr_delta >= 0 else 'drops')} by {abs(fr_delta)*100:.1f} pts"
        )
    if abs(cost_delta) >= 1.0:
        headline_parts.append(
            f"total cost {('up' if cost_delta > 0 else 'down')} {abs(cost_delta):.0f}"
        )
    if not headline_parts:
        headline = f"{scen_name} performs similarly to {base_name}."
    else:
        headline = f"{scen_name} vs {base_name}: " + ", ".join(headline_parts) + "."

    bullets: List[str] = []
    if "CSL" in deltas and abs(deltas["CSL"]) >= 0.005:
        bullets.append(f"CSL change: {'+' if deltas['CSL']>=0 else ''}{deltas['CSL']*100:.1f} pts")
    if "AvgInventory" in deltas and abs(deltas["AvgInventory"]) >= 1:
        bullets.append(f"Avg inventory {'↑' if deltas['AvgInventory']>0 else '↓'}{abs(deltas['AvgInventory']):.1f}")
    if "AvgBackorder" in deltas and abs(deltas["AvgBackorder"]) >= 0.5:
        bullets.append(f"Avg backlog {'↑' if deltas['AvgBackorder']>0 else '↓'}{abs(deltas['AvgBackorder']):.1f}")
    if "StockoutDays" in deltas and abs(deltas["StockoutDays"]) >= 1:
        bullets.append(f"Stockout days {'↑' if deltas['StockoutDays']>0 else '↓'}{abs(deltas['StockoutDays']):.0f}")

    if service_target is not None:
        fr = scenario.get("FR", 0.0)
        csl = scenario.get("CSL", 0.0)
        if fr < service_target or csl < service_target:
            bullets.append(
                f"⚠ service target {service_target:.2f} unmet (FR={fr:.2f}, CSL={csl:.2f})."
            )

    return {"headline": headline, "details": bullets}


# ---------------------------------------------------------------------------
# Two-echelon (store ↔ DC ↔ supplier)
# ---------------------------------------------------------------------------


def _draw_lt_or_fixed(lt_cfg: Dict, rng: np.random.Generator) -> int:
    if not lt_cfg:
        return 1
    cfg = dict(lt_cfg)
    if "type" not in cfg:
        cfg["type"] = "fixed"
    return draw_lead_time({"LeadTime": cfg}, rng)


def run_two_echelon(cfg: Dict) -> pd.DataFrame:
    days = int(cfg.get("N_DAYS", 30))
    if days <= 0:
        raise ValueError("N_DAYS must be positive")
    validate_config(cfg.get("Store", {}), required={
        "Policy": ("s", "S"),
        "Initial": ("on_hand",),
    })
    validate_config(cfg.get("DC", {}), required={
        "Policy": ("s", "S"),
        "Initial": ("on_hand",),
    })

    store_cfg = cfg.get("Store")
    dc_cfg = cfg.get("DC")
    if not store_cfg or not dc_cfg:
        raise KeyError("Configuration requires 'Store' and 'DC' sections")

    rng = _rng(cfg.get("seed"))

    store_on_hand = int(store_cfg.get("Initial", {}).get("on_hand", 0))
    store_backorder = int(store_cfg.get("Initial", {}).get("backorder", 0))
    store_on_order = 0
    store_pipeline: List[Tuple[int, int]] = []
    store_policy = store_cfg.get("Policy", {})
    store_s = int(store_policy.get("s", 0))
    store_S = int(store_policy.get("S", store_s))
    if store_S < store_s:
        raise ValueError("Store policy invalid: S must be >= s")
    store_costs = store_cfg.get("Costs", {})
    store_hold_cost = float(store_costs.get("holding_cost", 0.0))
    store_penalty = float(store_costs.get("stockout_penalty", 0.0))
    store_lt_cfg = store_cfg.get("LeadTime", {"type": "fixed", "days": 1})

    dc_on_hand = int(dc_cfg.get("Initial", {}).get("on_hand", 0))
    dc_backorder = int(dc_cfg.get("Initial", {}).get("backorder", 0))
    dc_on_order = 0
    supplier_pipeline: List[Tuple[int, int]] = []
    store_shipments_out: List[Tuple[int, int]] = []
    dc_policy = dc_cfg.get("Policy", {})
    dc_s = int(dc_policy.get("s", 0))
    dc_S = int(dc_policy.get("S", dc_s))
    if dc_S < dc_s:
        raise ValueError("DC policy invalid: S must be >= s")
    dc_costs = dc_cfg.get("Costs", {})
    dc_hold_cost = float(dc_costs.get("holding_cost", 0.0))
    dc_penalty = float(dc_costs.get("stockout_penalty", 0.0))
    dc_lt_cfg = dc_cfg.get("LeadTime", {"type": "fixed", "days": 5})

    rows: List[Dict[str, float]] = []

    for day in range(1, days + 1):
        store_arrivals_today = 0
        remaining_store_pipe: List[Tuple[int, int]] = []
        for arrive_day, qty in store_pipeline:
            if arrive_day == day:
                store_on_hand += qty
                store_on_order -= qty
                store_arrivals_today += qty
            else:
                remaining_store_pipe.append((arrive_day, qty))
        store_pipeline = remaining_store_pipe

        dc_arrivals_today = 0
        remaining_supplier: List[Tuple[int, int]] = []
        for arrive_day, qty in supplier_pipeline:
            if arrive_day == day:
                dc_on_hand += qty
                dc_on_order -= qty
                dc_arrivals_today += qty
            else:
                remaining_supplier.append((arrive_day, qty))
        supplier_pipeline = remaining_supplier

        demand_today = draw_demand(cfg, day, rng)
        store_served_backlog = min(store_on_hand, store_backorder)
        store_on_hand -= store_served_backlog
        store_backorder -= store_served_backlog

        store_served_today = min(store_on_hand, demand_today)
        store_on_hand -= store_served_today
        store_unfilled_today = demand_today - store_served_today
        store_backorder += store_unfilled_today

        store_stock_position = store_on_hand + store_on_order - store_backorder
        store_order_qty = 0
        if store_stock_position <= store_s:
            store_order_qty = max(0, int(store_S - store_stock_position))
            if store_order_qty > 0:
                store_on_order += store_order_qty
                dc_backorder += store_order_qty

        dc_ship_today = 0
        if dc_backorder > 0 and dc_on_hand > 0:
            dc_ship_today = min(dc_on_hand, dc_backorder)
            dc_on_hand -= dc_ship_today
            dc_backorder -= dc_ship_today
            lt_store = _draw_lt_or_fixed(store_lt_cfg, rng)
            store_pipeline.append((day + max(1, int(lt_store)), dc_ship_today))
            store_shipments_out.append((day, dc_ship_today))

        dc_stock_position = dc_on_hand + dc_on_order - dc_backorder
        supplier_order_qty = 0
        if dc_stock_position <= dc_s:
            supplier_order_qty = max(0, int(dc_S - dc_stock_position))
            if supplier_order_qty > 0:
                lt_sup = _draw_lt_or_fixed(dc_lt_cfg, rng)
                supplier_pipeline.append((day + max(1, int(lt_sup)), supplier_order_qty))
                dc_on_order += supplier_order_qty

        store_holding_cost = store_on_hand * store_hold_cost
        store_stockout_cost = store_unfilled_today * store_penalty
        dc_holding_cost = dc_on_hand * dc_hold_cost
        dc_backlog_cost = dc_backorder * dc_penalty

        rows.append({
            "Day": day,
            "Store_Demand": demand_today,
            "Store_ServedToday": store_served_today,
            "Store_UnfilledToday": store_unfilled_today,
            "Store_OnHandEnd": store_on_hand,
            "Store_BackorderEnd": store_backorder,
            "Store_OnOrderEnd": store_on_order,
            "Store_OrderPlaced": store_order_qty,
            "Store_Arrivals": store_arrivals_today,
            "Store_HoldingCost": store_holding_cost,
            "Store_StockoutCost": store_stockout_cost,
            "DC_OnHandEnd": dc_on_hand,
            "DC_BackorderEnd": dc_backorder,
            "DC_OnOrderEnd": dc_on_order,
            "DC_ShippedToStore": dc_ship_today,
            "DC_Arrivals": dc_arrivals_today,
            "DC_OrderPlaced": supplier_order_qty,
            "DC_HoldingCost": dc_holding_cost,
            "DC_BacklogCost": dc_backlog_cost,
        })

    return pd.DataFrame(rows)


def compute_two_echelon_kpis(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    store_demand = float(df["Store_Demand"].sum())
    store_served = float(df["Store_ServedToday"].sum())
    store_fr = store_served / store_demand if store_demand > 0 else 1.0
    store_days_stockout = int((df["Store_UnfilledToday"] > 0).sum())
    store_csl = 1.0 - store_days_stockout / len(df)

    store_orders = float(df["Store_OrderPlaced"].sum())
    dc_ship = float(df["DC_ShippedToStore"].sum())
    dc_service_rate = dc_ship / store_orders if store_orders > 0 else 1.0

    return {
        "Store_FR": store_fr,
        "Store_CSL": store_csl,
        "Store_AvgInventory": float(df["Store_OnHandEnd"].mean()),
        "Store_AvgBackorder": float(df["Store_BackorderEnd"].mean()),
        "Store_TotalHoldingCost": float(df["Store_HoldingCost"].sum()),
        "Store_TotalStockoutCost": float(df["Store_StockoutCost"].sum()),
        "DC_ServiceRate": dc_service_rate,
        "DC_AvgInventory": float(df["DC_OnHandEnd"].mean()),
        "DC_AvgBackorder": float(df["DC_BackorderEnd"].mean()),
        "DC_TotalHoldingCost": float(df["DC_HoldingCost"].sum()),
        "DC_TotalBacklogCost": float(df["DC_BacklogCost"].sum()),
    }


# ---------------------------------------------------------------------------
# Policy optimisation & counterfactual search
# ---------------------------------------------------------------------------


def _extract_policy(cfg: Dict) -> Tuple[int, int]:
    pol = cfg.get("Policy", {})
    return int(pol.get("s", 0)), int(pol.get("S", pol.get("s", 0)))


def _evaluate_policy(
    base_cfg: Dict,
    s_val: int,
    S_val: int,
    *,
    overrides: Optional[Dict] = None,
    n_runs: int = 1,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    overrides = overrides or {}
    policy_override = {"Policy": {"s": int(s_val), "S": int(S_val)}}
    merged_overrides = deep_merge(overrides, policy_override)
    if n_runs and n_runs > 1:
        runs_df, stats_df = run_monte_carlo(
            base_cfg,
            merged_overrides,
            n_runs=int(n_runs),
            base_seed=seed,
            scenario_name=None,
        )
        mean_stats = stats_df[stats_df["stat"] == "mean"]
        if not mean_stats.empty:
            kpis = mean_stats.iloc[0].to_dict()
            kpis.pop("stat", None)
        else:
            # fallback to deterministic run
            _, kpis = run_scenario(base_cfg, merged_overrides)
        return kpis, stats_df
    _, kpis = run_scenario(base_cfg, merged_overrides)
    return kpis, None


def optimize_policy_spsa(
    base_cfg: Dict,
    *,
    overrides: Optional[Dict] = None,
    objective: str = "total_cost",
    service_target: Optional[float] = None,
    n_runs: int = 1,
    iterations: int = 20,
    a: float = 20.0,
    c: float = 10.0,
    alpha: float = 0.602,
    gamma: float = 0.101,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Basic SPSA optimiser for (s, S) policies.

    Returns a dictionary containing history, best policy and KPIs. This is a
    light-weight implementation suitable for experimentation rather than a
    production optimiser.
    """

    rng = _rng(seed)
    s0, S0 = _extract_policy(base_cfg)
    theta = np.array([float(s0), float(S0)], dtype=float)
    best = None
    best_score = float("inf")
    history_rows = []

    for k in range(1, iterations + 1):
        a_k = a / ((k) ** alpha)
        c_k = c / ((k) ** gamma)
        delta = rng.choice([-1, 1], size=2)

        theta_plus = theta + c_k * delta
        theta_minus = theta - c_k * delta

        s_plus, S_plus = np.round(theta_plus).astype(int)
        s_minus, S_minus = np.round(theta_minus).astype(int)
        S_plus = max(S_plus, s_plus)
        S_minus = max(S_minus, s_minus)
        s_plus = max(0, s_plus)
        s_minus = max(0, s_minus)

        kpis_plus, _ = _evaluate_policy(base_cfg, s_plus, S_plus, overrides=overrides, n_runs=n_runs, seed=None)
        kpis_minus, _ = _evaluate_policy(base_cfg, s_minus, S_minus, overrides=overrides, n_runs=n_runs, seed=None)
        g_plus = _objective_value(kpis_plus, objective, service_target)
        g_minus = _objective_value(kpis_minus, objective, service_target)

        grad = (g_plus - g_minus) / (2 * c_k * delta)
        theta = theta - a_k * grad
        theta[0] = max(0, theta[0])
        theta[1] = max(theta[0], theta[1])

        s_cur, S_cur = np.round(theta).astype(int)
        kpis_cur, _ = _evaluate_policy(base_cfg, s_cur, S_cur, overrides=overrides, n_runs=n_runs, seed=None)
        score_cur = _objective_value(kpis_cur, objective, service_target)

        if score_cur < best_score:
            best_score = score_cur
            best = {
                "s": int(s_cur),
                "S": int(S_cur),
                "score": score_cur,
                "kpis": kpis_cur,
            }

        history_rows.append({
            "iter": k,
            "s": int(s_cur),
            "S": int(S_cur),
            "objective": score_cur,
            "FR": kpis_cur.get("FR"),
            "CSL": kpis_cur.get("CSL"),
            "TotalCost": kpis_cur.get("TotalCost"),
        })

    history_df = pd.DataFrame(history_rows)
    return {"best": best, "history": history_df}


def counterfactual_policy_search(
    base_cfg: Dict,
    *,
    target_fr: float = 0.95,
    step: int = 20,
    max_iter: int = 20,
    overrides: Optional[Dict] = None,
    n_runs: int = 1,
) -> Dict[str, object]:
    """Incrementally adjust (s,S) until the target fill rate is met."""

    overrides = overrides or {}
    s_cur, S_cur = _extract_policy(deep_merge(base_cfg, overrides))
    history: List[Dict[str, float]] = []

    for i in range(max_iter):
        kpis, stats = _evaluate_policy(
            base_cfg,
            s_cur,
            S_cur,
            overrides=overrides,
            n_runs=n_runs,
            seed=None,
        )
        fr = kpis.get("FR", 0.0)
        history.append({
            "iter": i,
            "s": s_cur,
            "S": S_cur,
            "FR": fr,
            "CSL": kpis.get("CSL"),
            "TotalCost": kpis.get("TotalCost"),
        })
        if fr >= target_fr:
            return {
                "status": "success",
                "iterations": history,
                "policy": {"s": s_cur, "S": S_cur},
                "kpis": kpis,
                "stats": stats,
            }
        # increase s and S while keeping S ≥ s
        s_cur += step
        S_cur = max(S_cur + step, s_cur)

    return {
        "status": "incomplete",
        "iterations": history,
        "policy": {"s": s_cur, "S": S_cur},
        "kpis": kpis,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _require_matplotlib() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting but is not available")


def save_inventory_plot(df: pd.DataFrame, title: str, out_png: str) -> None:
    _require_matplotlib()
    plt.figure(figsize=(10, 4))
    plt.plot(df["Day"], df["OnHandEnd"], lw=2, label="On-hand")
    orders = df[df["OrderPlaced"] > 0]
    arrivals = df[df["Arrivals"] > 0]
    if not orders.empty:
        plt.scatter(orders["Day"], orders["OnHandEnd"], marker="^", s=60, label="Order placed")
    if not arrivals.empty:
        plt.scatter(arrivals["Day"], arrivals["OnHandEnd"], marker="D", s=50, label="Delivery arrived")
    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("Units")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_cost_plot(df: pd.DataFrame, title: str, out_png: str) -> None:
    _require_matplotlib()
    plt.figure(figsize=(10, 4))
    plt.plot(df["Day"], df["HoldingCost"].cumsum(), lw=2, label="Cum Holding")
    plt.plot(df["Day"], df["StockoutCost"].cumsum(), lw=2, linestyle="--", label="Cum Stockout")
    plt.title(title)
    plt.xlabel("Day")
    plt.ylabel("Cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# Multi-node network simulation (Phase 7)
# ---------------------------------------------------------------------------


@dataclass
class NetworkNode:
    name: str
    kind: str  # "supplier", "dc", "store"
    policy_s: int
    policy_S: int
    on_hand: int
    backlog: int
    costs: Dict[str, float]


@dataclass
class NetworkLane:
    src: str
    dst: str
    lead_cfg: Dict[str, object]
    capacity: Optional[int] = None


def _build_network(cfg: Dict) -> Tuple[Dict[str, NetworkNode], List[NetworkLane]]:
    nodes_cfg = cfg.get("Nodes")
    lanes_cfg = cfg.get("Lanes")
    if not nodes_cfg or not lanes_cfg:
        raise KeyError("Network requires 'Nodes' and 'Lanes'")

    nodes: Dict[str, NetworkNode] = {}
    for node in nodes_cfg:
        name = node["name"]
        policy = node.get("Policy", {})
        initial = node.get("Initial", {})
        nodes[name] = NetworkNode(
            name=name,
            kind=node.get("type", "store"),
            policy_s=int(policy.get("s", 0)),
            policy_S=int(policy.get("S", policy.get("s", 0))),
            on_hand=int(initial.get("on_hand", 0)),
            backlog=int(initial.get("backorder", 0)),
            costs=node.get("Costs", {}),
        )

    lanes: List[NetworkLane] = []
    for lane in lanes_cfg:
        lanes.append(
            NetworkLane(
                src=lane["from"],
                dst=lane["to"],
                lead_cfg=lane.get("LeadTime", {"type": "fixed", "days": 1}),
                capacity=lane.get("capacity"),
            )
        )

    return nodes, lanes


def _network_demand(node: NetworkNode, day: int, cfg: Dict, rng: np.random.Generator) -> int:
    if node.kind == "store":
        return draw_demand(cfg, day, rng)
    return 0


def run_network(cfg: Dict) -> pd.DataFrame:
    days = int(cfg.get("N_DAYS", 30))
    if days <= 0:
        raise ValueError("N_DAYS must be positive")
    if "Nodes" not in cfg or "Lanes" not in cfg:
        raise KeyError("Network configuration requires 'Nodes' and 'Lanes'")

    nodes, lanes = _build_network(cfg)
    rng = _rng(cfg.get("seed"))

    lane_map: Dict[Tuple[str, str], NetworkLane] = {(lane.src, lane.dst): lane for lane in lanes}
    pipeline: Dict[Tuple[str, str], List[Tuple[int, int]]] = {
        (lane.src, lane.dst): [] for lane in lanes
    }
    lane_backlog: Dict[Tuple[str, str], int] = {
        (lane.src, lane.dst): 0 for lane in lanes
    }

    outgoing: Dict[str, List[str]] = {}
    incoming: Dict[str, List[str]] = {}
    for lane in lanes:
        outgoing.setdefault(lane.src, []).append(lane.dst)
        incoming.setdefault(lane.dst, []).append(lane.src)

    rows: List[Dict[str, object]] = []

    for day in range(1, days + 1):
        # Process arrivals on all lanes
        for lane_key, pipe in pipeline.items():
            arrivals_today = 0
            remaining: List[Tuple[int, int]] = []
            dst = lane_key[1]
            node_dst = nodes[dst]
            for arrive_day, qty in pipe:
                if arrive_day == day:
                    node_dst.on_hand += qty
                    arrivals_today += qty
                else:
                    remaining.append((arrive_day, qty))
            pipeline[lane_key] = remaining
            if arrivals_today > 0:
                rows.append({
                    "Day": day,
                    "Event": "Arrival",
                    "Lane": f"{lane_key[0]}->{lane_key[1]}",
                    "Quantity": arrivals_today,
                })

        # Demand at stores
        node_metrics: Dict[str, Dict[str, float]] = {}
        for name, node in nodes.items():
            demand = _network_demand(node, day, cfg, rng)
            served_backlog = min(node.on_hand, node.backlog)
            node.on_hand -= served_backlog
            node.backlog -= served_backlog

            served_today = min(node.on_hand, demand)
            node.on_hand -= served_today
            unfilled = demand - served_today
            node.backlog += unfilled

            node_metrics[name] = {
                "demand": demand,
                "served": served_today,
                "backlog": node.backlog,
                "on_hand": node.on_hand,
                "order_placed": 0,
            }

        # Place orders according to policy
        for name, node in nodes.items():
            s_val, S_val = node.policy_s, node.policy_S
            stock_position = node.on_hand - node.backlog
            if outgoing.get(name):
                stock_position += sum(sum(q for _, q in pipeline[(name, dst)]) for dst in outgoing[name])
            if incoming.get(name):
                stock_position += sum(-lane_backlog[(src, name)] for src in incoming[name])

            if stock_position <= s_val:
                order_qty = max(0, int(S_val - stock_position))
                node_metrics[name]["order_placed"] = order_qty
                if order_qty > 0 and incoming.get(name):
                    per_lane = order_qty // len(incoming[name])
                    remainder = order_qty % len(incoming[name])
                    for src in incoming[name]:
                        qty = per_lane + (1 if remainder > 0 else 0)
                        if qty <= 0:
                            continue
                        if remainder > 0:
                            remainder -= 1
                        lane_backlog[(src, name)] += qty
                elif order_qty > 0 and not incoming.get(name):
                    node.backlog += order_qty

        # Fulfil lane backlogs based on source inventory
        for (src, dst), backlog_qty in lane_backlog.items():
            if backlog_qty <= 0:
                continue
            src_node = nodes[src]
            lane = lane_map[(src, dst)]
            qty = min(backlog_qty, src_node.on_hand)
            if lane.capacity is not None:
                qty = min(qty, int(lane.capacity))
            if qty <= 0:
                continue
            src_node.on_hand -= qty
            lane_backlog[(src, dst)] -= qty
            delay = _draw_lt_or_fixed(lane.lead_cfg, rng)
            pipeline[(src, dst)].append((day + max(1, int(delay)), qty))
            rows.append({
                "Day": day,
                "Event": "Shipment",
                "Lane": f"{src}->{dst}",
                "Quantity": qty,
            })

        # Cost tracking
        for name, node in nodes.items():
            costs = node.costs
            hold_cost = float(costs.get("holding_cost", 0.0)) * node.on_hand
            backlog_cost = float(costs.get("backlog_penalty", costs.get("stockout_penalty", 0.0))) * node.backlog
            row = {
                "Day": day,
                "Node": name,
                "Kind": node.kind,
                "OnHand": node.on_hand,
                "Backlog": node.backlog,
                "Demand": node_metrics[name]["demand"],
                "Served": node_metrics[name]["served"],
                "OrderPlaced": node_metrics[name]["order_placed"],
                "HoldingCost": hold_cost,
                "BacklogCost": backlog_cost,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def compute_network_kpis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Node" not in df.columns:
        return pd.DataFrame()
    pivot = df.groupby("Node").agg({
        "Demand": "sum",
        "Served": "sum",
        "OnHand": "mean",
        "Backlog": "mean",
        "HoldingCost": "sum",
        "BacklogCost": "sum",
    }).reset_index()
    pivot["FillRate"] = pivot.apply(lambda r: r["Served"] / r["Demand"] if r["Demand"] > 0 else 1.0, axis=1)
    return pivot


__all__ = [
    "SCENARIO_REGISTRY",
    "SimulationState",
    "run_once",
    "run_scenario",
    "compute_kpis",
    "compare_scenarios",
    "draw_demand",
    "draw_lead_time",
    "ensure_dir",
    "save_inventory_plot",
    "save_cost_plot",
    "apply_preset",
    "run_monte_carlo",
    "summarize_kpis",
    "grid_policy_search",
    "generate_narrative",
    "run_two_echelon",
    "compute_two_echelon_kpis",
    "optimize_policy_spsa",
    "counterfactual_policy_search",
    "run_network",
    "compute_network_kpis",
    "validate_config",
    "profile_run",
]
