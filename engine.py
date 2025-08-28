# engine.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Random draws
# ---------------------------
def draw_demand(rng: np.random.Generator, dist: str, mu: float, sigma: float) -> int:
    if dist.lower() == "normal":
        x = rng.normal(mu, sigma)
        return max(0, int(round(x)))
    elif dist.lower() == "poisson":
        x = rng.poisson(mu)
        return int(x)
    else:
        raise ValueError(f"Unsupported demand dist: {dist}")

def draw_lead_time(rng: np.random.Generator, lt_model: Dict) -> int:
    """
    lt_model examples:
      {"type":"lognormal","mu":1.609437912,"sigma":0.3,"Lmax":60}   # ln(5)=1.6094...
      {"type":"discrete","pmf":{3:0.2,5:0.6,7:0.2}}
    """
    t = lt_model["type"].lower()
    if t == "lognormal":
        mu_log = float(lt_model["mu"])
        sigma = float(lt_model["sigma"])
        Lmax = int(lt_model.get("Lmax", 60))
        val = int(round(rng.lognormal(mean=mu_log, sigma=sigma)))
        return max(1, min(Lmax, val))
    elif t == "discrete":
        pmf: Dict[int, float] = lt_model["pmf"]
        days = list(pmf.keys())
        probs = np.array([pmf[k] for k in days], dtype=float)
        probs = probs / probs.sum()
        return int(rng.choice(days, p=probs))
    else:
        raise ValueError(f"Unsupported LT model: {t}")

# ---------------------------
# Core run (single item, (s,S), backorders)
# ---------------------------
def run_once(cfg: Dict) -> pd.DataFrame:
    """
    Per-day log columns:
    Day | Demand | ServedToday | UnfilledToday | OnHandEnd | BackorderEnd |
    Arrivals | OrderPlaced | HoldingCost | StockoutCost
    """
    N = int(cfg["N_DAYS"])
    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    # Params
    d_cfg = cfg["Demand"]
    mu_d = float(d_cfg["mu"]); sigma_d = float(d_cfg["sigma"]); d_dist = d_cfg["distribution"]
    lt_model = cfg["LeadTime"]
    pol = cfg["Policy"]; s = int(pol["s"]); S = int(pol["S"])
    c_cfg = cfg["Costs"]; ch = float(c_cfg["holding_cost"]); co = float(c_cfg["stockout_penalty"])
    init = cfg["Initial"]; on_hand = int(init["inventory0"]); backorder = int(init.get("backorder0", 0))

    # State
    outstanding: List[Tuple[int, int]] = []  # (arrival_day, qty)
    on_order = 0
    rows = []

    for day in range(1, N + 1):
        # 1) arrivals
        arrivals_today = 0
        if outstanding:
            keep = []
            for arrive, q in outstanding:
                if arrive == day:
                    arrivals_today += q
                    on_order -= q
                else:
                    keep.append((arrive, q))
            outstanding = keep
            on_hand += arrivals_today

        # 2) demand
        d = draw_demand(rng, d_dist, mu_d, sigma_d)

        # 3) serve backorders first
        serve_bk = min(on_hand, backorder)
        on_hand -= serve_bk
        backorder -= serve_bk

        # 4) serve today's demand
        serve_now = min(on_hand, d)
        on_hand -= serve_now
        unfilled = d - serve_now
        backorder += unfilled

        # 5) reorder check using stock position
        stock_position = on_hand + on_order - backorder
        order_placed = 0
        if stock_position <= s:
            needed = S - (on_hand + on_order)
            if needed > 0:
                L = draw_lead_time(rng, lt_model)
                outstanding.append((day + L, int(needed)))
                on_order += int(needed)
                order_placed = int(needed)

        # 6) costs
        holding_cost_day = on_hand * ch
        stockout_cost_day = unfilled * co

        # 7) log
        rows.append({
            "Day": day,
            "Demand": int(d),
            "ServedToday": int(serve_now),
            "UnfilledToday": int(unfilled),
            "OnHandEnd": int(on_hand),
            "BackorderEnd": int(backorder),
            "Arrivals": int(arrivals_today),
            "OrderPlaced": int(order_placed),
            "HoldingCost": float(holding_cost_day),
            "StockoutCost": float(stockout_cost_day),
        })

    return pd.DataFrame(rows)

# ---------------------------
# KPIs
# ---------------------------
def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    D_total = int(df["Demand"].sum())
    served_total = int(df["ServedToday"].sum())
    stockout_days = int(((df["UnfilledToday"] > 0) | (df["BackorderEnd"] > 0)).sum())
    N = int(df["Day"].max())

    avg_inventory = float(df["OnHandEnd"].mean())
    total_holding = float(df["HoldingCost"].sum())
    total_stockout = float(df["StockoutCost"].sum())
    total_cost = total_holding + total_stockout

    FR = served_total / D_total if D_total > 0 else 1.0            # α-service (fill rate)
    CSL = 1.0 - (stockout_days / N if N > 0 else 0.0)              # β-service (day-based)

    return {
        "FR": FR,
        "CSL": CSL,
        "AvgInventory": avg_inventory,
        "TotalHoldingCost": total_holding,
        "TotalStockoutCost": total_stockout,
        "TotalCost": total_cost,
        "StockoutDays": stockout_days,
        "OrdersPlacedCount": int((df["OrderPlaced"] > 0).sum()),
        "D_total": D_total,
        "Served_total": served_total,
        "N_days": N,
    }

# ---------------------------
# Scenario helpers
# ---------------------------
def deep_update(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

def run_scenario(name: str, base_cfg: Dict, overrides: Optional[Dict] = None):
    cfg = deep_update(base_cfg, overrides or {})
    df = run_once(cfg)
    kpis = compute_kpis(df)
    kpis["Scenario"] = name
    return df, kpis

def compare_scenarios(results: List[Dict], baseline_name: str = "baseline") -> pd.DataFrame:
    k = pd.DataFrame(results)
    cols = ["Scenario","FR","CSL","AvgInventory","TotalHoldingCost","TotalStockoutCost","TotalCost","StockoutDays","OrdersPlacedCount"]
    k = k[cols]
    base = k[k["Scenario"] == baseline_name].iloc[0]
    comp = k.copy()
    for col in cols[1:]:
        comp[f"{col}_delta"] = comp[col] - float(base[col])
        comp[f"{col}_delta_pct"] = np.where(base[col] != 0, 100.0*comp[f"{col}_delta"]/float(base[col]), np.nan)
    return comp

# ---------------------------
# Plots
# ---------------------------
def plot_inventory(df: pd.DataFrame, title: str = "Inventory vs Day"):
    plt.figure()
    plt.plot(df["Day"], df["OnHandEnd"])
    plt.xlabel("Day"); plt.ylabel("On-hand (end)"); plt.title(title); plt.grid(True); plt.tight_layout()

def plot_demand_served(df: pd.DataFrame, title: str = "Demand vs Served"):
    plt.figure()
    plt.plot(df["Day"], df["Demand"], label="Demand")
    plt.plot(df["Day"], df["ServedToday"], label="ServedToday")
    plt.xlabel("Day"); plt.ylabel("Units"); plt.title(title); plt.grid(True); plt.legend(); plt.tight_layout()

def plot_cum_cost(df: pd.DataFrame, title: str = "Cumulative Cost"):
    plt.figure()
    cum = (df["HoldingCost"] + df["StockoutCost"]).cumsum()
    plt.plot(df["Day"], cum)
    plt.xlabel("Day"); plt.ylabel("Cumulative Cost"); plt.title(title); plt.grid(True); plt.tight_layout()
