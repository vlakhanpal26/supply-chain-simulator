# engine/network.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import math

# -------------------- Data types --------------------
@dataclass
class LTModel:
    type: str                # "fixed" | "lognormal" | "discrete"
    p1: float                # for fixed: days; for lognormal: mu; for discrete: unused
    p2: float = 0.0          # for lognormal: sigma
    pmf: Optional[Dict[int, float]] = None  # for discrete

def draw_lead_time(rng: np.random.Generator, m: LTModel) -> int:
    if m.type == "fixed":
        return max(1, int(m.p1))
    elif m.type == "lognormal":
        val = int(round(rng.lognormal(mean=float(m.p1), sigma=float(m.p2))))
        return max(1, min(365, val))
    elif m.type == "discrete" and m.pmf:
        days = list(m.pmf.keys())
        probs = np.array([m.pmf[d] for d in days], dtype=float)
        probs /= probs.sum()
        return int(rng.choice(days, p=probs))
    else:
        raise ValueError("Unsupported lead time model")

def draw_demand(rng: np.random.Generator, dist: str, mu: float, sigma: float) -> int:
    if dist == "poisson":
        return int(rng.poisson(mu))
    # default normal
    return max(0, int(round(rng.normal(mu, sigma))))

# -------------------- Core engine --------------------
def run_network_once(
    config: dict,
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    items_df: pd.DataFrame,
    policies_df: pd.DataFrame,
    demand_df: Optional[pd.DataFrame] = None,   # optional fixed demand schedule
) -> pd.DataFrame:
    """
    Multi-echelon daily simulator (single item supported in starter).
    Echelons: supplier/factory/dc/customer.
    Policies: (s,S) at DC (starter).
    """
    rng = np.random.default_rng(int(config.get("seed", 42)))
    N_DAYS = int(config.get("N_DAYS", 30))
    demand_dist = config.get("Demand", {}).get("distribution", "normal")
    mu_d = float(config.get("Demand", {}).get("mu", 100))
    sigma_d = float(config.get("Demand", {}).get("sigma", 20))

    item = items_df.iloc[0]["item_id"]  # single SKU for now
    hold_cost = float(items_df.iloc[0]["holding_cost"])
    oos_penalty = float(items_df.iloc[0]["oos_penalty"])

    # State per node
    on_hand: Dict[str, int] = {}
    backorder: Dict[str, int] = {}
    on_order: Dict[str, int] = {}
    for _, r in nodes_df.iterrows():
        nid = str(r["node_id"]).strip()
        on_hand[nid] = int(r["init_on_hand"])
        backorder[nid] = 0
        on_order[nid] = 0

    # Lead time per arc
    lt: Dict[Tuple[str, str], LTModel] = {}
    for _, r in arcs_df.iterrows():
        a = str(r["from_id"]).strip(); b = str(r["to_id"]).strip()
        t = str(r["lt_type"]).strip().lower()
        if t == "discrete":
            # parse pmf from columns? in starter we stick to fixed
            pmf = None
            lt[(a,b)] = LTModel("discrete", 0.0, 0.0, pmf)
        elif t == "lognormal":
            lt[(a,b)] = LTModel("lognormal", float(r["lt_param1"]), float(r["lt_param2"]))
        else:
            lt[(a,b)] = LTModel("fixed", float(r["lt_param1"]))

    # Outstanding shipments: list of (arrival_day, to_node, qty)
    pipeline: List[Tuple[int, str, int]] = []

    # Policy (starter: only (s,S) at DC)
    pol = policies_df.iloc[0]
    policy_node = str(pol["node_id"]).strip()
    policy_type = str(pol["policy_type"]).strip()
    s = int(pol["s"]); S = int(pol["S"])

    logs = []
    for day in range(1, N_DAYS + 1):
        arrivals_today = 0
        orders_placed_today = 0

        # 1) Receive arrivals
        still = []
        for (arrive, to_node, qty) in pipeline:
            if arrive == day:
                on_hand[to_node] += qty
                arrivals_today += qty
                on_order[to_node] -= qty
            else:
                still.append((arrive, to_node, qty))
        pipeline = still

        # 2) Realize customer demand (only at customer nodes in starter)
        # If demand_df provided, use it; else draw from distribution and apply to all customers.
        dem_today = 0
        served_today = 0
        unfilled_today = 0

        if demand_df is not None and not demand_df.empty:
            dem_rows = demand_df[(demand_df["day"] == day) & (demand_df["node_id"].str.strip() == "C1")]
            if not dem_rows.empty:
                dem_today = int(dem_rows.iloc[0]["demand"])
            else:
                dem_today = draw_demand(rng, demand_dist, mu_d, sigma_d)
        else:
            dem_today = draw_demand(rng, demand_dist, mu_d, sigma_d)

        # Serve backorders at C1 first, then today’s demand
        cnode = "C1"
        # feed from DC to C1 if DC has stock: demand at customer is sourced from DC with 1-day arc DC->C1
        #   In this starter, we treat customer service directly from DC on-hand.
        # Serve BO at customer
        bo_serve = min(on_hand["DC"], backorder[cnode])
        on_hand["DC"] -= bo_serve
        backorder[cnode] -= bo_serve
        served_today += bo_serve

        # Serve today's demand
        serve_now = min(on_hand["DC"], dem_today)
        on_hand["DC"] -= serve_now
        served_today += serve_now
        unfilled = dem_today - serve_now
        backorder[cnode] += unfilled
        unfilled_today += unfilled

        # 3) Replenishment policy at DC → order from FAC
        stock_position = on_hand["DC"] + on_order["DC"] - backorder[cnode]
        if policy_type == "(s,S)" and stock_position <= s:
            order_qty = max(0, S - (on_hand["DC"] + on_order["DC"]))
            if order_qty > 0:
                L = draw_lead_time(rng, lt[("FAC","DC")])
                pipeline.append((day + L, "DC", order_qty))
                on_order["DC"] += order_qty
                orders_placed_today += order_qty

        # 4) Factory auto-requests from Supplier (push to keep FAC on-hand high)
        #   Simple rule: if FAC on-hand < DC on_order (use as proxy), request batch.
        fac_need = max(0, on_order["DC"] - on_hand["FAC"])
        if fac_need > 0:
            Lf = draw_lead_time(rng, lt[("SUP","FAC")])
            pipeline.append((day + Lf, "FAC", fac_need))
            on_order["FAC"] += fac_need

        # 5) Daily costs at DC (holding + customer stockout penalty)
        holding_cost = on_hand["DC"] * hold_cost
        stockout_cost = unfilled_today * oos_penalty

        logs.append({
            "Day": day,
            "Node": "DC",
            "Item": item,
            "Demand": dem_today,
            "Served": served_today,
            "Unfilled": unfilled_today,
            "OnHandEnd": on_hand["DC"],
            "BackorderEnd": backorder[cnode],
            "Arrivals": arrivals_today,
            "OrderPlaced": orders_placed_today,
            "HoldingCost": holding_cost,
            "StockoutCost": stockout_cost
        })

    df = pd.DataFrame(logs)

    # KPIs (single node focus for now)
    D_total = df["Demand"].sum()
    F_total = (df["Served"]).sum()
    days_stockout = int(((df["Unfilled"] > 0) | (df["BackorderEnd"] > 0)).sum())
    kpis = {
        "FR": 0.0 if D_total == 0 else F_total / D_total,
        "CSL": 1 - (days_stockout / len(df)),
        "AvgInventory": float(df["OnHandEnd"].mean()),
        "TotalHoldingCost": float(df["HoldingCost"].sum()),
        "TotalStockoutCost": float(df["StockoutCost"].sum()),
        "TotalCost": float((df["HoldingCost"] + df["StockoutCost"]).sum()),
        "StockoutDays": days_stockout,
        "OrdersPlacedCount": int((df["OrderPlaced"] > 0).sum())
    }
    return df, kpis
