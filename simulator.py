"""
Phase 1 — Single-echelon (s, S) simulator
One item, one DC, fixed lead time, demand ~ Normal(mu, sigma)
Outputs: daily log CSV + inventory plot
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Config you can tweak ----
CFG = {
    "N_DAYS": 30,
    "Demand": {"distribution": "normal", "mu": 100, "sigma": 20},
    "LeadTime": {"days": 2},             # fixed lead time
    "Policy": {"s": 575, "S": 1200},      # reorder point & order-up-to
    "Initial": {"on_hand": 500, "backorder": 0},
    "seed": 42,
    "Costs": {"holding_cost": 1.0, "stockout_penalty": 10.0},
}

# ---- Engine ----
@dataclass
class State:
    on_hand: int
    backorder: int
    on_order: int

def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))

def draw_demand(cfg: dict, r: np.random.Generator) -> int:
    mu = float(cfg["Demand"].get("mu", 100))
    sigma = float(cfg["Demand"].get("sigma", 20))
    return max(0, int(round(r.normal(mu, sigma))))

def run_once(cfg: dict) -> pd.DataFrame:
    days = int(cfg.get("N_DAYS", 30))
    L = int(cfg["LeadTime"].get("days", 5))
    s = int(cfg["Policy"]["s"]); S = int(cfg["Policy"]["S"])
    seed = int(cfg.get("seed", 42))
    assert S >= s, "Policy invalid: S must be >= s"

    r = rng(seed)
    st = State(on_hand=int(cfg["Initial"]["on_hand"]),
               backorder=int(cfg["Initial"]["backorder"]),
               on_order=0)

    pos: List[Tuple[int, int]] = []  # outstanding POs: (arrival_day, qty)
    rows = []
    hold_c = float(cfg["Costs"]["holding_cost"])
    oos_c  = float(cfg["Costs"]["stockout_penalty"])

    for day in range(1, days + 1):
        # 1) Receive arrivals
        arrivals_today = 0
        if pos:
            keep = []
            for arrive, q in pos:
                if arrive == day:
                    st.on_hand += q
                    arrivals_today += q
                    st.on_order -= q          # CRITICAL: decrement on_order on arrival
                else:
                    keep.append((arrive, q))
            pos = keep

        # 2) Realize demand
        d = draw_demand(cfg, r)

        # 3) Serve backorders first
        serve_bk = min(st.on_hand, st.backorder)
        st.on_hand -= serve_bk
        st.backorder -= serve_bk

        # 4) Serve today
        serve_now = min(st.on_hand, d)
        st.on_hand -= serve_now
        unfilled = d - serve_now
        st.backorder += unfilled

        # 5) Reorder (s,S)
        stock_position = st.on_hand + st.on_order - st.backorder
        order_placed = 0
        if stock_position <= s:
            order_qty = S - (st.on_hand + st.on_order)
            if order_qty > 0:
                pos.append((day + L, order_qty))
                st.on_order += order_qty
                order_placed = order_qty

        # 6) Costs + log
        hold_cost = st.on_hand * hold_c
        stockout_cost = unfilled * oos_c
        rows.append({
            "Day": day,
            "Demand": d,
            "ServedToday": serve_now,
            "UnfilledToday": unfilled,
            "OnHandEnd": st.on_hand,
            "BackorderEnd": st.backorder,
            "Arrivals": arrivals_today,
            "OrderPlaced": order_placed,
            "HoldingCost": hold_cost,
            "StockoutCost": stockout_cost
        })

    return pd.DataFrame(rows)

def compute_kpis(df: pd.DataFrame) -> dict:
    D = float(df["Demand"].sum())
    served = float(df["ServedToday"].sum())
    FR = (served / D) if D > 0 else 1.0
    days = int(df["Day"].max())
    days_stockout = int(((df["UnfilledToday"] > 0) | (df["BackorderEnd"] > 0)).sum())
    CSL = 1.0 - (days_stockout / days) if days > 0 else 1.0
    return {
        "FR": FR,
        "CSL": CSL,
        "AvgInventory": float(df["OnHandEnd"].mean()),
        "OrdersPlacedCount": int((df["OrderPlaced"] > 0).sum()),
    }

def plot_inventory(df: pd.DataFrame, title: str = "Inventory vs Day — Phase 1"):
    import matplotlib
    # Recommended for macOS if GUI issues: matplotlib.use("TkAgg")
    plt.figure(figsize=(10, 4))
    plt.plot(df["Day"], df["OnHandEnd"], lw=2, label="On-hand")
    o = df[df["OrderPlaced"] > 0]
    a = df[df["Arrivals"] > 0]
    if not o.empty:
        plt.scatter(o["Day"], o["OnHandEnd"], marker="^", s=60, label="Order placed")
    if not a.empty:
        plt.scatter(a["Day"], a["OnHandEnd"], marker="D", s=50, label="Delivery arrived")
    plt.xlabel("Day"); plt.ylabel("Units"); plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.show()

def main():
    df = run_once(CFG)
    k = compute_kpis(df)
    print("KPIs:", {k1: (round(v, 3) if isinstance(v, float) else v) for k1, v in k.items()})
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "baseline_log.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved log → {out_path}")
    plot_inventory(df)

if __name__ == "__main__":
    main()
