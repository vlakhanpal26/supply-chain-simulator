# simulator.py
import math
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_inventory(
    N: int = 60,            # number of days
    mu: float = 100.0,      # mean daily demand
    sigma: float = 20.0,    # std dev of daily demand
    initial_inventory: int = 500,
    s: int = 200,           # reorder point
    Q: int = 500,           # order quantity
    L: int = 3,             # lead time (days)
    seed: int = 42
) -> pd.DataFrame:
    """
    Simple (s, Q) lost-sales simulator with normal demand.
    Follows your steps exactly:
      - Step 2: demand ~ Normal(mu, sigma) per day
      - Step 3: subtract demand from inventory; if stock < demand, log stockout
                 if inventory <= s → place order (arrives after lead time L)
      - Step 4: track outstanding orders and deliver on arrival day
      - Step 5: log to DataFrame [Day, Inventory, Demand, OrdersPlaced, Deliveries]
                Inventory here is END-OF-DAY inventory.
    """
    rng = np.random.default_rng(seed)
    inventory = int(initial_inventory)
    outstanding: List[Tuple[int, int]] = []  # (arrival_day, qty)
    records = []

    for day in range(1, N + 1):
        # Step 4: check arrivals
        deliveries_today = 0
        if outstanding:
            keep = []
            for arrival_day, qty in outstanding:
                if arrival_day == day:
                    deliveries_today += qty
                else:
                    keep.append((arrival_day, qty))
            outstanding = keep
            inventory += deliveries_today

        # Step 2: generate demand (clip at 0 and round to int)
        demand = max(0, int(round(rng.normal(mu, sigma))))

        # Step 3: fulfill demand (lost sales if not enough)
        sales = min(inventory, demand)
        stockout_qty = max(0, demand - inventory)
        inventory -= sales  # leftover after sales

        # place an order if at/below reorder point
        orders_placed = 0
        if inventory <= s:
            arrival = day + L
            outstanding.append((arrival, Q))
            orders_placed = Q

        # Step 5: log results (Inventory = end-of-day)
        records.append({
            "Day": day,
            "Inventory": int(inventory),
            "Demand": int(demand),
            "OrdersPlaced": int(orders_placed),
            "Deliveries": int(deliveries_today),
            # optional debug you can uncomment later:
            # "StockoutQty": int(stockout_qty),
        })

    return pd.DataFrame(records)

def plot_inventory(df: pd.DataFrame) -> None:
    """Plot Inventory vs Time using Matplotlib."""
    plt.figure()
    plt.plot(df["Day"], df["Inventory"])
    plt.xlabel("Day")
    plt.ylabel("Inventory (end-of-day)")
    plt.title("Inventory vs Time")
    plt.grid(True)
    plt.tight_layout()

if __name__ == "__main__":
    # quick demo run if you execute: python simulator.py
    data = simulate_inventory(
        N=60, mu=100, sigma=20, initial_inventory=500, s=200, Q=500, L=3, seed=42
    )
    print(data.head())
    # To see the plot when running as a script, uncomment:
    # import matplotlib.pyplot as plt
    # plot_inventory(data)
    # plt.show()





    
