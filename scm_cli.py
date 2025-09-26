#!/usr/bin/env python3
"""Command-line runner for Supply Chain Simulator."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from engine import run_scenario, run_monte_carlo, summarize_kpis


def load_config(path: Path) -> dict:
    text = path.read_text()
    data = json.loads(text)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Supply Chain Simulator CLI")
    parser.add_argument("config", type=Path, help="Path to JSON config file")
    parser.add_argument("--monte-carlo", type=int, default=0, help="Number of MC runs")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    if args.monte_carlo and args.monte_carlo > 1:
        runs_df, stats_df = run_monte_carlo(cfg, n_runs=args.monte_carlo, base_seed=args.seed)
        print("Monte Carlo summary:")
        print(stats_df.round(3).to_string(index=False))
    else:
        df, kpis = run_scenario(cfg)
        print("KPIs:")
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in kpis.items()})
        out_path = Path("outputs/cli_run.csv")
        out_path.parent.mkdir(exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved log → {out_path}")
if __name__ == "__main__":
    main()
