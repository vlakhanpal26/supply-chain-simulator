# app/app.py — Supply Chain Simulator (polished dashboard)
import sys, math, io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ----- import engine from repo root -----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Your Phase-2 engine API
from engine import (
    apply_preset,
    compare_scenarios,
    compute_kpis,
    compute_two_echelon_kpis,
    generate_narrative,
    grid_policy_search,
    counterfactual_policy_search,
    optimize_policy_spsa,
    run_monte_carlo,
    run_scenario,
    run_two_echelon,
)

# ---------- Page + global style ----------
st.set_page_config(page_title="Supply Chain Simulator", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
* { font-family: Inter, -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
.hero{
  background: linear-gradient(180deg,#FFFFFF 0%,#F7F9FC 100%);
  border:1px solid #E5E7EB; border-radius:22px; padding:18px 22px; margin-bottom:14px;
  box-shadow:0 6px 18px rgba(18,38,63,.06);
}
.hero h1{ margin:0; font-weight:700; color:#0B1220; letter-spacing:.2px;}
.hero p{ margin:.25rem 0 0; color:#4B5563;}
.card{
  background:#FFFFFF; border:1px solid #E5E7EB; border-radius:18px; padding:16px 18px;
  box-shadow:0 2px 10px rgba(15,23,42,.04);
}
.kpi .label{ color:#6B7280; font-size:12px; letter-spacing:.02em; }
.kpi .value{ color:#0B1220; font-size:26px; font-weight:700; }
.stTabs [data-baseweb="tab"]{ background:#FFFFFF; border:1px solid #E5E7EB; border-radius:12px; padding:10px 14px; margin-right:6px;}
.stTabs [data-baseweb="tab-list"]{ gap: 6px; }
.small-muted{ color:#6B7280; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ---------- session state ----------
if "mc_results" not in st.session_state:
    st.session_state["mc_results"] = None  # (summary_df, runs)
if "advisor_results" not in st.session_state:
    st.session_state["advisor_results"] = None
if "network_results" not in st.session_state:
    st.session_state["network_results"] = None
if "advisor_spsa" not in st.session_state:
    st.session_state["advisor_spsa"] = None
if "advisor_counterfactual" not in st.session_state:
    st.session_state["advisor_counterfactual"] = None

# ---------- helpers ----------
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def clean_name(s: str) -> str:
    # Pretty scenario names (no underscores, title case)
    return s.replace("_"," ").strip().title()

def plot_inventory(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Day"], y=df["OnHandEnd"], mode="lines", name="On-hand", line=dict(width=3)))
    o = df.query("OrderPlaced>0"); a = df.query("Arrivals>0")
    if not o.empty:
        fig.add_trace(go.Scatter(x=o["Day"], y=o["OnHandEnd"], mode="markers", name="Order placed",
                                 marker=dict(symbol="triangle-up", size=10)))
    if not a.empty:
        fig.add_trace(go.Scatter(x=a["Day"], y=a["OnHandEnd"], mode="markers", name="Delivery arrived",
                                 marker=dict(symbol="diamond", size=9)))
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=10,r=10,t=60,b=10),
                      title=title, legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def plot_demand_served(df: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["Day"], y=df["Demand"], name="Demand", opacity=0.45))
    fig.add_trace(go.Scatter(x=df["Day"], y=df["ServedToday"], name="Served", mode="lines", line=dict(width=3)))
    fig.update_layout(template="plotly_white", barmode="overlay", height=320,
                      margin=dict(l=10,r=10,t=60,b=10), title=title, legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def plot_cumulative_cost(df: pd.DataFrame, title: str):
    cum_h = df["HoldingCost"].cumsum()
    cum_s = df["StockoutCost"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Day"], y=cum_h, name="Cum Holding", mode="lines", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=df["Day"], y=cum_s, name="Cum Stockout", mode="lines", line=dict(width=3, dash="dot")))
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=10,r=10,t=60,b=10),
                      title=title, legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def kpi_strip(k: dict):
    metrics = [
        ("Fill Rate (FR)", f"{k['FR']:.3f}"),
        ("Service Level (CSL)", f"{k['CSL']:.3f}"),
        ("Avg Inventory", f"{k['AvgInventory']:.1f}"),
        ("Avg Backorder", f"{k.get('AvgBackorder', 0.0):.1f}"),
        ("Total Cost", f"{k.get('TotalCost', k.get('TotalHoldingCost',0)+k.get('TotalStockoutCost',0)):.0f}"),
    ]
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(f'<div class="card kpi"><div class="label">{label}</div>'
                        f'<div class="value">{value}</div></div>', unsafe_allow_html=True)

# ---------- header ----------
st.markdown('<div class="hero"><h1>Supply Chain Simulator</h1>'
            '<p>Baseline & scenarios with clean KPIs, cost curves, and inventory dynamics.</p></div>', unsafe_allow_html=True)

# ---------- SIDEBAR (inputs) ----------
with st.sidebar:
    st.header("Inputs")

    # Demand
    mu = st.number_input("Demand mean (μ)", min_value=0.0, value=100.0, step=1.0)
    sigma = st.number_input("Demand std (σ)", min_value=0.0, value=20.0, step=1.0)
    dist = st.selectbox("Demand distribution", ["Normal","Poisson","Uniform"], index=0)

    # Lead time
    lt_model = st.selectbox("Lead-time model", ["Fixed","Variable (lognormal)","From table (discrete)"], index=0)
    lt_fixed = st.number_input("If fixed: days", min_value=1, value=5, step=1)
    lt_mean  = st.number_input("If variable: mean days", min_value=1.0, value=5.0, step=0.5)
    lt_cv    = st.number_input("If variable: CV (std/mean)", min_value=0.0, value=0.30, step=0.05)

    # Policy & Costs
    s_val = st.number_input("Reorder point (s)", min_value=0, value=200, step=10)
    S_val = st.number_input("Order-up-to (S)", min_value=0, value=500, step=10)
    hold  = st.number_input("Holding cost ($/u/day)", min_value=0.0, value=1.0, step=0.1)
    oos   = st.number_input("Stockout penalty ($/u)", min_value=0.0, value=10.0, step=1.0)
    days  = st.slider("Horizon (days)", min_value=7, max_value=365, value=60, step=1)
    seed  = st.number_input("Random seed", min_value=0, value=42, step=1)

    st.caption("Tip: If FR < 0.8, raise s or S (watch holding cost).")

    st.markdown("---")
    st.subheader("Scenarios")
    presets = {
        "Supplier delay (+2 days LT)":"supplier_delay_plus2",
        "Demand +15% (days 8–14)":"demand_spike_15pct",
        "Raise (s,S) by +100":"policy_raise_sS",
    }
    chosen = st.multiselect("Select presets to compare", list(presets.keys()))
    run_base = st.button("Run Baseline")
    run_scen = st.button("Run Selected Scenarios")
    mc_runs = st.number_input("Monte Carlo runs", min_value=20, max_value=2000, value=200, step=20,
                              help="Number of Monte Carlo replications for risk stats.")
    run_mc = st.button("Run Monte Carlo")
    st.markdown("---")
    st.subheader("Policy advisor")
    advisor_target = st.number_input("Service target (FR)", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
    advisor_span = st.number_input("Search span (± units)", min_value=10, max_value=1000, value=200, step=10,
                                   help="Generate candidates within ± span of current s and S.")
    advisor_step = st.number_input("Step size", min_value=5, max_value=200, value=50, step=5)
    advisor_objective = st.selectbox("Objective", ["total_cost","inventory","stockout","fill_rate"], index=0)
    advisor_mc = st.number_input("Advisor MC runs", min_value=0, max_value=1000, value=50, step=10,
                                 help="Set >1 to evaluate each policy via Monte Carlo.")
    run_advisor = st.button("Run Grid Advisor")
    spsa_iters = st.number_input("SPSA iterations", min_value=5, max_value=200, value=40, step=5)
    spsa_runs = st.number_input("SPSA MC runs", min_value=1, max_value=500, value=30, step=5)
    run_spsa = st.button("Run SPSA Optimizer")
    counter_target = st.number_input("Counterfactual target FR", min_value=0.5, max_value=1.0, value=0.95, step=0.01)
    counter_step = st.number_input("Counterfactual step", min_value=1, max_value=200, value=25, step=5)
    run_counter = st.button("Run Counterfactual Lift")
    st.markdown("---")
    st.subheader("Two-echelon network")
    store_lt_network = st.number_input("Store lead time from DC (days)", min_value=1, max_value=30, value=3, step=1)
    store_init = st.number_input("Store initial on-hand", min_value=0, max_value=5000, value=300, step=50)
    dc_s_val = st.number_input("DC reorder point", min_value=0, max_value=10000, value=900, step=50)
    dc_S_val = st.number_input("DC order-up-to", min_value=0, max_value=20000, value=1600, step=50)
    dc_lt_network = st.number_input("DC lead time from supplier", min_value=1, max_value=60, value=6, step=1)
    dc_hold = st.number_input("DC holding cost ($/u/day)", min_value=0.0, value=0.5, step=0.1)
    dc_penalty = st.number_input("DC backlog penalty ($/u)", min_value=0.0, value=8.0, step=0.5)
    run_network = st.button("Run Two-Echelon Simulation")

if S_val < s_val:
    st.error("Order-up-to level **S** must be ≥ **s**.")

# ---------- build baseline config dict ----------
cfg = {
    "N_DAYS": int(days),
    "Demand": {"distribution": dist.lower(), "mu": float(mu), "sigma": float(sigma)},
    "LeadTime": (
        {"type":"fixed","days":int(lt_fixed)} if lt_model=="Fixed" else
        {"type":"lognormal","mu": float(math.log(lt_mean) - 0.5*math.log(1+lt_cv**2)),
         "sigma": float(math.sqrt(math.log(1+lt_cv**2))), "Lmax": 60} if lt_model.startswith("Variable") else
        {"type":"discrete","pmf": {"3":0.2,"5":0.6,"7":0.2}}
    ),
    "Policy": {"s": int(s_val), "S": int(S_val)},
    "Costs": {"holding_cost": float(hold), "stockout_penalty": float(oos)},
    "Initial": {"on_hand": 500, "backorder": 0},
    "seed": int(seed),
}

# ---------- MAIN ----------
tabs = st.tabs(["Overview","Performance","Daily Log","Compare","Risk","Advisor","Network"])

df_base = None
kpi_base = None

if run_base:
    df_base, kpi_base = run_scenario(cfg)
    kpi_base["Scenario"] = "Baseline"
    st.session_state["mc_results"] = None
    st.session_state["advisor_results"] = None
    st.session_state["network_results"] = None
    st.session_state["advisor_spsa"] = None
    st.session_state["advisor_counterfactual"] = None
    with tabs[0]:
        st.success("Baseline run completed.")
        kpi_strip(kpi_base)
        plot_inventory(df_base, "Inventory Position & Events — Baseline")
        c1, c2 = st.columns([1,1])
        with c1:
            plot_cumulative_cost(df_base, "Cumulative Cost Breakdown — Baseline")
        with c2:
            plot_demand_served(df_base, "Customer Demand vs Orders Fulfilled — Baseline")
        st.download_button("Download Baseline Log (CSV)", to_csv_bytes(df_base), "baseline_log.csv", "text/csv")

    with tabs[1]:
        # rolling fill (7-day)
        rr = df_base["ServedToday"].rolling(7).sum() / df_base["Demand"].rolling(7).sum().replace(0,np.nan)
        fig = px.line(x=df_base["Day"], y=rr)
        fig.update_traces(name="7-day Fill", line=dict(width=3))
        fig.update_layout(template="plotly_white", yaxis=dict(range=[0,1.05]), title="Rolling Fill (7-day)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tabs[2]:
        st.dataframe(df_base, use_container_width=True, height=420)

if run_scen:
    # ensure baseline exists
    if df_base is None or kpi_base is None:
        df_base, kpi_base = run_scenario(cfg)
        kpi_base["Scenario"] = "Baseline"
    st.session_state["mc_results"] = None
    st.session_state["advisor_results"] = None
    st.session_state["network_results"] = None
    st.session_state["advisor_spsa"] = None
    st.session_state["advisor_counterfactual"] = None
    all_kpis = [dict(kpi_base)]
    overlays = {}

    st.subheader("Scenario Runs")
    for label in chosen:
        key = presets[label]
        scen_cfg = apply_preset(cfg, key)
        df_s, kpi_s = run_scenario(scen_cfg)
        kpi_s["Scenario"] = label
        overlays[label] = df_s

        st.markdown(f"#### {label}")
        kpi_strip(kpi_s)
        with st.expander(f"Charts — {label}", expanded=False):
            plot_inventory(df_s, f"Inventory — {label}")
            plot_demand_served(df_s, f"Demand vs Served — {label}")
            plot_cumulative_cost(df_s, f"Cumulative Cost — {label}")
        st.download_button(f"Download Log (CSV) — {label}", to_csv_bytes(df_s),
                           file_name=f"{key}_log.csv", mime="text/csv")

        all_kpis.append(dict(kpi_s))

    with tabs[3]:
        comp = compare_scenarios(all_kpis, baseline_name="Baseline").copy()
        # Show only the useful columns and round for readability
        keep = ["Scenario","FR","CSL","AvgInventory","AvgBackorder","TotalHoldingCost","TotalStockoutCost",
                "TotalCost","StockoutDays","OrdersPlacedCount",
                "FR_delta","CSL_delta","TotalCost_delta","StockoutDays_delta","AvgBackorder_delta"]
        comp = comp[[c for c in keep if c in comp.columns]]
        for c in comp.columns:
            if c != "Scenario":
                comp[c] = pd.to_numeric(comp[c], errors="ignore")
        comp = comp.round(3)
        st.dataframe(comp, use_container_width=True)
        st.download_button("Download KPI Comparison (CSV)", to_csv_bytes(comp), "kpi_compare.csv", "text/csv")
        if len(all_kpis) > 1 and kpi_base is not None:
            st.markdown("**Scenario Insights**")
            for row in all_kpis[1:]:
                story = generate_narrative(kpi_base, row)
                st.markdown(f"- **{row['Scenario']}** — {story['headline']}")
                for bullet in story["details"]:
                    st.markdown(f"  • {bullet}")

if run_mc:
    with st.spinner("Running Monte Carlo simulations..."):
        mc_stats_frames = []
        mc_runs_dict = {}

        base_runs, base_stats = run_monte_carlo(cfg, n_runs=int(mc_runs), base_seed=int(seed), scenario_name="Baseline")
        mc_runs_dict["Baseline"] = base_runs
        mc_stats_frames.append(base_stats)

        for label in chosen:
            key = presets[label]
            scen_cfg = apply_preset(cfg, key)
            seed_offset = (abs(hash(key)) % (2**31)) + int(seed)
            scen_runs, scen_stats = run_monte_carlo(
                scen_cfg,
                n_runs=int(mc_runs),
                base_seed=int(seed_offset),
                scenario_name=label,
            )
            mc_runs_dict[label] = scen_runs
            mc_stats_frames.append(scen_stats)

        mc_summary = pd.concat(mc_stats_frames, ignore_index=True)
        st.session_state["mc_results"] = {"summary": mc_summary, "runs": mc_runs_dict}

if run_advisor:
    if kpi_base is None:
        df_base, kpi_base = run_scenario(cfg)
        kpi_base["Scenario"] = "Baseline"
    span = max(0, int(advisor_span))
    step = max(1, int(advisor_step))
    s_center = int(s_val)
    S_center = int(S_val)
    s_candidates = list(range(max(0, s_center - span), s_center + span + step, step))
    S_candidates = list(range(max(s_center, S_center - span), S_center + span + step, step))
    if not s_candidates or not S_candidates:
        st.warning("No candidate policies generated; adjust span/step.")
    else:
        with st.spinner("Evaluating policies..."):
            advisor_df = grid_policy_search(
                cfg,
                s_candidates,
                S_candidates,
                service_target=float(advisor_target) if advisor_target > 0 else None,
                objective=advisor_objective,
                n_runs=int(advisor_mc) if advisor_mc > 1 else None,
                base_seed=int(seed),
            )
        st.session_state["advisor_results"] = {
            "results": advisor_df,
            "baseline": kpi_base,
        }

if run_spsa:
    if kpi_base is None:
        df_base, kpi_base = run_scenario(cfg)
        kpi_base["Scenario"] = "Baseline"
    with st.spinner("Running SPSA optimisation..."):
        spsa_out = optimize_policy_spsa(
            cfg,
            objective=advisor_objective,
            service_target=float(advisor_target),
            n_runs=int(spsa_runs),
            iterations=int(spsa_iters),
            seed=int(seed),
        )
    st.session_state["advisor_spsa"] = spsa_out

if run_counter:
    if kpi_base is None:
        df_base, kpi_base = run_scenario(cfg)
        kpi_base["Scenario"] = "Baseline"
    with st.spinner("Searching counterfactual policy..."):
        counter_out = counterfactual_policy_search(
            cfg,
            target_fr=float(counter_target),
            step=int(counter_step),
            overrides={},
            n_runs=int(max(1, min(30, spsa_runs))),
        )
    st.session_state["advisor_counterfactual"] = counter_out

if run_network:
    network_cfg = {
        "N_DAYS": int(days),
        "seed": int(seed),
        "Demand": {"distribution": dist.lower(), "mu": float(mu), "sigma": float(sigma)},
        "Nodes": [
            {
                "name": "Supplier",
                "type": "supplier",
                "Policy": {"s": 0, "S": int(max(dc_S_val * 2, 5000))},
                "Initial": {"on_hand": int(max(dc_S_val * 2, 10000)), "backorder": 0},
                "Costs": {"holding_cost": 0.1},
            },
            {
                "name": "DC",
                "type": "dc",
                "Policy": {"s": int(dc_s_val), "S": int(dc_S_val)},
                "Initial": {"on_hand": int(dc_S_val), "backorder": 0},
                "Costs": {"holding_cost": float(dc_hold), "backlog_penalty": float(dc_penalty)},
            },
            {
                "name": "Store",
                "type": "store",
                "Policy": {"s": int(s_val), "S": int(S_val)},
                "Initial": {"on_hand": int(store_init), "backorder": 0},
                "Costs": {"holding_cost": float(hold), "backlog_penalty": float(oos)},
            },
        ],
        "Lanes": [
            {"from": "Supplier", "to": "DC", "LeadTime": {"type": "fixed", "days": int(dc_lt_network)}},
            {"from": "DC", "to": "Store", "LeadTime": {"type": "fixed", "days": int(store_lt_network)}},
        ],
    }
    with st.spinner("Simulating network..."):
        net_df = run_network(network_cfg)
        net_kpis = compute_network_kpis(net_df)
    st.session_state["network_results"] = {
        "config": network_cfg,
        "log": net_df,
        "kpis": net_kpis,
    }

with tabs[4]:
    st.subheader("Monte Carlo Risk Insights")
    mc_state = st.session_state.get("mc_results")
    if not mc_state:
        st.info("Run Monte Carlo to see risk metrics.")
    else:
        summary = mc_state["summary"]
        focus_stats = summary[summary["stat"].isin(["mean", "std", "q05", "q95", "risk"])].copy()
        focus_stats = focus_stats.reset_index(drop=True)
        st.dataframe(focus_stats, use_container_width=True)
        st.download_button(
            "Download Monte Carlo Summary (CSV)",
            to_csv_bytes(focus_stats),
            "monte_carlo_summary.csv",
            "text/csv",
        )

        risk_row = focus_stats[focus_stats["stat"] == "risk"].iloc[0] if (focus_stats["stat"] == "risk").any() else None
        if risk_row is not None:
            c_cols = st.columns(3)
            if "TotalCost_var95" in risk_row:
                c_cols[0].metric("Cost VaR 95%", f"{risk_row['TotalCost_var95']:.0f}")
            if "TotalCost_cvar95" in risk_row:
                c_cols[1].metric("Cost CVaR 95%", f"{risk_row['TotalCost_cvar95']:.0f}")
            if "FR_prob_below_90" in risk_row:
                c_cols[2].metric("FR < 90% prob", f"{risk_row['FR_prob_below_90']*100:.1f}%")

        runs_dict = mc_state["runs"]
        scenario_choice = st.selectbox("Scenario", list(runs_dict.keys()))
        metric_choice = st.selectbox("Metric", ["FR", "CSL", "TotalCost", "StockoutDays"])
        run_df = runs_dict[scenario_choice]
        fig = px.histogram(run_df, x=metric_choice, nbins=40,
                           title=f"Distribution of {metric_choice} — {scenario_choice}")
        fig.update_layout(template="plotly_white", height=320)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.download_button(
            f"Download Runs (CSV) — {scenario_choice}",
            to_csv_bytes(run_df),
            f"{scenario_choice.lower().replace(' ', '_')}_mc_runs.csv",
            "text/csv",
        )

        mean_stats = summary[summary["stat"] == "mean"].copy()
        if "Scenario" in mean_stats.columns and mean_stats["Scenario"].nunique() > 1 and "TotalCost" in mean_stats.columns:
            baseline_mask = mean_stats["Scenario"].str.lower() == "baseline"
            if baseline_mask.any():
                baseline_cost = float(mean_stats[baseline_mask].iloc[0]["TotalCost"])
                delta = mean_stats.copy()
                delta["CostDelta"] = delta["TotalCost"] - baseline_cost
                fig_delta = px.bar(
                    delta[~baseline_mask],
                    x="Scenario",
                    y="CostDelta",
                    title="Mean Total Cost delta vs Baseline",
                    color="CostDelta",
                    color_continuous_scale="RdBu"
                )
                fig_delta.update_layout(template="plotly_white", height=340)
                st.plotly_chart(fig_delta, use_container_width=True, config={"displayModeBar": False})

with tabs[5]:
    st.subheader("Policy Advisor")
    advisor_state = st.session_state.get("advisor_results")
    if not advisor_state:
        st.info("Configure span and objectives in the sidebar, then run the advisor.")
    else:
        advisor_df = advisor_state["results"].copy()
        if advisor_df.empty:
            st.warning("Advisor produced no feasible policies. Expand the grid or relax constraints.")
        else:
            top_view = advisor_df.head(10).round(3)
            st.dataframe(top_view, use_container_width=True)
            st.download_button(
                "Download Advisor Results (CSV)",
                to_csv_bytes(advisor_df),
                "policy_advisor_results.csv",
                "text/csv",
            )
            baseline_kpi = advisor_state["baseline"]
            best_row = advisor_df.iloc[0]
            rec_policy = {
                "Scenario": "Advisor",
                "FR": best_row.get("FR"),
                "CSL": best_row.get("CSL"),
                "TotalCost": best_row.get("TotalCost"),
                "AvgInventory": best_row.get("AvgInventory"),
                "AvgBackorder": best_row.get("AvgBackorder"),
                "StockoutDays": best_row.get("StockoutDays", 0),
            }
            story = generate_narrative(baseline_kpi, rec_policy, service_target=float(advisor_target))
            st.markdown(f"**Recommendation:** s = {int(best_row['s'])}, S = {int(best_row['S'])}")
            st.markdown(f"*{story['headline']}*")
            for bullet in story["details"]:
                st.markdown(f"- {bullet}")

    spsa_state = st.session_state.get("advisor_spsa")
    if spsa_state:
        st.markdown("---")
        st.markdown("### SPSA Optimiser")
        best = spsa_state.get("best", {})
        if best:
            st.markdown(f"**Best policy:** s = {best['s']}, S = {best['S']} (objective {best['score']:.1f})")
            best_kpi = best.get("kpis", {})
            cols = st.columns(3)
            cols[0].metric("FR", f"{best_kpi.get('FR', float('nan')):.3f}")
            cols[1].metric("CSL", f"{best_kpi.get('CSL', float('nan')):.3f}")
            cols[2].metric("Total cost", f"{best_kpi.get('TotalCost', float('nan')):.0f}")
        history_df = spsa_state.get("history")
        if isinstance(history_df, pd.DataFrame) and not history_df.empty:
            st.line_chart(history_df.set_index("iter")["objective"], height=220)
            st.download_button(
                "Download SPSA History (CSV)",
                to_csv_bytes(history_df),
                "advisor_spsa_history.csv",
                "text/csv",
            )

    counter_state = st.session_state.get("advisor_counterfactual")
    if counter_state:
        st.markdown("---")
        st.markdown("### Counterfactual Lift")
        status = counter_state.get("status")
        policy = counter_state.get("policy", {})
        st.markdown(f"Status: **{status}** · Proposed policy s = {policy.get('s')}, S = {policy.get('S')}")
        counter_kpi = counter_state.get("kpis", {})
        cols = st.columns(3)
        cols[0].metric("FR", f"{counter_kpi.get('FR', float('nan')):.3f}")
        cols[1].metric("CSL", f"{counter_kpi.get('CSL', float('nan')):.3f}")
        cols[2].metric("Total cost", f"{counter_kpi.get('TotalCost', float('nan')):.0f}")
        iterations = counter_state.get("iterations", [])
        if iterations:
            df_iter = pd.DataFrame(iterations)
            st.dataframe(df_iter, use_container_width=True, height=220)
            st.download_button(
                "Download Counterfactual Trace (CSV)",
                to_csv_bytes(df_iter),
                "advisor_counterfactual.csv",
                "text/csv",
            )

with tabs[6]:
    st.subheader("Two-Echelon Results")
    network_state = st.session_state.get("network_results")
    if not network_state:
        st.info("Set network parameters in the sidebar and run the simulation.")
    else:
        kpis_df = network_state["kpis"]
        if isinstance(kpis_df, pd.DataFrame) and not kpis_df.empty:
            st.dataframe(kpis_df.round(3), use_container_width=True, height=200)
        else:
            st.warning("No KPIs computed for the network run.")

        net_df = network_state["log"]
        st.dataframe(net_df.tail(30), use_container_width=True, height=300)
        st.download_button(
            "Download Two-Echelon Log (CSV)",
            to_csv_bytes(net_df),
            "two_echelon_log.csv",
            "text/csv",
        )

st.caption("Designed with Streamlit theming + Plotly templates. Charts use a consistent white template for clarity.")
