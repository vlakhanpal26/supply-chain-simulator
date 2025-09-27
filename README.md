# Supply Chain Simulator

A research-grade playground for analysing stochastic supply chains. It supports
single-echelon (s, S) policies, backorders, multi-echelon networks, scenario
analysis, Monte Carlo risk metrics, and automated policy search—exposed via a
Streamlit dashboard, CLI, and batch report workflow.

## Features
- **Simulation Engines** – deterministic or random demand, stochastic lead
times, backlog handling, supplier redemption, and multi-node networks.
- **Analytics & Reporting** – KPI tables, risk metrics (probabilities,
VaR/CVaR), Monte Carlo ensembles, tornado summaries, and PDF reports with
charts + narrative findings.
- **Policy Advisory** – grid search, SPSA optimisation, counterfactual "lift"
that suggests how far to adjust (s,S) to hit target service levels.
- **User Interfaces** – Streamlit app, batch script (`run_phase2.py`), and a
lightweight CLI (`scm_cli.py`).
- **Artifacts** – CSV logs, KPI comparisons, risk summaries, advisor histories,
and ready-to-share PDF decks.

## Quick Start
```bash
git clone <repo-url>
cd Supply\ Chain\ Simulator
conda create -n scsim python=3.12
conda activate scsim
pip install -r requirements.txt
```

### CLI
```bash
python scm_cli.py configs/example_baseline.json
python scm_cli.py configs/example_baseline.json --monte-carlo 250 --seed 42
```

### Streamlit Dashboard
```bash
streamlit run app/app.py
```

### Batch Reporting
```bash
python run_phase2.py
open outputs/phase2_report.pdf
```

## Project Layout
```
app/              Streamlit UI
engine_core.py    Simulation core (single, two-echelon, network)
engine.py         Modern wrapper + public API
engine2.py        Legacy wrapper for phase-2 notebooks/scripts
run_phase2.py     Batch scenario runner + reporting
report.py         ReportLab PDF generator
scm_cli.py        Command line entry point
tests/            Automated regression/unit tests
outputs/          Generated artifacts (logs, KPIs, reports)
configs/          Example JSON configs
```

## Configuration
- Minimum keys for single-echelon runs:
  ```json
  {
    "N_DAYS": 30,
    "Demand": {"distribution": "normal", "mu": 100, "sigma": 20},
    "LeadTime": {"type": "discrete", "pmf": {"3":0.2,"5":0.6,"7":0.2}},
    "Policy": {"s": 575, "S": 1200},
    "Initial": {"on_hand": 500, "backorder": 0},
    "Costs": {"holding_cost": 1.0, "stockout_penalty": 10.0},
    "seed": 42
  }
  ```
- Networks specify nodes/lanes; see `engine_core.run_network` docstring or the
sample config below.

## Development
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
python run_phase2.py
streamlit run app/app.py
```
- Add dependencies to `requirements.txt`; `reportlab` is required for PDF
generation.
- Follow standard Black/flake8 style (add `pyproject.toml` if desired).

## Roadmap
- [x] Single-echelon simulator
- [x] Monte Carlo analytics & PDF reporting
- [x] Streamlit dashboard
- [x] Two-echelon (supplier→DC→store)
- [x] Network simulator + SPSA advisor
- [ ] Multi-item and capacity-aware networks
- [ ] Automated optimisation across scenarios
- [ ] CI pipeline & coverage reporting
```
