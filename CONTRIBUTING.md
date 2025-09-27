# Contributing

Thanks for considering a contribution! This project blends research and
portfolio work, so we try to keep the repo in a clean, reproducible state.

## Getting Started
1. Create/activate the working environment (conda recommended):
   ```bash
   conda create -n scsim python=3.12
   conda activate scsim
   pip install -r requirements.txt
   ```
2. Run the regression tests:
   ```bash
   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
   ```
3. Generate the benchmark artifacts (optional, but keeps the outputs current):
   ```bash
   python run_phase2.py
   ```

## Guidelines
- Keep dependencies in `requirements.txt`; `reportlab` is mandatory for PDF
  output and tests.
- Stick to a clean, comprehensible code style—Black defaults are fine. Add
  docstrings for new modules/functions.
- Include tests for new behaviour. Seeded short-horizon simulations make good
  regression tests.
- Update the README or configs when you introduce new features or parameters.

## Pull Requests
- Create a topic branch from `main` (e.g., `feature/spsa-tuning`).
- Keep commits tidy; squash or rebase if you end up with noisy fixups.
- Describe the change, include manual/automated test results, and attach
  screenshots/GIFs for UI updates.
- Wait for tests to pass before requesting review.

Thanks again—happy building!
