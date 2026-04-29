
# bh-agent-model: Brock-Hommes Agent-Based Asset Pricing Model

This repository implements the Brock–Hommes Adaptive Belief System, an Agent-Based Modeling framework for behavioral finance research. It explores how competition between heterogeneous forecasting heuristics drives financial market dynamics.

## Project Structure

```text
├── src/bh_agent_model
│   ├── analysis/                         # Research & Experimentation
│   │   ├── bh_sensitivity_analysis/      # Global Sensitivity (Sobol indices)
│   │   ├── bh_strategy_simulation_base/  # Base simulation framework
│   │   ├── bh_strategy_simulation_real_data/ # Empirical replay (S&P 500)
│   │   └── bh_strategy_simulation_synthetic_data/ # Endogenous dynamics
│   ├── streamlit/
│   │   └── streamlit_app.py              # Interactive Streamlit presentation app
│   └── utils/
│       ├── base/
│       │   ├── agents.py                 # Trader types and demand rules
│       │   ├── markets.py                # Market clearing & switching
│       │   ├── math_ops.py               # Numerical helpers, including stable softmax
│       │   └── models.py                 # Main orchestrators
│       ├── helper/                       # Shared utility functions
│       └── load_data/                    # yfinance data ingestion
├── tests/                                # Testing Suite
│   └── unit/
│       ├── utils/base/
│       │   ├── test_agents.py            # Demand & forecast validation
│       │   └── test_markets.py           # Price clearing & softmax tests
│       └── test_dummy.py                 # CI/CD sanity checks
├── .gitignore                            # Environment & cache filters
├── .pre-commit-config.yaml               # Git hooks for code quality
├── .ruff.toml                            # Linting & formatting rules
├── conftest.py                           # Global test fixtures
├── pytest.ini                            # Pytest configuration
└── requirements.txt                      # Project dependencies

---
```

## Getting Started (For us users)

### 1. Clone the repository

```bash
git clone https://github.com/adaptive-markets-lab/bh-agent-model.git
cd bh-agent-model
```

### 2. Create environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -e .
```

### 4. Run the Streamlit app

After installing the package in editable mode:

```bash
streamlit run src/bh_agent_model/streamlit/streamlit_app.py
````
---

## Code Quality (Pre-commit + Ruff)

This project uses **pre-commit** with **Ruff** for linting and formatting.

### Install pre-commit

```bash
pip install pre-commit
```

### Install git hooks

```bash
pre-commit install
```

### Run checks manually

```bash
pre-commit run --all-files
```

### Update hooks

```bash
pre-commit autoupdate
```

---

## Development Workflow

Follow this workflow for all contributions.

### 1. Create an Issue

* Describe the feature, bug, or task
* Keep scope clear and concise

---

### 2. Create a Branch

Branch must correspond to the issue number.

```text
<issue-number>-<short-description>
```

#### Examples

```text
25-create-bh-strategy-simulation
23-refactor-agents
19-load-yfinance-data
```

```bash
git checkout main
git pull origin main
git checkout -b 25-create-bh-strategy-simulation
```

---

### 3. Make Changes

* Keep changes scoped to the issue
* Add/update tests if needed
* Ensure pre-commit passes

---

### 4. Commit Convention

Use the following format:

```text
<type> #<issue-number>: <message>
```

#### Types

* `feat` → new feature
* `fix` → bug fix
* `refactor` → code changes without behavior change
* `docs` → documentation
* `test` → tests
* `chore` → maintenance

#### Examples

```text
feat #25: add BH strategy simulation
fix #19: handle missing ticker data
refactor #23: simplify agent logic
docs #31: update README
test #12: add market tests
chore #18: update pre-commit config
```

---

### 5. Push Branch

```bash
git push origin 25-create-bh-strategy-simulation
```

---

### 6. Open Pull Request

* Link the issue (e.g., `Closes #25`)
* Describe changes clearly
* Request review

---

### 7. Merge

* Address review comments
* Ensure CI passes
* Merge after approval

---

## Docstring Style (Google)

All Python code must follow **Google-style docstrings**.

### Function Example

```python
def calculate_price(quantity: int, unit_price: float) -> float:
    """Calculate total price.

    Args:
        quantity: Number of units.
        unit_price: Price per unit.

    Returns:
        Total value.
    """
    return quantity * unit_price
```

### Class Example

```python
class Market:
    """Market environment.

    Args:
        name: Market name.
        tickers: Supported tickers.
    """

    def __init__(self, name: str, tickers: list[str]) -> None:
        self.name = name
        self.tickers = tickers
```

### Rules

* Use `"""` triple double quotes
* Start with a one-line summary
* Add a blank line before sections
* Use:

  * `Args:`
  * `Returns:`
  * `Raises:` (if needed)

---

## PR Checklist

Before opening a PR:

* Issue created
* Branch follows naming convention
* Code updated
* Tests added/updated (if needed)
* `pre-commit run --all-files` passes
* Commit messages follow convention
* Docstrings follow Google style
* PR linked to issue

---

