# TRP Time-Varying VAR (TVP-VAR) Application

## Overview

This repository implements a **time-varying parameter VAR (TVP-VAR)** framework using a **Kalman filter**, with an interactive **Streamlit application** as the primary interface.

The application is designed to explore how macroeconomic relationships (e.g., GDP, inflation, exchange rates) respond to **exogenous shocks** (such as climate or ENSO indicators), while allowing those relationships to **evolve over time** in a controlled and interpretable way.

At present:

- The **Streamlit app** is the primary entry point (`streamlit run apps/streamlit_TRP.py`)
- The macro panel **`data/gvar_panel_streamlit.csv`** is a **fixed input** (not rebuilt in this repo)
- VAR / TVP-VAR estimation for the main tabs runs in memory from that CSV
- Structural-break views use precomputed pickle / LLM artifacts under `structural_break/` and `Dash_Input/`

See `data/README.md` for input file expectations.

---

## Key Features

- Fixed VAR and time-varying VAR (TVP-VAR) estimation
- Kalman-filter–based coefficient evolution
- Strong anchoring of TVP-VAR to a fixed VAR baseline
- Scenario-based impulse response functions (GIRFs) for exogenous shocks
- Interactive comparison of fixed VAR vs TVP-VAR results
- Clear separation between numerical core, analysis logic, and user interface

---

## Project Structure

```
project_root/
│
├── tvp/                     # Core numerical routines (math only)
│   ├── __init__.py
│   ├── kalman_var.py        # Kalman filter / TVP-VAR implementation
│   └── irf.py               # IRF utilities (companion form)
│
├── analysis/                # Model orchestration & analysis logic
│   ├── __init__.py
│   ├── config.py            # Variable definitions & configuration
│   ├── estimate_var.py      # Fixed VAR and TVP-VAR estimation
│   ├── scenario_irf.py      # Scenario-based GIRFs
│   └── results.py           # Result container classes (TVPVARResult)
│   └── rolling_average.py   # compare truth with rolling average
│
├── analysis/validation/
│   ├── __init__.py
│   ├── simulate_data.py      # all synthetic generators
│   ├── run_simulation.py     # all model runners
│   └── compare_betas.py      # all comparisons & plots
│
├── apps/                    # Interactive applications
│   └── streamlit_TRP.py     # Main Streamlit app (entry point)
│
└── README.md
```

### Architectural principles

- `tvp/` contains **only numerical code** (no pandas, no statsmodels, no Streamlit)
- `analysis/` contains orchestration logic (pandas, statsmodels, scenarios)
- `apps/` contains presentation and UI logic only
- Lower layers never import from higher layers (to avoid circular dependencies)

---

## How to Run the Application

### Prerequisites

You will need Python 3.9+ and the following packages:

- numpy
- pandas
- scipy
- statsmodels
- streamlit

(If a `requirements.txt` file is added later, this section can be updated.)

---

### Run the Streamlit App

From the project root directory:

```bash
streamlit run apps/streamlit_TRP.py
```

This launches the interactive application in your browser.

All results are computed dynamically during execution and displayed in the app.

---

## What This Project Does *Not* Do (Yet)

- No CSV or file-based outputs are written to disk
- No automated batch estimation scripts
- No command-line interface
- No scheduled or reproducible pipelines

These may be added in the future, but the current design prioritizes **interactive exploration**.

---

## Design Notes

### Fixed VAR as an Anchor

The TVP-VAR implementation is intentionally **anchored to a fixed VAR**:

- Fixed VAR coefficients are used as the initial state
- The estimated covariance of the VAR coefficients is used as a prior
- Kalman filter hyperparameters are chosen to allow **small, controlled deviations**

This ensures that TVP-VAR results:
- closely resemble fixed VAR results in stable periods
- diverge only when the data consistently support regime-specific changes

---

### Sensitivity to Hyperparameters

TVP-VAR results depend on Kalman filter hyperparameters, including:

- Forgetting factor (`λ`)
- Measurement noise (`R`)
- Prior covariance scaling

This sensitivity is expected and reflects weak identification in near-unit-root macroeconomic systems.  
The application is designed to make this sensitivity **explicit and controllable**, rather than hidden.

---

## Debugging and Development

- The Streamlit app serves as the primary **integration test**
- Some analysis modules are **not intended to be run as standalone scripts**
- Circular-import errors during refactoring are expected until module boundaries stabilize
- If the Streamlit app runs correctly, the overall architecture is functionally consistent

---

## Current Status

- Core Kalman / TVP-VAR implementation complete
- Streamlit app running successfully
- Modular architecture in place
- No persistent outputs yet (by design)
- README is a first-pass and will evolve with the project

---

## Possible Extensions

- Batch scripts that write CSV outputs
- Simulation-based unit tests
- Export of IRFs and forecasts
- Formal documentation of Kalman calibration choices
- Policy-ready reporting outputs

---

## Notes

This repository is under active development.  
The current emphasis is on **correctness, transparency, and architectural clarity**, rather than production automation.
