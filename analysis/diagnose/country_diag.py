"""Manual country forecast diagnostic.

Edit the country/config block below, run this file, and inspect the Matplotlib
plots. Set SAVE_COUNTRY_PICKLE = True only after approving the plotted result.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("GVAR_IMPORT_ONLY", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "trp_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "trp_cache"))

import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIAG_DIR = ROOT / "analysis" / "diagnose"
DASH_OUTPUT_DIR = ROOT / "analysis" / "Dash_Output"
for path in (ROOT, DIAG_DIR, DASH_OUTPUT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import baseline_intervention_kf_diag as bid  # noqa: E402
import gvar_kf_forecast as gkf  # noqa: E402


def _activate_interactive_backend() -> None:
    """Use a GUI backend for this manual diagnostic script when available."""
    if os.environ.get("COUNTRY_DIAG_NONINTERACTIVE") == "1":
        return
    import matplotlib.pyplot as _plt

    if "agg" not in matplotlib.get_backend().lower():
        return

    preferred = os.environ.get("COUNTRY_DIAG_MPL_BACKEND")
    candidates = [preferred] if preferred else ["MacOSX", "TkAgg", "QtAgg"]
    for backend in candidates:
        if not backend:
            continue
        try:
            _plt.switch_backend(backend)
            return
        except Exception:
            continue


_activate_interactive_backend()
import matplotlib.pyplot as plt


# ============================================================================
# Manual configuration
# ============================================================================

COUNTRY = "THA"
CONFIG = {
    "data_vintage": "new",
    "external_variables": ["HeatDryF"],
    "drop_last_n": 0,
    "append_new_n": 0,  # rows appended before fitting, matching baseline/intervention semantics
}
COEFF_METHOD = "last"  # "last", "avg4", or "avg8"

MAX_EM_ITER = 8
EM_DAMPING = 0.1
P0_SCALE = 0.1
P0_FIXED = False
Q0_SCALE = 0.1
Q0_FIXED = False
R0_SCALE = 1
R0_FIXED = False

# Separate from coefficient fitting. When enabled, the fitted coefficients are
# kept fixed, then the forecast state is advanced through newer partially
# observed quarters using actual GDP/CPI/FX/EX values where available and model
# predictions only where a variable is missing.
USE_OBSERVED_FORWARD_INIT = True
OBSERVED_FORWARD_PANEL_VINTAGE = "new"  # "new" uses the latest panel observations

DISPLAY_VARS = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]

# Keep False while diagnosing. When True, the current fitted result replaces the
# selected scenario in Dash_Input/country_forecasts/{COUNTRY}.pkl.
SAVE_COUNTRY_PICKLE = False


EXPORT_SCENARIO = "approved"
COUNTRY_FORECAST_DIR = ROOT / "Dash_Input" / "country_forecasts"
LARGE_FORECAST_PICKLE = ROOT / "Dash_Input" / "gvar_forecast_results.pkl"


def _panel_path(vintage: str) -> Path:
    paths = {"old": bid.OLD_PANEL, "new": bid.NEW_PANEL}
    if vintage not in paths:
        raise ValueError(f"Unknown panel vintage={vintage!r}; expected old or new.")
    return paths[vintage]


def _set_bid_globals(country: str) -> None:
    bid.COUNTRY = country
    bid.MAX_EM_ITER = int(MAX_EM_ITER)
    bid.EM_DAMPING = float(EM_DAMPING)
    bid.P0_SCALE = float(P0_SCALE)
    bid.P0_FIXED = bool(P0_FIXED)
    bid.Q0_SCALE = float(Q0_SCALE)
    bid.Q0_FIXED = bool(Q0_FIXED)
    bid.R0_SCALE = float(R0_SCALE)
    bid.R0_FIXED = bool(R0_FIXED)


def _load_panels() -> dict[str, pd.DataFrame]:
    return {"old": bid._load_panel("old"), "new": bid._load_panel("new")}


def _prep_dict(prep: bid.PreparedRun) -> dict:
    return {
        "g": prep.g,
        "quarters": prep.quarters,
        "Yd": prep.y,
        "Xd": prep.x,
        "ENDO_use": prep.endo_use,
        "EXO_use": prep.exo_use,
    }


def _build_forecast_exog(country: str, exo_vars: list[str]) -> pd.DataFrame:
    known = set(getattr(gkf, "CLIMATE_TOGGLE_VARS", []))
    if set(exo_vars).issubset(known):
        return gkf.build_forecast_exo_df_multi(
            exo_vars,
            country,
            enso_map=gkf.FORECAST_ENSO_MEAN,
            target_quarters=gkf.FORECAST_TARGET_QUARTERS,
        )

    raw = pd.read_csv(bid.NEW_PANEL, usecols=lambda c: c in {"country", "quarter", *exo_vars})
    raw["quarter"] = pd.to_datetime(raw["quarter"]).dt.to_period("Q").dt.to_timestamp()
    hist = raw.loc[raw["country"].astype(str).eq(country)].groupby("quarter", as_index=False).first().set_index("quarter")
    rows = []
    for q in gkf.FORECAST_TARGET_QUARTERS:
        tq = pd.Timestamp(q).to_period("Q").to_timestamp()
        row = {"target_quarter": tq, "exo_quarter": tq}
        for var in exo_vars:
            val = np.nan
            source = "missing"
            if var in hist.columns and tq in hist.index:
                val = pd.to_numeric(pd.Series([hist.loc[tq, var]]), errors="coerce").iloc[0]
                if np.isfinite(val):
                    source = "panel"
            if not np.isfinite(val) and (var == "ENSO" or var.startswith("ENSO_")):
                val = gkf.FORECAST_ENSO_MEAN.get(str(tq.to_period("Q")), np.nan)
                if np.isfinite(val):
                    source = "forecast"
            row[var] = float(val) if np.isfinite(val) else np.nan
            row[f"{var}_source"] = source
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_coeff_method(fc: dict, coeff_method: str) -> dict:
    out = dict(fc)
    if coeff_method != "last":
        method_pack = fc.get("forecast_methods", {}).get(coeff_method)
        if not isinstance(method_pack, dict):
            raise RuntimeError(f"Forecast method {coeff_method!r} is unavailable")
        for key in ("y_hat", "y_hat_enso0", "y_hat_iod0", "y_hat_heat0", "heat0_var", "y_lower", "y_upper"):
            if key in method_pack:
                out[key] = method_pack[key]
    out["selected_coeff_method"] = coeff_method
    return out


def run_forecast(country: str = COUNTRY, config: dict = CONFIG, coeff_method: str = COEFF_METHOD) -> tuple[bid.PreparedRun, dict, dict]:
    _set_bid_globals(country)
    prep = bid._prepare_run("manual", dict(config), _load_panels())
    res = bid._run_kf_em_trace(prep)
    if res.get("status") != "ok":
        raise RuntimeError(f"{country} fit failed: {res.get('reason')}")

    exo_fc = _build_forecast_exog(country, prep.exo_use)
    previous_ragged_init = bool(gkf.RAGGED_EDGE_FORECAST_INIT)
    previous_panel_path = gkf.gp.PATH
    gkf.RAGGED_EDGE_FORECAST_INIT = bool(USE_OBSERVED_FORWARD_INIT)
    if USE_OBSERVED_FORWARD_INIT:
        gkf.gp.PATH = str(_panel_path(OBSERVED_FORWARD_PANEL_VINTAGE))
    try:
        fc = gkf.forecast_country_from_em(
            _prep_dict(prep),
            res,
            exo_fc,
            country=country,
            scenario_name="mean",
        )
    finally:
        gkf.RAGGED_EDGE_FORECAST_INIT = previous_ragged_init
        gkf.gp.PATH = previous_panel_path
    if fc is None:
        raise RuntimeError(f"No forecast returned for {country}")
    return prep, res, _apply_coeff_method(fc, coeff_method)


def _raw_y(fc: dict, arr) -> np.ndarray:
    return np.asarray(arr, dtype=float) * np.asarray(fc["y_sd"]) + np.asarray(fc["y_mu"])


def _history_frame(prep: bid.PreparedRun, country: str, var: str, fc: dict | None = None) -> pd.DataFrame:
    if fc is not None and var in list(fc.get("ENDO_use", [])) and fc.get("hist_y_raw") is not None:
        j = list(fc["ENDO_use"]).index(var)
        return pd.DataFrame(
            {
                "quarter": pd.to_datetime(fc.get("hist_quarters", []), errors="coerce"),
                "value": np.asarray(fc["hist_y_raw"], dtype=float)[:, j],
            }
        ).dropna(subset=["quarter", "value"])

    panel = prep.g.copy()
    if var not in panel.columns:
        return pd.DataFrame(columns=["quarter", "value"])
    return pd.DataFrame(
        {
            "quarter": pd.to_datetime(panel["quarter"], errors="coerce"),
            "value": pd.to_numeric(panel[var], errors="coerce"),
        }
    ).dropna(subset=["quarter", "value"])


def plot_dashboard_like_forecast(prep: bid.PreparedRun, fc: dict, country: str = COUNTRY) -> None:
    endo = list(fc["ENDO_use"])
    vars_use = [v for v in DISPLAY_VARS if v in endo]
    quarters = pd.to_datetime(fc["fc_quarters"])
    y = _raw_y(fc, fc["y_hat"])
    y_lower = _raw_y(fc, fc["y_lower"]) if fc.get("y_lower") is not None else None
    y_upper = _raw_y(fc, fc["y_upper"]) if fc.get("y_upper") is not None else None

    cf_series = []
    if fc.get("y_hat_enso0") is not None:
        cf_series.append(("No ENSO", _raw_y(fc, fc["y_hat_enso0"])))
    if fc.get("y_hat_iod0") is not None:
        cf_series.append(("No IOD", _raw_y(fc, fc["y_hat_iod0"])))
    if fc.get("y_hat_heat0") is not None:
        heat_label = fc.get("heat0_var") or "HeatDry"
        cf_series.append((f"No {heat_label}", _raw_y(fc, fc["y_hat_heat0"])))

    fig, axes = plt.subplots(len(vars_use), 1, figsize=(12, 3.0 * len(vars_use)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, var in zip(axes, vars_use):
        j = endo.index(var)
        hist = _history_frame(prep, country, var, fc)
        if not hist.empty:
            ax.plot(hist["quarter"], hist["value"], color="#555555", linewidth=1.4, label="observed/completed history")
        ax.plot(quarters, y[:, j], color="C0", marker="o", linewidth=1.8, label="forecast")
        if y_lower is not None and y_upper is not None:
            ax.fill_between(quarters, y_lower[:, j], y_upper[:, j], color="C0", alpha=0.16, label="95% CI")
        for label, arr in cf_series:
            ax.plot(quarters, arr[:, j], linestyle="--", linewidth=1.5, label=label)
        ax.axvline(pd.Timestamp(fc["hist_last_quarter"]), color="#777777", linestyle=":", linewidth=1.0)
        ax.set_title(f"{country} {var}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(
        f"{country} manual diagnostic | exo={'+'.join(fc['EXO_use'])} | coeff={fc.get('selected_coeff_method', COEFF_METHOD)}",
        y=0.995,
    )
    fig.tight_layout()


def print_summary(fc: dict) -> None:
    y = _raw_y(fc, fc["y_hat"])
    q = pd.to_datetime(fc["fc_quarters"])
    rows = []
    for j, var in enumerate(fc["ENDO_use"]):
        rows.append(
            {
                "variable": var,
                "forecast_2026_mean": float(np.nanmean(y[q.year == 2026, j])) if np.any(q.year == 2026) else np.nan,
                "forecast_last": float(y[-1, j]),
                "hist_last_quarter": pd.Timestamp(fc["hist_last_quarter"]).to_period("Q").strftime("%YQ%q"),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


def save_country_pickle(fc: dict, country: str = COUNTRY, scenario: str = EXPORT_SCENARIO) -> Path:
    """Write one Core-country forecast pickle for dashboard loading."""
    COUNTRY_FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COUNTRY_FORECAST_DIR / f"{country}.pkl"

    country_bundle = None
    if out_path.exists():
        with out_path.open("rb") as f:
            country_bundle = pickle.load(f)
    if not isinstance(country_bundle, dict) or country_bundle.get("country") != country:
        country_bundle = {
            "format": "core_country_forecast_v1",
            "country": country,
            "source_pickle": str(LARGE_FORECAST_PICKLE),
            "source_format": None,
            "source_active_scenario": scenario,
            "forecasts": {},
        }

    if LARGE_FORECAST_PICKLE.exists():
        try:
            with LARGE_FORECAST_PICKLE.open("rb") as f:
                large = pickle.load(f)
            if isinstance(large, dict):
                country_bundle.setdefault("source_pickle", str(LARGE_FORECAST_PICKLE))
                country_bundle["source_format"] = large.get("format")
                country_bundle["source_active_scenario"] = large.get("active_scenario", scenario)
        except Exception:
            pass

    export_fc = dict(fc)
    export_fc["scenario_name"] = scenario
    export_fc["manual_config"] = dict(CONFIG)
    export_fc["selected_coeff_method"] = fc.get("selected_coeff_method", COEFF_METHOD)
    export_fc["use_observed_forward_init"] = bool(USE_OBSERVED_FORWARD_INIT)
    export_fc["observed_forward_panel_vintage"] = OBSERVED_FORWARD_PANEL_VINTAGE

    forecasts = dict(country_bundle.get("forecasts") or {})
    forecasts[scenario] = export_fc
    country_bundle["format"] = "core_country_forecast_v1"
    country_bundle["country"] = country
    country_bundle["forecasts"] = forecasts

    with out_path.open("wb") as f:
        pickle.dump(country_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


if __name__ == "__main__":
    prep_, res_, fc_ = run_forecast()
    print_summary(fc_)
    plot_dashboard_like_forecast(prep_, fc_)
    if SAVE_COUNTRY_PICKLE:
        saved_path = save_country_pickle(fc_)
        print(f"Saved dashboard country pickle: {saved_path}")
    plt.show()
