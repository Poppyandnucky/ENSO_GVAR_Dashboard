"""Manual country forecast configuration and approved-pickle writer.

This file intentionally does not run candidate screens. It is now a small
orchestration layer around the baseline/intervention diagnostic fitter and the
production forecast/pickle schema.
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


# ============================================================================
# Manual single-country run
# ============================================================================

COUNTRY = "CHL"
CONFIG = {
    "mode": "baseline_intervention",
    "intervention": {
        "data_vintage": "new",
        "external_variables": ["HeatDry"],
        "drop_last_n": 0,
        "append_new_n": 0,
    },
    "max_em_iter": 10,
    "p0_scale": 0.01,
    "p0_fixed": False,
    "q0_scale": 0.5,
    "q0_fixed": False,
    "r0_scale": 2.0,
    "r0_fixed": False,
    "coeff_method": "last",
}
SAVE_RESULT = False


# ============================================================================
# Final approved country configs
# ============================================================================

DEFAULT_OLD_FIT_SCALES = {
    "p0_scale": 0.01,
    "q0_scale": 0.5,
    "r0_scale": 2.0,
    "q0_fixed": False,
    "r0_fixed": False,
}

OLD_FIT_COUNTRIES = {
    "BRA": {"external_variables": ["ENSO"], "lag": 1, "model": "KF", "coeff_method": "last", "max_em_iter": 10, "update_p0": True, "append_new_n": 1},
    "COL": {"external_variables": ["HeatDry"], "lag": 1, "model": "KF", "coeff_method": "last", "max_em_iter": 10, "update_p0": True, "append_new_n": 1},
    "MEX": {"external_variables": ["HeatDry"], "lag": 1, "model": "KF", "coeff_method": "last", "max_em_iter": 10, "update_p0": False, "append_new_n": 1},
    "EGY": {"external_variables": ["ENSO"], "lag": 1, "model": "KF", "coeff_method": "last", "max_em_iter": 10, "update_p0": True, "append_new_n": 1},
    "IND": {"external_variables": ["ENSO"], "lag": 1, "model": "KF", "coeff_method": "avg4", "max_em_iter": 10, "update_p0": True, "append_new_n": 1},
    "THA": {"external_variables": ["HeatDry", "OIL_YoY"], "lag": 1, "model": "KF", "coeff_method": "avg4", "max_em_iter": 10, "update_p0": True, "append_new_n": 1},
    "IDN": {"external_variables": ["HeatDry"], "lag": 1, "model": "KF", "coeff_method": "last", "max_em_iter": 10, "update_p0": False, "append_new_n": 1},
    "PER": {"external_variables": ["HeatDry", "OIL_YoY"], "lag": 1, "model": "KF", "coeff_method": "avg4", "max_em_iter": 10, "update_p0": False, "append_new_n": 1},
}

APPROVED_COUNTRY_SPECS = {
    **{
        country: {
            "mode": "old_fit_newest_quarter_init",
            "data_vintage": "old",
            "external_variables": spec["external_variables"],
            "lag": spec["lag"],
            "model": spec["model"],
            "coeff_method": spec["coeff_method"],
            "max_em_iter": spec["max_em_iter"],
            "append_new_n": spec.get("append_new_n", 1),
            "p0_fixed": not bool(spec["update_p0"]),
            **DEFAULT_OLD_FIT_SCALES,
        }
        for country, spec in OLD_FIT_COUNTRIES.items()
    },
    "CHL": {
        "mode": "baseline_intervention",
        "intervention": {"data_vintage": "new", "external_variables": ["HeatDry"], "drop_last_n": 0, "append_new_n": 0},
        "max_em_iter": 10,
        "p0_scale": 0.01,
        "p0_fixed": False,
        "q0_scale": 0.5,
        "q0_fixed": False,
        "r0_scale": 2.0,
        "r0_fixed": False,
        "coeff_method": "last",
    },
    "KEN": {
        "mode": "baseline_intervention",
        "intervention": {"data_vintage": "new", "external_variables": ["HeatDry", "OIL_YoY"], "drop_last_n": 0, "append_new_n": 0},
        "max_em_iter": 10,
        "p0_scale": 0.1,
        "p0_fixed": False,
        "q0_scale": 0.5,
        "q0_fixed": False,
        "r0_scale": 2.0,
        "r0_fixed": False,
        "coeff_method": "last",
    },
    "PHL": {
        "mode": "baseline_intervention",
        "intervention": {"data_vintage": "old", "external_variables": ["HeatDry", "OIL_YoY"], "drop_last_n": 0, "append_new_n": 1},
        "max_em_iter": 10,
        "p0_scale": 0.01,
        "p0_fixed": True,
        "q0_scale": 0.1,
        "q0_fixed": False,
        "r0_scale": 2.0,
        "r0_fixed": False,
        "coeff_method": "last",
    },
    # ZAF was not included in the pasted 11-country config. This preserves the
    # previously approved simple refit choice so the dashboard pickle remains complete.
    "ZAF": {
        "mode": "baseline_intervention",
        "intervention": {"data_vintage": "new", "external_variables": ["HeatDryYoY"], "drop_last_n": 0, "append_new_n": 0},
        "max_em_iter": 8,
        "p0_scale": 0.01,
        "p0_fixed": False,
        "q0_scale": 0.1,
        "q0_fixed": False,
        "r0_scale": 2.0,
        "r0_fixed": False,
        "coeff_method": "last",
    },
}

COUNTRIES_TO_RUN = ["BRA", "CHL", "COL", "MEX", "KEN", "ZAF", "IND", "IDN", "THA", "PER", "PHL", "EGY"]

APPROVED_PICKLE_PATH = ROOT / "Dash_Input" / "gvar_forecast_results.pkl"


_SAVE_COUNTRY_FIELDS = {
    "ENDO_use",
    "EXO_use",
    "fc_quarters",
    "hist_last_quarter",
    "gap_quarters",
    "period_type",
    "scenario_start_quarter",
    "y_mu",
    "y_sd",
    "y_hat",
    "y_hat_enso0",
    "y_hat_iod0",
    "y_hat_heat0",
    "heat0_var",
    "y_lower",
    "y_upper",
    "forecast_methods",
    "exo_forecast",
    "enso_scenario",
    "iod_scenario",
    "climate_variant",
    "setting",
    "approved_country_spec",
    "selected_coeff_method",
    "approved_settings",
}

_SAVE_METHOD_FIELDS = {
    "method",
    "theta_window",
    "y_hat",
    "y_hat_enso0",
    "y_hat_iod0",
    "y_hat_heat0",
    "heat0_var",
    "y_lower",
    "y_upper",
}


def _set_bid_globals(country: str, spec: dict) -> None:
    bid.COUNTRY = country
    bid.MAX_EM_ITER = int(spec["max_em_iter"])
    bid.P0_SCALE = float(spec["p0_scale"])
    bid.P0_FIXED = bool(spec["p0_fixed"])
    bid.Q0_SCALE = float(spec["q0_scale"])
    bid.Q0_FIXED = bool(spec["q0_fixed"])
    bid.R0_SCALE = float(spec["r0_scale"])
    bid.R0_FIXED = bool(spec["r0_fixed"])


def _load_panels() -> dict[str, pd.DataFrame]:
    return {"old": bid._load_panel("old"), "new": bid._load_panel("new")}


def _fit_from_config(country: str, sample_config: dict, spec: dict, panels: dict[str, pd.DataFrame]) -> tuple[bid.PreparedRun, dict]:
    _set_bid_globals(country, spec)
    prep = bid._prepare_run(spec.get("fit_name", "fit"), sample_config, panels)
    res = bid._run_kf_em_trace(prep)
    if res.get("status") != "ok":
        raise RuntimeError(f"{country} fit failed: {res.get('reason')}")
    return prep, res


def _prep_dict(prep: bid.PreparedRun) -> dict:
    return {
        "g": prep.g,
        "quarters": prep.quarters,
        "Yd": prep.y,
        "Xd": prep.x,
        "ENDO_use": prep.endo_use,
        "EXO_use": prep.exo_use,
    }


def _apply_coeff_method(fc: dict, coeff_method: str) -> dict:
    out = dict(fc)
    if coeff_method != "last":
        method_pack = fc.get("forecast_methods", {}).get(coeff_method)
        if not isinstance(method_pack, dict):
            raise RuntimeError(f"Forecast method {coeff_method!r} is unavailable")
        for key in ("y_hat", "y_hat_enso0", "y_hat_iod0", "y_hat_heat0", "heat0_var", "y_lower", "y_upper", "forecast_stability"):
            if key in method_pack:
                out[key] = method_pack[key]
    out["selected_coeff_method"] = coeff_method
    return out


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


def _forecast_from_fit(
    country: str,
    prep: bid.PreparedRun,
    res: dict,
    spec: dict,
    *,
    ragged_panel_path: Path | None,
    ragged_max_quarters: int | None,
) -> dict:
    exo_fc = _build_forecast_exog(country, prep.exo_use)
    fc = gkf.forecast_country_from_em(
        _prep_dict(prep),
        res,
        exo_fc,
        country=country,
        scenario_name="mean",
        ragged_panel_path=ragged_panel_path,
        ragged_max_quarters=ragged_max_quarters,
    )
    if fc is None:
        raise RuntimeError(f"No forecast returned for {country}")
    fc = _apply_coeff_method(fc, str(spec.get("coeff_method", "last")))
    fc["setting"] = _setting_label(country, spec)
    fc["approved_country_spec"] = True
    fc["approved_settings"] = dict(spec)
    return fc


def _setting_label(country: str, spec: dict) -> str:
    exo = "+".join(spec.get("external_variables") or spec.get("intervention", {}).get("external_variables", []))
    return (
        f"{country} | {spec['mode']} | {exo} | coeff={spec.get('coeff_method', 'last')} | "
        f"EM={spec['max_em_iter']} | P0_FIXED={spec['p0_fixed']} | "
        f"Q0_FIXED={spec['q0_fixed']} | R0_FIXED={spec['r0_fixed']}"
    )


def _run_one(country: str, spec: dict, panels: dict[str, pd.DataFrame] | None = None) -> dict:
    panels = panels or _load_panels()
    mode = spec["mode"]
    if mode == "old_fit_newest_quarter_init":
        sample_config = {
            "data_vintage": "old",
            "external_variables": list(spec["external_variables"]),
            "drop_last_n": 0,
            "append_new_n": 0,
        }
        prep, res = _fit_from_config(country, sample_config, spec, panels)
        return _forecast_from_fit(
            country,
            prep,
            res,
            spec,
            ragged_panel_path=bid.NEW_PANEL,
            ragged_max_quarters=int(spec.get("append_new_n", 1)),
        )

    if mode == "baseline_intervention":
        sample_config = dict(spec["intervention"])
        fit_spec = dict(spec, external_variables=list(sample_config["external_variables"]), fit_name="intervention")
        prep, res = _fit_from_config(country, sample_config, fit_spec, panels)
        return _forecast_from_fit(country, prep, res, fit_spec, ragged_panel_path=None, ragged_max_quarters=0)

    raise ValueError(f"Unknown mode for {country}: {mode!r}")


def _prune_for_dashboard(fc: dict) -> dict:
    source = dict(fc)
    coeff_method = source.get("selected_coeff_method", "last")
    if coeff_method != "last":
        method_pack = fc.get("forecast_methods", {}).get(coeff_method)
        if isinstance(method_pack, dict):
            source.update({k: v for k, v in method_pack.items() if k in _SAVE_METHOD_FIELDS})
    out = {k: v for k, v in source.items() if k in _SAVE_COUNTRY_FIELDS}
    methods = {}
    for name, pack in fc.get("forecast_methods", {}).items():
        if isinstance(pack, dict):
            methods[name] = {k: v for k, v in pack.items() if k in _SAVE_METHOD_FIELDS}
    out["forecast_methods"] = methods
    out["approved_country_spec"] = True
    out["climate_variant"] = out.get("climate_variant") or "+".join(out.get("EXO_use", []))
    return out


def save_selected_country_result(country: str, fc: dict, pickle_path: Path = APPROVED_PICKLE_PATH) -> Path:
    if pickle_path.exists():
        with pickle_path.open("rb") as f:
            bundle = pickle.load(f)
        if not isinstance(bundle, dict):
            bundle = {}
    else:
        bundle = {}

    scenarios = bundle.setdefault("scenarios", {})
    approved = scenarios.setdefault(
        "approved",
        {
            "config": {
                "format": "approved_country_forecasts_v1",
                "description": "Country-by-country approved forecasts",
            },
            "per_country": {},
        },
    )
    approved.setdefault("config", {})["format"] = "approved_country_forecasts_v1"
    approved.setdefault("per_country", {})[country] = _prune_for_dashboard(fc)
    approved["exo_forecast"] = fc.get("exo_forecast")

    bundle["format"] = "approved_country_forecasts_v1"
    bundle["active_scenario"] = "approved"
    bundle["climate_variants"] = {
        "approved": {
            "label": "Approved country-specific specifications",
            "scenarios": {"approved": approved},
        }
    }
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with pickle_path.open("wb") as f:
        pickle.dump(bundle, f)
    return pickle_path


def run_manual_country(country: str = COUNTRY, config: dict = CONFIG, *, save_result: bool = SAVE_RESULT) -> dict:
    fc = _run_one(country, config)
    if save_result:
        save_selected_country_result(country, fc)
    return fc


def build_approved_dashboard_pickle(
    approved_specs: dict[str, dict] = APPROVED_COUNTRY_SPECS,
    *,
    pickle_path: Path = APPROVED_PICKLE_PATH,
) -> Path:
    panels = _load_panels()
    per_country = {}
    exo_forecast = None
    for country in COUNTRIES_TO_RUN:
        if country not in approved_specs:
            raise KeyError(f"Missing approved country config for {country}")
        print(f"[RUN] {country}: {_setting_label(country, approved_specs[country])}")
        fc = _run_one(country, approved_specs[country], panels)
        per_country[country] = _prune_for_dashboard(fc)
        exo_forecast = fc.get("exo_forecast")

    approved = {
        "config": {
            "format": "approved_country_forecasts_v1",
            "description": "Approved country-specific forecasts generated by simplified country_diag.py",
            "approved_country_specs": approved_specs,
        },
        "per_country": per_country,
        "exo_forecast": exo_forecast,
    }
    bundle = {
        "format": "approved_country_forecasts_v1",
        "active_scenario": "approved",
        "scenarios": {"approved": approved},
        "climate_variants": {
            "approved": {
                "label": "Approved country-specific specifications",
                "scenarios": {"approved": approved},
            }
        },
    }
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with pickle_path.open("wb") as f:
        pickle.dump(bundle, f)
    return pickle_path


def _forecast_summary(fc: dict) -> pd.DataFrame:
    y = np.asarray(fc["y_hat"], dtype=float) * np.asarray(fc["y_sd"]) + np.asarray(fc["y_mu"])
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
    return pd.DataFrame(rows)


if __name__ == "__main__":
    fc = run_manual_country()
    print(_forecast_summary(fc).to_string(index=False))
    if SAVE_RESULT:
        print(f"[SAVED] {APPROVED_PICKLE_PATH}")
