"""Dashboard-faithful country diagnostic for the GVAR/KF forecast model.

This script is intentionally a thin diagnostic layer around the production
dashboard modules. It does not reimplement the Kalman filter, EM loop, data
preparation, rolling VARX track, recursive forecast, Monte Carlo bands, or
zero-driver counterfactual logic. The counterfactuals use the same fitted
selected model and set the selected climate driver to zero.

Inspection notes before implementation
--------------------------------------
Directly reused production functions/settings:
    - structural_break/GVAR_LLM_pickle.py
        _load_panel_df_cached, _prepare_country_panel_cached,
        _run_kf_em_cached, PATH, COL_COUNTRY, COL_TIME, ENDO, EXO, lags.
    - trp/kalman_core.py
        run_kf_em via gp._run_kf_em_cached, including VARX init, KF/EM,
        Q/R updates, smoother, final fixed-Q/R forward pass.
    - analysis/Dash_Output/gvar_kf_forecast.py
        build_forecast_exo_df, build_forecast_exo_df_multi,
        forecast_country_from_em, _insample_kf_track, _varx_insample_track,
        _roll_kf_forecast, _roll_forecast_path, _build_H, _spectral_radius,
        _stabilize_theta_for_forecast, run_forecast_for_countries, main.

Hard-coded production settings currently observed:
    - gp.ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
    - gp.EXO = ["ENSO", "OIL_YoY"]
    - gp.lags = 1
    - gp._run_kf_em_cached uses window=40, tol=1e-4, em_damping=0.0,
      update_P0=True by default, and final covariance_mode="fixed".
    - gp.MAX_EM_ITER = 50 for the structural-break pickle builder.
    - gkf.MAX_EM_ITER = 10 for forecast generation.
    - gkf.STABILIZE_FORECAST = False and FORECAST_MAX_EIGENVALUE = 0.98.
    - Forecast bands use beta/Q random-walk Monte Carlo, not R/P observation
      noise; counterfactual paths do not get their own bands.

Configurable here, while still passing through production code:
    - selected country, displayed variables, section switches
    - max_em_iter, update_P0, coefficient method, forecast horizon
    - default forecast spec, HeatDry diagnostic spec
    - optional full production forecast-pickle generation step

Known places that would differ if changed:
    - Using max_em_iter=50 here while comparing to gkf forecast output generated
      with gkf.MAX_EM_ITER=10 will differ. Default below uses gkf.MAX_EM_ITER.
    - Counterfactual plots are same-model scenarios; they do not refit a
      no-climate model.

Breakpoint guide for VS Code
----------------------------
1. Set a breakpoint at the line marked:
       BREAKPOINT 1: after model generation / forecast objects exist
   Run once to generate the selected-country objects, then inspect `ctx`.
2. Enable only the switches you want below, or set breakpoints at:
       BREAKPOINT 2: model fit
       BREAKPOINT 3: counterfactual
       BREAKPOINT 4: stability
       BREAKPOINT 5: decomposition
       BREAKPOINT 6: coefficients
       BREAKPOINT 7: residuals
3. A common run is:
       RUN_MODEL_FIT = True
       RUN_COUNTERFACTUAL = True
       all later switches = False
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "trp_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "trp_cache"))

import matplotlib

_BACKEND_BEFORE_PROD_IMPORT = matplotlib.get_backend()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GVAR_IMPORT_ONLY", "1")
STRUCTURAL_BREAK_DIR = ROOT / "structural_break"
DASH_OUTPUT_DIR = ROOT / "analysis" / "Dash_Output"
for p in (STRUCTURAL_BREAK_DIR, DASH_OUTPUT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import GVAR_LLM_pickle as gp  # noqa: E402
import gvar_kf_forecast as gkf  # noqa: E402

# gvar_kf_forecast is a batch script and calls matplotlib.use("Agg") on import.
# Restore the caller backend so VS Code/Jupyter can display figures normally.
matplotlib.use(_BACKEND_BEFORE_PROD_IMPORT)


# ============================================================
# 1. CONFIGURATION
# ============================================================
COUNTRY = "KEN"  # BRA, CHL, COL, IDN, IND, MEX, PHL, THA,PER, ZAF,EGY,KEN
DISPLAY_VARS = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]

# MAX_EM_ITER = gkf.MAX_EM_ITER
MAX_EM_ITER = 10
UPDATE_P0 = False  # True = update P0 from EM; False = use initial P0 from VARX.
FORECAST_SCENARIO = "mean"
FORECAST_HORIZON = None  # None = production target quarters after history.
COEFF_METHOD = "avg4"  # production forecast methods: last, avg4, avg8.
SELECTED_EXO = ["HeatDry","OIL_YoY"]  # examples: ["ENSO"], ["HeatDry"], ["HeatDryF"], ["ENSO", "OIL_YoY"]
SETTING = ""  # optional override; blank builds e.g. "ENSO | lag1 | KF | coeff=last | EM=10"

RUN_CANDIDATE_SCREEN = False  # runs a small candidate screen for the selected country and saves to CSV
CANDIDATE_COUNTRIES = [COUNTRY]
CANDIDATE_EXO_SPECS = {
    "ENSO": ["ENSO"],
    "HeatDry": ["HeatDry"],
    "HeatDry+OIL": ["HeatDry", "OIL_YoY"],
}
CANDIDATE_MAX_EM_ITER = [10, 30, 50]
CANDIDATE_COEFF_METHODS = ["last", "avg4"]
CANDIDATE_UPDATE_P0 = [True, False]
CANDIDATE_OUTPUT = ROOT / "analysis" / "diagnose" / "output" / "country_diag_candidate_screen.csv"

RUN_EXISTING_FORECAST_PICKLE_STEP = False
RUN_MODEL_FIT = True
RUN_COUNTERFACTUAL = True
RUN_STABILITY = True
RUN_DECOMPOSITION = True
RUN_COEFFICIENTS = True
RUN_RESIDUALS = False
SHOW_FIGURES = True
SAVE_APPROVED_RESULT = True  # saves a dashboard-ready pickle with the selected country and spec
APPROVED_PICKLE_PATH = ROOT / "Dash_Input" / "gvar_forecast_results.pkl"

BENCHMARK_2026 = {
    "BRA": (1.9, 4.0),
    "MEX": (1.6, 3.9),
    "CHL": (2.4, 2.9),
    "PHL": (4.1, 4.3),
    "IND": (6.5, 4.7),
    "IDN": (5.0, 3.0),
    "PER": (2.8, 2.5),
    "THA": (1.5, 0.9),
    "COL": (2.3, 5.9),
    "KEN": (4.5, 5.9),
    "EGY": (4.2, 13.2),
    "ZAF": (1.0, 3.9),
}


# ============================================================
# 2. RUN / LOAD MODEL
# ============================================================
def _selected_indices(endo_use: list[str]) -> list[int]:
    return [endo_use.index(v) for v in DISPLAY_VARS if v in endo_use]


def _raw_scale(fc: dict, arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=float) * np.asarray(fc["y_sd"]) + np.asarray(fc["y_mu"])


def _setting_string(exo_vars: list[str], *, max_em_iter: int = MAX_EM_ITER, update_p0: bool = UPDATE_P0, coeff_method: str = COEFF_METHOD) -> str:
    if SETTING:
        return SETTING
    spec = "+".join(exo_vars) if exo_vars else "no climate"
    p0 = "P0=EM" if update_p0 else "P0=init"
    return f"{spec} | lag{gp.lags} | KF | coeff={coeff_method} | EM={max_em_iter} | {p0}"


def _fc_setting_label(fc: dict) -> str:
    setting = fc.get("setting")
    if setting:
        return str(setting)
    exo = "+".join(fc.get("EXO_use", [])) or "no climate"
    return f"{exo} | lag{gp.lags} | KF | coeff={COEFF_METHOD} | EM={MAX_EM_ITER}"


def _ctx_setting_label(ctx: dict) -> str:
    return _fc_setting_label(ctx["fc"])


def _build_selected_exo_forecast(country: str, exo_vars: list[str]) -> pd.DataFrame:
    known_multi = set(getattr(gkf, "CLIMATE_TOGGLE_VARS", []))
    if set(exo_vars).issubset(known_multi):
        return gkf.build_forecast_exo_df_multi(
            exo_vars,
            country,
            enso_map=gkf.FORECAST_ENSO_SCENARIOS.get(FORECAST_SCENARIO, gkf.FORECAST_ENSO_MEAN),
            target_quarters=gkf.FORECAST_TARGET_QUARTERS,
        )

    raw = pd.read_csv(gp.PATH, usecols=lambda c: c in {"country", "quarter", *exo_vars})
    raw["quarter"] = pd.to_datetime(raw["quarter"]).dt.to_period("Q").dt.to_timestamp()
    hist = raw.loc[raw["country"].astype(str).eq(country)].groupby("quarter", as_index=False).first().set_index("quarter")
    rows = []
    for q in gkf.FORECAST_TARGET_QUARTERS:
        tq = pd.Timestamp(q).to_period("Q").to_timestamp()
        row = {"target_quarter": tq, "exo_quarter": tq}
        for var in exo_vars:
            val = np.nan
            if var in hist.columns and tq in hist.index:
                val = pd.to_numeric(pd.Series([hist.loc[tq, var]]), errors="coerce").iloc[0]
            row[var] = float(val) if np.isfinite(val) else np.nan
            row[f"{var}_source"] = "panel" if np.isfinite(row[var]) else "missing"
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_coeff_method(fc: dict, coeff_method: str) -> dict:
    fc = dict(fc)
    if coeff_method != "last":
        method_pack = fc.get("forecast_methods", {}).get(coeff_method)
        if not isinstance(method_pack, dict):
            raise RuntimeError(f"Forecast method {coeff_method!r} is unavailable")
        for k in ("y_hat", "y_hat_enso0", "y_hat_iod0", "y_hat_heat0", "heat0_var", "y_lower", "y_upper", "forecast_stability"):
            if k in method_pack:
                fc[k] = method_pack[k]
    fc["selected_coeff_method"] = coeff_method
    return fc


def _run_forecast_with_settings(
    country: str,
    exo_vars: list[str],
    *,
    max_em_iter: int,
    update_p0: bool,
    coeff_method: str,
) -> tuple[dict, dict, dict]:
    prep, res = gp._run_kf_em_cached(
        PATH=gp.PATH,
        country=country,
        COL_COUNTRY=gp.COL_COUNTRY,
        COL_TIME=gp.COL_TIME,
        ENDO=gp.ENDO,
        EXO=exo_vars,
        lags=gp.lags,
        min_T=gp.lags + 5,
        max_em_iter=max_em_iter,
        exo_mode="all",
        update_P0=update_p0,
    )
    if prep is None or res is None:
        raise RuntimeError(f"No production EM result for {country}")
    exo_fc = _build_selected_exo_forecast(country, exo_vars)
    if FORECAST_HORIZON is not None:
        exo_fc = exo_fc.head(int(FORECAST_HORIZON)).copy()
    fc = gkf.forecast_country_from_em(prep, res, exo_fc, country=country, scenario_name=FORECAST_SCENARIO)
    if fc is None:
        raise RuntimeError(f"No production forecast for {country}")
    fc = _apply_coeff_method(fc, coeff_method)
    fc["setting"] = _setting_string(exo_vars, max_em_iter=max_em_iter, update_p0=update_p0, coeff_method=coeff_method)
    fc["approved_country_spec"] = True
    fc["approved_settings"] = {
        "country": country,
        "exo_vars": list(exo_vars),
        "max_em_iter": int(max_em_iter),
        "update_P0": bool(update_p0),
        "coeff_method": str(coeff_method),
        "counterfactual": "same fitted model, selected climate driver set to raw 0",
    }
    return prep, res, fc


def _run_selected_forecast(country: str, exo_vars: list[str]) -> tuple[dict, dict, dict]:
    return _run_forecast_with_settings(
        country,
        exo_vars,
        max_em_iter=MAX_EM_ITER,
        update_p0=UPDATE_P0,
        coeff_method=COEFF_METHOD,
    )


def _run_exo_forecast(country: str, exo_vars: list[str], scenario_name: str) -> tuple[dict, dict, dict] | None:
    prep, res = gp._run_kf_em_cached(
        PATH=gp.PATH,
        country=country,
        COL_COUNTRY=gp.COL_COUNTRY,
        COL_TIME=gp.COL_TIME,
        ENDO=gp.ENDO,
        EXO=exo_vars,
        lags=gp.lags,
        min_T=gp.lags + 5,
        max_em_iter=MAX_EM_ITER,
        exo_mode="all",
        update_P0=UPDATE_P0,
    )
    if prep is None or res is None:
        return None
    exo_fc = gkf.build_forecast_exo_df_multi(exo_vars, country, target_quarters=gkf.FORECAST_TARGET_QUARTERS)
    if FORECAST_HORIZON is not None:
        exo_fc = exo_fc.head(int(FORECAST_HORIZON)).copy()
    fc = gkf.forecast_country_from_em(prep, res, exo_fc, country=country, scenario_name=scenario_name)
    if fc is None:
        return None
    fc["setting"] = f"{'+'.join(exo_vars)} | lag{gp.lags} | KF | coeff={COEFF_METHOD} | EM={MAX_EM_ITER}"
    return prep, res, fc


def build_context(country: str = COUNTRY) -> dict:
    if RUN_EXISTING_FORECAST_PICKLE_STEP:
        print("[RUN] existing production forecast pickle step: gkf.main(plot_only=False)")
        gkf.main(plot_only=False)

    prep, res, fc = _run_selected_forecast(country, SELECTED_EXO)
    # BREAKPOINT 1: after model generation / forecast objects exist.
    return {
        "country": country,
        "prep": prep,
        "res": res,
        "fc": fc,
    }


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


def _prune_for_dashboard(fc: dict) -> dict:
    source = dict(fc)
    if COEFF_METHOD != "last":
        method_pack = fc.get("forecast_methods", {}).get(COEFF_METHOD)
        if isinstance(method_pack, dict):
            source.update({k: v for k, v in method_pack.items() if k in _SAVE_METHOD_FIELDS})
    out = {k: v for k, v in source.items() if k in _SAVE_COUNTRY_FIELDS}
    methods = {}
    for name, pack in fc.get("forecast_methods", {}).items():
        if isinstance(pack, dict):
            methods[name] = {k: v for k, v in pack.items() if k in _SAVE_METHOD_FIELDS}
    out["forecast_methods"] = methods
    out["setting"] = fc.get("setting") or _setting_string(list(fc.get("EXO_use", SELECTED_EXO)))
    out["approved_country_spec"] = True
    out["climate_variant"] = out.get("climate_variant") or "+".join(out.get("EXO_use", []))
    return out


def save_approved_country_result(ctx: dict, pickle_path: Path = APPROVED_PICKLE_PATH) -> Path:
    country = str(ctx["country"])
    fc = _prune_for_dashboard(ctx["fc"])

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
                "description": "Country-by-country forecasts approved from analysis/diagnose/country_diag.py",
            },
            "per_country": {},
        },
    )
    approved.setdefault("config", {})["format"] = "approved_country_forecasts_v1"
    approved.setdefault("per_country", {})[country] = fc
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


# ============================================================
# 3. MODEL FIT
# ============================================================
def plot_model_fit(ctx: dict) -> pd.DataFrame:
    fc = ctx["fc"]
    endo = fc["ENDO_use"]
    idx = _selected_indices(endo)

    kf = fc["kf_insample"]
    varx = fc["varx_insample"]

    q = pd.to_datetime(kf["quarters"])
    y_raw = _raw_scale(fc, kf["y"])
    yhat_raw = _raw_scale(fc, kf["y_hat"])
    lo_raw = _raw_scale(fc, kf["y_lower"])
    hi_raw = _raw_scale(fc, kf["y_upper"])
    varx_raw = _raw_scale(fc, varx["y_hat"])

    fig, axes = plt.subplots(len(idx), 1, figsize=(12, 3.2 * len(idx)), sharex=True)
    axes = np.atleast_1d(axes)
    rows = []
    for ax, j in zip(axes, idx):
        ax.plot(q, y_raw[:, j], color="black", linewidth=1.4, label="observed")
        ax.plot(q, yhat_raw[:, j], color="C0", linewidth=1.3, label="KF 1-step")
        ax.fill_between(q, lo_raw[:, j], hi_raw[:, j], color="C0", alpha=0.18, label="KF CI")
        ax.plot(pd.to_datetime(varx["quarters"]), varx_raw[:, j], color="C2", linewidth=1.0, label="rolling VARX")
        ax.set_ylabel(endo[j])
        ax.grid(alpha=0.25)

        for name, pred in (("KF 1-step", yhat_raw[:, j]), ("VARX 1-step", varx_raw[:, j])):
            actual = y_raw[:, j]
            ok = np.isfinite(actual) & np.isfinite(pred)
            rows.append(
                {
                    "variable": endo[j],
                    "model": name,
                    "RMSE": float(np.sqrt(np.mean((actual[ok] - pred[ok]) ** 2))),
                    "MAE": float(np.mean(np.abs(actual[ok] - pred[ok]))),
                    "note": "in-sample one-step track; not true real-time OOS for full-sample KF",
                }
            )

    axes[0].legend(fontsize=8, ncol=3)
    fig.suptitle(f"{ctx['country']} model fit\nselected spec: {_ctx_setting_label(ctx)}")
    fig.tight_layout()
    return pd.DataFrame(rows)


# ============================================================
# 4. COUNTERFACTUAL
# ============================================================
def plot_counterfactual(ctx: dict, *, driver: str = "ENSO") -> None:
    fc = ctx["fc"]
    if driver not in fc["EXO_use"]:
        print(f"[SKIP] {driver} is not in selected spec: {_fc_setting_label(fc)}")
        return
    if driver == "ENSO":
        key = "y_hat_enso0"
    elif driver == "IOD":
        key = "y_hat_iod0"
    elif driver in {"HeatDry", "HeatDryF"}:
        key = "y_hat_heat0"
    else:
        print(f"[SKIP] no zero-driver counterfactual is defined for {driver}")
        return
    cf = fc.get(key)
    if cf is None:
        print(f"[SKIP] no production {driver}=0 counterfactual in forecast object")
        return

    endo = fc["ENDO_use"]
    idx = _selected_indices(endo)
    fq = pd.to_datetime(fc["fc_quarters"])
    y = _raw_scale(fc, fc["y_hat"])
    y0 = _raw_scale(fc, cf)
    lo = _raw_scale(fc, fc["y_lower"])
    hi = _raw_scale(fc, fc["y_upper"])
    hist_q = pd.to_datetime(fc.get("hist_quarters", []))
    hist_y = np.asarray(fc.get("hist_y_raw", []), dtype=float)
    hist_start = pd.Timestamp("2020-01-01")
    hist_mask = hist_q >= hist_start if len(hist_q) else []

    fig, axes = plt.subplots(len(idx), 1, figsize=(12, 3.4 * len(idx)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, j in zip(axes, idx):
        if len(hist_q) and hist_y.size:
            ax.plot(hist_q[hist_mask], hist_y[hist_mask, j], color="black", linewidth=1.3, label="observed since 2020")
            ax.axvline(fq[0], color="gray", linestyle=":", linewidth=0.9)
        ax.plot(fq, y[:, j], color="C0", marker="o", label=f"forecast with {driver}")
        ax.fill_between(fq, lo[:, j], hi[:, j], color="C0", alpha=0.2, label="forecast CI")
        ax.plot(fq, y0[:, j], color="C3", marker="x", linestyle="--", label=f"{driver}=0")
        ax.set_ylabel(endo[j])
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{ctx['country']} {driver} scenario vs zero-driver counterfactual\nselected spec: {_fc_setting_label(fc)}")
    fig.tight_layout()


# ============================================================
# 5. STABILITY
# ============================================================
def _rho_series(fc: dict) -> pd.DataFrame:
    theta = np.asarray(fc["theta_trace"], dtype=float)
    q = pd.to_datetime(fc["theta_trace_quarters"])
    mY = len(fc["ENDO_use"])
    mX = len(fc["EXO_use"])
    m = gp.lags * mY + mX
    rows = []
    for qt, th in zip(q, theta):
        B = th.reshape(mY, m, order="C")
        blocks = [B[:, i * mY : (i + 1) * mY] for i in range(gp.lags)]
        rows.append({"quarter": qt, "spectral_radius": gkf._spectral_radius(blocks)})
    return pd.DataFrame(rows)


def plot_stability(ctx: dict) -> pd.DataFrame:
    fc = ctx["fc"]
    rho = _rho_series(fc)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rho["quarter"], rho["spectral_radius"], color="C0", linewidth=1.4)
    ax.axhline(1.0, color="C3", linestyle="--", linewidth=1.0)
    ax.set_title(f"{ctx['country']} endogenous transition spectral radius\nselected spec: {_ctx_setting_label(ctx)}")
    ax.set_ylabel("max |eigenvalue|")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    info = fc.get("forecast_stability", {})
    print(
        "Terminal spectral radius used for forecasting: "
        f"before={info.get('rho_before')}, after={info.get('rho_after')}, "
        f"stabilize_applied={info.get('applied')}"
    )
    return rho


def _companion_matrix_from_forecast(fc: dict, coeff_method: str | None = None) -> np.ndarray:
    coeff_method = coeff_method or fc.get("selected_coeff_method") or COEFF_METHOD
    method_pack = fc.get("forecast_methods", {}).get(coeff_method) or fc.get("forecast_methods", {}).get("last")
    if not isinstance(method_pack, dict) or "theta" not in method_pack:
        theta = np.asarray(fc["theta_last"], dtype=float)
    else:
        theta = np.asarray(method_pack["theta"], dtype=float)
    mY = len(fc["ENDO_use"])
    mX = len(fc["EXO_use"])
    lags = int(gp.lags)
    m = lags * mY + mX
    B = theta.reshape(mY, m, order="C")
    lag_blocks = [B[:, i * mY : (i + 1) * mY] for i in range(lags)]
    if lags == 1:
        return lag_blocks[0]
    top = np.hstack(lag_blocks)
    lower = np.hstack([np.eye((lags - 1) * mY), np.zeros(((lags - 1) * mY, mY))])
    return np.vstack([top, lower])


def plot_finite_horizon_amplification(ctx: dict) -> pd.DataFrame:
    fc = ctx["fc"]
    H = len(fc.get("fc_quarters", [])) or int(FORECAST_HORIZON or 0)
    if H <= 0:
        print("[SKIP] no forecast horizon available for finite-horizon amplification")
        return pd.DataFrame()

    A = _companion_matrix_from_forecast(fc)
    Ak = np.eye(A.shape[0])
    rows = []
    for k in range(1, H + 1):
        Ak = Ak @ A
        rows.append({"step": k, "spectral_norm_Ak": float(np.linalg.norm(Ak, ord=2))})
    out = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(out["step"], out["spectral_norm_Ak"], color="C4", marker="o", linewidth=1.4)
    ax.axhline(1.0, color="C3", linestyle="--", linewidth=1.0)
    ax.set_title(f"{ctx['country']} finite-horizon endogenous amplification\nselected spec: {_ctx_setting_label(ctx)}")
    ax.set_xlabel("Forecast step k")
    ax.set_ylabel(r"$||A^k||_2$ using companion matrix")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    print(f"Max ||A^k||_2 over forecast horizon: {out['spectral_norm_Ak'].max():.4f}")
    print(f"||A^H||_2 at H={H}: {out['spectral_norm_Ak'].iloc[-1]:.4f}")
    return out


# ============================================================
# 6. EFFECT DECOMPOSITION
# ============================================================
def climate_effect_decomposition(ctx: dict, *, driver: str = "ENSO") -> pd.DataFrame:
    fc = ctx["fc"]
    if driver == "ENSO":
        cf_key = "y_hat_enso0"
    elif driver == "IOD":
        cf_key = "y_hat_iod0"
    elif driver in {"HeatDry", "HeatDryF"}:
        cf_key = "y_hat_heat0"
    else:
        return pd.DataFrame()
    if fc.get(cf_key) is None or driver not in fc["EXO_use"]:
        return pd.DataFrame()

    coeff_method = fc.get("selected_coeff_method") or COEFF_METHOD
    theta = np.asarray(fc["forecast_methods"][coeff_method]["theta"], dtype=float).reshape(-1, 1)
    mY = len(fc["ENDO_use"])
    mX = len(fc["EXO_use"])
    m = gp.lags * mY + mX
    B = theta.reshape(mY, m, order="C")
    A = B[:, : gp.lags * mY]
    C = B[:, gp.lags * mY :]

    exo = fc["exo_forecast"]
    mu_x = np.nanmean(ctx["prep"]["g"][fc["EXO_use"]].to_numpy(float), axis=0)
    sd_x = np.nanstd(ctx["prep"]["g"][fc["EXO_use"]].to_numpy(float), axis=0) + 1e-8
    z = np.column_stack([(pd.to_numeric(exo[c], errors="coerce").to_numpy(float) - mu_x[i]) / sd_x[i] for i, c in enumerate(fc["EXO_use"])])
    z0 = z.copy()
    j_driver = fc["EXO_use"].index(driver)
    z0[:, j_driver] = (0.0 - mu_x[j_driver]) / sd_x[j_driver]

    total = np.asarray(fc["y_hat"], dtype=float) - np.asarray(fc[cf_key], dtype=float)
    prev_delta_lags = [np.zeros(mY)]
    rows = []
    for h, qt in enumerate(pd.to_datetime(fc["fc_quarters"])):
        direct = C @ (z[h] - z0[h])
        propagated = A @ np.concatenate(prev_delta_lags)
        for k, var in enumerate(fc["ENDO_use"]):
            rows.append(
                {
                    "quarter": qt,
                    "variable": var,
                    "direct": direct[k] * fc["y_sd"][k],
                    "propagated": propagated[k] * fc["y_sd"][k],
                    "total": total[h, k] * fc["y_sd"][k],
                }
            )
        prev_delta_lags = [total[h].copy()] + prev_delta_lags[: gp.lags - 1]
    return pd.DataFrame(rows)


# ============================================================
# 6B. SMALL CANDIDATE SCREEN, USING THIS SCRIPT ONLY
# ============================================================
def _counterfactual_keys(fc: dict) -> list[tuple[str, str]]:
    out = []
    if "ENSO" in fc["EXO_use"] and fc.get("y_hat_enso0") is not None:
        out.append(("ENSO", "y_hat_enso0"))
    if "HeatDry" in fc["EXO_use"] and fc.get("y_hat_heat0") is not None:
        out.append(("HeatDry", "y_hat_heat0"))
    return out


def _forecast_shape_metrics(fc: dict) -> dict:
    endo = fc["ENDO_use"]
    y = _raw_scale(fc, fc["y_hat"])
    lo = _raw_scale(fc, fc["y_lower"])
    hi = _raw_scale(fc, fc["y_upper"])
    q = pd.to_datetime(fc["fc_quarters"])
    out = {}
    for var in ("GDP_YoY", "CPI_YoY"):
        if var not in endo:
            continue
        j = endo.index(var)
        vals = y[:, j]
        out[f"{var}_2026"] = float(np.nanmean(vals[q.year == 2026])) if np.any(q.year == 2026) else np.nan
        out[f"{var}_min"] = float(np.nanmin(vals))
        out[f"{var}_max"] = float(np.nanmax(vals))
        out[f"{var}_last"] = float(vals[-1])
        out[f"{var}_ci_max_width"] = float(np.nanmax(hi[:, j] - lo[:, j]))
        d = np.diff(vals)
        out[f"{var}_monotone_steps"] = int(max(np.sum(d > 0), np.sum(d < 0))) if len(d) else 0
    cf_max = 0.0
    for _, key in _counterfactual_keys(fc):
        cf = _raw_scale(fc, fc[key])
        cols = [endo.index(v) for v in ("GDP_YoY", "CPI_YoY") if v in endo]
        if cols:
            cf_max = max(cf_max, float(np.nanmax(np.abs(y[:, cols] - cf[:, cols]))))
    out["counterfactual_max_abs_gdp_cpi"] = cf_max
    return out


def _candidate_row(country: str, spec_name: str, exo_vars: list[str], max_em_iter: int, update_p0: bool, coeff_method: str) -> dict:
    try:
        _, _, fc = _run_forecast_with_settings(
            country,
            exo_vars,
            max_em_iter=max_em_iter,
            update_p0=update_p0,
            coeff_method=coeff_method,
        )
        metrics = _forecast_shape_metrics(fc)
        A = _companion_matrix_from_forecast(fc, coeff_method)
        Ak = np.eye(A.shape[0])
        norms = []
        for _ in range(len(fc["fc_quarters"])):
            Ak = Ak @ A
            norms.append(float(np.linalg.norm(Ak, ord=2)))
        rho = fc.get("forecast_stability", {}).get("rho_before", np.nan)
        gdp = metrics.get("GDP_YoY_2026", np.nan)
        cpi = metrics.get("CPI_YoY_2026", np.nan)
        bgdp, bcpi = BENCHMARK_2026.get(country, (np.nan, np.nan))
        bench_dist = abs(gdp - bgdp) + 0.6 * abs(cpi - bcpi) if np.isfinite(gdp) and np.isfinite(cpi) else np.inf
        flags = []
        if not np.isfinite(rho) or rho >= 1.0:
            flags.append("rho>=1")
        if max(norms, default=np.nan) > 8:
            flags.append("finite-horizon amplification")
        if max(abs(metrics.get("GDP_YoY_min", 0)), abs(metrics.get("GDP_YoY_max", 0)), abs(metrics.get("CPI_YoY_min", 0)), abs(metrics.get("CPI_YoY_max", 0))) > 40:
            flags.append("GDP/CPI extreme")
        if max(metrics.get("GDP_YoY_ci_max_width", 0), metrics.get("CPI_YoY_ci_max_width", 0)) > 25:
            flags.append("CI wide")
        if metrics["counterfactual_max_abs_gdp_cpi"] > 10:
            flags.append("counterfactual large")
        overall = "KEEP" if not flags and bench_dist <= 4 else ("BORDERLINE" if not flags else "REJECT")
        return {
            "country": country,
            "spec": spec_name,
            "exo": "+".join(exo_vars),
            "max_em_iter": max_em_iter,
            "update_P0": update_p0,
            "coeff_method": coeff_method,
            "overall": overall,
            "rho": rho,
            "max_norm_Ak": max(norms, default=np.nan),
            "bench_dist": bench_dist,
            "reason": "; ".join(flags) or "ok",
            **metrics,
        }
    except Exception as exc:
        return {
            "country": country,
            "spec": spec_name,
            "exo": "+".join(exo_vars),
            "max_em_iter": max_em_iter,
            "update_P0": update_p0,
            "coeff_method": coeff_method,
            "overall": "REJECT",
            "reason": f"{type(exc).__name__}: {exc}",
        }


def run_candidate_screen() -> pd.DataFrame:
    rows = []
    for country in CANDIDATE_COUNTRIES:
        for spec_name, exo_vars in CANDIDATE_EXO_SPECS.items():
            for max_em_iter in CANDIDATE_MAX_EM_ITER:
                for update_p0 in CANDIDATE_UPDATE_P0:
                    for coeff_method in CANDIDATE_COEFF_METHODS:
                        print(f"[SCREEN] {country} {spec_name} EM={max_em_iter} P0={update_p0} coeff={coeff_method}")
                        rows.append(_candidate_row(country, spec_name, exo_vars, max_em_iter, update_p0, coeff_method))
    out = pd.DataFrame(rows)
    if not out.empty:
        CANDIDATE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(CANDIDATE_OUTPUT, index=False)
        sort_cols = ["country", "overall", "bench_dist", "max_norm_Ak"]
        print(f"\n[SAVED] {CANDIDATE_OUTPUT}")
        print("\n== Shortlist ==")
        for country, g in out.sort_values(sort_cols).groupby("country"):
            keep = g[g["overall"].isin(["KEEP", "BORDERLINE"])].head(3)
            print(f"\n{country}")
            if keep.empty:
                print(g[["spec", "max_em_iter", "update_P0", "coeff_method", "overall", "reason"]].head(5).to_string(index=False))
            else:
                cols = ["spec", "max_em_iter", "update_P0", "coeff_method", "overall", "rho", "max_norm_Ak", "GDP_YoY_2026", "CPI_YoY_2026", "counterfactual_max_abs_gdp_cpi", "reason"]
                print(keep[[c for c in cols if c in keep.columns]].to_string(index=False))
    return out


def plot_decomposition(ctx: dict, *, driver: str = "ENSO") -> pd.DataFrame:
    df = climate_effect_decomposition(ctx, driver=driver)
    if df.empty:
        print(f"[SKIP] no decomposition for {driver}")
        return df
    idx_vars = [v for v in DISPLAY_VARS if v in set(df["variable"])]
    fig, axes = plt.subplots(len(idx_vars), 1, figsize=(12, 3.0 * len(idx_vars)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, var in zip(axes, idx_vars):
        g = df[df["variable"] == var]
        ax.plot(g["quarter"], g["direct"], label="direct climate effect")
        ax.plot(g["quarter"], g["propagated"], label="propagated endogenous effect")
        ax.plot(g["quarter"], g["total"], color="black", linewidth=1.4, label="total scenario difference")
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_ylabel(var)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{ctx['country']} {driver} direct vs propagated effect\nselected spec: {_ctx_setting_label(ctx)}")
    fig.tight_layout()
    return df


# ============================================================
# 7. COEFFICIENT TRAJECTORY
# ============================================================
def plot_climate_coefficients(ctx: dict, *, driver: str = "ENSO") -> None:
    fc = ctx["fc"]
    if driver not in fc["EXO_use"]:
        print(f"[SKIP] no {driver} coefficient trajectory")
        return
    endo = fc["ENDO_use"]
    idx = _selected_indices(endo)
    exo_idx = fc["EXO_use"].index(driver)
    theta = np.asarray(fc["theta_trace"], dtype=float)
    q = pd.to_datetime(fc["theta_trace_quarters"])
    mY = len(endo)
    m = gp.lags * mY + len(fc["EXO_use"])
    off = gp.lags * mY + exo_idx

    fig, ax = plt.subplots(figsize=(12, 4.8))
    for j in idx:
        k = j * m + off
        ax.plot(q, theta[:, k], linewidth=1.2, label=f"{driver} -> {endo[j]}")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title(f"{ctx['country']} {driver} coefficient trajectories\nselected spec: {_fc_setting_label(fc)}")
    ax.set_ylabel("standardized coefficient")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()


# ============================================================
# 8. RESIDUAL DIAGNOSTICS
# ============================================================
def residual_diagnostics(ctx: dict, *, rolling_window: int = 8) -> pd.DataFrame:
    kf = ctx["fc"]["kf_insample"]
    endo = ctx["fc"]["ENDO_use"]
    idx = _selected_indices(endo)
    q = pd.to_datetime(kf["quarters"])
    resid = (np.asarray(kf["y"], dtype=float) - np.asarray(kf["y_hat"], dtype=float)) * np.asarray(ctx["fc"]["y_sd"])
    rows = []

    fig, axes = plt.subplots(len(idx), 1, figsize=(12, 3.0 * len(idx)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, j in zip(axes, idx):
        r = pd.Series(resid[:, j], index=q)
        ax.plot(r.index, r.values, color="C0", label="residual")
        ax.plot(r.rolling(rolling_window).std(), color="C3", label=f"rolling std ({rolling_window}Q)")
        acf1 = r.autocorr(lag=1)
        lb_p = np.nan
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox

            lb_p = float(acorr_ljungbox(r.dropna(), lags=[4], return_df=True)["lb_pvalue"].iloc[0])
        except Exception:
            pass
        rows.append({"variable": endo[j], "mean": float(r.mean()), "std": float(r.std()), "acf1": float(acf1), "ljung_box_p_lag4": lb_p})
        ax.set_title(f"{ctx['country']} {endo[j]} residuals\nselected spec: {_ctx_setting_label(ctx)}\nacf1={acf1:.3f}, Ljung-Box p(4)={lb_p:.3g}")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if RUN_CANDIDATE_SCREEN:
        run_candidate_screen()
        if SHOW_FIGURES:
            plt.show()
        raise SystemExit(0)

    ctx = build_context(COUNTRY)
    print(f"\n[SELECTED SPEC] {COUNTRY}: {_ctx_setting_label(ctx)}")

    if RUN_MODEL_FIT:
        # BREAKPOINT 2: model fit.
        print("\n== A. Model fit ==")
        print(plot_model_fit(ctx).to_string(index=False))

    if RUN_COUNTERFACTUAL:
        # BREAKPOINT 3: counterfactual.
        print("\n== B. Climate scenario vs counterfactual ==")
        for driver in ctx["fc"]["EXO_use"]:
            plot_counterfactual(ctx, driver=driver)

    if RUN_STABILITY:
        # BREAKPOINT 4: stability.
        print("\n== C. Dynamic stability ==")
        print(plot_stability(ctx).tail().to_string(index=False))
        print("\n== C2. Finite-horizon endogenous amplification ==")
        print(plot_finite_horizon_amplification(ctx).to_string(index=False))

    if RUN_DECOMPOSITION:
        # BREAKPOINT 5: decomposition.
        print("\n== D. Direct vs propagated climate effect ==")
        for driver in ctx["fc"]["EXO_use"]:
            decomp = plot_decomposition(ctx, driver=driver)
            if not decomp.empty:
                print(decomp.tail(8).to_string(index=False))

    if RUN_COEFFICIENTS:
        # BREAKPOINT 6: coefficients.
        print("\n== E. Climate coefficient trajectory ==")
        for driver in ctx["fc"]["EXO_use"]:
            plot_climate_coefficients(ctx, driver=driver)

    if RUN_RESIDUALS:
        # BREAKPOINT 7: residuals.
        print("\n== F. Residual diagnostics ==")
        print(residual_diagnostics(ctx).to_string(index=False))

    if SAVE_APPROVED_RESULT:
        path = save_approved_country_result(ctx)
        print(f"\n[SAVED] approved {COUNTRY} result -> {path}")
        print(f"[SETTING] {ctx['fc'].get('setting')}")

    if SHOW_FIGURES:
        plt.show()
