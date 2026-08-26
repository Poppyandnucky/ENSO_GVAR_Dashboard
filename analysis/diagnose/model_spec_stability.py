"""Standalone model-specification and stability testing tool.

Kept entirely separate from the dashboard and from `structural_break/GVAR_LLM_pickle.py`.
Does not modify, import, or execute any dashboard code, and never writes a file the
dashboard reads. Plots are returned as `matplotlib.figure.Figure` objects -- nothing
is saved to disk; call `.show()` yourself (or let Jupyter/VS Code display them).

What this reuses vs. reimplements
----------------------------------
Kalman/EM math (unchanged, imported directly from `trp/kalman_core.py`):
    `init_from_varx_rolling`, `kf_e_step_store`, `rts_smoother`,
    `em_m_step_update`, `kalman_multilag_filter`, `_kf_log_likelihood`.
    This file never redefines any of this math -- see `run_kf_em_alt` below,
    which only re-orchestrates these same calls (same shape as
    `trp.kalman_core.run_kf_em`), so shrinkage/ridge/other approximations can
    be tried without touching production code.

Data loading (unchanged): `trp.inputs.load_gvar_panel`.

Forecast/stability helpers -- reproduced, not imported, from
`analysis/Dash_Output/gvar_kf_forecast.py`, so behavior matches production
exactly without pulling in `GVAR_LLM_pickle.py`'s dashboard-coupled import-time
execution:
    `_companion_matrix`, `_spectral_radius`  (verbatim)
    `_stabilize_theta_for_forecast`          (verbatim math; the module-level
                                               STABILIZE_FORECAST toggle is an
                                               explicit `enabled` arg here so
                                               every spec can report rho
                                               regardless of whether scaling
                                               is applied)
    `_build_H`, `_roll_forecast_path`        (verbatim -- fixed-beta, "ordinary
                                               VAR" forward rollout; no Q/R
                                               Monte Carlo banding, per request)
    `_insample_kf_track`                     (verbatim construction: y_hat =
                                               H @ theta_pred, S = H P_pred H' + R)
    zero-driver counterfactual                (same idea as production's
                                               `_zero_driver_counterfactual`:
                                               climate columns forced to raw 0
                                               over the forecast horizon only,
                                               history untouched)

Climate specifications (new -- pure data-prep, no kalman_core changes):
    Every spec below only changes which columns are built into the exogenous
    matrix `Z` before it reaches `kf_e_step_store`/`kalman_multilag_filter`.
    Those functions always read `Z_all[t-1, :]` as the exogenous regressor for
    predicting Y_t, so "ENSO at lag L" is obtained by pre-shifting a column by
    (L-1) periods before it is passed in -- see `_driver_lag_columns`.

    Specs are declared in `SPEC_DEFS` as {driver: pattern} -- e.g.
    {"ENSO": "lag2", "OIL_YoY": "lag1"} -- and grouped into five families:
    ENSO-only, HeatDry-only, Oil+ENSO (jointly lag-swept), Oil+HeatDry
    (jointly lag-swept), and ENSO+HeatDry (jointly lag-swept, no oil).
    `oil_enso_lag1` reproduces the dashboard's actual default spec
    (EXO=["ENSO","OIL_YoY"], both at their implicit lag 1).

Terminal-coefficient sourcing mirrors
`analysis/Dash_Output/gvar_kf_forecast.py::forecast_country_from_em` exactly:
    - final_filtered      <- e_step_store["theta_filt"][last valid t]
                             (production's "last" method)
    - avg_filtered_{5,10}q <- windowed mean of res["theta_est"], the final,
                             fully-converged-Q/R forward pass (production's
                             "avg4"/"avg8", just with a 5/10Q window)
    - final_smoothed / avg_smoothed_{5,10}q <- no production precedent (the
                             dashboard never forecasts from smoothed
                             coefficients); sourced from
                             e_step_store["theta_smooth"], the only smoothed
                             trajectory the EM loop produces.

Usage
-----
    python analysis/diagnose/model_spec_stability.py --country KEN
    python analysis/diagnose/model_spec_stability.py --country KEN --specs enso_baseline,enso_lag2 --plot

Or from a notebook / VS Code interactive window:
    from analysis.diagnose.model_spec_stability import run_spec_sweep, ...
    df = run_spec_sweep("KEN")
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trp.kalman_core import (  # noqa: E402
    NumericalInstabilityError,
    init_from_varx_rolling,
    kf_e_step_store,
    rts_smoother,
    em_m_step_update,
    kalman_multilag_filter,
    _kf_log_likelihood,
)
from trp.inputs import load_gvar_panel  # noqa: E402

# ---------------------------------------------------------------------------
# Production forecast-band / ENSO-scenario source, imported directly (not
# reimplemented). This module originally avoided importing
# analysis/Dash_Output/gvar_kf_forecast.py to sidestep
# structural_break/GVAR_LLM_pickle.py's dashboard-coupled import-time
# execution. Verified (2026-08): with GVAR_IMPORT_ONLY=1 -- the same guard
# gvar_kf_forecast.py already sets before importing GVAR_LLM_pickle -- both
# imports are cheap (<1s combined) and write no pickle/PDF. Importing
# directly here means the Monte Carlo forecast band (_roll_kf_forecast) and
# the ENSO scenario tables (FORECAST_ENSO_MEAN/MIN/MAX) are byte-identical
# to what the dashboard computes, not a parallel reimplementation to keep
# in sync by hand.
#
# gvar_kf_forecast.py calls matplotlib.use("Agg") at its own import time
# (it's meant to run headless/batch) -- that's a *global* backend switch,
# so without guarding it here, importing this module would silently kill
# inline plot rendering for whatever imported us (e.g. a notebook). Capture
# the caller's backend first and restore it right after.
# ---------------------------------------------------------------------------
os.environ.setdefault("GVAR_IMPORT_ONLY", "1")
_DASH_OUTPUT_DIR = _ROOT / "analysis" / "Dash_Output"
if str(_DASH_OUTPUT_DIR) not in sys.path:
    sys.path.insert(0, str(_DASH_OUTPUT_DIR))
import matplotlib  # noqa: E402
_MPL_BACKEND_BEFORE_GKF_IMPORT = matplotlib.get_backend()
import gvar_kf_forecast as gkf  # noqa: E402
matplotlib.use(_MPL_BACKEND_BEFORE_GKF_IMPORT)

# ---------------------------------------------------------------------------
# Production config mirror (see module docstring). Keep in sync with
# `structural_break/GVAR_LLM_pickle.py` (ENDO/EXO/lags) and
# `analysis/Dash_Output/gvar_kf_forecast.py` (MAX_EM_ITER/Z_CI/STABILIZE_*)
# if those change.
# ---------------------------------------------------------------------------
COL_COUNTRY = "country"
COL_TIME = "quarter"
ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
LAGS = 1                          # endogenous VAR lag order -- fixed; only the
                                   # exogenous/climate columns vary across specs.
WINDOW = 40                       # VARX rolling burn-in window
MIN_T = LAGS + 5                  # matches production's `min_T=gp.lags + 5`
MAX_EM_ITER = 10                  # matches production's MAX_EM_ITER
Z_CI = 1.96                       # matches production's Z_CI
STABILIZE_FORECAST = False        # matches production's STABILIZE_FORECAST default
FORECAST_MAX_EIGENVALUE = 0.98    # matches production's FORECAST_MAX_EIGENVALUE
FORECAST_MC_SIMULATIONS = gkf.FORECAST_MC_SIMULATIONS  # single source of truth
FORECAST_MC_SEED = gkf.FORECAST_MC_SEED                # single source of truth


# ---------------------------------------------------------------------------
# Climate specifications -- pure exogenous-matrix construction (item 1,2,5,6).
# ---------------------------------------------------------------------------
def _lag_series(x: np.ndarray, k: int) -> np.ndarray:
    """Shift x forward by k periods: out[t] = x[t-k]. Leading k entries are NaN."""
    x = np.asarray(x, dtype=float)
    if k == 0:
        return x.copy()
    out = np.full_like(x, np.nan, dtype=float)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def _driver_lag_columns(x: np.ndarray, driver: str, lags: tuple[int, ...]) -> dict[str, np.ndarray]:
    """Column for 'driver at lag L relative to Y_t'. L=1 reproduces the baseline
    (kalman_core already applies one implicit lag via Z_all[t-1]); L=2 needs an
    extra period of shifting so that Z_all[t-1] evaluates to driver[t-2], etc.
    """
    cols = {}
    for L in lags:
        label = driver if L == 1 else f"{driver}_lag{L}"
        cols[label] = _lag_series(x, L - 1)
    return cols


def _driver_pos_neg_columns(x: np.ndarray, driver: str) -> dict[str, np.ndarray]:
    """Split into positive-part / negative-part regressors (sum back to x)."""
    base = _lag_series(x, 0)
    return {
        f"{driver}_pos": np.maximum(base, 0.0),
        f"{driver}_neg": np.minimum(base, 0.0),
    }


def _driver_threshold_columns(x: np.ndarray, driver: str, threshold: float = 0.5) -> dict[str, np.ndarray]:
    """El Nino / La Nina threshold split at +-threshold; |x| < threshold -> both 0 (neutral)."""
    base = _lag_series(x, 0)
    elnino = np.where(base >= threshold, base, 0.0)
    lanina = np.where(base <= -threshold, base, 0.0)
    return {f"{driver}_elnino": elnino, f"{driver}_lanina": lanina}


LAG_PATTERNS: dict[str, tuple[int, ...]] = {
    "lag1": (1,),
    "lag2": (2,),
    "lag1_2": (1, 2),
}


def _pattern_columns(x: np.ndarray, driver: str, pattern: str) -> dict[str, np.ndarray]:
    """Dispatch a single driver's raw series to the requested column pattern."""
    if pattern in LAG_PATTERNS:
        return _driver_lag_columns(x, driver, LAG_PATTERNS[pattern])
    if pattern == "pos_neg":
        return _driver_pos_neg_columns(x, driver)
    if pattern == "threshold_0p5":
        return _driver_threshold_columns(x, driver, 0.5)
    raise ValueError(f"unknown column pattern {pattern!r}")


# Each spec is {driver: pattern}; the exogenous matrix only ever contains the
# drivers named here (no other driver is silently added). Grouped into the
# five requested families -- ENSO alone, HeatDry alone, Oil+ENSO jointly
# lag-swept, Oil+HeatDry jointly lag-swept, ENSO+HeatDry jointly lag-swept
# (no oil). `oil_enso_lag1` is the dashboard's actual production spec
# (EXO=["ENSO","OIL_YoY"], both at their implicit lag 1).
SPEC_DEFS: dict[str, dict[str, str]] = {
    # ---- ENSO only ----
    "enso_lag1": {"ENSO": "lag1"},
    "enso_lag2": {"ENSO": "lag2"},
    "enso_lag1_2": {"ENSO": "lag1_2"},
    "enso_pos_neg": {"ENSO": "pos_neg"},
    "enso_threshold_0p5": {"ENSO": "threshold_0p5"},
    # ---- HeatDry only (3 raw-column variants from yield_climate_total.csv) ----
    "heatdry_lag1": {"HeatDry": "lag1"},
    "heatdry_lag2": {"HeatDry": "lag2"},
    "heatdry_lag1_2": {"HeatDry": "lag1_2"},
    "heatdryF_lag1": {"HeatDryF": "lag1"},
    "heatdryyoy_lag1": {"HeatDryYoY": "lag1"},
    # ---- Oil price + ENSO, jointly lag-swept ----
    "oil_enso_lag1": {"OIL_YoY": "lag1", "ENSO": "lag1"},
    "oil_enso_lag2": {"OIL_YoY": "lag2", "ENSO": "lag2"},
    "oil_enso_lag1_2": {"OIL_YoY": "lag1_2", "ENSO": "lag1_2"},
    # ---- Oil price + HeatDry, jointly lag-swept ----
    "oil_heatdry_lag1": {"OIL_YoY": "lag1", "HeatDry": "lag1"},
    "oil_heatdry_lag2": {"OIL_YoY": "lag2", "HeatDry": "lag2"},
    "oil_heatdry_lag1_2": {"OIL_YoY": "lag1_2", "HeatDry": "lag1_2"},
    # ---- ENSO + HeatDry, jointly lag-swept (no oil) ----
    "enso_heatdry_lag1": {"ENSO": "lag1", "HeatDry": "lag1"},
    "enso_heatdry_lag1_2": {"ENSO": "lag1_2", "HeatDry": "lag1_2"},
}

# Which of each spec's drivers count as "climate" (zeroed in the no-climate
# scenario-impact counterfactual) vs. an economic control (OIL_YoY, left
# untouched in both the "with climate" and "no climate" paths).
SPEC_CLIMATE_DRIVERS: dict[str, tuple[str, ...]] = {
    name: tuple(d for d in defn if d != "OIL_YoY") for name, defn in SPEC_DEFS.items()
}


# ---------------------------------------------------------------------------
# Data prep (standalone -- does not import GVAR_LLM_pickle.py)
# ---------------------------------------------------------------------------
def _prepare_country_for_spec(country: str, spec_name: str) -> dict | None:
    df = load_gvar_panel()
    g = df[df[COL_COUNTRY] == country].sort_values(COL_TIME).reset_index(drop=True)
    if g.empty:
        return None

    endo_use = [c for c in ENDO if c in g.columns and g[c].notna().any()]
    if not endo_use:
        return None
    spec_def = SPEC_DEFS[spec_name]
    if any(c not in g.columns for c in spec_def):
        return None

    exo_cols: dict[str, np.ndarray] = {}
    driver_of_col: dict[str, str] = {}
    for driver, pattern in spec_def.items():
        cols = _pattern_columns(g[driver].to_numpy(float), driver, pattern)
        exo_cols.update(cols)
        for c in cols:
            driver_of_col[c] = driver
    exo_use = list(exo_cols.keys())

    Yd_raw = g[endo_use].to_numpy(float)
    Xd_raw = np.column_stack([exo_cols[c] for c in exo_use])

    mask = np.isfinite(Yd_raw).all(axis=1) & np.isfinite(Xd_raw).all(axis=1)
    g = g.loc[mask].reset_index(drop=True)
    Yd_raw = Yd_raw[mask]
    Xd_raw = Xd_raw[mask]
    if len(g) < MIN_T:
        return None

    mu_y = np.nanmean(Yd_raw, axis=0)
    sd_y = np.nanstd(Yd_raw, axis=0) + 1e-8
    Yd = (Yd_raw - mu_y) / sd_y

    mu_x = np.nanmean(Xd_raw, axis=0)
    sd_x = np.nanstd(Xd_raw, axis=0) + 1e-8
    Xd = (Xd_raw - mu_x) / sd_x

    return {
        "g": g,
        "Yd": Yd,
        "Xd": Xd,
        "Yd_raw": Yd_raw,
        "Xd_raw": Xd_raw,
        "quarters": pd.to_datetime(g[COL_TIME].values),
        "ENDO_use": endo_use,
        "EXO_use": exo_use,
        "mu_y": mu_y,
        "sd_y": sd_y,
        "mu_x": mu_x,
        "sd_x": sd_x,
        "spec_name": spec_name,
        "country": country,
        "driver_of_col": driver_of_col,
    }


# ---------------------------------------------------------------------------
# Alternative model-running function -- same structure/orchestration as
# `trp.kalman_core.run_kf_em`, built from the same primitives, with injection
# points for approximations (shrinkage, ridge, ...) so those can be tried here
# without touching trp/kalman_core.py or the dashboard.
# ---------------------------------------------------------------------------
def run_kf_em_alt(
    Y: np.ndarray,
    Z: np.ndarray,
    lags: int = LAGS,
    window: int = WINDOW,
    ridge: float = 1e-6,
    max_em_iter: int = MAX_EM_ITER,
    tol: float = 1e-4,
    em_damping: float = 0.0,
    eps: float = 1e-8,
    verbose: bool = False,
    update_P0: bool = True,
    q_shrinkage: float = 0.0,
    r_ridge: float = 0.0,
) -> dict:
    """Same VAR-burn-in -> EM loop -> final-forward-pass orchestration as
    `trp.kalman_core.run_kf_em`, calling the identical math primitives in the
    identical order. `q_shrinkage`/`r_ridge` default to 0.0, which reproduces
    `run_kf_em` exactly; set them > 0 to try shrinking each EM-updated Q toward
    its diagonal, or adding ridge to R, without editing kalman_core.py.
    """
    theta0, Q, R, P0, m, p = init_from_varx_rolling(Y=Y, Z=Z, lags=lags, window=window, ridge=ridge, eps=eps)
    P0_init = P0.copy()

    history = {"trace_Q": [], "trace_R": [], "log_likelihood": [], "obj": []}
    last_obj = np.inf
    best_pack = None

    for it in range(max_em_iter):
        theta_filt, P_filt, theta_pred, P_pred, H_list, y_list, valid_mask = kf_e_step_store(
            Y=Y, Z_all=Z, theta0=theta0, Q=Q, R=R, P0=P0, lags=lags, eps=eps, em_iteration=it + 1,
        )
        theta_smooth, P_smooth, J_hist = rts_smoother(
            theta_filt, P_filt, theta_pred, P_pred, valid_mask, eps=eps, em_iteration=it + 1,
        )
        Q_new, R_new = em_m_step_update(
            Y=Y, H_list=H_list, y_list=y_list, valid_mask=valid_mask,
            theta_smooth=theta_smooth, P_smooth=P_smooth, J_hist=J_hist,
            Q_old=Q, R_old=R, em_damping=em_damping, eps=eps, em_iteration=it + 1,
        )
        if q_shrinkage > 0.0:
            Q_new = (1.0 - q_shrinkage) * Q_new + q_shrinkage * np.diag(np.diag(Q_new))
        if r_ridge > 0.0:
            R_new = R_new + r_ridge * np.eye(R_new.shape[0])

        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) > 0:
            theta0 = theta_smooth[valid_idx[0]].reshape(-1, 1)
            if update_P0:
                P0 = P_smooth[valid_idx[0]]
        Q, R = Q_new, R_new

        obj = 0.0
        cnt = 0
        for t in valid_idx:
            e = y_list[t] - H_list[t] @ theta_smooth[t].reshape(-1, 1)
            obj += float((e.T @ e).item())
            cnt += 1
        obj = obj / max(cnt, 1)
        ll = _kf_log_likelihood(H_list, y_list, valid_mask, theta_pred, P_pred, R, eps=eps)

        history["trace_Q"].append(float(np.trace(Q)))
        history["trace_R"].append(float(np.trace(R)))
        history["log_likelihood"].append(ll)
        history["obj"].append(obj)
        if verbose:
            print(f"[EM-alt] iter={it + 1:02d} obj={obj:.6f} ll={ll:.3f}")

        best_pack = {
            "theta_filt": theta_filt, "P_filt": P_filt, "theta_pred": theta_pred, "P_pred": P_pred,
            "theta_smooth": theta_smooth, "P_smooth": P_smooth, "J_hist": J_hist,
            "valid_mask": valid_mask, "H_list": H_list, "y_list": y_list,
        }

        if abs(last_obj - obj) < tol:
            if verbose:
                print(f"[EM-alt] converged at iter={it + 1}")
            break
        last_obj = obj

    rmse, e_raw, theta_est, Y_pred, P_hist, Q_trace, R_trace = kalman_multilag_filter(
        Y=Y, Z_all=Z, Q0=Q, R0=R, P0=P0, theta0=theta0, lags=lags, eps=eps, covariance_mode="fixed",
    )

    return {
        "rmse": rmse, "e_raw": e_raw, "theta_est": theta_est, "theta_trace": theta_est,
        "Y_pred": Y_pred, "P_hist": P_hist, "Q_trace": Q_trace, "R_trace": R_trace,
        "theta0": theta0, "Q": Q, "R": R, "P0": P0, "P0_init": P0_init, "update_P0": update_P0,
        "m": m, "p": p, "em_history": history, "e_step_store": best_pack,
        "q_shrinkage": q_shrinkage, "r_ridge": r_ridge,
    }


# ---------------------------------------------------------------------------
# Stability diagnostics -- ported verbatim from
# analysis/Dash_Output/gvar_kf_forecast.py (see module docstring).
# ---------------------------------------------------------------------------
def _companion_matrix(a_blocks: list[np.ndarray]) -> np.ndarray:
    mY = a_blocks[0].shape[0]
    lags = len(a_blocks)
    comp = np.zeros((mY * lags, mY * lags), dtype=float)
    comp[:mY, : mY * lags] = np.hstack(a_blocks)
    if lags > 1:
        comp[mY:, :-mY] = np.eye(mY * (lags - 1))
    return comp


def _spectral_radius(a_blocks: list[np.ndarray]) -> float:
    comp = _companion_matrix(a_blocks)
    eig = np.linalg.eigvals(comp)
    return float(np.max(np.abs(eig))) if eig.size else 0.0


def _stabilize_theta_for_forecast(
    theta: np.ndarray,
    *,
    mY: int,
    m: int,
    lags: int,
    max_eigenvalue: float = FORECAST_MAX_EIGENVALUE,
    enabled: bool = STABILIZE_FORECAST,
) -> tuple[np.ndarray, dict]:
    """Scale endogenous lag coefficients if the forecast companion is unstable.

    Same math as gvar_kf_forecast.py::_stabilize_theta_for_forecast; the
    module-level STABILIZE_FORECAST toggle is replaced by an explicit
    `enabled` argument (no numerical difference) so this tool can report
    rho_before/rho_after for every spec regardless of whether scaling is
    actually applied.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1, 1)
    B = theta.reshape(mY, m, order="C").copy()
    a_blocks = [B[:, i * mY : (i + 1) * mY].copy() for i in range(lags)]
    rho_before = _spectral_radius(a_blocks)

    info = {
        "enabled": bool(enabled), "applied": False, "max_eigenvalue": float(max_eigenvalue),
        "rho_before": rho_before, "rho_after": rho_before, "scale": 1.0,
    }
    if (not enabled) or (not np.isfinite(rho_before)) or rho_before <= max_eigenvalue:
        return theta, info

    scale = max_eigenvalue / max(rho_before, 1e-12)
    for i in range(lags):
        B[:, i * mY : (i + 1) * mY] *= scale
    a_blocks_after = [B[:, i * mY : (i + 1) * mY].copy() for i in range(lags)]
    rho_after = _spectral_radius(a_blocks_after)
    info.update({"applied": True, "rho_after": rho_after, "scale": float(scale)})
    return B.reshape(-1, 1, order="C"), info


# ---------------------------------------------------------------------------
# Terminal-coefficient variants (item 3) -- sourced exactly as production
# sources them (see module docstring for the "last"/"avg4"/"avg8" mapping).
# ---------------------------------------------------------------------------
TERMINAL_METHODS = (
    "final_filtered",
    "final_smoothed",
    "avg_filtered_5q",
    "avg_filtered_10q",
    "avg_smoothed_5q",
    "avg_smoothed_10q",
)


def terminal_coefficient_variants(
    res: dict,
    *,
    mY: int,
    m: int,
    lags: int = LAGS,
    stabilize: bool = STABILIZE_FORECAST,
    max_eigenvalue: float = FORECAST_MAX_EIGENVALUE,
    windows: dict[str, int] | None = None,
) -> dict[str, dict]:
    windows = windows or {"avg_filtered_5q": 5, "avg_filtered_10q": 10, "avg_smoothed_5q": 5, "avg_smoothed_10q": 10}
    pack = res.get("e_step_store")
    if not pack:
        return {}
    valid = np.asarray(pack["valid_mask"], dtype=bool)
    if not np.any(valid):
        return {}
    valid_idx = np.where(valid)[0]
    last_t = int(valid_idx[-1])

    theta_est = np.asarray(res.get("theta_est"), dtype=float)
    est_valid_idx = None
    if theta_est.ndim == 2 and theta_est.shape[0] == len(valid):
        est_mask = valid & np.isfinite(theta_est).all(axis=1)
        est_valid_idx = np.where(est_mask)[0]

    theta_smooth = pack["theta_smooth"]

    def _avg(traj: np.ndarray, idx: np.ndarray | None, window: int) -> np.ndarray | None:
        if idx is None or len(idx) < window:
            return None
        pick = idx[-window:]
        return traj[pick].mean(axis=0).reshape(-1, 1)

    raw = {
        "final_filtered": pack["theta_filt"][last_t].reshape(-1, 1),
        "final_smoothed": theta_smooth[last_t].reshape(-1, 1),
        "avg_filtered_5q": _avg(theta_est, est_valid_idx, windows["avg_filtered_5q"]),
        "avg_filtered_10q": _avg(theta_est, est_valid_idx, windows["avg_filtered_10q"]),
        "avg_smoothed_5q": _avg(theta_smooth, valid_idx, windows["avg_smoothed_5q"]),
        "avg_smoothed_10q": _avg(theta_smooth, valid_idx, windows["avg_smoothed_10q"]),
    }

    out = {}
    for name, theta_raw in raw.items():
        if theta_raw is None:
            continue
        theta_stab, stability = _stabilize_theta_for_forecast(
            theta_raw, mY=mY, m=m, lags=lags, max_eigenvalue=max_eigenvalue, enabled=stabilize,
        )
        out[name] = {"theta_raw": theta_raw, "theta": theta_stab, "stability": stability}
    return out


def stability_summary(variants: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, v in variants.items():
        rows.append({
            "method": name,
            "spectral_radius": v["stability"]["rho_before"],
            "exceeds_unit_circle": bool(v["stability"]["rho_before"] > 1.0),
            "rho_after_stabilize": v["stability"]["rho_after"],
            "stabilize_applied": v["stability"]["applied"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Simple residual check (item 4, "very simple") -- built on the same
# construction as gvar_kf_forecast.py::_insample_kf_track.
# ---------------------------------------------------------------------------
def _insample_kf_track(pack: dict, R: np.ndarray, quarters: np.ndarray, mY: int, *, z_ci: float = Z_CI, eps: float = 1e-8) -> dict:
    """In-sample one-step-ahead KF path: y_hat = H @ theta_pred, S = H P_pred H' + R.
    Same construction as gvar_kf_forecast.py::_insample_kf_track; also used
    directly as the "historical one-step-ahead forecast" plot data (item 6).
    """
    valid = np.asarray(pack["valid_mask"], dtype=bool)
    n = len(valid)
    R = np.asarray(R, dtype=float)
    y_hat = np.full((n, mY), np.nan)
    y_lo = np.full((n, mY), np.nan)
    y_hi = np.full((n, mY), np.nan)
    z_std = np.full((n, mY), np.nan)

    for t in np.where(valid)[0]:
        H = pack["H_list"][t]
        if H is None:
            continue
        theta_pr = np.asarray(pack["theta_pred"][t], dtype=float).reshape(-1, 1)
        P_pr = np.asarray(pack["P_pred"][t], dtype=float)
        yh = (H @ theta_pr).ravel()
        S = H @ P_pr @ H.T + R
        S = 0.5 * (S + S.T) + eps * np.eye(mY)
        se = np.sqrt(np.clip(np.diag(S), 0.0, None))
        y_hat[t, :] = yh
        y_lo[t, :] = yh - z_ci * se
        y_hi[t, :] = yh + z_ci * se
        y_t = np.asarray(pack["y_list"][t], dtype=float).ravel()
        z_std[t, :] = (y_t - yh) / np.clip(se, eps, None)

    return {
        "quarters": pd.to_datetime(quarters), "valid_mask": valid,
        "y_hat": y_hat, "y_lower": y_lo, "y_upper": y_hi, "z_std": z_std,
    }


def simple_residual_check(track: dict, ENDO_use: list[str]) -> pd.DataFrame:
    """Minimal residual diagnostic per endogenous variable: mean/std of the
    standardized one-step residual, % beyond +-1.96, and lag-1 autocorrelation.
    Deliberately not a full innovation-diagnostics suite (see
    analysis/validation/kf_em_diagnostics.py for that) -- just enough to flag
    obvious bias, over/under-dispersion, or leftover autocorrelation.
    """
    valid = track["valid_mask"]
    z = track["z_std"][valid]
    rows = []
    for j, var in enumerate(ENDO_use):
        zj = z[:, j]
        zj = zj[np.isfinite(zj)]
        if len(zj) < 2:
            continue
        lag1_acf = float(np.corrcoef(zj[:-1], zj[1:])[0, 1]) if len(zj) > 2 else np.nan
        rows.append({
            "variable": var,
            "n": len(zj),
            "mean_std_resid": float(np.mean(zj)),
            "std_std_resid": float(np.std(zj, ddof=1)),
            "pct_outside_1_96": float(np.mean(np.abs(zj) > Z_CI) * 100),
            "lag1_autocorr": lag1_acf,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Forward, fixed-terminal-coefficient rollout -- ported verbatim from
# gvar_kf_forecast.py::_build_H / _roll_forecast_path. No Kalman updates in
# the forward pass: theta is held fixed ("turned into an ordinary VAR"), per
# the requested terminal-coefficient comparison.
# ---------------------------------------------------------------------------
def _build_H(Y_lags: list[np.ndarray], z_row: np.ndarray, mY: int, m: int, lags: int) -> np.ndarray:
    pieces = [np.asarray(y_l, dtype=float).ravel() for y_l in Y_lags]
    if z_row is not None and len(z_row) > 0:
        pieces.append(np.asarray(z_row, dtype=float).ravel())
    x_t = np.concatenate(pieces)
    H = np.zeros((mY, mY * m))
    for j in range(mY):
        H[j, j * m : (j + 1) * m] = x_t
    return H


def _roll_forecast_path(*, theta: np.ndarray, y_buf0: list[np.ndarray], z_fc: np.ndarray, mY: int, m: int, lags: int) -> np.ndarray:
    theta = np.asarray(theta, dtype=float).reshape(-1, 1)
    y_buf = [y.copy() for y in y_buf0]
    n_fc = z_fc.shape[0]
    y_hat = np.full((n_fc, mY), np.nan)
    for h in range(n_fc):
        z_row = z_fc[h, :]
        if not np.isfinite(z_row).all():
            z_row = np.where(np.isfinite(z_row), z_row, 0.0)
        H = _build_H(y_buf, z_row, mY, m, lags)
        yhat = (H @ theta).ravel()
        y_hat[h, :] = yhat
        y_buf = [yhat.copy()] + y_buf[:-1]
    return y_hat


def _roll_forecast_with_band(
    *,
    theta: np.ndarray,
    Q: np.ndarray,
    y_buf0: list[np.ndarray],
    z_fc: np.ndarray,
    mY: int,
    m: int,
    lags: int,
    with_ci: bool = True,
    mc_simulations: int = FORECAST_MC_SIMULATIONS,
    mc_seed: int = FORECAST_MC_SEED,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Point forecast plus the beta/Q Monte Carlo uncertainty band -- calls
    gvar_kf_forecast.py's own `_roll_kf_forecast` directly (not a
    reimplementation), so this is byte-identical to what the dashboard
    computes for the exact same theta/Q/z_fc. `P0` and `R` are accepted by
    that function but intentionally unused (band excludes observation noise
    and treats the terminal beta as known -- see its docstring); passed as
    zeros here since they don't affect the result. Set `with_ci=False` to
    get just the deterministic rollout (matches production's treatment of
    zero-driver counterfactuals, which never get their own band).
    """
    theta0 = np.asarray(theta, dtype=float).reshape(-1, 1)
    P0 = np.zeros((theta0.shape[0], theta0.shape[0]))
    R = np.zeros((mY, mY))
    return gkf._roll_kf_forecast(
        theta0=theta0, P0=P0, Q=Q, R=R, y_buf0=y_buf0, z_fc=z_fc,
        mY=mY, m=m, lags=lags, z_ci=gkf.Z_CI, eps=eps, with_ci=with_ci,
        mc_simulations=mc_simulations, mc_seed=mc_seed,
    )


def _extend_raw_driver(hist_raw: np.ndarray, n_fc: int, *, mode: str = "last") -> np.ndarray:
    """Simple, explicit forward extension of a raw driver series for the
    scenario rollout: 'last' holds it flat at the last observed value. This
    tool does not reproduce the dashboard's ENSO/IOD scenario-input tables --
    pass `future_raw` to `build_scenario_exog` for a specific future path.
    """
    if mode == "zero":
        return np.zeros(n_fc, dtype=float)
    return np.full(n_fc, hist_raw[-1], dtype=float)


def build_scenario_exog(prep: dict, spec_name: str, *, n_fc: int, future_raw: dict[str, np.ndarray] | None = None) -> dict:
    """Standardized forward exogenous matrices for the scenario-impact plot:
    'z_fc' (spec's normal forward path) and 'z_fc_no_climate' (the spec's
    climate-driver columns -- SPEC_CLIMATE_DRIVERS[spec_name], i.e. everything
    except OIL_YoY -- forced to raw 0 over the forecast horizon only; same
    idea as gvar_kf_forecast.py's zero-driver counterfactual. OIL_YoY, when
    present, is left untouched in both paths since it is an economic control,
    not a climate driver. History is untouched in either case.
    """
    g = prep["g"]
    spec_def = SPEC_DEFS[spec_name]
    climate_drivers = SPEC_CLIMATE_DRIVERS[spec_name]
    future_raw = future_raw or {}

    n_hist = len(g)
    full_raw: dict[str, np.ndarray] = {}
    for driver in spec_def:
        hist = g[driver].to_numpy(float)
        fut = future_raw.get(driver)
        if fut is None:
            fut = _extend_raw_driver(hist, n_fc)
        full_raw[driver] = np.concatenate([hist, fut])

    climate_cols_full: dict[str, np.ndarray] = {}
    for driver, pattern in spec_def.items():
        climate_cols_full.update(_pattern_columns(full_raw[driver], driver, pattern))

    exo_use = prep["EXO_use"]
    driver_of_col = prep["driver_of_col"]
    z_fc_raw = np.zeros((n_fc, len(exo_use)), dtype=float)
    z_fc_raw_no_climate = np.zeros((n_fc, len(exo_use)), dtype=float)
    for j, col in enumerate(exo_use):
        vals = climate_cols_full[col][n_hist:]
        z_fc_raw[:, j] = vals
        z_fc_raw_no_climate[:, j] = vals if driver_of_col[col] not in climate_drivers else 0.0

    mu_x, sd_x = prep["mu_x"], prep["sd_x"]
    z_fc = (z_fc_raw - mu_x) / sd_x
    z_fc_no_climate = (z_fc_raw_no_climate - mu_x) / sd_x
    return {"z_fc": z_fc, "z_fc_no_climate": z_fc_no_climate, "climate_drivers": climate_drivers}


def run_scenario_impact(
    prep: dict,
    variants: dict[str, dict],
    method: str,
    spec_name: str,
    *,
    n_fc: int = 8,
    future_raw: dict[str, np.ndarray] | None = None,
    Q: np.ndarray | None = None,
    mc_simulations: int = FORECAST_MC_SIMULATIONS,
    mc_seed: int = FORECAST_MC_SEED,
) -> dict:
    """Forward rollout with a fixed terminal theta (ordinary-VAR style): the
    spec's forward exogenous path vs. the zero-climate-driver counterfactual.

    Pass `Q` (the EM-estimated process-noise covariance, e.g. `res["Q"]`) to
    also get a beta/Q Monte Carlo uncertainty band (`y_hat_climate_lo/hi`)
    around the main forecast, via `_roll_forecast_with_band` ->
    gvar_kf_forecast.py's own `_roll_kf_forecast`. Matching production's
    `forecast_country_from_em` exactly: only the main forecast gets a band --
    the zero-driver counterfactual (`y_hat_no_climate`) is always the
    deterministic rollout only (`y_hat_no_climate_lo/hi` stay None), same as
    production's `y_hat_enso0`/`y_hat_iod0`/`y_hat_heat0` (computed with
    with_ci=False; no band exists there to compare against). Omitting `Q`
    (the default) reproduces the old point-forecast-only behavior exactly.
    """
    ENDO_use, EXO_use = prep["ENDO_use"], prep["EXO_use"]
    mY, mX = len(ENDO_use), len(EXO_use)
    m = LAGS * mY + mX
    theta = variants[method]["theta"]

    Yd = prep["Yd"]
    y_buf0 = [Yd[-i, :].copy() for i in range(1, LAGS + 1)]
    scen = build_scenario_exog(prep, spec_name, n_fc=n_fc, future_raw=future_raw)

    if Q is not None:
        y_hat_climate, y_hat_climate_lo, y_hat_climate_hi = _roll_forecast_with_band(
            theta=theta, Q=Q, y_buf0=y_buf0, z_fc=scen["z_fc"], mY=mY, m=m, lags=LAGS,
            with_ci=True, mc_simulations=mc_simulations, mc_seed=mc_seed,
        )
    else:
        y_hat_climate = _roll_forecast_path(theta=theta, y_buf0=y_buf0, z_fc=scen["z_fc"], mY=mY, m=m, lags=LAGS)
        y_hat_climate_lo = y_hat_climate_hi = None

    y_hat_no_climate = _roll_forecast_path(
        theta=theta, y_buf0=y_buf0, z_fc=scen["z_fc_no_climate"], mY=mY, m=m, lags=LAGS
    )

    last_q = pd.to_datetime(prep["quarters"][-1])
    fc_quarters = pd.period_range(last_q.to_period("Q") + 1, periods=n_fc, freq="Q").to_timestamp()

    return {
        "fc_quarters": fc_quarters,
        "y_hat_climate": y_hat_climate,
        "y_hat_no_climate": y_hat_no_climate,
        "y_hat_climate_lo": y_hat_climate_lo,
        "y_hat_climate_hi": y_hat_climate_hi,
        "climate_drivers": scen["climate_drivers"],
        "method": method,
        "spec_name": spec_name,
    }


# ---------------------------------------------------------------------------
# Plots (item 6) -- returned as Figures, never saved.
# ---------------------------------------------------------------------------
def plot_historical_one_step(prep: dict, track: dict, *, title_prefix: str = "") -> plt.Figure:
    ENDO_use = prep["ENDO_use"]
    quarters = pd.to_datetime(prep["quarters"])
    valid = track["valid_mask"]
    fig, axes = plt.subplots(len(ENDO_use), 1, figsize=(11, 3.2 * len(ENDO_use)), sharex=True)
    if len(ENDO_use) == 1:
        axes = [axes]
    for ax, var, j in zip(axes, ENDO_use, range(len(ENDO_use))):
        ax.plot(quarters[valid], prep["Yd"][valid, j], color="black", label="observed (standardized)")
        ax.plot(quarters[valid], track["y_hat"][valid, j], linestyle="--", label="one-step-ahead y_hat")
        ax.fill_between(quarters[valid], track["y_lower"][valid, j], track["y_upper"][valid, j], alpha=0.2, label="95% band")
        ax.set_title(f"{var}: historical one-step-ahead forecast")
        ax.legend(fontsize=8)
    fig.suptitle(f"{title_prefix}Historical one-step-ahead forecasts".strip())
    fig.tight_layout()
    return fig


def plot_scenario_impact(prep: dict, scenario: dict, *, hist_quarters_shown: int = 20) -> plt.Figure:
    ENDO_use = prep["ENDO_use"]
    quarters = pd.to_datetime(prep["quarters"])[-hist_quarters_shown:]
    hist_y = prep["Yd"][-hist_quarters_shown:]
    fc_q = scenario["fc_quarters"]
    driver_label = "+".join(scenario["climate_drivers"])
    fig, axes = plt.subplots(len(ENDO_use), 1, figsize=(11, 3.2 * len(ENDO_use)), sharex=True)
    if len(ENDO_use) == 1:
        axes = [axes]
    for ax, var, j in zip(axes, ENDO_use, range(len(ENDO_use))):
        ax.plot(quarters, hist_y[:, j], color="black", label="history (standardized)")
        ax.plot(fc_q, scenario["y_hat_climate"][:, j], color="tab:blue", label=f"forecast w/ {driver_label}")
        ax.plot(fc_q, scenario["y_hat_no_climate"][:, j], color="tab:red", linestyle="--", label=f"counterfactual: {driver_label}=0")
        ax.axvline(quarters[-1], color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"{var}: scenario impact -- {scenario['method']} theta, spec={scenario['spec_name']}")
        ax.legend(fontsize=8)
    fig.suptitle("Scenario impact: forecast vs. no-climate-driver counterfactual")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sweep driver -- runs each (spec x terminal method) once per country and
# returns a tidy comparison table (spectral radius + simple residual check).
# No method is flagged as "best" -- that judgment is left to you.
# ---------------------------------------------------------------------------
def run_spec_sweep(
    country: str,
    *,
    spec_names: list[str] | None = None,
    terminal_methods: tuple[str, ...] = TERMINAL_METHODS,
    max_em_iter: int = MAX_EM_ITER,
    update_P0: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    spec_names = spec_names or list(SPEC_DEFS.keys())
    rows = []
    for spec_name in spec_names:
        prep = _prepare_country_for_spec(country, spec_name)
        if prep is None:
            print(f"[SKIP] {country}/{spec_name}: insufficient data")
            continue

        mY, mX = len(prep["ENDO_use"]), len(prep["EXO_use"])
        m = LAGS * mY + mX
        try:
            res = run_kf_em_alt(
                Y=prep["Yd"], Z=prep["Xd"], lags=LAGS, window=WINDOW,
                max_em_iter=max_em_iter, update_P0=update_P0, verbose=verbose,
            )
        except NumericalInstabilityError as exc:
            print(f"[SKIP] {country}/{spec_name}: numerical instability ({exc})")
            continue
        if res["e_step_store"] is None:
            print(f"[SKIP] {country}/{spec_name}: EM produced no valid pass")
            continue

        variants = terminal_coefficient_variants(res, mY=mY, m=m, lags=LAGS)
        track = _insample_kf_track(res["e_step_store"], res["R"], prep["quarters"], mY)
        resid_df = simple_residual_check(track, prep["ENDO_use"])
        resid_summary = {
            "resid_mean_abs": float(resid_df["mean_std_resid"].abs().mean()) if len(resid_df) else np.nan,
            "resid_pct_outside_1_96": float(resid_df["pct_outside_1_96"].mean()) if len(resid_df) else np.nan,
            "resid_lag1_autocorr_mean": float(resid_df["lag1_autocorr"].mean()) if len(resid_df) else np.nan,
        }

        for method in terminal_methods:
            if method not in variants:
                continue
            v = variants[method]
            rows.append({
                "country": country,
                "spec": spec_name,
                "method": method,
                "n_obs": int(np.sum(res["e_step_store"]["valid_mask"])),
                "log_likelihood_final": res["em_history"]["log_likelihood"][-1],
                "spectral_radius": v["stability"]["rho_before"],
                "exceeds_unit_circle": bool(v["stability"]["rho_before"] > 1.0),
                **resid_summary,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Batch plot saving -- one historical one-step-ahead figure and one
# scenario-impact figure (for a fixed terminal-coefficient `method`) per
# spec, written to disk. `plot_historical_one_step`/`plot_scenario_impact`
# above stay Figure-returning for interactive/notebook use; this is the
# save-to-disk entry point for a full spec sweep.
# ---------------------------------------------------------------------------
def save_all_spec_plots(
    country: str,
    *,
    spec_names: list[str] | None = None,
    method: str = "final_filtered",
    n_fc: int = 8,
    max_em_iter: int = MAX_EM_ITER,
    update_P0: bool = True,
    output_dir: str | Path | None = None,
) -> list[Path]:
    spec_names = spec_names or list(SPEC_DEFS.keys())
    out_dir = Path(output_dir) if output_dir else (_ROOT / "analysis" / "diagnose" / "output" / country / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for spec_name in spec_names:
        prep = _prepare_country_for_spec(country, spec_name)
        if prep is None:
            print(f"[SKIP] {country}/{spec_name}: insufficient data")
            continue

        mY, mX = len(prep["ENDO_use"]), len(prep["EXO_use"])
        m = LAGS * mY + mX
        try:
            res = run_kf_em_alt(
                Y=prep["Yd"], Z=prep["Xd"], lags=LAGS, window=WINDOW,
                max_em_iter=max_em_iter, update_P0=update_P0,
            )
        except NumericalInstabilityError as exc:
            print(f"[SKIP] {country}/{spec_name}: numerical instability ({exc})")
            continue
        if res["e_step_store"] is None:
            print(f"[SKIP] {country}/{spec_name}: EM produced no valid pass")
            continue

        variants = terminal_coefficient_variants(res, mY=mY, m=m, lags=LAGS)
        if method not in variants:
            print(f"[SKIP] {country}/{spec_name}: '{method}' unavailable (not enough valid quarters)")
            continue

        track = _insample_kf_track(res["e_step_store"], res["R"], prep["quarters"], mY)
        fig1 = plot_historical_one_step(prep, track, title_prefix=f"{country} [{spec_name}] ")
        p1 = out_dir / f"{spec_name}__historical_one_step.png"
        fig1.savefig(p1, dpi=140)
        plt.close(fig1)
        written.append(p1)

        scenario = run_scenario_impact(prep, variants, method, spec_name, n_fc=n_fc)
        fig2 = plot_scenario_impact(prep, scenario)
        p2 = out_dir / f"{spec_name}__scenario_impact_{method}.png"
        fig2.savefig(p2, dpi=140)
        plt.close(fig2)
        written.append(p2)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli():
    parser = argparse.ArgumentParser(
        description="Standalone model-spec / stability sweep for one country "
                    "(reuses production KF/EM math; no dashboard coupling)."
    )
    parser.add_argument("--country", required=True, help="ISO3 country code, e.g. KEN")
    parser.add_argument("--specs", default="all", help=f"comma-separated spec names, or 'all'. Options: {list(SPEC_DEFS)}")
    parser.add_argument("--terminal-methods", default="all", help=f"comma-separated terminal-coefficient methods, or 'all'. Options: {TERMINAL_METHODS}")
    parser.add_argument("--max-iter", type=int, default=MAX_EM_ITER)
    parser.add_argument("--no-update-p0", action="store_true")
    parser.add_argument("--plot", action="store_true", help="also build the historical one-step-ahead and scenario-impact figures for the first spec/method (shown, not saved)")
    parser.add_argument("--save-plots", action="store_true", help="save the historical one-step-ahead and scenario-impact figures for every spec in this run to analysis/diagnose/output/<country>/plots/ (PNG)")
    parser.add_argument("--plot-method", default="final_filtered", help=f"terminal-coefficient method used for the saved scenario-impact figures. Options: {TERMINAL_METHODS}")
    parser.add_argument("--n-fc", type=int, default=8, help="scenario-impact forecast horizon in quarters")
    parser.add_argument("--output", default=None, help="CSV path to save the comparison table (default: analysis/diagnose/output/<country>_spec_sweep.csv)")
    parser.add_argument("--no-save", action="store_true", help="print only, do not write a CSV")
    args = parser.parse_args()

    spec_names = None if args.specs == "all" else [s.strip() for s in args.specs.split(",")]
    terminal_methods = TERMINAL_METHODS if args.terminal_methods == "all" else tuple(s.strip() for s in args.terminal_methods.split(","))

    summary = run_spec_sweep(
        args.country, spec_names=spec_names, terminal_methods=terminal_methods,
        max_em_iter=args.max_iter, update_P0=not args.no_update_p0,
    )
    pd.set_option("display.width", 160)
    print(summary.to_string(index=False) if not summary.empty else "[no results]")

    if not summary.empty and not args.no_save:
        out_path = Path(args.output) if args.output else (_ROOT / "analysis" / "diagnose" / "output" / f"{args.country}_spec_sweep.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_path, index=False)
        print(f"\n[SAVE] {out_path}")

    if args.plot and not summary.empty:
        first_spec = summary.iloc[0]["spec"]
        first_method = summary.iloc[0]["method"]
        prep = _prepare_country_for_spec(args.country, first_spec)
        mY, mX = len(prep["ENDO_use"]), len(prep["EXO_use"])
        m = LAGS * mY + mX
        res = run_kf_em_alt(
            Y=prep["Yd"], Z=prep["Xd"], lags=LAGS, window=WINDOW,
            max_em_iter=args.max_iter, update_P0=not args.no_update_p0,
        )
        variants = terminal_coefficient_variants(res, mY=mY, m=m, lags=LAGS)
        track = _insample_kf_track(res["e_step_store"], res["R"], prep["quarters"], mY)
        plot_historical_one_step(prep, track, title_prefix=f"{args.country} [{first_spec}] ")
        scenario = run_scenario_impact(prep, variants, first_method, first_spec, n_fc=args.n_fc)
        plot_scenario_impact(prep, scenario)
        plt.show()

    if args.save_plots and not summary.empty:
        written = save_all_spec_plots(
            args.country, spec_names=spec_names, method=args.plot_method, n_fc=args.n_fc,
            max_em_iter=args.max_iter, update_P0=not args.no_update_p0,
        )
        if written:
            print(f"\n[SAVE] {len(written)} figures -> {written[0].parent}")


if __name__ == "__main__":
    _cli()
