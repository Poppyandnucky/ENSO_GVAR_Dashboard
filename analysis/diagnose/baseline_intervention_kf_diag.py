"""Baseline-vs-intervention KF/EM sensitivity diagnostic.

This script is intentionally separate from the dashboard workflow. It does not
modify production data, dashboard files, forecast pickles, or country_diag.py.

Default run is a small BRA-only test:
    python analysis/diagnose/baseline_intervention_kf_diag.py

Edit the CONFIG block below to test old/new vintages, terminal-quarter changes,
and external-variable combinations.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("GVAR_IMPORT_ONLY", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "trp_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "trp_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DASH_OUTPUT = ROOT / "analysis" / "Dash_Output"
for path in (ROOT, DASH_OUTPUT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from trp.kalman_core import (  # noqa: E402
    NumericalInstabilityError,
    _kf_log_likelihood,
    em_m_step_update,
    init_from_varx_rolling,
    kalman_multilag_filter,
    kf_e_step_store,
    rts_smoother,
)

import gvar_kf_forecast as gkf  # noqa: E402


# ============================================================================
# CONFIG
# ============================================================================

COUNTRY = "PHL"

BASELINE = {
    "data_vintage": "old",
    "external_variables": ["ENSO"],
    "drop_last_n": 0,
    "append_new_n": 0,
}

INTERVENTION = {
    "data_vintage": "old",
    "external_variables": ["HeatDry","OIL_YoY"],
    "drop_last_n": 0,
    "append_new_n": 1,
}

MAX_EM_ITER = 10

P0_SCALE = 0.01
P0_FIXED = True

Q0_SCALE = 0.1
Q0_FIXED = False

R0_SCALE = 2
R0_FIXED = False

FLAG_PLOT_EM_ITER = True
FLAG_PLOT_COEFF_PATH = True
FLAG_PLOT_ORIGINAL_DATA_COMPARISON = True
FLAG_PLOT_VARX_COEFF = False
FLAG_PLOT_FORECAST_COMPARISON = True
FLAG_PLOT_CLIMATE_COUNTERFACTUAL = True

FLAG_SHOW_PLOTS = True
FLAG_SAVE_PLOTS = True


# ============================================================================
# CONSTANTS
# ============================================================================

OLD_PANEL = ROOT / "analysis" / "gvar_panel_streamlit_old (8 + EGY + PER).csv"
NEW_PANEL = ROOT / "analysis" / "gvar_panel_streamlit (8 + EGY + PER).csv"
OUT_DIR = ROOT / "analysis" / "diagnose" / "output" / "baseline_intervention_kf_diag"
PLOT_DIR = OUT_DIR / "plots"

ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
LAGS = 1
WINDOW = 26
RIDGE = 1e-9
EPS = 1e-8
FORECAST_VARS = ["GDP_YoY", "CPI_YoY"]


@dataclass
class PreparedRun:
    name: str
    config: dict
    g: pd.DataFrame
    quarters: pd.DatetimeIndex
    endo_use: list[str]
    exo_use: list[str]
    y_raw: np.ndarray
    x_raw: np.ndarray
    y: np.ndarray
    x: np.ndarray
    y_mu: np.ndarray
    y_sd: np.ndarray
    x_mu: np.ndarray
    x_sd: np.ndarray
    theta0_init: np.ndarray
    q0_init: np.ndarray
    r0_init: np.ndarray
    p0_init: np.ndarray
    m: int
    p: int
    rolling_theta: np.ndarray
    rolling_quarters: pd.DatetimeIndex


def _load_panel(vintage: str) -> pd.DataFrame:
    paths = {"old": OLD_PANEL, "new": NEW_PANEL}
    if vintage not in paths:
        raise ValueError(f"Unknown data_vintage={vintage!r}; expected old or new.")
    df = pd.read_csv(paths[vintage], parse_dates=["quarter"])
    df["quarter"] = pd.to_datetime(df["quarter"]).dt.to_period("Q").dt.to_timestamp()
    df["_data_vintage"] = vintage
    return df


def _validate_columns(df: pd.DataFrame, columns: list[str], vintage: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        candidates = [
            c
            for c in df.columns
            if c in {"HeatDry", "HeatDryYoY", "HeatDryF", "ENSO", "OIL_YoY", "IOD"}
            or "ENSO" in c
            or "RONI" in c
            or "HeatDry" in c
        ]
        raise KeyError(
            f"{vintage} panel is missing requested columns {missing}. "
            f"Available likely external columns: {candidates}"
        )


def _complete_country_rows(df: pd.DataFrame, country: str, required: list[str]) -> pd.DataFrame:
    g = df[df["country"].eq(country)].sort_values("quarter").copy()
    if g.empty:
        raise ValueError(f"No rows found for country={country!r}.")
    mask = np.isfinite(g[required].to_numpy(float)).all(axis=1)
    return g.loc[mask].reset_index(drop=True)


def _construct_sample(name: str, config: dict, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    vintage = config["data_vintage"]
    exo_use = list(config["external_variables"])
    required = ENDO + exo_use
    base = panels[vintage]
    _validate_columns(base, required, vintage)

    g = _complete_country_rows(base, COUNTRY, required)
    drop_last_n = int(config.get("drop_last_n", 0))
    if drop_last_n < 0:
        raise ValueError("drop_last_n must be >= 0.")
    if drop_last_n:
        if drop_last_n >= len(g):
            raise ValueError(f"drop_last_n={drop_last_n} removes all rows for {name}.")
        g = g.iloc[:-drop_last_n].copy()

    append_jobs: list[tuple[str, int]] = []
    if int(config.get("append_new_n", 0)) > 0:
        append_jobs.append(("new", int(config["append_new_n"])))
    if int(config.get("append_old_n", 0)) > 0:
        append_jobs.append(("old", int(config["append_old_n"])))
    if int(config.get("append_other_n", 0)) > 0:
        other = "new" if vintage == "old" else "old"
        append_jobs.append((other, int(config["append_other_n"])))

    if append_jobs:
        last_q = pd.Timestamp(g["quarter"].max())
        for append_vintage, append_n in append_jobs:
            append_panel = panels[append_vintage]
            _validate_columns(append_panel, required, append_vintage)
            pool = _complete_country_rows(append_panel, COUNTRY, required)
            pool = pool[pool["quarter"] > last_q].sort_values("quarter").head(append_n)
            if len(pool) < append_n:
                raise ValueError(
                    f"{name}: requested {append_n} appended rows from {append_vintage}, "
                    f"but only found {len(pool)} after {last_q.to_period('Q')}."
                )
            g = pd.concat([g, pool], ignore_index=True).sort_values("quarter").reset_index(drop=True)
            last_q = pd.Timestamp(g["quarter"].max())

    return g


def _rolling_varx_coefficients(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, my = y.shape
    mx = x.shape[1]
    m = LAGS * my + mx
    theta = []
    idx = []
    for t in range(LAGS + 1, n):
        s0 = max(LAGS + 1, t - WINDOW)
        rows = t - s0
        if rows <= 0:
            continue
        x_train = np.zeros((rows, m))
        y_train = np.zeros((rows, my))
        for k, s in enumerate(range(s0, t)):
            reg_y = np.concatenate([y[s - i, :] for i in range(1, LAGS + 1)])
            x_train[k, :] = np.concatenate([reg_y, x[s - 1, :]])
            y_train[k, :] = y[s, :]
        coef = np.linalg.solve(x_train.T @ x_train + RIDGE * np.eye(m), x_train.T @ y_train)
        theta.append(coef.T.reshape(-1, order="C"))
        idx.append(t)
    return np.asarray(theta), np.asarray(idx, dtype=int)


def _prepare_run(name: str, config: dict, panels: dict[str, pd.DataFrame]) -> PreparedRun:
    g = _construct_sample(name, config, panels)
    exo_use = list(config["external_variables"])
    endo_use = [c for c in ENDO if c in g.columns]

    y_raw = g[endo_use].to_numpy(float)
    x_raw = g[exo_use].to_numpy(float)
    y_mu = np.nanmean(y_raw, axis=0)
    y_sd = np.nanstd(y_raw, axis=0) + EPS
    x_mu = np.nanmean(x_raw, axis=0)
    x_sd = np.nanstd(x_raw, axis=0) + EPS
    y = (y_raw - y_mu) / y_sd
    x = (x_raw - x_mu) / x_sd

    theta0, q0, r0, p0, m, p = init_from_varx_rolling(
        Y=y, Z=x, lags=LAGS, window=WINDOW, ridge=RIDGE, eps=EPS
    )
    rolling_theta, rolling_idx = _rolling_varx_coefficients(y, x)
    quarters = pd.DatetimeIndex(pd.to_datetime(g["quarter"]))
    rolling_quarters = pd.DatetimeIndex(quarters[rolling_idx]) if len(rolling_idx) else pd.DatetimeIndex([])

    return PreparedRun(
        name=name,
        config=config,
        g=g,
        quarters=quarters,
        endo_use=endo_use,
        exo_use=exo_use,
        y_raw=y_raw,
        x_raw=x_raw,
        y=y,
        x=x,
        y_mu=y_mu,
        y_sd=y_sd,
        x_mu=x_mu,
        x_sd=x_sd,
        theta0_init=theta0,
        q0_init=q0,
        r0_init=r0,
        p0_init=p0,
        m=m,
        p=p,
        rolling_theta=rolling_theta,
        rolling_quarters=rolling_quarters,
    )


def _eig_min_max(a: np.ndarray) -> tuple[float, float]:
    vals = np.linalg.eigvalsh(0.5 * (a + a.T))
    return float(vals.min()), float(vals.max())


def _max_abs_aligned(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape or a.size == 0:
        return np.nan
    return float(np.nanmax(np.abs(a - b)))


def _run_kf_em_trace(prep: PreparedRun) -> dict:
    theta0 = prep.theta0_init.copy()
    q_init = Q0_SCALE * prep.q0_init
    r_init = R0_SCALE * prep.r0_init
    p0_init = P0_SCALE * prep.p0_init
    q = q_init.copy()
    r = r_init.copy()
    p0 = p0_init.copy()

    history_rows = []
    coeff_by_iter = {}
    status = "ok"
    reason = ""

    def record(iteration: int, obj=np.nan, ll=np.nan, pack=None):
        q_min, q_max = _eig_min_max(q)
        r_min, r_max = _eig_min_max(r)
        p_min, p_max = _eig_min_max(p0)
        history_rows.append(
            {
                "run": prep.name,
                "iteration": iteration,
                "obj": obj,
                "log_likelihood": ll,
                "trace_Q": float(np.trace(q)),
                "abs_trace_Q": float(abs(np.trace(q))),
                "trace_R": float(np.trace(r)),
                "abs_trace_R": float(abs(np.trace(r))),
                "trace_P0": float(np.trace(p0)),
                "abs_trace_P0": float(abs(np.trace(p0))),
                "eig_min_Q": q_min,
                "eig_max_Q": q_max,
                "eig_min_R": r_min,
                "eig_max_R": r_max,
                "eig_min_P0": p_min,
                "eig_max_P0": p_max,
            }
        )
        if pack is None:
            _, _, theta_est, _, _, _, _ = kalman_multilag_filter(
                Y=prep.y,
                Z_all=prep.x,
                Q0=q,
                R0=r,
                P0=p0,
                theta0=theta0,
                lags=LAGS,
                eps=EPS,
                covariance_mode="fixed",
            )
        else:
            theta_est = pack["theta_smooth"]
        coeff_by_iter[iteration] = theta_est.copy()

    record(0)

    last_obj = np.inf
    best_pack = None
    for iteration in range(1, MAX_EM_ITER + 1):
        try:
            theta_filt, p_filt, theta_pred, p_pred, h_list, y_list, valid_mask = kf_e_step_store(
                Y=prep.y,
                Z_all=prep.x,
                theta0=theta0,
                Q=q,
                R=r,
                P0=p0,
                lags=LAGS,
                eps=EPS,
                em_iteration=iteration,
            )
            theta_smooth, p_smooth, j_hist = rts_smoother(
                theta_filt,
                p_filt,
                theta_pred,
                p_pred,
                valid_mask,
                eps=EPS,
                em_iteration=iteration,
            )
            q_new, r_new = em_m_step_update(
                Y=prep.y,
                H_list=h_list,
                y_list=y_list,
                valid_mask=valid_mask,
                theta_smooth=theta_smooth,
                P_smooth=p_smooth,
                J_hist=j_hist,
                Q_old=q,
                R_old=r,
                em_damping=0.0,
                eps=EPS,
                em_iteration=iteration,
            )

            valid_idx = np.where(valid_mask)[0]
            if len(valid_idx):
                theta0 = theta_smooth[valid_idx[0]].reshape(-1, 1)
                if not P0_FIXED:
                    p0 = p_smooth[valid_idx[0]]
            if not Q0_FIXED:
                q = q_new
            if not R0_FIXED:
                r = r_new

            obj = 0.0
            cnt = 0
            for t in valid_idx:
                err = y_list[t] - h_list[t] @ theta_smooth[t].reshape(-1, 1)
                obj += float((err.T @ err).item())
                cnt += 1
            obj /= max(cnt, 1)
            ll = _kf_log_likelihood(h_list, y_list, valid_mask, theta_pred, p_pred, r, eps=EPS)

            best_pack = {
                "theta_filt": theta_filt,
                "P_filt": p_filt,
                "theta_pred": theta_pred,
                "P_pred": p_pred,
                "theta_smooth": theta_smooth,
                "P_smooth": p_smooth,
                "J_hist": j_hist,
                "valid_mask": valid_mask,
                "H_list": h_list,
                "y_list": y_list,
            }
            record(iteration, obj=obj, ll=ll, pack=best_pack)
            if abs(last_obj - obj) < 1e-4:
                break
            last_obj = obj
        except NumericalInstabilityError as exc:
            status = "failed"
            reason = str(exc)
            break

    rmse, e_raw, theta_est, y_pred, p_hist, q_trace, r_trace = kalman_multilag_filter(
        Y=prep.y,
        Z_all=prep.x,
        Q0=q,
        R0=r,
        P0=p0,
        theta0=theta0,
        lags=LAGS,
        eps=EPS,
        covariance_mode="fixed",
    )

    return {
        "status": status,
        "reason": reason,
        "history": pd.DataFrame(history_rows),
        "coeff_by_iter": coeff_by_iter,
        "theta0": theta0,
        "Q": q,
        "R": r,
        "P0": p0,
        "Q_init_scaled": q_init,
        "R_init_scaled": r_init,
        "P0_init_scaled": p0_init,
        "rmse": rmse,
        "e_raw": e_raw,
        "theta_est": theta_est,
        "Y_pred": y_pred,
        "P_hist": p_hist,
        "Q_trace": q_trace,
        "R_trace": r_trace,
        "e_step_store": best_pack,
    }


def _sample_audit(base: PreparedRun, intervention: PreparedRun) -> tuple[pd.DataFrame, dict]:
    rows = []
    for prep in (base, intervention):
        q = prep.quarters.to_period("Q").astype(str).tolist()
        rows.append(
            {
                "run": prep.name,
                "country": COUNTRY,
                "data_vintage": prep.config["data_vintage"],
                "external_variables": ",".join(prep.exo_use),
                "first_estimation_quarter": q[0] if q else "",
                "last_estimation_quarter": q[-1] if q else "",
                "n_complete_observations": len(q),
                "quarters_present": ";".join(q),
            }
        )

    bq = set(base.quarters)
    iq = set(intervention.quarters)
    shared = sorted(bq & iq)
    only_base = sorted(bq - iq)
    only_int = sorted(iq - bq)
    shared_cols = [c for c in ENDO + base.exo_use if c in intervention.g.columns and c in base.g.columns]
    shared_equal = True
    max_shared_diff = 0.0
    changed_cols = []
    if shared and shared_cols:
        bg = base.g.set_index("quarter")
        ig = intervention.g.set_index("quarter")
        for col in shared_cols:
            diff = (
                pd.to_numeric(bg.loc[shared, col], errors="coerce").to_numpy(float)
                - pd.to_numeric(ig.loc[shared, col], errors="coerce").to_numpy(float)
            )
            finite = np.isfinite(diff)
            col_max = float(np.nanmax(np.abs(diff[finite]))) if finite.any() else np.nan
            if np.isfinite(col_max):
                max_shared_diff = max(max_shared_diff, col_max)
                if col_max > 0:
                    shared_equal = False
                    changed_cols.append(col)

    info = {
        "quarters_only_in_baseline": [q.to_period("Q").strftime("%YQ%q") for q in only_base],
        "quarters_only_in_intervention": [q.to_period("Q").strftime("%YQ%q") for q in only_int],
        "shared_observations_numerically_identical": bool(shared_equal),
        "shared_observation_max_abs_diff": float(max_shared_diff),
        "shared_changed_columns": changed_cols,
        "shared_compared_columns": shared_cols,
    }
    audit = pd.DataFrame(rows)
    for key, value in info.items():
        audit[key] = ";".join(map(str, value)) if isinstance(value, list) else value
    return audit, info


def _coeff_names(endo_use: list[str], exo_use: list[str]) -> list[str]:
    regressors = []
    for lag in range(1, LAGS + 1):
        regressors.extend([f"{v}_lag{lag}" for v in endo_use])
    regressors.extend(exo_use)
    return [f"{y}<-{x}" for y in endo_use for x in regressors]


def _aligned_coeff_diff(
    base_coeff: np.ndarray,
    int_coeff: np.ndarray,
    base: PreparedRun,
    intervention: PreparedRun,
) -> float:
    if base.exo_use != intervention.exo_use or base.endo_use != intervention.endo_use:
        return np.nan
    base_df = pd.DataFrame(base_coeff, index=base.quarters)
    int_df = pd.DataFrame(int_coeff, index=intervention.quarters)
    shared = base_df.index.intersection(int_df.index)
    if shared.empty:
        return np.nan
    return _max_abs_aligned(base_df.loc[shared].to_numpy(), int_df.loc[shared].to_numpy())


def _plot_original_data(base: PreparedRun, intervention: PreparedRun) -> None:
    cols = base.endo_use + sorted(set(base.exo_use) | set(intervention.exo_use))
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(11, max(2.2 * n, 5)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        if col in base.g.columns:
            ax.plot(base.quarters, base.g[col], label="baseline", lw=1.5)
        if col in intervention.g.columns:
            ax.plot(intervention.quarters, intervention.g[col], label="intervention", lw=1.2, ls="--")
        ax.set_title(col)
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    _finish_plot(fig, "original_data_comparison.png")


def _plot_em_history(base_res: dict, int_res: dict) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for label, res in [("baseline", base_res), ("intervention", int_res)]:
        h = res["history"]
        axes[0].plot(h["iteration"], h["abs_trace_Q"], marker="o", label=label)
        axes[1].plot(h["iteration"], h["abs_trace_R"], marker="o", label=label)
        axes[2].plot(h["iteration"], h["abs_trace_P0"], marker="o", label=label)
    for ax, title in zip(axes, ["abs trace(Q)", "abs trace(R)", "abs trace(P0)"]):
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("EM iteration, 0 = scaled VARX initialization")
    fig.tight_layout()
    _finish_plot(fig, "em_iteration_traces.png")


def _plot_rolling_varx(base: PreparedRun, intervention: PreparedRun) -> None:
    if base.exo_use != intervention.exo_use:
        return
    names = _coeff_names(base.endo_use, base.exo_use)
    picks = [i for i, name in enumerate(names) if any(f"<-{x}" in name for x in base.exo_use)]
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 1, figsize=(12, max(2.1 * len(picks), 5)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, picks):
        ax.plot(base.rolling_quarters, base.rolling_theta[:, idx], label="baseline")
        ax.plot(intervention.rolling_quarters, intervention.rolling_theta[:, idx], label="intervention", ls="--")
        ax.set_title(names[idx])
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    _finish_plot(fig, "rolling_varx_exogenous_coefficients.png")


def _plot_final_coefficients(base: PreparedRun, base_res: dict, intervention: PreparedRun, int_res: dict) -> None:
    if base.exo_use != intervention.exo_use:
        return
    names = _coeff_names(base.endo_use, base.exo_use)
    picks = [
        i
        for i, name in enumerate(names)
        if name.split("<-")[0] in ENDO and any(f"<-{x}" in name for x in base.exo_use)
    ]
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 1, figsize=(12, max(2.1 * len(picks), 5)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, picks):
        ax.plot(base.quarters, base_res["theta_est"][:, idx], label="baseline")
        ax.plot(intervention.quarters, int_res["theta_est"][:, idx], label="intervention", ls="--")
        ax.set_title(names[idx])
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    _finish_plot(fig, "final_kf_exogenous_coefficients.png")


def _forecast_from_result(prep: PreparedRun, res: dict) -> pd.DataFrame:
    pack = res.get("e_step_store")
    if not pack or not np.asarray(pack["valid_mask"], bool).any():
        return pd.DataFrame()
    exo_fc = gkf.build_forecast_exo_df_multi(
        prep.exo_use,
        COUNTRY,
        enso_map=gkf.FORECAST_ENSO_MEAN,
        target_quarters=gkf.FORECAST_TARGET_QUARTERS,
    )
    last_p = prep.quarters[-1].to_period("Q")
    exo_fc = exo_fc[
        exo_fc["target_quarter"].apply(lambda x: pd.Timestamp(x).to_period("Q") > last_p)
    ].copy()
    if exo_fc.empty or any(c not in exo_fc.columns for c in prep.exo_use):
        return pd.DataFrame()
    exo_fc = exo_fc.dropna(subset=prep.exo_use).reset_index(drop=True)
    if exo_fc.empty:
        return pd.DataFrame()

    valid_idx = np.where(np.asarray(pack["valid_mask"], bool))[0]
    last_t = int(valid_idx[-1])
    theta = np.asarray(pack["theta_filt"][last_t]).reshape(-1, 1)
    p = np.asarray(pack["P_filt"][last_t])
    z_fc = np.zeros((len(exo_fc), len(prep.exo_use)))
    for j, col in enumerate(prep.exo_use):
        raw = pd.to_numeric(exo_fc[col], errors="coerce").to_numpy(float)
        z_fc[:, j] = (raw - prep.x_mu[j]) / prep.x_sd[j]

    y_buf0 = [prep.y[-i, :].copy() for i in range(1, LAGS + 1)]
    y_hat, y_lo, y_hi = gkf._roll_kf_forecast(
        theta0=theta,
        P0=p,
        Q=res["Q"],
        R=res["R"],
        y_buf0=y_buf0,
        z_fc=z_fc,
        mY=len(prep.endo_use),
        m=prep.m,
        lags=LAGS,
        z_ci=gkf.Z_CI,
        eps=EPS,
        with_ci=True,
        mc_simulations=min(100, gkf.FORECAST_MC_SIMULATIONS),
        mc_seed=gkf.FORECAST_MC_SEED,
    )
    y_raw = y_hat * prep.y_sd + prep.y_mu
    lo_raw = y_lo * prep.y_sd + prep.y_mu
    hi_raw = y_hi * prep.y_sd + prep.y_mu
    rows = []
    for h, q in enumerate(exo_fc["target_quarter"]):
        for j, var in enumerate(prep.endo_use):
            rows.append(
                {
                    "run": prep.name,
                    "quarter": q,
                    "variable": var,
                    "forecast": y_raw[h, j],
                    "lower": lo_raw[h, j],
                    "upper": hi_raw[h, j],
                }
            )
    return pd.DataFrame(rows)


def _climate_counterfactual_drivers(exo_use: list[str]) -> list[str]:
    out = []
    for col in exo_use:
        if col == "OIL_YoY":
            continue
        if col in {"ENSO", "IOD", "HeatDry", "HeatDryF", "HeatDryYoY"} or "ENSO" in col or "RONI" in col or "HeatDry" in col:
            out.append(col)
    return out


def _forecast_scenarios_from_result(prep: PreparedRun, res: dict) -> pd.DataFrame:
    pack = res.get("e_step_store")
    if not pack or not np.asarray(pack["valid_mask"], bool).any():
        return pd.DataFrame()
    exo_fc = gkf.build_forecast_exo_df_multi(
        prep.exo_use,
        COUNTRY,
        enso_map=gkf.FORECAST_ENSO_MEAN,
        target_quarters=gkf.FORECAST_TARGET_QUARTERS,
    )
    last_p = prep.quarters[-1].to_period("Q")
    exo_fc = exo_fc[
        exo_fc["target_quarter"].apply(lambda x: pd.Timestamp(x).to_period("Q") > last_p)
    ].copy()
    if exo_fc.empty or any(c not in exo_fc.columns for c in prep.exo_use):
        return pd.DataFrame()
    exo_fc = exo_fc.dropna(subset=prep.exo_use).reset_index(drop=True)
    if exo_fc.empty:
        return pd.DataFrame()

    valid_idx = np.where(np.asarray(pack["valid_mask"], bool))[0]
    last_t = int(valid_idx[-1])
    theta = np.asarray(pack["theta_filt"][last_t]).reshape(-1, 1)
    p = np.asarray(pack["P_filt"][last_t])

    z_factual = np.zeros((len(exo_fc), len(prep.exo_use)))
    for j, col in enumerate(prep.exo_use):
        raw = pd.to_numeric(exo_fc[col], errors="coerce").to_numpy(float)
        z_factual[:, j] = (raw - prep.x_mu[j]) / prep.x_sd[j]

    z_counterfactual = z_factual.copy()
    cf_drivers = _climate_counterfactual_drivers(prep.exo_use)
    for driver in cf_drivers:
        j = prep.exo_use.index(driver)
        z_counterfactual[:, j] = (0.0 - prep.x_mu[j]) / prep.x_sd[j]

    y_buf0 = [prep.y[-i, :].copy() for i in range(1, LAGS + 1)]
    rows = []
    for scenario, z_fc in [("climate", z_factual), ("counterfactual_zero_climate", z_counterfactual)]:
        y_hat, y_lo, y_hi = gkf._roll_kf_forecast(
            theta0=theta,
            P0=p,
            Q=res["Q"],
            R=res["R"],
            y_buf0=[v.copy() for v in y_buf0],
            z_fc=z_fc,
            mY=len(prep.endo_use),
            m=prep.m,
            lags=LAGS,
            z_ci=gkf.Z_CI,
            eps=EPS,
            with_ci=True,
            mc_simulations=min(100, gkf.FORECAST_MC_SIMULATIONS),
            mc_seed=gkf.FORECAST_MC_SEED,
        )
        y_raw = y_hat * prep.y_sd + prep.y_mu
        lo_raw = y_lo * prep.y_sd + prep.y_mu
        hi_raw = y_hi * prep.y_sd + prep.y_mu
        for h, q in enumerate(exo_fc["target_quarter"]):
            for j, var in enumerate(prep.endo_use):
                rows.append(
                    {
                        "run": prep.name,
                        "scenario": scenario,
                        "counterfactual_drivers": ",".join(cf_drivers),
                        "quarter": q,
                        "variable": var,
                        "forecast": y_raw[h, j],
                        "lower": lo_raw[h, j],
                        "upper": hi_raw[h, j],
                    }
                )
    return pd.DataFrame(rows)


def _plot_climate_counterfactual(prep: PreparedRun, res: dict) -> None:
    fc = _forecast_scenarios_from_result(prep, res)
    if fc.empty:
        return
    drivers = sorted(set(fc["counterfactual_drivers"].dropna()))
    driver_label = drivers[0] if drivers else ""
    if not driver_label:
        return

    fig, axes = plt.subplots(len(FORECAST_VARS), 1, figsize=(10, 6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, var in zip(axes, FORECAST_VARS):
        d = fc[fc["variable"].eq(var)]
        for scenario, g in d.groupby("scenario"):
            if scenario == "climate":
                label = f"{prep.name} climate path"
                color = "C0"
                ls = "-"
                marker = "o"
            else:
                label = f"{prep.name} {driver_label}=0"
                color = "C3"
                ls = "--"
                marker = "x"
            ax.plot(g["quarter"], g["forecast"], marker=marker, markersize=3, linestyle=ls, color=color, label=label)
            ax.fill_between(g["quarter"], g["lower"], g["upper"], color=color, alpha=0.10)
        ax.set_title(f"{prep.name.title()} {var}: climate scenario vs counterfactual")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    _finish_plot(fig, f"{prep.name}_climate_vs_counterfactual_forecast.png")

    fc.to_csv(OUT_DIR / f"{prep.name}_climate_vs_counterfactual_forecast.csv", index=False)


def _plot_forecasts(fc: pd.DataFrame) -> None:
    if fc.empty:
        return
    fig, axes = plt.subplots(len(FORECAST_VARS), 1, figsize=(10, 6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, var in zip(axes, FORECAST_VARS):
        d = fc[fc["variable"].eq(var)]
        for run, g in d.groupby("run"):
            ax.plot(g["quarter"], g["forecast"], marker="o", label=run)
            ax.fill_between(g["quarter"], g["lower"], g["upper"], alpha=0.12)
        ax.set_title(f"{var} forecast comparison")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    _finish_plot(fig, "forecast_comparison_gdp_cpi.png")


def _finish_plot(fig: plt.Figure, filename: str) -> None:
    if FLAG_SAVE_PLOTS:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOT_DIR / filename, dpi=180, bbox_inches="tight")
    if FLAG_SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def _comparison_summary(base: PreparedRun, base_res: dict, intervention: PreparedRun, int_res: dict, fc: pd.DataFrame) -> pd.DataFrame:
    rolling_diff = np.nan
    if base.exo_use == intervention.exo_use:
        bd = pd.DataFrame(base.rolling_theta, index=base.rolling_quarters)
        idf = pd.DataFrame(intervention.rolling_theta, index=intervention.rolling_quarters)
        shared = bd.index.intersection(idf.index)
        if not shared.empty:
            rolling_diff = _max_abs_aligned(bd.loc[shared].to_numpy(), idf.loc[shared].to_numpy())

    final_coeff_diff = _aligned_coeff_diff(base_res["theta_est"], int_res["theta_est"], base, intervention)
    fc_pivot = fc.pivot_table(index=["quarter", "variable"], columns="run", values="forecast")
    fc_diffs = {}
    for var in FORECAST_VARS:
        if {"baseline", "intervention"}.issubset(fc_pivot.columns):
            d = fc_pivot.xs(var, level="variable", drop_level=False)
            fc_diffs[f"{var}_forecast_max_abs_diff"] = float(
                np.nanmax(np.abs(d["intervention"] - d["baseline"]))
            )
        else:
            fc_diffs[f"{var}_forecast_max_abs_diff"] = np.nan

    row = {
        "country": COUNTRY,
        "baseline_sample_size": len(base.quarters),
        "intervention_sample_size": len(intervention.quarters),
        "baseline_last_estimation_quarter": base.quarters[-1].to_period("Q").strftime("%YQ%q"),
        "intervention_last_estimation_quarter": intervention.quarters[-1].to_period("Q").strftime("%YQ%q"),
        "baseline_initial_trace_Q0": float(np.trace(base_res["Q_init_scaled"])),
        "baseline_initial_trace_R0": float(np.trace(base_res["R_init_scaled"])),
        "baseline_initial_trace_P0": float(np.trace(base_res["P0_init_scaled"])),
        "intervention_initial_trace_Q0": float(np.trace(int_res["Q_init_scaled"])),
        "intervention_initial_trace_R0": float(np.trace(int_res["R_init_scaled"])),
        "intervention_initial_trace_P0": float(np.trace(int_res["P0_init_scaled"])),
        "baseline_final_trace_Q": float(np.trace(base_res["Q"])),
        "baseline_final_trace_R": float(np.trace(base_res["R"])),
        "baseline_final_trace_P0": float(np.trace(base_res["P0"])),
        "intervention_final_trace_Q": float(np.trace(int_res["Q"])),
        "intervention_final_trace_R": float(np.trace(int_res["R"])),
        "intervention_final_trace_P0": float(np.trace(int_res["P0"])),
        "max_abs_rolling_varx_coeff_diff": rolling_diff,
        "max_abs_final_kf_coeff_path_diff": final_coeff_diff,
    }
    row.update(fc_diffs)
    return pd.DataFrame([row])


def _coeff_diff_by_iter(base: PreparedRun, base_res: dict, intervention: PreparedRun, int_res: dict) -> pd.DataFrame:
    rows = []
    for iteration in sorted(set(base_res["coeff_by_iter"]) & set(int_res["coeff_by_iter"])):
        rows.append(
            {
                "iteration": iteration,
                "max_abs_coeff_path_diff": _aligned_coeff_diff(
                    base_res["coeff_by_iter"][iteration],
                    int_res["coeff_by_iter"][iteration],
                    base,
                    intervention,
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = {"old": _load_panel("old"), "new": _load_panel("new")}

    print("Available external columns in old panel:")
    print([c for c in panels["old"].columns if c not in {"country", "quarter"} and c not in ENDO])
    print("Available external columns in new panel:")
    print([c for c in panels["new"].columns if c not in {"country", "quarter"} and c not in ENDO])

    base = _prepare_run("baseline", BASELINE, panels)
    intervention = _prepare_run("intervention", INTERVENTION, panels)

    audit, audit_info = _sample_audit(base, intervention)
    audit.to_csv(OUT_DIR / "sample_audit.csv", index=False)
    print("\nSample audit")
    print(audit.drop(columns=["quarters_present"]).to_string(index=False))
    print("Quarters only in baseline:", audit_info["quarters_only_in_baseline"])
    print("Quarters only in intervention:", audit_info["quarters_only_in_intervention"])
    print("Shared observations identical:", audit_info["shared_observations_numerically_identical"])
    print("Shared max abs diff:", audit_info["shared_observation_max_abs_diff"])

    base_res = _run_kf_em_trace(base)
    int_res = _run_kf_em_trace(intervention)
    pd.concat([base_res["history"], int_res["history"]], ignore_index=True).to_csv(
        OUT_DIR / "em_iteration_traces.csv", index=False
    )

    fc = pd.concat(
        [_forecast_from_result(base, base_res), _forecast_from_result(intervention, int_res)],
        ignore_index=True,
    )
    if not fc.empty:
        fc.to_csv(OUT_DIR / "forecast_comparison.csv", index=False)

    summary = _comparison_summary(base, base_res, intervention, int_res, fc)
    summary.to_csv(OUT_DIR / "baseline_intervention_summary.csv", index=False)
    _coeff_diff_by_iter(base, base_res, intervention, int_res).to_csv(
        OUT_DIR / "coefficient_difference_by_em_iteration.csv", index=False
    )

    if FLAG_PLOT_ORIGINAL_DATA_COMPARISON:
        _plot_original_data(base, intervention)
    if FLAG_PLOT_VARX_COEFF:
        _plot_rolling_varx(base, intervention)
    if FLAG_PLOT_EM_ITER:
        _plot_em_history(base_res, int_res)
    if FLAG_PLOT_COEFF_PATH:
        _plot_final_coefficients(base, base_res, intervention, int_res)
    if FLAG_PLOT_FORECAST_COMPARISON:
        _plot_forecasts(fc)
    if FLAG_PLOT_CLIMATE_COUNTERFACTUAL:
        _plot_climate_counterfactual(base, base_res)
        _plot_climate_counterfactual(intervention, int_res)

    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nWrote diagnostic outputs to: {OUT_DIR}")
    if FLAG_SAVE_PLOTS:
        print(f"Wrote plots to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
