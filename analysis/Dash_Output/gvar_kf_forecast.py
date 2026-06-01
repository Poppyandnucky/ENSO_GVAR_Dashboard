"""
Multi-step KF forecast after EM estimation (GVAR_LLM_pickle pipeline).

Uses the final EM/KF state and user-supplied exogenous forecasts (ENSO, COMMODITY).
Forecast bands are generated offline by Monte Carlo draws from the last filtered
parameter covariance P. Each simulated beta is held fixed across the future path
(Q = R = 0 in the forecast simulation), so uncertainty enters through beta and
the nonlinear recursion of lagged predicted y.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os

os.environ.setdefault("GVAR_IMPORT_ONLY", "1")
_ROOT = Path(__file__).resolve().parents[2]
_STRUCTURAL_BREAK_DIR = _ROOT / "structural_break"
if str(_STRUCTURAL_BREAK_DIR) not in sys.path:
    sys.path.insert(0, str(_STRUCTURAL_BREAK_DIR))
import GVAR_LLM_pickle as gp  # noqa: E402

# -----------------------------------------------------------------------------
# User forecast inputs (edit here)
# -----------------------------------------------------------------------------
FORECAST_ENSO_MEAN = {
    "2025Q3": -0.6055,
    "2025Q4": -0.9034,
    "2026Q1": -0.6499,
    "2026Q2": 0.4286,
    "2026Q3": 1.7784,
    "2026Q4": 2.2875,
    "2027Q1": 1.8143,
}

FORECAST_ENSO_MIN = {
    "2025Q3": -0.6055,
    "2025Q4": -0.9034,
    "2026Q1": -0.6499,
    "2026Q2": 0.4286,
    "2026Q3": 0.9385,
    "2026Q4": 1.4430,
    "2027Q1": 0.8575,
}

FORECAST_ENSO_MAX = {
    "2025Q3": -0.6055,
    "2025Q4": -0.9034,
    "2026Q1": -0.6499,
    "2026Q2": 0.4286,
    "2026Q3": 2.0330,
    "2026Q4": 3.0678,
    "2027Q1": 2.2604,
}

FORECAST_ENSO_SCENARIOS: dict[str, dict[str, float]] = {
    "mean": FORECAST_ENSO_MEAN,
    "min": FORECAST_ENSO_MIN,
    "max": FORECAST_ENSO_MAX,
}
# Backward-compatible alias (mean scenario).
FORECAST_ENSO = FORECAST_ENSO_MEAN

# COMMODITY_YoY forecast (quarter-end dates)
FORECAST_COMMODITY = {
    "2025-03-31": -1.520088,
    "2025-06-30": 2.171268,
    "2025-09-30": 5.290896,
    "2025-12-31": 7.293260,
    "2026-03-31": 8.207627,
    "2026-06-30": 8.339559,
    "2026-09-30": 8.049436,
    "2026-12-31": 7.628315,
    "2027-03-31": 7.255114,
}

# If True, also forecast 2025Q1–Q2 etc. when COMMODITY exists (missing ENSO → 0 in z-space).
EXTEND_TARGETS_TO_COMMODITY_RANGE = True


def _quarter_start(ts) -> pd.Timestamp:
    """Match panel `quarter` column: period start (e.g. 2024Q4 → 2024-10-01)."""
    return pd.to_datetime(ts).to_period("Q").to_timestamp()


def _all_forecast_target_quarters() -> list[pd.Timestamp]:
    enso_keys: set[str] = set()
    for enso_map in FORECAST_ENSO_SCENARIOS.values():
        enso_keys.update(enso_map)
    enso_q = {pd.Period(str(k).strip().upper(), freq="Q").to_timestamp() for k in enso_keys}
    comm_q = {_quarter_start(pd.to_datetime(k)) for k in FORECAST_COMMODITY}
    if EXTEND_TARGETS_TO_COMMODITY_RANGE:
        p_min = min(comm_q)
        p_max = max(enso_q | comm_q)
    else:
        p_min = min(enso_q)
        p_max = max(enso_q)
    targets: list[pd.Timestamp] = []
    p = p_min.to_period("Q")
    p_end = p_max.to_period("Q")
    while p <= p_end:
        targets.append(p.to_timestamp())
        p += 1
    return targets


Z_CI = 1.96
FORECAST_MC_SIMULATIONS = 500
FORECAST_MC_SEED = 20260531
FORECAST_MC_METHOD = "beta_q_random_walk_monte_carlo"
FORECAST_MC_BAND_METHOD = "mean_plus_minus_one_std"
FORECAST_MC_INTERVAL_LABEL = "+/- 1 standard deviation"
# Legacy single-scenario output dirs (mean); multi-scenario uses _scenario_output_dirs().
OUTPUT_DIR = Path("Dash_Output/forecast")
OUTPUT_DIR_KF_TRACK = Path("Dash_Output/forecast_kf_track")
OUTPUT_DIR_VARX_TRACK = Path("Dash_Output/forecast_varx_track")
FORECAST_PICKLE_PATH = Path("Dash_Input/gvar_forecast_results.pkl")


def _scenario_output_dirs(scenario: str) -> dict[str, Path]:
    """Per ENSO scenario: forecast / kf_track / varx_track under Dash_Output/forecast_enso_{scenario}/."""
    base = Path(f"Dash_Output/forecast_enso_{scenario}")
    return {
        "forecast": base / "forecast",
        "kf_track": base / "kf_track",
        "varx_track": base / "varx_track",
    }
MAX_EM_ITER = 10
VARX_ROLLING_WINDOW = 40
# Open-loop multi-step forecasts can explode when the last filtered VAR block has
# roots outside the unit circle. Keep the fitted in-sample path untouched, but
# stabilize the forecast-only transition before recursive rollout.
STABILIZE_FORECAST = False
FORECAST_MAX_EIGENVALUE = 0.98
# Only show the last N quarters of history so the forecast segment is visible.
HIST_PLOT_QUARTERS = 24
# KF track plots: in-sample segment starts at this quarter (period start).
KF_TRACK_PLOT_START = "1990-01-01"

# Restrict forecast/track plots to this short list of countries.
# Original full list comes from `gp.countries_to_run` (kept as fallback below, commented).
DEFAULT_COUNTRIES: list[str] = [
    "BRA",  # Brazil
    "CHL",  # Chile
    "COL",  # Colombia
    "MEX",  # Mexico
    "KEN",  # Kenya
    "ZAF",  # South Africa
    "IND",  # India
    "IDN",  # Indonesia
    "THA",  # Thailand
    "PER",  # Peru
    "PHL",  # Philippines
    "EGY",  # Egypt
]


def _period_ts(q: str | pd.Timestamp) -> pd.Timestamp:
    if isinstance(q, pd.Timestamp):
        return _quarter_start(q)
    s = str(q).strip().upper().replace("-", "")
    if len(s) == 6 and s[4] == "Q":
        return pd.Period(s, freq="Q").to_timestamp()
    return _quarter_start(q)


FORECAST_TARGET_QUARTERS = _all_forecast_target_quarters()


def _panel_exog_history() -> pd.DataFrame:
    """Load unique quarterly exogenous values from the panel when available."""
    try:
        raw = pd.read_csv(gp.PATH, usecols=["quarter", "ENSO", "COMMODITY_YoY"])
    except Exception as exc:
        print(f"[WARN] could not load panel exogenous history from {gp.PATH}: {exc}")
        return pd.DataFrame(columns=["quarter", "ENSO", "COMMODITY_YoY"])
    raw["quarter"] = pd.to_datetime(raw["quarter"]).dt.to_period("Q").dt.to_timestamp()
    hist = (
        raw.groupby("quarter", as_index=False)[["ENSO", "COMMODITY_YoY"]]
        .first()
        .sort_values("quarter")
    )
    return hist


def build_forecast_exo_df(
    enso_map: dict[str, float] | None = None,
    commodity_map: dict[str, float] | None = None,
    target_quarters: list[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Build exogenous panel, preferring actual panel values before forecast values."""
    enso_map = enso_map or FORECAST_ENSO_MEAN
    commodity_map = commodity_map or FORECAST_COMMODITY

    enso_s = {_period_ts(k): float(v) for k, v in enso_map.items()}
    comm_s = {_quarter_start(pd.to_datetime(k)): float(v) for k, v in commodity_map.items()}
    exog_hist = _panel_exog_history()
    enso_actual = dict(
        zip(exog_hist["quarter"], pd.to_numeric(exog_hist["ENSO"], errors="coerce"))
    )
    comm_actual = dict(
        zip(exog_hist["quarter"], pd.to_numeric(exog_hist["COMMODITY_YoY"], errors="coerce"))
    )

    if target_quarters is None:
        target_quarters = list(FORECAST_TARGET_QUARTERS)
    else:
        target_quarters = [_period_ts(q) for q in target_quarters]

    rows = []
    for tq in target_quarters:
        zq = (tq.to_period("Q") - 1).to_timestamp()
        enso_val = enso_actual.get(tq, np.nan)
        comm_val = comm_actual.get(zq, np.nan)
        enso_source = "panel" if np.isfinite(enso_val) else "forecast"
        comm_source = "panel" if np.isfinite(comm_val) else "forecast"
        rows.append(
            {
                "target_quarter": tq,
                "exo_quarter": zq,
                # ENSO is keyed by target quarter; COMMODITY is used with lag t-1.
                "ENSO": enso_val if enso_source == "panel" else enso_s.get(tq, np.nan),
                "COMMODITY_YoY": comm_val if comm_source == "panel" else comm_s.get(zq, np.nan),
                "ENSO_source": enso_source,
                "COMMODITY_YoY_source": comm_source,
            }
        )
    out = pd.DataFrame(rows).sort_values("target_quarter").reset_index(drop=True)
    if out[["ENSO", "COMMODITY_YoY"]].isna().any().any():
        missing = out[out[["ENSO", "COMMODITY_YoY"]].isna().any(axis=1)]
        print("[WARN] forecast exo missing values for some lags:")
        print(missing.to_string(index=False))
    return out


def _marginal_bounds_from_S(
    yhat: np.ndarray,
    S: np.ndarray,
    *,
    z_ci: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Marginal 95% bounds per endogenous: yhat ± z * sqrt(diag(S))."""
    se = np.sqrt(np.clip(np.diag(S), 0.0, None))
    yhat = np.asarray(yhat, dtype=float).ravel()
    return yhat - z_ci * se, yhat + z_ci * se


def _insample_kf_track(
    prep: dict,
    pack: dict,
    R: np.ndarray,
    *,
    z_ci: float = Z_CI,
    eps: float = 1e-8,
) -> dict:
    """In-sample one-step-ahead KF path: y_hat = H @ theta_pred, S = H P_pred H' + R."""
    valid = np.asarray(pack["valid_mask"], dtype=bool)
    Yd = np.asarray(prep["Yd"], dtype=float)
    quarters = pd.to_datetime(prep["quarters"])
    mY = Yd.shape[1]
    R = np.asarray(R, dtype=float)

    y_hat = np.full_like(Yd, np.nan)
    y_lo = np.full_like(Yd, np.nan)
    y_hi = np.full_like(Yd, np.nan)

    for t in np.where(valid)[0]:
        H = pack["H_list"][t]
        if H is None:
            continue
        theta_pr = np.asarray(pack["theta_pred"][t], dtype=float).reshape(-1, 1)
        P_pr = np.asarray(pack["P_pred"][t], dtype=float)
        yh = (H @ theta_pr).ravel()
        S = H @ P_pr @ H.T + R
        S = 0.5 * (S + S.T) + eps * np.eye(mY)
        lo, hi = _marginal_bounds_from_S(yh, S, z_ci=z_ci)
        y_hat[t, :] = yh
        y_lo[t, :] = lo
        y_hi[t, :] = hi

    mask = valid & np.isfinite(Yd).all(axis=1)
    q = pd.to_datetime([_quarter_start(x) for x in quarters[mask]])
    return {
        "quarters": q,
        "y": Yd[mask],
        "y_hat": y_hat[mask],
        "y_lower": y_lo[mask],
        "y_upper": y_hi[mask],
    }

def _varx_window_residual_se(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    coef: np.ndarray,
    *,
    min_obs: int = 3,
) -> np.ndarray:
    """Per-equation residual std from the current rolling training window."""
    resid = Y_train - X_train @ coef
    mY = Y_train.shape[1]
    se = np.full(mY, np.nan)
    for j in range(mY):
        r = resid[:, j]
        r = r[np.isfinite(r)]
        if r.size >= min_obs:
            se[j] = float(np.std(r, ddof=1))
        elif r.size >= 1:
            se[j] = float(np.std(r, ddof=0))
    fallback = np.nanstd(resid, axis=0)
    se = np.where(np.isfinite(se) & (se > 0), se, fallback)
    se = np.where(np.isfinite(se) & (se > 0), se, 1e-8)
    return se


def _varx_insample_track(
    prep: dict,
    *,
    kf_mask: np.ndarray | None = None,
    window: int = VARX_ROLLING_WINDOW,
    ridge: float = 1e-6,
    z_ci: float = Z_CI,
) -> dict:
    """Rolling VARX 1-step path; 95% band = y_hat ± z * residual SE in each fit window."""
    Yd = np.asarray(prep["Yd"], dtype=float)
    Xd = np.asarray(prep["Xd"], dtype=float)
    lags = int(gp.lags)
    n, mY = Yd.shape
    m = lags * mY + Xd.shape[1]

    y_hat = np.full((n, mY), np.nan)
    y_lo = np.full((n, mY), np.nan)
    y_hi = np.full((n, mY), np.nan)

    for t in range(lags + 1, n - 1):
        s0 = max(lags + 1, t - 1 - window + 1) if window is not None else lags + 1
        T = t - s0
        if T <= 0:
            continue

        X_train = np.zeros((T, m))
        Y_train = np.zeros((T, mY))
        for k, s in enumerate(range(s0, t)):
            reg_y = np.concatenate([Yd[s - i, :] for i in range(1, lags + 1)], axis=0)
            X_train[k, :] = np.concatenate([reg_y, Xd[s - 1, :]], axis=0)
            Y_train[k, :] = Yd[s, :]

        coef = np.linalg.solve(X_train.T @ X_train + ridge * np.eye(m), X_train.T @ Y_train)
        reg_next = np.concatenate([Yd[t + 1 - i, :] for i in range(1, lags + 1)], axis=0)
        yhat = np.concatenate([reg_next, Xd[t, :]], axis=0) @ coef
        se = _varx_window_residual_se(X_train, Y_train, coef)
        y_hat[t + 1, :] = yhat
        y_lo[t + 1, :] = yhat - z_ci * se
        y_hi[t + 1, :] = yhat + z_ci * se

    if kf_mask is None:
        kf_mask = np.isfinite(Yd).all(axis=1) & np.isfinite(y_hat).any(axis=1)
    else:
        kf_mask = np.asarray(kf_mask, dtype=bool)

    quarters = pd.to_datetime(prep["quarters"])
    q = pd.to_datetime([_quarter_start(x) for x in quarters[kf_mask]])
    return {
        "quarters": q,
        "y": Yd[kf_mask],
        "y_hat": y_hat[kf_mask],
        "y_lower": y_lo[kf_mask],
        "y_upper": y_hi[kf_mask],
    }


def _varx_multistep_forecast(
    prep: dict,
    *,
    last_t: int,
    z_fc: np.ndarray,
    window: int = VARX_ROLLING_WINDOW,
    ridge: float = 1e-6,
    z_ci: float = Z_CI,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-step VARX; band = y_hat ± z * residual SE from each step's rolling window."""
    Yd = np.asarray(prep["Yd"], dtype=float)
    Xd = np.asarray(prep["Xd"], dtype=float)
    lags = int(gp.lags)
    n_fc = z_fc.shape[0]
    mY = Yd.shape[1]
    m = lags * mY + Xd.shape[1]

    Y_work = np.vstack([Yd, np.full((n_fc, mY), np.nan)])
    Z_work = np.vstack([Xd, z_fc])
    y_hat_fc = np.full((n_fc, mY), np.nan)
    y_lo_fc = np.full((n_fc, mY), np.nan)
    y_hi_fc = np.full((n_fc, mY), np.nan)

    for h in range(n_fc):
        t = last_t + h
        s0 = max(lags + 1, t - 1 - window + 1) if window is not None else lags + 1
        T = t - s0
        if T <= 0:
            continue

        X_train = np.zeros((T, m))
        Y_train = np.zeros((T, mY))
        for k, s in enumerate(range(s0, t)):
            reg_y = np.concatenate([Y_work[s - i, :] for i in range(1, lags + 1)], axis=0)
            X_train[k, :] = np.concatenate([reg_y, Z_work[s - 1, :]], axis=0)
            Y_train[k, :] = Y_work[s, :]

        coef = np.linalg.solve(X_train.T @ X_train + ridge * np.eye(m), X_train.T @ Y_train)
        se = _varx_window_residual_se(X_train, Y_train, coef)
        reg_next = np.concatenate([Y_work[t + 1 - i, :] for i in range(1, lags + 1)], axis=0)
        z_row = z_fc[h, :]
        if not np.isfinite(z_row).all():
            z_row = np.where(np.isfinite(z_row), z_row, 0.0)
        yhat = np.concatenate([reg_next, z_row], axis=0) @ coef
        Y_work[t + 1, :] = yhat
        y_hat_fc[h, :] = yhat
        y_lo_fc[h, :] = yhat - z_ci * se
        y_hi_fc[h, :] = yhat + z_ci * se

    return y_hat_fc, y_lo_fc, y_hi_fc


def _roll_forecast_path(
    *,
    theta: np.ndarray,
    y_buf0: list[np.ndarray],
    z_fc: np.ndarray,
    mY: int,
    m: int,
    lags: int,
) -> np.ndarray:
    """Recursive deterministic forecast for one fixed beta draw."""
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


def _chol_lower_psd(cov: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Lower factor for a nearly-PSD covariance, with eigenvalue fallback."""
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    eye = np.eye(cov.shape[0])
    for scale in (1.0, 10.0, 100.0, 1000.0):
        try:
            return np.linalg.cholesky(cov + scale * eps * eye)
        except np.linalg.LinAlgError:
            continue

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(np.sqrt(vals))


def _roll_kf_forecast(
    *,
    theta0: np.ndarray,
    P0: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    y_buf0: list[np.ndarray],
    z_fc: np.ndarray,
    mY: int,
    m: int,
    lags: int,
    z_ci: float,
    eps: float,
    with_ci: bool = True,
    mc_simulations: int = FORECAST_MC_SIMULATIONS,
    mc_seed: int = FORECAST_MC_SEED,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Multi-step forecast; optional bands are offline beta/Q Monte Carlo.

    P0, R, and z_ci are kept in the signature for backward compatibility. The
    forecast band intentionally sets P=0 and R=0, starts from the last filtered
    beta as known, and evolves beta forward with the Kalman-filter Q.
    """
    _ = (P0, R, z_ci)
    theta = np.asarray(theta0, dtype=float).reshape(-1, 1)
    y_hat = _roll_forecast_path(
        theta=theta,
        y_buf0=y_buf0,
        z_fc=z_fc,
        mY=mY,
        m=m,
        lags=lags,
    )
    if not with_ci:
        return y_hat, None, None

    n_sim = max(1, int(mc_simulations))
    L = _chol_lower_psd(Q, eps=eps)
    rng = np.random.default_rng(int(mc_seed))
    paths = np.full((n_sim, z_fc.shape[0], mY), np.nan)
    n_pairs = int(np.ceil(n_sim / 2))
    raw_draws = rng.standard_normal((theta.shape[0], n_pairs, z_fc.shape[0]))
    draws = np.concatenate([raw_draws, -raw_draws], axis=1)[:, :n_sim, :]
    for s in range(n_sim):
        beta_draw = theta.copy()
        y_buf = [y.copy() for y in y_buf0]
        for h in range(z_fc.shape[0]):
            beta_draw = beta_draw + L @ draws[:, [s], h]
            z_row = z_fc[h, :]
            if not np.isfinite(z_row).all():
                z_row = np.where(np.isfinite(z_row), z_row, 0.0)
            H = _build_H(y_buf, z_row, mY, m, lags)
            yhat_s = (H @ beta_draw).ravel()
            paths[s, h, :] = yhat_s
            y_buf = [yhat_s.copy()] + y_buf[:-1]

    sim_sd = np.nanstd(paths, axis=0, ddof=1)
    y_lo = y_hat - sim_sd
    y_hi = y_hat + sim_sd
    return y_hat, y_lo, y_hi


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
) -> tuple[np.ndarray, dict]:
    """Scale endogenous lag coefficients if the forecast companion is unstable."""
    theta = np.asarray(theta, dtype=float).reshape(-1, 1)
    B = theta.reshape(mY, m, order="C").copy()
    a_blocks = [B[:, i * mY : (i + 1) * mY].copy() for i in range(lags)]
    rho_before = _spectral_radius(a_blocks)

    info = {
        "enabled": bool(STABILIZE_FORECAST),
        "applied": False,
        "max_eigenvalue": float(max_eigenvalue),
        "rho_before": rho_before,
        "rho_after": rho_before,
        "scale": 1.0,
    }
    if (not STABILIZE_FORECAST) or (not np.isfinite(rho_before)) or rho_before <= max_eigenvalue:
        return theta, info

    scale = max_eigenvalue / max(rho_before, 1e-12)
    for i in range(lags):
        B[:, i * mY : (i + 1) * mY] *= scale
    a_blocks_after = [B[:, i * mY : (i + 1) * mY].copy() for i in range(lags)]
    rho_after = _spectral_radius(a_blocks_after)
    info.update(
        {
            "applied": True,
            "rho_after": rho_after,
            "scale": float(scale),
        }
    )
    return B.reshape(-1, 1, order="C"), info


def _build_H(Y_lags: list[np.ndarray], z_row: np.ndarray, mY: int, m: int, lags: int) -> np.ndarray:
    pieces = []
    for y_l in Y_lags:
        pieces.append(np.asarray(y_l, dtype=float).ravel())
    if z_row is not None and len(z_row) > 0:
        pieces.append(np.asarray(z_row, dtype=float).ravel())
    x_t = np.concatenate(pieces)
    H = np.zeros((mY, mY * m))
    for j in range(mY):
        H[j, j * m : (j + 1) * m] = x_t
    return H


def forecast_country_from_em(
    prep: dict,
    res: dict,
    exo_fc: pd.DataFrame,
    *,
    country: str = "",
    scenario_name: str = "mean",
    z_ci: float = Z_CI,
    eps: float = 1e-8,
) -> dict | None:
    """Roll KF forward using last EM state; no measurement updates in forecast."""
    pack = res.get("e_step_store")
    if not pack:
        return None

    valid = np.asarray(pack["valid_mask"], dtype=bool)
    if not np.any(valid):
        return None
    last_t = int(np.where(valid)[0][-1])

    theta = np.asarray(pack["theta_filt"][last_t], dtype=float).reshape(-1, 1)
    P = np.asarray(pack["P_filt"][last_t], dtype=float)
    Q = np.asarray(res["Q"], dtype=float)
    R = np.asarray(res["R"], dtype=float)

    Yd = np.asarray(prep["Yd"], dtype=float)
    Xd = np.asarray(prep["Xd"], dtype=float)
    g = prep["g"]
    ENDO_use = list(prep["ENDO_use"])
    EXO_use = list(prep["EXO_use"])
    lags = int(gp.lags)
    mY = len(ENDO_use)
    mX = len(EXO_use)
    m = lags * mY + mX
    theta_raw = theta.copy()
    theta, forecast_stability = _stabilize_theta_for_forecast(
        theta,
        mY=mY,
        m=m,
        lags=lags,
    )

    mu_x = np.nanmean(g[EXO_use].to_numpy(float), axis=0)
    sd_x = np.nanstd(g[EXO_use].to_numpy(float), axis=0) + 1e-8
    mu_y = np.nanmean(g[ENDO_use].to_numpy(float), axis=0)
    sd_y = np.nanstd(g[ENDO_use].to_numpy(float), axis=0) + 1e-8

    hist_mask_pre = np.isfinite(Yd).all(axis=1)
    last_hist_p = pd.to_datetime(prep["quarters"][hist_mask_pre][-1]).to_period("Q")
    exo_fc = exo_fc[
        exo_fc["target_quarter"].apply(lambda x: pd.Timestamp(x).to_period("Q") > last_hist_p)
    ].reset_index(drop=True)
    if exo_fc.empty:
        return None

    hist_q = pd.to_datetime(prep["quarters"])
    hist_mask = np.isfinite(Yd).all(axis=1)
    hist_q = hist_q[hist_mask]
    Y_hist = Yd[hist_mask]

    # Standardize forecast exogenous using in-sample scaling.
    z_fc = np.zeros((len(exo_fc), mX), dtype=float)
    for j, col in enumerate(EXO_use):
        raw = pd.to_numeric(exo_fc[col], errors="coerce").to_numpy(float)
        z_fc[:, j] = (raw - mu_x[j]) / sd_x[j]
    z_fc = np.where(np.isfinite(z_fc), z_fc, 0.0)

    y_buf0 = [Yd[last_t - i, :].copy() for i in range(1, lags + 1)]
    seed_key = f"{country}:{scenario_name}"
    mc_seed = FORECAST_MC_SEED + sum(ord(ch) for ch in seed_key)
    y_hat, y_lo, y_hi = _roll_kf_forecast(
        theta0=theta,
        P0=P,
        Q=Q,
        R=R,
        y_buf0=y_buf0,
        z_fc=z_fc,
        mY=mY,
        m=m,
        lags=lags,
        z_ci=z_ci,
        eps=eps,
        with_ci=True,
        mc_seed=mc_seed,
    )

    y_hat_enso0 = None
    if "ENSO" in EXO_use:
        enso_j = EXO_use.index("ENSO")
        z_fc_enso0 = z_fc.copy()
        z_fc_enso0[:, enso_j] = 0.0
        y_hat_enso0, _, _ = _roll_kf_forecast(
            theta0=theta,
            P0=P,
            Q=Q,
            R=R,
            y_buf0=y_buf0,
            z_fc=z_fc_enso0,
            mY=mY,
            m=m,
            lags=lags,
            z_ci=z_ci,
            eps=eps,
            with_ci=False,
        )

    fc_q = pd.to_datetime([_quarter_start(x) for x in exo_fc["target_quarter"].values])
    if "ENSO_source" in exo_fc.columns:
        scenario_mask = exo_fc["ENSO_source"].astype(str).eq("forecast").to_numpy()
    else:
        scenario_mask = np.ones(len(fc_q), dtype=bool)
    period_type = np.where(scenario_mask, "Scenario forecast", "Gap fill / nowcast")
    scenario_start_quarter = fc_q[scenario_mask][0] if np.any(scenario_mask) else fc_q[0]
    last_hist_p = pd.Period(hist_q[-1], freq="Q")
    fc_start_p = pd.Period(fc_q[0], freq="Q") if len(fc_q) else None
    if fc_start_p is not None:
        gap_quarters = max(0, fc_start_p.ordinal - last_hist_p.ordinal - 1)
    else:
        gap_quarters = 0

    kf_mask = valid & np.isfinite(Yd).all(axis=1)
    kf_insample = _insample_kf_track(prep, pack, R, z_ci=z_ci, eps=eps)
    varx_insample = _varx_insample_track(prep, kf_mask=kf_mask, z_ci=z_ci)
    varx_y_hat, varx_y_lo, varx_y_hi = _varx_multistep_forecast(
        prep, last_t=last_t, z_fc=z_fc, z_ci=z_ci
    )

    return {
        "ENDO_use": ENDO_use,
        "EXO_use": EXO_use,
        "hist_quarters": hist_q,
        "hist_y": Y_hist,
        "hist_y_raw": g.loc[hist_mask, ENDO_use].to_numpy(float),
        "hist_last_quarter": hist_q[-1],
        "fc_quarters": fc_q,
        "gap_quarters": gap_quarters,
        "period_type": period_type.tolist(),
        "scenario_start_quarter": scenario_start_quarter,
        "y_mu": mu_y,
        "y_sd": sd_y,
        "y_hat": y_hat,
        "y_hat_enso0": y_hat_enso0,
        "y_lower": y_lo,
        "y_upper": y_hi,
        "forecast_uncertainty": {
            "method": FORECAST_MC_METHOD,
            "simulations": FORECAST_MC_SIMULATIONS,
            "seed": mc_seed,
            "band_method": FORECAST_MC_BAND_METHOD,
            "interval": FORECAST_MC_INTERVAL_LABEL,
            "antithetic_draws": True,
            "uses_P": False,
            "uses_Q": True,
            "uses_R": False,
            "initial_beta_assumed_known": True,
        },
        "kf_insample": kf_insample,
        "varx_insample": varx_insample,
        "varx_y_hat": varx_y_hat,
        "varx_y_lower": varx_y_lo,
        "varx_y_upper": varx_y_hi,
        "theta_last": theta.ravel(),
        "theta_last_raw": theta_raw.ravel(),
        "forecast_stability": forecast_stability,
        "P_last": P,
        "Q": Q,
        "R": R,
        "exo_forecast": exo_fc,
    }


def _countries_from_pickle(pickle_path: Path = Path("Dash_Input/gvar_pipeline_results.pkl")) -> list[str] | None:
    if not pickle_path.is_file():
        return None
    with pickle_path.open("rb") as f:
        bundle = pickle.load(f)
    cfg = bundle.get("config", {})
    for key in ("COUNTRIES_FOR_COEF_PLOT", "countries_to_run"):
        if key in cfg and cfg[key]:
            return list(cfg[key])
    return None


def run_forecast_for_countries(
    countries: list[str] | None = None,
    *,
    path: str | None = None,
    max_em_iter: int = MAX_EM_ITER,
    enso_map: dict[str, float] | None = None,
    enso_scenario: str = "mean",
) -> dict:
    path = path or gp.PATH
    # Original behavior used the full panel: list(gp.countries_to_run)
    # countries = countries or _countries_from_pickle() or list(gp.countries_to_run)
    countries = countries or list(DEFAULT_COUNTRIES)
    enso_map = enso_map or FORECAST_ENSO_SCENARIOS.get(enso_scenario, FORECAST_ENSO_MEAN)
    exo_fc = build_forecast_exo_df(enso_map=enso_map, target_quarters=FORECAST_TARGET_QUARTERS)

    out: dict = {
        "config": {
            "PATH": path,
            "lags": gp.lags,
            "ENDO": gp.ENDO,
            "EXO": gp.EXO,
            "z_ci": Z_CI,
            "enso_scenario": enso_scenario,
            "stabilize_forecast": STABILIZE_FORECAST,
            "forecast_max_eigenvalue": FORECAST_MAX_EIGENVALUE,
            "forecast_uncertainty_method": FORECAST_MC_METHOD,
            "forecast_mc_simulations": FORECAST_MC_SIMULATIONS,
            "forecast_mc_seed": FORECAST_MC_SEED,
            "forecast_mc_band_method": FORECAST_MC_BAND_METHOD,
            "forecast_mc_interval": FORECAST_MC_INTERVAL_LABEL,
            "forecast_mc_antithetic_draws": True,
            "forecast_ci_uses_P": False,
            "forecast_ci_uses_Q": True,
            "forecast_ci_uses_R": False,
            "forecast_initial_beta_assumed_known": True,
            "forecast_target_quarters": [pd.Timestamp(x).isoformat() for x in FORECAST_TARGET_QUARTERS],
        },
        "exo_forecast": exo_fc,
        "per_country": {},
    }

    for country in countries:
        prep, res = gp._run_kf_em_cached(
            PATH=path,
            country=country,
            COL_COUNTRY=gp.COL_COUNTRY,
            COL_TIME=gp.COL_TIME,
            ENDO=gp.ENDO,
            EXO=gp.EXO,
            lags=gp.lags,
            min_T=gp.lags + 5,
            max_em_iter=max_em_iter,
            exo_mode="all",
        )
        if prep is None or res is None:
            print(f"[SKIP] {country}: no EM result")
            continue
        fc = forecast_country_from_em(
            prep,
            res,
            exo_fc,
            country=country,
            scenario_name=enso_scenario,
        )
        if fc is None:
            print(f"[SKIP] {country}: forecast failed")
            continue
        fc["enso_scenario"] = enso_scenario
        out["per_country"][country] = fc
        print(f"[OK] {country} [{enso_scenario}]: forecast steps={len(fc['fc_quarters'])}")

    return out


def _empty_scenario_bundle(path: str, scenario: str, exo_fc: pd.DataFrame) -> dict:
    return {
        "config": {
            "PATH": path,
            "lags": gp.lags,
            "ENDO": gp.ENDO,
            "EXO": gp.EXO,
            "z_ci": Z_CI,
            "enso_scenario": scenario,
            "stabilize_forecast": STABILIZE_FORECAST,
            "forecast_max_eigenvalue": FORECAST_MAX_EIGENVALUE,
            "forecast_uncertainty_method": FORECAST_MC_METHOD,
            "forecast_mc_simulations": FORECAST_MC_SIMULATIONS,
            "forecast_mc_seed": FORECAST_MC_SEED,
            "forecast_mc_band_method": FORECAST_MC_BAND_METHOD,
            "forecast_mc_interval": FORECAST_MC_INTERVAL_LABEL,
            "forecast_mc_antithetic_draws": True,
            "forecast_ci_uses_P": False,
            "forecast_ci_uses_Q": True,
            "forecast_ci_uses_R": False,
            "forecast_initial_beta_assumed_known": True,
            "forecast_target_quarters": [pd.Timestamp(x).isoformat() for x in FORECAST_TARGET_QUARTERS],
        },
        "exo_forecast": exo_fc,
        "per_country": {},
    }


def run_forecast_all_enso_scenarios(
    countries: list[str] | None = None,
    *,
    path: str | None = None,
    max_em_iter: int = MAX_EM_ITER,
) -> dict[str, dict]:
    """Run mean/min/max ENSO scenarios; EM once per country, forecast path varies by exo only."""
    path = path or gp.PATH
    countries = countries or list(DEFAULT_COUNTRIES)
    exo_by_scenario = {
        scenario: build_forecast_exo_df(enso_map=enso_map, target_quarters=FORECAST_TARGET_QUARTERS)
        for scenario, enso_map in FORECAST_ENSO_SCENARIOS.items()
    }
    bundles = {
        scenario: _empty_scenario_bundle(path, scenario, exo_fc)
        for scenario, exo_fc in exo_by_scenario.items()
    }

    for country in countries:
        prep, res = gp._run_kf_em_cached(
            PATH=path,
            country=country,
            COL_COUNTRY=gp.COL_COUNTRY,
            COL_TIME=gp.COL_TIME,
            ENDO=gp.ENDO,
            EXO=gp.EXO,
            lags=gp.lags,
            min_T=gp.lags + 5,
            max_em_iter=max_em_iter,
            exo_mode="all",
        )
        if prep is None or res is None:
            print(f"[SKIP] {country}: no EM result")
            continue
        for scenario, exo_fc in exo_by_scenario.items():
            fc = forecast_country_from_em(
                prep,
                res,
                exo_fc,
                country=country,
                scenario_name=scenario,
            )
            if fc is None:
                print(f"[SKIP] {country} [{scenario}]: forecast failed")
                continue
            fc["enso_scenario"] = scenario
            bundles[scenario]["per_country"][country] = fc
            print(f"[OK] {country} [{scenario}]: forecast steps={len(fc['fc_quarters'])}")

    return bundles


def plot_forecast_paths(
    forecast_bundle: dict,
    *,
    output_dir: str | Path = OUTPUT_DIR,
    countries: list[str] | None = None,
    hist_plot_quarters: int = HIST_PLOT_QUARTERS,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per = forecast_bundle.get("per_country", {})
    use_countries = countries or sorted(per.keys())

    warned_enso0 = False
    for country in use_countries:
        d = per.get(country)
        if not d:
            continue
        if d.get("y_hat_enso0") is None and not warned_enso0:
            print("[WARN] y_hat_enso0 missing in bundle; rerun forecast (not --plot-only) to draw red line")
            warned_enso0 = True
        endo = d["ENDO_use"]
        mY = len(endo)
        ncols = 2
        nrows = int(np.ceil(mY / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), sharex=True)
        axes = np.atleast_1d(axes).flatten()

        hq = pd.to_datetime([_quarter_start(x) for x in d["hist_quarters"]])
        fq = pd.to_datetime([_quarter_start(x) for x in d["fc_quarters"]])
        n_hist = min(hist_plot_quarters, len(hq))
        hq_win = hq[-n_hist:]
        y_win = d["hist_y"][-n_hist:, :]
        gap_q = int(d.get("gap_quarters", 0))

        for j in range(mY):
            ax = axes[j]
            ax.plot(hq_win, y_win[:, j], color="black", linewidth=1.6, label="history")
            ax.plot(fq, d["y_hat"][:, j], color="C0", linewidth=1.6, marker="o", markersize=4, label="forecast")
            y_enso0 = d.get("y_hat_enso0")
            if y_enso0 is not None:
                ax.plot(
                    fq,
                    y_enso0[:, j],
                    color="red",
                    linewidth=1.6,
                    linestyle="-",
                    marker="x",
                    markersize=4,
                    label="ENSO=0 (z-space)" if j == 0 else None,
                )
            ax.fill_between(
                fq,
                d["y_lower"][:, j],
                d["y_upper"][:, j],
                color="C0",
                alpha=0.30,
                label="95% band" if j == 0 else None,
            )
            ax.axvline(hq[-1], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_title(endo[j])
            ax.grid(alpha=0.25)

        for k in range(mY, len(axes)):
            axes[k].axis("off")

        gap_note = f" | gap={gap_q}Q (no panel / no exo)" if gap_q > 0 else ""
        scen = forecast_bundle.get("config", {}).get("enso_scenario", "")
        scen_note = f" | ENSO {scen}" if scen else ""
        fig.suptitle(
            f"{country} — last {n_hist}Q + forecast{scen_note}{gap_note}",
            y=1.02,
            fontsize=10,
        )
        axes[0].legend(loc="upper left", fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_dir / f"forecast_{country}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def _coverage_fraction(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not np.any(ok):
        return float("nan")
    inside = (y[ok] >= lo[ok]) & (y[ok] <= hi[ok])
    return float(np.mean(inside))


def _plot_track_paths(
    country: str,
    d: dict,
    *,
    model: str,
    output_dir: Path,
    track_plot_start: str | pd.Timestamp,
    z_ci: float,
) -> None:
    """One figure per model: mY rows x 1 col; KF or VARX residual-based 95% bands."""
    ins = d["kf_insample"] if model == "kf" else d.get("varx_insample")
    if not ins:
        return

    endo = d["ENDO_use"]
    mY = len(endo)
    fig, axes = plt.subplots(mY, 1, figsize=(11, 3.2 * mY), sharex=True, squeeze=False)
    axes = axes.flatten()

    iq = pd.to_datetime(ins["quarters"])
    iy = np.asarray(ins["y"], dtype=float)
    ilo = np.asarray(ins["y_lower"], dtype=float)
    ihi = np.asarray(ins["y_upper"], dtype=float)
    ihat = np.asarray(ins["y_hat"], dtype=float)

    fq = pd.to_datetime([_quarter_start(x) for x in d["fc_quarters"]])
    fhat = np.asarray(d["varx_y_hat" if model == "varx" else "y_hat"], dtype=float)
    if model == "varx":
        flo = np.asarray(d["varx_y_lower"], dtype=float)
        fhi = np.asarray(d["varx_y_upper"], dtype=float)
    else:
        flo = np.asarray(d["y_lower"], dtype=float)
        fhi = np.asarray(d["y_upper"], dtype=float)
    fenso0 = d.get("y_hat_enso0") if model == "kf" else None

    t0 = pd.Timestamp(track_plot_start)
    ins_mask = iq >= t0
    iq_w = iq[ins_mask]
    iy_w = iy[ins_mask, :]
    ihat_w = ihat[ins_mask, :]
    ilo_w = ilo[ins_mask, :]
    ihi_w = ihi[ins_mask, :]

    t_fc0 = fq[0] if len(fq) else None
    gap_q = int(d.get("gap_quarters", 0))
    line_color = "C0" if model == "kf" else "C2"
    band_color = "C0" if model == "kf" else "C2"
    band_tag = "KF" if model == "kf" else "VARX"
    ins_label = "KF 1-step" if model == "kf" else "VARX 1-step"
    fc_label = "KF forecast" if model == "kf" else "VARX forecast"

    for j in range(mY):
        ax = axes[j]
        ax.plot(iq_w, iy_w[:, j], color="black", linewidth=1.6, label="actual")
        ax.plot(iq_w, ihat_w[:, j], color=line_color, linewidth=1.4, label=ins_label if j == 0 else None)
        ax.fill_between(
            iq_w,
            ilo_w[:, j],
            ihi_w[:, j],
            color=band_color,
            alpha=0.22,
            label=f"95% band {band_tag} (in-sample)" if j == 0 else None,
        )

        out_ins = ~((iy_w[:, j] >= ilo_w[:, j]) & (iy_w[:, j] <= ihi_w[:, j]))
        if np.any(out_ins):
            ax.scatter(
                iq_w[out_ins],
                iy_w[out_ins, j],
                s=28,
                color="crimson",
                zorder=5,
                label=f"actual outside {band_tag} band" if j == 0 else None,
            )

        ax.plot(
            fq,
            fhat[:, j],
            color=line_color,
            linewidth=1.6,
            marker="o",
            markersize=4,
            label=fc_label if j == 0 else None,
        )
        ax.fill_between(
            fq,
            flo[:, j],
            fhi[:, j],
            color=band_color,
            alpha=0.30,
            label=f"95% band {band_tag} (forecast)" if j == 0 else None,
        )

        if fenso0 is not None:
            y0 = np.asarray(fenso0, dtype=float)
            out_fc0 = (y0[:, j] < flo[:, j]) | (y0[:, j] > fhi[:, j])
            ax.plot(
                fq,
                y0[:, j],
                color="red",
                linewidth=1.6,
                linestyle="-",
                marker="x",
                markersize=4,
                label="ENSO=0 (z-space)" if j == 0 else None,
            )
            if np.any(out_fc0):
                ax.scatter(
                    fq[out_fc0],
                    y0[out_fc0, j],
                    s=36,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=1.2,
                    zorder=6,
                )

        if len(iq_w):
            ax.axvline(iq_w[-1], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        if t_fc0 is not None:
            ax.axvline(t_fc0, color="C1", linestyle=":", linewidth=0.9, alpha=0.85)

        cov_ins = _coverage_fraction(iy_w[:, j], ilo_w[:, j], ihi_w[:, j])
        ax.set_ylabel(endo[j])
        ax.set_title(f"actual in {band_tag} band: {cov_ins:.0%}")
        ax.grid(alpha=0.25)

    gap_note = f" | gap={gap_q}Q" if gap_q > 0 else ""
    model_title = "KF" if model == "kf" else "VARX"
    band_note = "KF S=HP'H+R" if model == "kf" else "VARX rolling residual SE"
    scen = d.get("enso_scenario") or ""
    scen_note = f" | ENSO {scen}" if scen else ""
    fig.suptitle(
        f"{country} — {model_title} track from {t0.year} ({band_note}, z={z_ci:g}){scen_note}{gap_note}",
        y=1.01,
        fontsize=10,
    )
    axes[0].legend(loc="upper left", fontsize=7, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    prefix = "forecast_kf_track" if model == "kf" else "forecast_varx_track"
    fig.savefig(output_dir / f"{prefix}_{country}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_kf_track_paths(
    forecast_bundle: dict,
    *,
    output_dir: str | Path = OUTPUT_DIR_KF_TRACK,
    countries: list[str] | None = None,
    track_plot_start: str | pd.Timestamp = KF_TRACK_PLOT_START,
) -> None:
    """KF track + KF 95% bands; one row per endogenous variable."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per = forecast_bundle.get("per_country", {})
    use_countries = countries or sorted(per.keys())
    z_ci = float(forecast_bundle.get("config", {}).get("z_ci", Z_CI))

    warned = False
    for country in use_countries:
        d = per.get(country)
        if not d or not d.get("kf_insample"):
            if not warned:
                print("[WARN] kf_insample missing; rerun forecast for track plots")
                warned = True
            continue
        _plot_track_paths(
            country, d, model="kf", output_dir=output_dir, track_plot_start=track_plot_start, z_ci=z_ci
        )


def plot_forecast_varx_track_paths(
    forecast_bundle: dict,
    *,
    output_dir: str | Path = OUTPUT_DIR_VARX_TRACK,
    countries: list[str] | None = None,
    track_plot_start: str | pd.Timestamp = KF_TRACK_PLOT_START,
) -> None:
    """VARX track + VARX residual-based 95% bands; one row per endogenous variable."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per = forecast_bundle.get("per_country", {})
    use_countries = countries or sorted(per.keys())
    z_ci = float(forecast_bundle.get("config", {}).get("z_ci", Z_CI))

    warned = False
    for country in use_countries:
        d = per.get(country)
        if not d or not d.get("varx_insample") or d.get("varx_y_lower") is None:
            if not warned:
                print("[WARN] varx track/bounds missing; rerun forecast or --plot-only to backfill")
                warned = True
            continue
        _plot_track_paths(
            country, d, model="varx", output_dir=output_dir, track_plot_start=track_plot_start, z_ci=z_ci
        )


def _backfill_track_data(prep: dict, res: dict, d: dict) -> None:
    """Fill kf_insample / varx_insample / varx_y_hat on an existing per-country record."""
    pack = res.get("e_step_store")
    if pack is None:
        return
    valid = np.asarray(pack["valid_mask"], dtype=bool)
    if not np.any(valid):
        return
    last_t = int(np.where(valid)[0][-1])
    R = np.asarray(d.get("R", res.get("R")), dtype=float)
    z_ci = float(Z_CI)

    kf_mask = valid & np.isfinite(np.asarray(prep["Yd"], dtype=float)).all(axis=1)
    if not d.get("kf_insample"):
        d["kf_insample"] = _insample_kf_track(prep, pack, R, z_ci=z_ci)
    need_varx = (
        not d.get("varx_insample")
        or d.get("varx_y_hat") is None
        or d.get("varx_y_lower") is None
        or not (d.get("varx_insample") or {}).get("y_lower")
    )
    if need_varx and d.get("exo_forecast") is not None:
        d["varx_insample"] = _varx_insample_track(prep, kf_mask=kf_mask, z_ci=z_ci)
        exo_fc = d["exo_forecast"]
        g = prep["g"]
        EXO_use = list(prep["EXO_use"])
        mu_x = np.nanmean(g[EXO_use].to_numpy(float), axis=0)
        sd_x = np.nanstd(g[EXO_use].to_numpy(float), axis=0) + 1e-8
        mX = len(EXO_use)
        z_fc = np.zeros((len(exo_fc), mX), dtype=float)
        for j, col in enumerate(EXO_use):
            raw = pd.to_numeric(exo_fc[col], errors="coerce").to_numpy(float)
            z_fc[:, j] = (raw - mu_x[j]) / sd_x[j]
        z_fc = np.where(np.isfinite(z_fc), z_fc, 0.0)
        vy, vlo, vhi = _varx_multistep_forecast(prep, last_t=last_t, z_fc=z_fc, z_ci=z_ci)
        d["varx_y_hat"] = vy
        d["varx_y_lower"] = vlo
        d["varx_y_upper"] = vhi


def enrich_kf_insample_if_missing(
    bundle: dict,
    *,
    path: str | None = None,
    max_em_iter: int = MAX_EM_ITER,
) -> dict:
    """Backfill kf/varx track fields from EM when loading an older forecast pickle."""
    per = bundle.get("per_country", {})
    if not per:
        return bundle

    path = path or bundle.get("config", {}).get("PATH") or gp.PATH
    for country, d in per.items():
        if not d:
            continue
        need = (
            not d.get("kf_insample")
            or not d.get("varx_insample")
            or d.get("varx_y_hat") is None
            or d.get("varx_y_lower") is None
            or not (d.get("varx_insample") or {}).get("y_lower")
        )
        if not need:
            continue
        prep, res = gp._run_kf_em_cached(
            PATH=path,
            country=country,
            COL_COUNTRY=gp.COL_COUNTRY,
            COL_TIME=gp.COL_TIME,
            ENDO=gp.ENDO,
            EXO=gp.EXO,
            lags=gp.lags,
            min_T=gp.lags + 5,
            max_em_iter=max_em_iter,
            exo_mode="all",
        )
        if prep is None or res is None:
            continue
        _backfill_track_data(prep, res, d)
        print(f"[OK] backfilled track data: {country}")
    return bundle


def save_forecast_pickle(bundle: dict, path: str | Path = FORECAST_PICKLE_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"[PICKLE] Saved: {path}")


def _plot_bundle_all_charts(bundle: dict, *, scenario: str | None = None) -> None:
    """Write forecast / KF track / VARX track PNGs for one scenario bundle."""
    scenario = scenario or bundle.get("config", {}).get("enso_scenario", "mean")
    dirs = _scenario_output_dirs(scenario)
    plot_forecast_paths(bundle, output_dir=dirs["forecast"])
    plot_forecast_kf_track_paths(bundle, output_dir=dirs["kf_track"])
    plot_forecast_varx_track_paths(bundle, output_dir=dirs["varx_track"])


def main(*, plot_only: bool = False) -> dict:
    if plot_only:
        with FORECAST_PICKLE_PATH.open("rb") as f:
            saved = pickle.load(f)
        print(f"[INFO] plot-only from {FORECAST_PICKLE_PATH}")
        if "scenarios" in saved:
            for scenario, bundle in saved["scenarios"].items():
                bundle = enrich_kf_insample_if_missing(bundle)
                _plot_bundle_all_charts(bundle, scenario=scenario)
            return saved
        saved = enrich_kf_insample_if_missing(saved)
        _plot_bundle_all_charts(saved)
        return saved

    scenario_bundles = run_forecast_all_enso_scenarios()
    saved = {"scenarios": scenario_bundles}
    save_forecast_pickle(saved)
    for scenario, bundle in scenario_bundles.items():
        _plot_bundle_all_charts(bundle, scenario=scenario)
    return saved


if __name__ == "__main__":
    import sys

    main(plot_only="--plot-only" in sys.argv)
