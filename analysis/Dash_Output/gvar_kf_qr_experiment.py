"""
LLM structural-break–aware Q/R experiment for GVAR Kalman filter.

E-step: Q_t scales only the endogenous-coefficient submatrix of Q by s_Q,t;
exo coefficients (ENSO, COMMODITY, …) use unscaled Q.

Three scenarios (same figures; three lines):
- baseline: s_Q = s_R = 1.
- hard_scale: on break-year quarters, **only that endogenous Q submatrix** is multiplied by
  break_scale; exogenous (ENSO/COMMODITY) θ block of Q is unchanged. **R** is scaled by the same
  scalar on those quarters (s_R = s_Q).
- score_adaptive: same s_Q,t applies only to the **endogenous-θ submatrix of Q** (lagged GDP/CPI/…
  blocks); **ENSO & COMMODITY θ entries keep Q unchanged**. Formula per quarter as below; s_R = 1.

M-step: Q moments for endo indices divided by s_Q,t; exo indices not divided.
R moments divided by s_R,t (1 for score_adaptive).
EM runs at least MIN_EM_ITER iterations after s_t is fixed from LLM (hard) or from scores (adaptive).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
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
import gvar_kf_forecast as gkf  # noqa: E402
from llm_break_visualization import build_break_structures  # noqa: E402

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PIPELINE_PICKLE_PATH = _ROOT / "structural_break" / "gvar_pipeline_results.pkl"
OUTPUT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR_COEF = OUTPUT_ROOT / "qr_experiment" / "climate_coeff"
OUTPUT_DIR_TRACK = OUTPUT_ROOT / "qr_experiment" / "kf_track"
MIN_EM_ITER = 50
MAX_EM_ITER = 50
EM_TOL = 1e-4
EM_DAMPING = 0.2
VARX_INIT_WINDOW = 40
EPS = 1e-8

# Q and R multiplier on the 4 quarters of each LLM structural-break year.
BREAK_QR_SCALE = 10.0
MANUAL_ENSO_BREAKS = [
    {"country": "IND", "break_year": 2020},
    {"country": "IND", "break_year": 2021},
    {"country": "IND", "break_year": 2022},
    {"country": "IDN", "break_year": 2020},
    {"country": "IDN", "break_year": 2021},
    {"country": "IDN", "break_year": 2022},
]

# Plot window and shading (QR experiment figures only).
PLOT_START = pd.Timestamp("1990-01-01")
BREAK_SHADE_COLOR = "#5DA5DA"
HIGHLIGHT_2014_START = pd.Timestamp("2014-01-01")
HIGHLIGHT_2016_END = pd.Timestamp("2017-01-01")  # exclusive: covers calendar 2014–2016

# Restrict QR experiment to this short list of countries.
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
]

# (label, policy id, linestyle, color) — policy: baseline | hard_scale | score_adaptive
SCENARIOS: list[tuple[str, str, str, str]] = [
    ("baseline (s_t=1, EM)", "baseline", "-", "C0"),
    (f"LLM break s_t×{BREAK_QR_SCALE:g} + EM", "hard_scale", "--", "C1"),
    ("LLM Q adaptive (score×conf), R fixed", "score_adaptive", "-.", "C2"),
]


def _quarter_start(ts) -> pd.Timestamp:
    return gkf._quarter_start(ts)


def load_llm_df(pickle_path: Path = PIPELINE_PICKLE_PATH) -> pd.DataFrame:
    if not pickle_path.is_file():
        raise FileNotFoundError(f"Pipeline pickle not found: {pickle_path}")
    with pickle_path.open("rb") as f:
        bundle = pickle.load(f)
    llm_pack = bundle.get("llm_integration") or {}
    llm_df = llm_pack.get("llm_df")
    if llm_df is None or llm_df.empty:
        raise ValueError(f"No llm_integration.llm_df in {pickle_path}")
    return apply_manual_enso_breaks(llm_df.copy())


def apply_manual_enso_breaks(llm_df: pd.DataFrame) -> pd.DataFrame:
    """Overlay manually curated ENSO break years used by the QR experiment."""
    out = llm_df.copy()
    defaults = {
        "n_docs": 0,
        "status": "manual",
        "error_message": np.nan,
        "source_file": "manual_enso_breaks",
        "break_supported": 1,
        "confidence": 5,
        "break_type": "climate_shock, enso",
        "duration": "3 year",
        "climate_related": 1,
        "summary": "Manual ENSO break-year overlay for QR experiment.",
    }
    for col, val in defaults.items():
        if col not in out.columns:
            out[col] = val if not isinstance(val, float) or np.isfinite(val) else np.nan

    rows = []
    for item in MANUAL_ENSO_BREAKS:
        row = {col: defaults.get(col, np.nan) for col in out.columns}
        row.update(
            {
                "country": item["country"],
                "break_year": int(item["break_year"]),
                "break_supported": 1,
                "confidence": 5,
                "break_type": "climate_shock, enso",
                "duration": "3 year",
                "climate_related": 1,
                "raw_output": (
                    "break_supported: 1\n"
                    "break_type: climate_shock, enso\n"
                    "duration: 3 year\n"
                    "climate_related: 1\n"
                    "confidence: 5\n"
                    "summary: Manual ENSO break-year overlay for QR experiment."
                ),
            }
        )
        rows.append(row)

    manual = pd.DataFrame(rows, columns=out.columns)
    out = pd.concat([out, manual], ignore_index=True)
    out["country"] = out["country"].astype(str).str.upper()
    out["break_year"] = pd.to_numeric(out["break_year"], errors="coerce")
    manual_keys = {(r["country"], int(r["break_year"])) for r in rows}
    manual_rank = []
    for r in out[["country", "break_year"]].itertuples(index=False):
        if pd.isna(r.break_year):
            manual_rank.append(0)
            continue
        manual_rank.append(1 if (str(r.country).upper(), int(r.break_year)) in manual_keys else 0)
    out["_manual_rank"] = manual_rank
    out = (
        out.sort_values(["country", "break_year", "_manual_rank"])
        .drop_duplicates(["country", "break_year"], keep="last")
        .drop(columns=["_manual_rank"])
        .reset_index(drop=True)
    )
    return out


def _countries_from_pickle(pickle_path: Path = PIPELINE_PICKLE_PATH) -> list[str] | None:
    if not pickle_path.is_file():
        return None
    with pickle_path.open("rb") as f:
        bundle = pickle.load(f)
    cfg = bundle.get("config", {})
    for key in ("COUNTRIES_FOR_COEF_PLOT", "countries_to_run"):
        if key in cfg and cfg[key]:
            return list(cfg[key])
    return None


def _confidence_to_a(confidence: float) -> float:
    """LLM confidence → a in Q multiplier clip(1 + a*st, 1, 10); 0 if unsupported."""
    if confidence is None or not np.isfinite(confidence):
        return 0.0
    c = int(round(float(confidence)))
    if c < 3:
        return 0.6
    if c < 4:
        return 0.6
    if c == 4:
        return 0.8
    if c >= 5:
        return 1.0
    return 0.6


def baseline_break_diagnostics_from_res(res: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same three series as pickle `build_em_composite_break_score_panel` (from RTS pack + R)."""
    pack = res.get("e_step_store")
    if pack is None:
        return np.array([]), np.array([]), np.array([])
    R = np.asarray(res["R"], dtype=float)
    innov, dcoef, fsgap = gp.compute_kf_smoother_diagnostics(R=R, pack=pack, eps=EPS)
    return (
        np.asarray(innov, dtype=float),
        np.asarray(dcoef, dtype=float),
        np.asarray(fsgap, dtype=float),
    )


def _safe_z_quarterly(x: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score over time (full sample), NaN-safe."""
    x = np.asarray(x, dtype=float)
    fin = np.isfinite(x)
    out = np.zeros_like(x, dtype=float)
    if not np.any(fin):
        return out
    mu = float(np.nanmean(x[fin]))
    sd = float(np.nanstd(x[fin]))
    if not np.isfinite(sd) or sd < 1e-12:
        return out
    out[fin] = (x[fin] - mu) / sd
    return out


def composite_break_intensity_per_quarter(
    innov: np.ndarray,
    coef_change: np.ndarray,
    filter_smoother_gap: np.ndarray,
) -> np.ndarray:
    """
    Per-quarter intensity = |(z_innov + z_coef + z_fs) / 3| (same composite as pickle, then magnitude).
    """
    zi = _safe_z_quarterly(innov)
    zc = _safe_z_quarterly(coef_change)
    zf = _safe_z_quarterly(filter_smoother_gap)
    comp = (zi + zc + zf) / 3.0
    return np.abs(comp)


def build_qr_scale_score_adaptive(
    quarters: np.ndarray | pd.DatetimeIndex,
    innov: np.ndarray,
    coef_change: np.ndarray,
    filter_smoother_gap: np.ndarray,
    country: str,
    supported_llm: pd.DataFrame,
) -> np.ndarray:
    """
    Per-time-step scalars s_Q,t for `run_kf_em_qr_scale`: they multiply **only** the endogenous
    submatrix of Q (see `endo_coeff_index_mask` + `Q_matrix_at_break_scale`). Coefficients on
    ENSO/COMMODITY in θ never get an extra Q scale from this vector.

    For each LLM-supported break year Y: calendar years Y and Y+1 (e.g. 2018→2018–2019, 8 quarters).
    Each quarter t in that window: s_Q,t = clip(1 + a * st_t, 1, 10), st_t from the three
    diagnostics (z-composite, then |·|). R scale is fixed at 1 for this policy. Overlapping
    breaks: max s_Q per quarter.
    """
    n = len(quarters)
    qscale = np.ones(n, dtype=float)
    innov = np.asarray(innov, dtype=float).reshape(-1)
    cc = np.asarray(coef_change, dtype=float).reshape(-1)
    fsg = np.asarray(filter_smoother_gap, dtype=float).reshape(-1)
    if n == 0 or not (len(innov) == n == len(cc) == len(fsg)):
        return qscale

    intensity_t = composite_break_intensity_per_quarter(innov, cc, fsg)

    if supported_llm is None or supported_llm.empty:
        return qscale

    years = pd.to_datetime(quarters).year.to_numpy()
    sub = supported_llm[supported_llm["country"].astype(str).str.upper() == str(country).upper()]
    for row in sub.itertuples(index=False):
        Y = int(getattr(row, "break_year", 0))
        if Y <= 0:
            continue
        conf = float(getattr(row, "confidence", 0.0))
        a = _confidence_to_a(conf)
        if a <= 0.0:
            continue
        m = np.isin(years, [Y, Y + 1])
        idx = np.where(m)[0]
        for i in idx:
            st = float(intensity_t[i])
            s_val = float(np.clip(1.0 + a * st, 1.0, 10.0))
            qscale[i] = max(qscale[i], s_val)
    return qscale


def em_result_with_score_adaptive(
    prep: dict,
    *,
    break_years: list[int] | None,
    qr_scale_Q: np.ndarray,
    max_em_iter: int = MAX_EM_ITER,
    min_em_iter: int = MIN_EM_ITER,
    verbose: bool = False,
) -> dict:
    """EM with Q_t endo-block × s_Q,t; R_t ≡ R (no scaling)."""
    n = len(prep["Yd"])
    qr_R = np.ones(n, dtype=float)
    res = run_kf_em_qr_scale(
        prep,
        qr_scale_Q,
        qr_scale_R_t=qr_R,
        max_em_iter=max_em_iter,
        min_em_iter=min_em_iter,
        verbose=verbose,
    )
    res["break_years"] = list(break_years or [])
    res["use_llm_break_qr"] = True
    res["qr_policy"] = "score_adaptive"
    res["break_qr_scale"] = float(np.nanmax(qr_scale_Q)) if len(qr_scale_Q) else 1.0
    res["n_break_quarters"] = int(np.sum(qr_scale_Q > 1.0 + 1e-9))
    return res


def build_qr_scale_by_time(
    quarters: np.ndarray | pd.DatetimeIndex,
    break_years: list[int] | None,
    *,
    break_scale: float = BREAK_QR_SCALE,
) -> np.ndarray:
    """Per-row scale: 1.0 normally; break_scale on all quarters in break years."""
    n = len(quarters)
    scale = np.ones(n, dtype=float)
    if not break_years:
        return scale
    years = pd.to_datetime(quarters).year.to_numpy()
    break_set = {int(y) for y in break_years}
    scale[np.isin(years, list(break_set))] = float(break_scale)
    return scale


def endo_coeff_index_mask(mY: int, lags: int, mX: int) -> np.ndarray:
    """Which θ indices get process-noise scale: True = lagged endogenous only; False = ENSO, COMMODITY, …"""
    m = lags * mY + mX
    p = m * mY
    mask = np.zeros(p, dtype=bool)
    for eq in range(mY):
        for off in range(lags * mY):
            mask[eq * m + off] = True
    return mask


def Q_matrix_at_break_scale(Q: np.ndarray, s: float, endo_mask: np.ndarray) -> np.ndarray:
    """Q_t: multiply only the endogenous-coefficient submatrix of Q by s; exo block unchanged."""
    Q = np.asarray(Q, dtype=float)
    if float(s) == 1.0:
        return Q.copy()
    Q_t = Q.copy()
    idx = np.where(endo_mask)[0]
    if idx.size > 0:
        Q_t[np.ix_(idx, idx)] *= float(s)
    return Q_t


def em_m_step_update_qr_scale(
    Y: np.ndarray,
    H_list: list,
    y_list: list,
    valid_mask: np.ndarray,
    theta_smooth: np.ndarray,
    P_smooth: np.ndarray,
    qr_scale_t: np.ndarray,
    endo_mask: np.ndarray,
    Q_old: np.ndarray,
    R_old: np.ndarray,
    *,
    qr_scale_R_t: np.ndarray | None = None,
    em_damping: float = EM_DAMPING,
    eps: float = EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    M-step with Q_t = block-diag scale on endo coeffs only; R_t = s_R,t R.

    Q: E[ΔθΔθ'] / s_Q,t for endogenous state indices only; exo indices use /1.
    R <- mean_t E[(y-Hx)(y-Hx)' + HPH'] / s_R,t
    """
    _, p = theta_smooth.shape
    mY = Y.shape[1]
    endo_mask = np.asarray(endo_mask, dtype=bool).reshape(-1)
    qr_scale_t = np.asarray(qr_scale_t, dtype=float).reshape(-1)
    if qr_scale_R_t is None:
        qr_scale_R_t = qr_scale_t
    qr_scale_R_t = np.asarray(qr_scale_R_t, dtype=float).reshape(-1)

    R_acc = np.zeros((mY, mY))
    cntR = 0
    for t in np.where(valid_mask)[0]:
        H_t = H_list[t]
        y_t = y_list[t]
        if H_t is None or y_t is None:
            continue
        s_r = max(float(qr_scale_R_t[t]), eps)
        x_t = theta_smooth[t].reshape(-1, 1)
        resid = y_t - H_t @ x_t
        R_t = (resid @ resid.T + H_t @ P_smooth[t] @ H_t.T) / s_r
        R_acc += R_t
        cntR += 1
    R_new = (R_acc / cntR) if cntR > 0 else R_old.copy()

    valid_idx = np.where(valid_mask)[0]
    Q_acc_diag = np.zeros(p)
    cntQ = 0
    for k in range(1, len(valid_idx)):
        t = valid_idx[k]
        t0 = valid_idx[k - 1]
        s_q = max(float(qr_scale_t[t]), eps)
        d = theta_smooth[t] - theta_smooth[t0]
        mom = d**2 + 0.5 * (np.diag(P_smooth[t]) + np.diag(P_smooth[t0]))
        q_diag_t = mom.copy()
        q_diag_t[endo_mask] = mom[endo_mask] / s_q
        Q_acc_diag += q_diag_t
        cntQ += 1

    if cntQ > 0:
        Q_new_diag = Q_acc_diag / cntQ
    else:
        Q_new_diag = np.diag(Q_old)

    Q_old_diag = np.diag(Q_old)
    Q_diag = em_damping * Q_old_diag + (1.0 - em_damping) * Q_new_diag
    Q_diag = np.minimum(Q_diag, 10.0 * Q_old_diag)
    Q_diag *= 0.8
    Q_diag = np.maximum(Q_diag, eps)
    Q = np.diag(Q_diag)

    R = em_damping * R_old + (1.0 - em_damping) * R_new
    R = 0.5 * (R + R.T) + eps * np.eye(mY)

    trR_old = max(np.trace(R_old), eps)
    trQ_old = max(np.trace(Q_old), eps)
    if np.trace(R) > 10.0 * trR_old:
        R *= 10.0 * trR_old / max(np.trace(R), eps)
    if np.trace(Q) > 10.0 * trQ_old:
        Q *= 10.0 * trQ_old / max(np.trace(Q), eps)

    return Q, R


def run_kf_em_qr_scale(
    prep: dict,
    qr_scale_t: np.ndarray,
    *,
    qr_scale_R_t: np.ndarray | None = None,
    max_em_iter: int = MAX_EM_ITER,
    min_em_iter: int = MIN_EM_ITER,
    tol: float = EM_TOL,
    em_damping: float = EM_DAMPING,
    verbose: bool = False,
) -> dict:
    """EM: Q_t scales endo-coeff block only; R_t = s_R,t R; M-step matches."""
    Y = np.asarray(prep["Yd"], dtype=float)
    Z = np.asarray(prep["Xd"], dtype=float)
    lags = int(gp.lags)
    mY = Y.shape[1]
    mX = Z.shape[1] if Z is not None and Z.size > 0 else 0
    endo_mask = endo_coeff_index_mask(mY, lags, mX)
    qr_scale_t = np.asarray(qr_scale_t, dtype=float).reshape(-1)
    if qr_scale_R_t is None:
        qr_scale_R_t = qr_scale_t
    qr_scale_R_t = np.asarray(qr_scale_R_t, dtype=float).reshape(-1)
    if len(qr_scale_t) != len(Y):
        raise ValueError("qr_scale_t length must match Y rows")
    if len(qr_scale_R_t) != len(Y):
        raise ValueError("qr_scale_R_t length must match Y rows")

    theta0, Q, R, P0, m, p = gp.init_from_varx_rolling(
        Y=Y,
        Z=Z,
        lags=lags,
        window=VARX_INIT_WINDOW,
        ridge=1e-6,
        eps=EPS,
    )

    n_iter = max(int(max_em_iter), int(min_em_iter))
    history = {"trace_Q": [], "trace_R": [], "obj": []}
    last_obj = np.inf
    best_pack = None

    for it in range(n_iter):
        pack = kf_e_step_store_qr_scale(
            Y, Z, theta0, Q, R, P0, qr_scale_t, endo_mask, lags=lags, eps=EPS, qr_scale_R_t=qr_scale_R_t
        )
        theta_smooth, P_smooth, J_hist = gp.rts_smoother(
            pack["theta_filt"],
            pack["P_filt"],
            pack["theta_pred"],
            pack["P_pred"],
            pack["valid_mask"],
            eps=EPS,
        )
        pack["theta_smooth"] = theta_smooth
        pack["P_smooth"] = P_smooth
        pack["J_hist"] = J_hist

        Q, R = em_m_step_update_qr_scale(
            Y=Y,
            H_list=pack["H_list"],
            y_list=pack["y_list"],
            valid_mask=pack["valid_mask"],
            theta_smooth=theta_smooth,
            P_smooth=P_smooth,
            qr_scale_t=qr_scale_t,
            endo_mask=endo_mask,
            Q_old=Q,
            R_old=R,
            qr_scale_R_t=qr_scale_R_t,
            em_damping=em_damping,
            eps=EPS,
        )

        valid_idx = np.where(pack["valid_mask"])[0]
        if len(valid_idx) > 0:
            theta0 = theta_smooth[valid_idx[0]].reshape(-1, 1)

        obj = 0.0
        cnt = 0
        for t in valid_idx:
            H_t = pack["H_list"][t]
            y_t = pack["y_list"][t]
            if H_t is None or y_t is None:
                continue
            e = y_t - H_t @ theta_smooth[t].reshape(-1, 1)
            obj += float((e.T @ e).item())
            cnt += 1
        obj = obj / max(cnt, 1)

        history["trace_Q"].append(float(np.trace(Q)))
        history["trace_R"].append(float(np.trace(R)))
        history["obj"].append(obj)

        if verbose:
            print(
                f"[EM+scale] iter={it+1:02d}/{n_iter}, obj={obj:.6f}, "
                f"trQ={np.trace(Q):.6e}, trR={np.trace(R):.6e}"
            )

        best_pack = pack

        if (it + 1) >= min_em_iter and abs(last_obj - obj) < tol:
            if verbose:
                print(f"[EM+scale] converged at iter={it+1}, |Δobj|={abs(last_obj-obj):.3e}")
            break
        last_obj = obj

    rmse, e_raw, theta_est, Y_pred = kalman_filter_vecm_qr_scale(
        Y,
        Z,
        Q,
        R,
        P0,
        theta0,
        qr_scale_t,
        endo_mask,
        lags=lags,
        eps=EPS,
        qr_scale_R_t=qr_scale_R_t,
    )

    return {
        "rmse": rmse,
        "e_raw": e_raw,
        "theta_est": theta_est,
        "Y_pred": Y_pred,
        "theta0": theta0,
        "Q": Q,
        "R": R,
        "P0": P0,
        "m": m,
        "p": p,
        "em_history": history,
        "e_step_store": best_pack,
        "qr_scale_t": qr_scale_t,
        "qr_scale_R_t": qr_scale_R_t,
        "endo_coeff_mask": endo_mask,
        "em_iterations": len(history["obj"]),
    }


def kf_e_step_store_qr_scale(
    Y: np.ndarray,
    Z_all: np.ndarray,
    theta0: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    P0: np.ndarray,
    qr_scale_t: np.ndarray,
    endo_mask: np.ndarray,
    *,
    lags: int,
    eps: float = EPS,
    qr_scale_R_t: np.ndarray | None = None,
):
    """KF: Q_t scales endo-coeff block of Q by s_Q,t; R_t = s_R,t R."""
    n, mY = Y.shape
    mX = Z_all.shape[1] if Z_all is not None and Z_all.size > 0 else 0
    m = lags * mY + mX
    p = m * mY

    theta = np.asarray(theta0, dtype=float).reshape(p, 1)
    P = np.asarray(P0, dtype=float).copy()
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    qr_scale_t = np.asarray(qr_scale_t, dtype=float).reshape(-1)
    if qr_scale_R_t is None:
        qr_scale_R_t = qr_scale_t
    qr_scale_R_t = np.asarray(qr_scale_R_t, dtype=float).reshape(-1)
    endo_mask = np.asarray(endo_mask, dtype=bool).reshape(-1)
    I_p = np.eye(p)

    theta_filt = np.full((n, p), np.nan)
    P_filt = np.zeros((n, p, p))
    theta_pred = np.full((n, p), np.nan)
    P_pred = np.zeros((n, p, p))
    H_list: list = [None] * n
    y_list: list = [None] * n
    valid_mask = np.zeros(n, dtype=bool)
    qr_scale_used = np.ones(n, dtype=float)
    qr_scale_R_used = np.ones(n, dtype=float)

    for t in range(lags + 1, n):
        pieces = [Y[t - i, :] for i in range(1, lags + 1)]
        if mX > 0:
            pieces.append(Z_all[t - 1, :])
        X_t = np.concatenate(pieces)

        H_t = np.zeros((mY, p))
        for j in range(mY):
            idx = slice(j * m, (j + 1) * m)
            H_t[j, idx] = X_t

        s_q = float(qr_scale_t[t])
        s_r = float(qr_scale_R_t[t])
        qr_scale_used[t] = s_q
        qr_scale_R_used[t] = s_r
        Q_t = Q_matrix_at_break_scale(Q, s_q, endo_mask)
        R_t = R * s_r

        theta_pr = theta
        P_pr = P + Q_t

        y_t = Y[t, :].reshape(-1, 1)
        S = H_t @ P_pr @ H_t.T + R_t
        S = 0.5 * (S + S.T) + eps * np.eye(mY)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = P_pr @ H_t.T @ S_inv

        innovation = y_t - H_t @ theta_pr
        theta = theta_pr + K @ innovation
        P = (I_p - K @ H_t) @ P_pr
        P = 0.5 * (P + P.T) + eps * np.eye(p)

        theta_pred[t, :] = theta_pr.ravel()
        P_pred[t, :, :] = P_pr
        theta_filt[t, :] = theta.ravel()
        P_filt[t, :, :] = P
        H_list[t] = H_t
        y_list[t] = y_t
        valid_mask[t] = True

    pack = {
        "theta_filt": theta_filt,
        "P_filt": P_filt,
        "theta_pred": theta_pred,
        "P_pred": P_pred,
        "H_list": H_list,
        "y_list": y_list,
        "valid_mask": valid_mask,
        "qr_scale_t": qr_scale_used,
        "qr_scale_R_t": qr_scale_R_used,
        "endo_coeff_mask": endo_mask,
    }
    return pack


def kalman_filter_vecm_qr_scale(
    Y: np.ndarray,
    Z_all: np.ndarray,
    Q0: np.ndarray,
    R0: np.ndarray,
    P0: np.ndarray,
    theta0: np.ndarray,
    qr_scale_t: np.ndarray,
    endo_mask: np.ndarray,
    *,
    lags: int,
    eps: float = EPS,
    qr_scale_R_t: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Forward KF; Q_t endo-block scaled by s_Q,t; R_t = s_R,t R."""
    n, mY = Y.shape
    mX = Z_all.shape[1] if Z_all is not None and Z_all.size > 0 else 0
    m = lags * mY + mX
    p = m * mY

    theta = np.asarray(theta0, dtype=float).reshape(p, 1)
    P = np.asarray(P0, dtype=float).copy()
    Q = np.asarray(Q0, dtype=float)
    R = np.asarray(R0, dtype=float)
    qr_scale_t = np.asarray(qr_scale_t, dtype=float).reshape(-1)
    if qr_scale_R_t is None:
        qr_scale_R_t = qr_scale_t
    qr_scale_R_t = np.asarray(qr_scale_R_t, dtype=float).reshape(-1)
    endo_mask = np.asarray(endo_mask, dtype=bool).reshape(-1)
    I_p = np.eye(p)

    theta_est = np.zeros((n, p))
    Y_pred = np.full((n, mY), np.nan)
    e_raw = np.full((n, mY), np.nan)

    for t in range(lags + 1, n):
        pieces = [Y[t - i, :] for i in range(1, lags + 1)]
        if mX > 0:
            pieces.append(Z_all[t - 1, :])
        X_t = np.concatenate(pieces)

        H_t = np.zeros((mY, p))
        for j in range(mY):
            idx = slice(j * m, (j + 1) * m)
            H_t[j, idx] = X_t

        s_q = float(qr_scale_t[t])
        s_r = float(qr_scale_R_t[t])
        Q_t = Q_matrix_at_break_scale(Q, s_q, endo_mask)
        R_t = R * s_r

        theta_pr = theta
        P_pr = P + Q_t

        y_t = Y[t, :].reshape(-1, 1)
        S = H_t @ P_pr @ H_t.T + R_t
        S = 0.5 * (S + S.T) + eps * np.eye(mY)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = P_pr @ H_t.T @ S_inv

        y_hat = H_t @ theta_pr
        innovation = y_t - y_hat

        theta = theta_pr + K @ innovation
        P = (I_p - K @ H_t) @ P_pr
        P = 0.5 * (P + P.T) + eps * np.eye(p)

        theta_est[t, :] = theta.ravel()
        Y_pred[t, :] = y_hat.ravel()
        e_raw[t, :] = innovation.ravel()

    valid = ~np.isnan(e_raw).any(axis=1)
    rmse = float(np.sqrt(np.nanmean(e_raw[valid] ** 2))) if np.any(valid) else float("nan")
    return rmse, e_raw, theta_est, Y_pred


def insample_kf_track_qr_scale(
    prep: dict,
    pack: dict,
    R: np.ndarray,
    *,
    z_ci: float = gkf.Z_CI,
    eps: float = EPS,
) -> dict:
    """In-sample 1-step path with R_t = R * qr_scale_t[t] from the filter pack."""
    valid = np.asarray(pack["valid_mask"], dtype=bool)
    Yd = np.asarray(prep["Yd"], dtype=float)
    quarters = pd.to_datetime(prep["quarters"])
    mY = Yd.shape[1]
    R = np.asarray(R, dtype=float)
    qr_scale_t = np.asarray(pack.get("qr_scale_t", np.ones(len(Yd))), dtype=float)
    qr_scale_R = np.asarray(pack.get("qr_scale_R_t", qr_scale_t), dtype=float)

    y_hat = np.full_like(Yd, np.nan)
    y_lo = np.full_like(Yd, np.nan)
    y_hi = np.full_like(Yd, np.nan)

    for t in np.where(valid)[0]:
        H = pack["H_list"][t]
        if H is None:
            continue
        theta_pr = np.asarray(pack["theta_pred"][t], dtype=float).reshape(-1, 1)
        P_pr = np.asarray(pack["P_pred"][t], dtype=float)
        R_t = R * float(qr_scale_R[t])
        yh = (H @ theta_pr).ravel()
        S = H @ P_pr @ H.T + R_t
        S = 0.5 * (S + S.T) + eps * np.eye(mY)
        lo, hi = gkf._marginal_bounds_from_S(yh, S, z_ci=z_ci)
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


def forecast_country_with_pack(
    prep: dict,
    res: dict,
    pack: dict,
    exo_fc: pd.DataFrame,
    *,
    z_ci: float = gkf.Z_CI,
    eps: float = EPS,
) -> dict | None:
    """Multi-step forecast using filter terminal state; rollout uses baseline Q,R."""
    valid = np.asarray(pack["valid_mask"], dtype=bool)
    if not np.any(valid):
        return None
    last_t = int(np.where(valid)[0][-1])

    theta = np.asarray(pack["theta_filt"][last_t], dtype=float).reshape(-1, 1)
    P = np.asarray(pack["P_filt"][last_t], dtype=float)
    Q = np.asarray(res["Q"], dtype=float)
    R = np.asarray(res["R"], dtype=float)

    Yd = np.asarray(prep["Yd"], dtype=float)
    g = prep["g"]
    ENDO_use = list(prep["ENDO_use"])
    EXO_use = list(prep["EXO_use"])
    lags = int(gp.lags)
    mY = len(ENDO_use)
    mX = len(EXO_use)
    m = lags * mY + mX
    theta_raw = theta.copy()
    theta, forecast_stability = gkf._stabilize_theta_for_forecast(
        theta,
        mY=mY,
        m=m,
        lags=lags,
    )

    mu_x = np.nanmean(g[EXO_use].to_numpy(float), axis=0)
    sd_x = np.nanstd(g[EXO_use].to_numpy(float), axis=0) + 1e-8

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

    z_fc = np.zeros((len(exo_fc), mX), dtype=float)
    for j, col in enumerate(EXO_use):
        raw = pd.to_numeric(exo_fc[col], errors="coerce").to_numpy(float)
        z_fc[:, j] = (raw - mu_x[j]) / sd_x[j]
    z_fc = np.where(np.isfinite(z_fc), z_fc, 0.0)

    y_buf0 = [Yd[last_t - i, :].copy() for i in range(1, lags + 1)]
    y_hat, y_lo, y_hi = gkf._roll_kf_forecast(
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
    )
    kf_mask = valid & np.isfinite(Yd).all(axis=1)
    varx_insample = gkf._varx_insample_track(prep, kf_mask=kf_mask, z_ci=z_ci)
    varx_y_hat, varx_y_lo, varx_y_hi = gkf._varx_multistep_forecast(
        prep,
        last_t=last_t,
        z_fc=z_fc,
        z_ci=z_ci,
    )

    fc_q = pd.to_datetime([_quarter_start(x) for x in exo_fc["target_quarter"].values])
    last_hist_p = pd.Period(hist_q[-1], freq="Q")
    fc_start_p = pd.Period(fc_q[0], freq="Q") if len(fc_q) else None
    gap_quarters = max(0, fc_start_p.ordinal - last_hist_p.ordinal - 1) if fc_start_p else 0

    kf_insample = insample_kf_track_qr_scale(prep, pack, R, z_ci=z_ci, eps=eps)

    return {
        "ENDO_use": ENDO_use,
        "EXO_use": EXO_use,
        "hist_quarters": hist_q,
        "hist_y": Y_hist,
        "fc_quarters": fc_q,
        "gap_quarters": gap_quarters,
        "y_hat": y_hat,
        "y_lower": y_lo,
        "y_upper": y_hi,
        "kf_insample": kf_insample,
        "varx_insample": varx_insample,
        "varx_y_hat": varx_y_hat,
        "varx_y_lower": varx_y_lo,
        "varx_y_upper": varx_y_hi,
        "theta_last": theta.ravel(),
        "theta_last_raw": theta_raw.ravel(),
        "forecast_stability": forecast_stability,
        "Q": Q,
        "R": R,
    }


def em_result_with_qr_policy(
    prep: dict,
    *,
    break_years: list[int] | None,
    use_llm_break_qr: bool,
    break_scale: float = BREAK_QR_SCALE,
    max_em_iter: int = MAX_EM_ITER,
    min_em_iter: int = MIN_EM_ITER,
    verbose: bool = False,
) -> dict:
    """Run EM (≥min_em_iter): Q_t scales endo coeffs only; R_t=s_t R on break quarters."""
    quarters = pd.to_datetime(prep["quarters"])
    if use_llm_break_qr:
        qr_scale_t = build_qr_scale_by_time(quarters, break_years, break_scale=break_scale)
    else:
        qr_scale_t = np.ones(len(quarters), dtype=float)

    res = run_kf_em_qr_scale(
        prep,
        qr_scale_t,
        max_em_iter=max_em_iter,
        min_em_iter=min_em_iter,
        verbose=verbose,
    )
    res["break_years"] = list(break_years or [])
    res["use_llm_break_qr"] = use_llm_break_qr
    res["break_qr_scale"] = break_scale if use_llm_break_qr else 1.0
    res["n_break_quarters"] = int(np.sum(qr_scale_t > 1.0))
    return res


def extract_enso_coeff_series(
    prep: dict,
    res: dict,
) -> tuple[pd.DatetimeIndex, list[tuple[str, np.ndarray]]]:
    """One ENSO coefficient per endogenous equation (4 series)."""
    g = prep["g"]
    ENDO_use = list(prep["ENDO_use"])
    EXO_use = list(prep["EXO_use"])
    lags = int(gp.lags)
    mY = len(ENDO_use)
    mX = len(EXO_use)
    m = lags * mY + mX

    theta_est = np.asarray(res["theta_est"], dtype=float)
    valid = ~np.isnan(theta_est).any(axis=1)
    quarters = pd.to_datetime(g[gp.COL_TIME].values)
    q_valid = quarters[valid]

    if "ENSO" not in EXO_use:
        return q_valid, []

    idx_enso = EXO_use.index("ENSO")
    series: list[tuple[str, np.ndarray]] = []
    p = theta_est.shape[1]
    for eq in range(mY):
        off = lags * mY + idx_enso
        k = eq * m + off
        if k < p:
            series.append((f"{ENDO_use[eq]}<-ENSO", theta_est[valid, k]))
    return q_valid, series


def shade_period_2014_2016(ax, *, zorder: int = 0) -> None:
    """Blue diagonal hatch for calendar years 2014–2016 (background)."""
    ax.axvspan(
        HIGHLIGHT_2014_START,
        HIGHLIGHT_2016_END,
        facecolor=BREAK_SHADE_COLOR,
        alpha=0.10,
        hatch="///",
        edgecolor="#4A90C4",
        linewidth=0.5,
        zorder=zorder,
    )


def shade_structural_break_years(
    ax,
    country: str,
    break_dict: dict[str, list[int]],
    climate_flag: dict[tuple[str, int], int] | None = None,
    *,
    zorder: int = 1,
) -> None:
    """Solid blue shading for each LLM-supported break year (no shock-type split)."""
    del climate_flag  # kept for call-site compatibility
    for y in break_dict.get(country, []):
        start = pd.Timestamp(year=int(y), month=1, day=1)
        end = start + pd.DateOffset(months=12)
        ax.axvspan(
            start,
            end,
            color=BREAK_SHADE_COLOR,
            alpha=0.22,
            zorder=zorder,
        )


def apply_qr_plot_background(
    ax,
    country: str,
    break_dict: dict[str, list[int]] | None,
    climate_flag: dict[tuple[str, int], int] | None = None,
) -> None:
    """2014–2016 hatched band (back) + LLM break years (solid blue)."""
    del climate_flag
    shade_period_2014_2016(ax, zorder=0)
    if break_dict is not None:
        shade_structural_break_years(ax, country, break_dict, zorder=1)


def _mask_from_plot_start(times) -> np.ndarray:
    t = pd.to_datetime(np.asarray(times))
    return t >= PLOT_START


def _style_plot_xaxis(ax, *, plot_start: pd.Timestamp = PLOT_START) -> None:
    """Year labels every 2 years (less crowded than quarterly month labels)."""
    ax.set_xlim(left=plot_start)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, which="major", axis="x", alpha=0.35)


def _scenario_label_by_policy(policy: str) -> str:
    for label, pol, _, _ in SCENARIOS:
        if pol == policy:
            return label
    raise KeyError(f"scenario policy not found: {policy}")


def run_qr_experiment_for_country(
    country: str,
    *,
    path: str | None = None,
    exo_fc: pd.DataFrame | None = None,
    break_years: list[int] | None = None,
    max_em_iter: int = MAX_EM_ITER,
    break_scale: float = BREAK_QR_SCALE,
    supported_llm: pd.DataFrame | None = None,
) -> dict | None:
    path = path or gp.PATH
    if exo_fc is None:
        exo_fc = gkf.build_forecast_exo_df(target_quarters=gkf.FORECAST_TARGET_QUARTERS)
    if supported_llm is None:
        supported_llm = pd.DataFrame()

    prep = gp._prepare_country_panel_cached(
        PATH=path,
        country=country,
        COL_COUNTRY=gp.COL_COUNTRY,
        COL_TIME=gp.COL_TIME,
        ENDO=list(gp.ENDO),
        EXO=list(gp.EXO),
        min_T=gp.lags + 5,
    )
    if prep is None:
        return None

    em_hi = max(max_em_iter, MIN_EM_ITER)
    res_base = em_result_with_qr_policy(
        prep,
        break_years=break_years,
        use_llm_break_qr=False,
        break_scale=break_scale,
        max_em_iter=em_hi,
        min_em_iter=MIN_EM_ITER,
        verbose=False,
    )
    innov, dcoef, fsgap = baseline_break_diagnostics_from_res(res_base)
    quarters = pd.to_datetime(prep["quarters"])

    scenarios: dict[str, dict] = {}
    for label, policy, ls, color in SCENARIOS:
        if policy == "baseline":
            res_s = res_base
        elif policy == "hard_scale":
            res_s = em_result_with_qr_policy(
                prep,
                break_years=break_years,
                use_llm_break_qr=True,
                break_scale=break_scale,
                max_em_iter=em_hi,
                min_em_iter=MIN_EM_ITER,
                verbose=False,
            )
        elif policy == "score_adaptive":
            qr_Q = build_qr_scale_score_adaptive(
                quarters, innov, dcoef, fsgap, country, supported_llm
            )
            res_s = em_result_with_score_adaptive(
                prep,
                break_years=break_years,
                qr_scale_Q=qr_Q,
                max_em_iter=em_hi,
                min_em_iter=MIN_EM_ITER,
                verbose=False,
            )
        else:
            raise ValueError(f"unknown scenario policy: {policy}")

        pack = res_s["e_step_store"]
        q_valid, enso_series = extract_enso_coeff_series(prep, res_s)
        fc = forecast_country_with_pack(prep, res_s, pack, exo_fc)
        scenarios[label] = {
            "policy": policy,
            "use_llm_break_qr": policy != "baseline",
            "break_years": list(break_years or []),
            "n_break_quarters": int(res_s.get("n_break_quarters", 0)),
            "res": res_s,
            "coeff_quarters": q_valid,
            "enso_coeff_series": enso_series,
            "forecast": fc,
        }

    return {
        "country": country,
        "break_years": list(break_years or []),
        "prep": prep,
        "scenarios": scenarios,
    }


def plot_climate_coeff_qr_comparison(
    country_pack: dict,
    *,
    output_dir: Path = OUTPUT_DIR_COEF,
    compare_policy: str = "hard_scale",
    break_dict: dict | None = None,
    climate_flag: dict | None = None,
) -> None:
    """Four ENSO coefficient subplots: baseline vs one selected Q-rescale policy."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    country = country_pack["country"]
    scenarios = country_pack["scenarios"]
    base_label = _scenario_label_by_policy("baseline")
    comp_label = _scenario_label_by_policy(compare_policy)
    selected_labels = [base_label, comp_label]
    base_series = scenarios[base_label]["enso_coeff_series"]
    if not base_series:
        print(f"[SKIP] {country}: no ENSO coefficients")
        return

    n = len(base_series)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for j, (coef_lbl, _) in enumerate(base_series):
        ax = axes[j]
        if break_dict is not None:
            apply_qr_plot_background(ax, country, break_dict)

        for label, policy, ls, color in SCENARIOS:
            if label not in selected_labels:
                continue
            sc = scenarios[label]
            qv = pd.to_datetime(np.asarray(sc["coeff_quarters"]))
            t_mask = _mask_from_plot_start(qv)
            qv_w = qv[t_mask]
            n_bq = sc.get("n_break_quarters", 0)
            for lbl, vals in sc["enso_coeff_series"]:
                if lbl != coef_lbl:
                    continue
                leg = label if j == 0 else None
                if policy != "baseline":
                    leg = f"{label} ({n_bq} quarters)"
                ax.plot(
                    qv_w,
                    np.asarray(vals, dtype=float)[t_mask],
                    linestyle=ls,
                    color=color,
                    linewidth=1.4,
                    marker="o",
                    markersize=2,
                    alpha=0.9,
                    label=leg,
                )
                break

        ax.set_title(coef_lbl)
        ax.set_ylabel("coefficient")
        ax.grid(True, which="major", axis="y", alpha=0.25)
        _style_plot_xaxis(ax)

    for k in range(n, len(axes)):
        axes[k].axis("off")

    cname = gp.country_display_name(country, gp.ISO3_TO_COUNTRY)
    yrs = country_pack.get("break_years") or []
    yr_note = f" | LLM break years: {yrs}" if yrs else " | no LLM break years"
    cmp_name = "hard-scale vs baseline" if compare_policy == "hard_scale" else "adaptive-scale vs baseline"
    fig.suptitle(f"{cname} — ENSO coefficients ({cmp_name}){yr_note}", y=1.02, fontsize=11)
    axes[0].legend(loc="upper left", fontsize=7, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / f"climate_coeff_qr_{country}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_qr_kf_track_comparison(
    country_pack: dict,
    *,
    output_dir: Path = OUTPUT_DIR_TRACK,
    compare_policy: str = "hard_scale",
    track_plot_start: str | pd.Timestamp = PLOT_START,
    break_dict: dict | None = None,
    climate_flag: dict | None = None,
    z_ci: float = gkf.Z_CI,
) -> None:
    """KF track: baseline vs one selected Q-rescale policy (no uncertainty band)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    country = country_pack["country"]
    scenarios = country_pack["scenarios"]
    base_label = _scenario_label_by_policy("baseline")
    comp_label = _scenario_label_by_policy(compare_policy)
    selected_labels = [base_label, comp_label]
    base_fc = scenarios[base_label].get("forecast")
    if not base_fc or not base_fc.get("kf_insample"):
        print(f"[SKIP] {country}: missing baseline forecast/kf_insample")
        return

    endo = base_fc["ENDO_use"]
    mY = len(endo)
    fig, axes = plt.subplots(
        mY,
        2,
        figsize=(17, 3.2 * mY),
        sharex=True,
        squeeze=False,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    t0 = pd.Timestamp(track_plot_start)

    for j in range(mY):
        ax_kf = axes[j, 0]
        ax_varx = axes[j, 1]
        for ax in (ax_kf, ax_varx):
            if break_dict is not None:
                apply_qr_plot_background(ax, country, break_dict)

        hist_end_q = None
        for label, policy, ls, color in SCENARIOS:
            if label not in selected_labels:
                continue
            fc = scenarios[label].get("forecast")
            if not fc:
                continue
            ins = fc["kf_insample"]
            iq = pd.to_datetime(ins["quarters"])
            ins_mask = iq >= t0
            iq_w = iq[ins_mask]
            iy_w = np.asarray(ins["y"], dtype=float)[ins_mask, :]
            ihat_w = np.asarray(ins["y_hat"], dtype=float)[ins_mask, :]
            ilo_w = np.asarray(ins["y_lower"], dtype=float)[ins_mask, :]
            ihi_w = np.asarray(ins["y_upper"], dtype=float)[ins_mask, :]

            ax_kf.plot(
                iq_w,
                iy_w[:, j],
                color="black",
                linewidth=1.6,
                label="actual" if j == 0 else None,
            )
            tag = "KF 1-step" if policy == "baseline" else f"KF 1-step ({label})"
            ax_kf.plot(
                iq_w,
                ihat_w[:, j],
                color=color,
                linestyle=ls,
                linewidth=1.4,
                label=tag if j == 0 else None,
            )

            fq = pd.to_datetime([_quarter_start(x) for x in fc["fc_quarters"]])
            fhat = np.asarray(fc["y_hat"], dtype=float)
            fc_tag = "forecast" if policy == "baseline" else f"forecast ({label})"
            ax_kf.plot(
                fq,
                fhat[:, j],
                color=color,
                linestyle=ls,
                linewidth=1.6,
                marker="o",
                markersize=4,
                label=fc_tag if j == 0 else None,
            )
            if hist_end_q is None and len(iq_w):
                hist_end_q = iq_w[-1]

        varx_ins = base_fc.get("varx_insample")
        if varx_ins:
            vq = pd.to_datetime(varx_ins["quarters"])
            vmask = vq >= t0
            vq_w = vq[vmask]
            vy_w = np.asarray(varx_ins["y"], dtype=float)[vmask, :]
            vhat_w = np.asarray(varx_ins["y_hat"], dtype=float)[vmask, :]
            ax_varx.plot(
                vq_w,
                vy_w[:, j],
                color="black",
                linewidth=1.6,
                label="actual" if j == 0 else None,
            )
            base_ins = base_fc["kf_insample"]
            bq = pd.to_datetime(base_ins["quarters"])
            bmask = bq >= t0
            bhat_w = np.asarray(base_ins["y_hat"], dtype=float)[bmask, :]
            ax_varx.plot(
                bq[bmask],
                bhat_w[:, j],
                color="C0",
                linewidth=1.4,
                label="baseline KF 1-step" if j == 0 else None,
            )
            fq = pd.to_datetime([_quarter_start(x) for x in base_fc["fc_quarters"]])
            base_fhat = np.asarray(base_fc["y_hat"], dtype=float)
            ax_varx.plot(
                fq,
                base_fhat[:, j],
                color="C0",
                linestyle="-",
                linewidth=1.6,
                marker="o",
                markersize=4,
                label="baseline KF forecast" if j == 0 else None,
            )
            ax_varx.plot(
                vq_w,
                vhat_w[:, j],
                color="C2",
                linestyle="-.",
                linewidth=1.4,
                label="rolling VARX 1-step" if j == 0 else None,
            )

            varx_fhat = base_fc.get("varx_y_hat")
            if varx_fhat is not None:
                ax_varx.plot(
                    fq,
                    np.asarray(varx_fhat, dtype=float)[:, j],
                    color="C2",
                    linestyle="-.",
                    linewidth=1.6,
                    marker="o",
                    markersize=4,
                    label="VARX forecast" if j == 0 else None,
                )
            if hist_end_q is None and len(vq_w):
                hist_end_q = vq_w[-1]

        if hist_end_q is not None:
            for ax in (ax_kf, ax_varx):
                ax.axvline(hist_end_q, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax_kf.set_ylabel(endo[j])
        ax_kf.grid(alpha=0.25)
        ax_varx.grid(alpha=0.25)
        if j == 0:
            ax_kf.set_title("KF: baseline vs QR")
            ax_varx.set_title("Baseline KF vs rolling VARX")
        _style_plot_xaxis(ax_kf, plot_start=t0)
        _style_plot_xaxis(ax_varx, plot_start=t0)

    cname = gp.country_display_name(country, gp.ISO3_TO_COUNTRY)
    cmp_name = "hard-scale vs baseline" if compare_policy == "hard_scale" else "adaptive-scale vs baseline"
    fig.suptitle(
        f"{cname} — KF track ({cmp_name}, baseline band fixed) "
        f"(from {t0.year}, z={z_ci:g}) | blue = LLM break years; hatch = 2014–2016",
        y=1.01,
        fontsize=10,
    )
    axes[0, 0].legend(loc="upper left", fontsize=7, ncol=2)
    axes[0, 1].legend(loc="upper left", fontsize=7, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / f"kf_track_qr_{country}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_and_plot(
    countries: list[str] | None = None,
    *,
    path: str | None = None,
    pipeline_pickle: Path = PIPELINE_PICKLE_PATH,
    coef_dir: Path = OUTPUT_DIR_COEF,
    track_dir: Path = OUTPUT_DIR_TRACK,
    break_scale: float = BREAK_QR_SCALE,
    max_em_iter: int = MAX_EM_ITER,
) -> dict:
    # Original behavior used the full panel: list(gp.countries_to_run)
    # countries = countries or _countries_from_pickle(pipeline_pickle) or list(gp.countries_to_run)
    countries = countries or list(DEFAULT_COUNTRIES)
    llm_df = load_llm_df(pipeline_pickle)
    break_dict, climate_flag, supported = build_break_structures(llm_df)

    exo_fc = gkf.build_forecast_exo_df(target_quarters=gkf.FORECAST_TARGET_QUARTERS)
    em_iter = max(int(max_em_iter), MIN_EM_ITER)
    out: dict = {
        "countries": countries,
        "break_scale": break_scale,
        "max_em_iter": em_iter,
        "per_country": {},
    }

    for country in countries:
        break_years = break_dict.get(country, [])
        pack = run_qr_experiment_for_country(
            country,
            path=path,
            exo_fc=exo_fc,
            break_years=break_years,
            break_scale=break_scale,
            max_em_iter=em_iter,
            supported_llm=supported,
        )
        if pack is None:
            print(f"[SKIP] {country}: EM failed")
            continue
        plot_climate_coeff_qr_comparison(
            pack,
            output_dir=Path(coef_dir) / "hard_vs_baseline",
            compare_policy="hard_scale",
            break_dict=break_dict,
            climate_flag=climate_flag,
        )
        plot_qr_kf_track_comparison(
            pack,
            output_dir=Path(track_dir) / "hard_vs_baseline",
            compare_policy="hard_scale",
            break_dict=break_dict,
            climate_flag=climate_flag,
        )
        plot_climate_coeff_qr_comparison(
            pack,
            output_dir=Path(coef_dir) / "adaptive_vs_baseline",
            compare_policy="score_adaptive",
            break_dict=break_dict,
            climate_flag=climate_flag,
        )
        plot_qr_kf_track_comparison(
            pack,
            output_dir=Path(track_dir) / "adaptive_vs_baseline",
            compare_policy="score_adaptive",
            break_dict=break_dict,
            climate_flag=climate_flag,
        )
        out["per_country"][country] = {k: v for k, v in pack.items() if k != "prep"}
        n_hard = pack["scenarios"][SCENARIOS[1][0]].get("n_break_quarters", 0)
        n_sc = pack["scenarios"][SCENARIOS[2][0]].get("n_break_quarters", 0)
        print(
            f"[OK] {country}: break years={break_years}, "
            f"hard-scale Q quarters={n_hard}, score-adaptive Q>1 quarters={n_sc}"
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KF with Q/R relaxed only on LLM structural-break year quarters"
    )
    parser.add_argument(
        "--countries",
        type=str,
        default="",
        help="Comma-separated ISO3 list (default: from pipeline pickle)",
    )
    parser.add_argument(
        "--pickle",
        type=str,
        default=str(PIPELINE_PICKLE_PATH),
        help="Pipeline pickle with llm_integration",
    )
    parser.add_argument(
        "--break-scale",
        type=float,
        default=BREAK_QR_SCALE,
        help="s_t on LLM break-year quarters (default 10)",
    )
    parser.add_argument(
        "--max-em-iter",
        type=int,
        default=MAX_EM_ITER,
        help=f"EM iterations (at least {MIN_EM_ITER})",
    )
    args = parser.parse_args()
    selected = [x.strip() for x in args.countries.split(",") if x.strip()] or None
    break_scale = float(args.break_scale)
    max_em = max(int(args.max_em_iter), MIN_EM_ITER)
    SCENARIOS[1] = (f"LLM break s_t×{break_scale:g} + EM", "hard_scale", "--", "C1")
    run_and_plot(
        countries=selected,
        pipeline_pickle=Path(args.pickle),
        break_scale=break_scale,
        max_em_iter=max_em,
    )


if __name__ == "__main__":
    main()
