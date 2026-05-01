"""
Code Map (line ranges, current version):
- 1-41: imports and global setup.
- 43-234: climate data prep and merge helpers.
- 235-793: core KF/EM math (VARX init, KF, RTS, diagnostics, EM M-step).
- 794-935: EM runner (`run_kf_em`).
- 936-1551: legacy country loader / TVPKF / legacy plots.
- 1552-1979: panel data config and heatmap evaluation pipeline.
- 1984-2338: coefficient / diagnostics / Q-R trace plotting functions.
- 2344-2379: main PDF output for core EM plots (`PLOTS_PDF_PATH`).
- 2382-2482: LLM mapping + break-score panel builder.
- 2484-2568: optional LLM overlay / stats / map outputs (`ENABLE_LLM_INTEGRATION`).
- 2571-2765: breakscore plotting and drop-year refit utilities.
- 2766-end: optional refit pass and output mode flags (`ENABLE_LLM_REFIT_PDF`).

Debug tip:
- If output seems wrong, start from `run_kf_em`, `build_em_break_score_panel`,
  `build_drop_quarters_dict_from_llm_points`, and final output flag blocks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

try:
    from llm_break_visualization import (
        plot_structural_break_score_with_llm_overlay,
        plot_break_supported_ratio,
        plot_break_type_distribution,
        build_time_slice_maps,
    )
    _HAS_LLM_VIS = True
except Exception as _e:
    _HAS_LLM_VIS = False
    print(f"[WARN] llm_break_visualization import failed: {_e}")

# ======================================
#        CELL 1: Climate data
# ======================================

climate_df = pd.read_csv(
    "climate_indice.csv",
    parse_dates=["date"],           # force date parsing
    dayfirst=False,                 # month/day/year
)
CLIMATE_VARS_BY_COUNTRY = {
    "India": {"SOI": 4},           # SOI leads by 2 quarters
    "Brazil": {"NINO3+4": 1},
    # "Chile": {"SOI": 3, "copper": 1},
    "Chile": {"SOI": 3},
    "Indonesia": {"NINO1+2": 4},
    "Mexico": {"NINO3+4": 1},
    "Peru": {"SOI": 1},
    "Philippines": {"SOI": 1},
    "South Africa": {"SOI": 2},
    "Thailand": {"SOI": 2}
}


def load_trade_balance_q_from_Qstring(
    filepath,
    date_col="date",
    value_col="trade",
    take_diff=False
):
    df = pd.read_csv(filepath)

    q_index = pd.PeriodIndex(df[date_col], freq="Q")

    df.index = q_index.to_timestamp("Q")

    df = df[[value_col]].astype(float)

    if take_diff:
        df[value_col] = df[value_col].diff()

    return df

def prepare_climate_quarterly(df_climate: pd.DataFrame,
                              date_col: str = 'date',
                              climate_vars: list[str] | None = None,
                              agg='mean',
                              quarter_label='right',
                              normalize: bool = False) -> pd.DataFrame:
    df = df_climate.copy()
    
    # Handle date index / date column
    if date_col and date_col in df.columns:
        df.index = pd.to_datetime(df[date_col], errors='coerce')
        df = df.drop(columns=[date_col])
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try to detect a date-like column
        for c in df.columns:
            if 'Date' in str(c).lower():
                df.index = pd.to_datetime(df[c], errors='coerce')
                df = df.drop(columns=[c])
                break
    
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
    
    # Align to month-end (monthly data)
    df.index = df.index.to_period('M').to_timestamp('M')
    
    # Select climate variables
    if climate_vars is None:
        value_cols = list(df.columns)
    else:
        value_cols = [c for c in climate_vars if c in df.columns]
    
    if not value_cols:
        raise ValueError("No valid climate variable columns found")
    
    df = df[value_cols]
    
    # Aggregate within month (if duplicate monthly rows exist)
    if agg == 'mean':
        df_m = df.groupby(df.index).mean()
    elif agg == 'sum':
        df_m = df.groupby(df.index).sum()
    else:
        df_m = df.groupby(df.index).agg(agg)
    
    # Convert to quarterly frequency (mean)
    df_q = df_m.resample('QE', label='right', closed='right').mean()
    
    # Lag by one quarter (= 3 monthly steps)
    df_q_lag1 = df_q.shift(1)
    
    # Optional z-score normalization
    if normalize:
        df_q_lag1 = (df_q_lag1 - df_q_lag1.mean()) / df_q_lag1.std()
    
    return df_q_lag1


def remove_enso_from_Y(Y_df: pd.DataFrame) -> pd.DataFrame:

    Y = Y_df.copy()
    if 'ensos' in Y.columns:
        Y = Y.drop(columns=['ensos'])
    return Y


def merge_Y_with_climate_q(Y_levels_df: pd.DataFrame,
                           Y_date_col: str | None = None,
                           climate_q_df: pd.DataFrame | None = None,
                           how: str = 'inner') -> pd.DataFrame:

    Y = Y_levels_df.copy()
    
    if Y_date_col and Y_date_col in Y.columns:
        Y.index = pd.to_datetime(Y[Y_date_col], errors='coerce')
        Y = Y.drop(columns=[Y_date_col], errors='ignore')
    elif not isinstance(Y.index, pd.DatetimeIndex):
        Y.index = pd.to_datetime(Y.index, errors='coerce')
    
    # Align to quarter-end
    Y.index = Y.index.to_period('Q').to_timestamp('Q')
    Y = Y.sort_index()
    
    # Merge climate data
    if climate_q_df is not None:
        Cq = climate_q_df.copy()
        if not isinstance(Cq.index, pd.DatetimeIndex):
            Cq.index = pd.to_datetime(Cq.index, errors='coerce')
        Cq.index = Cq.index.to_period('Q').to_timestamp('Q')
        Cq = Cq.sort_index()
        merged = Y.join(Cq, how=how)
    else:
        merged = Y
    
    return merged
    

# ======================================
#   CELL 2: attach climate to country
# ======================================

def attach_climate_to_X(Y_levels_df, external_dfs, country, climate_vars_dict):

    var_lag_map = climate_vars_dict.get(country, {})
    climate_X_all = pd.DataFrame(index=Y_levels_df.index)

    for var_name, lag_q in var_lag_map.items():

        df_c = None

        # Find this variable across multiple external sources
        for df in external_dfs:
            if var_name in df.columns:
                df_c = df[["date", var_name]].copy()
                break

        if df_c is None:
            raise ValueError(f"Variable {var_name} not found in any external source")

        # Keep the downstream logic unchanged
        df_c["date"] = pd.to_datetime(df_c["date"], errors="coerce")
        df_c = df_c.dropna().set_index("date")

        inferred_freq = pd.infer_freq(df_c.index)

        if inferred_freq in ["Q", "QS", "QE"] or df_c.index.to_period("Q").nunique() == len(df_c):
            c_q = df_c.copy()
            c_q.index = c_q.index.to_period("Q").to_timestamp("Q")
        else:
            c_q = prepare_climate_quarterly(
                df_c.reset_index(),
                climate_vars=[var_name],
                normalize=False
            )

        c_q_lagged = c_q.shift(lag_q)

        merged = merge_Y_with_climate_q(
            Y_levels_df,
            climate_q_df=c_q_lagged
        )

        climate_X_all[var_name] = merged[var_name]

    # ===== Print merged climate variable diagnostics =====
    print(f"\n{'='*60}")
    print(f"[CLIMATE MERGE DONE] Country: {country}")
    print(f"{'='*60}")
    if len(climate_X_all.columns) == 0:
        print("Warning: no climate variable configuration for this country")
    else:
        print(f"Merged climate variables: {list(climate_X_all.columns)}")
        print(f"\nClimate variable shape: {climate_X_all.shape}")
        print(f"Time range: {climate_X_all.index.min()} to {climate_X_all.index.max()}")
        print(f"\nClimate variable summary:")
        print(climate_X_all.describe())
        print(f"\nFirst 5 rows:")
        print(climate_X_all.head())
        print(f"\nLast 5 rows:")
        print(climate_X_all.tail())
        # Check missing values
        missing_info = climate_X_all.isnull().sum()
        if missing_info.sum() > 0:
            print(f"\nMissing value counts:")
            print(missing_info[missing_info > 0])
        else:
            print(f"\nNo missing values")
    print(f"{'='*60}\n")

    return climate_X_all

# ---------- helpers ----------
def intersect_stable(a_list, b_list):
    b_index = {v: i for i, v in enumerate(b_list)}
    ia, ib, c_vals = [], [], []
    for i, v in enumerate(a_list):
        if v in b_index:
            ia.append(i)
            ib.append(b_index[v])
            c_vals.append(v)
    return c_vals, np.array(ia, dtype=int), np.array(ib, dtype=int)

# ---------- VARX benchmark ----------
def varx_rolling_predict(Y: np.ndarray,
                         Z: np.ndarray,
                         lags: int = 2,
                         window: int | None = None,
                         ridge: float = 1e-6):
    n, mY = Y.shape
    mX = Z.shape[1]
    m = lags * mY + mX
    Y_pred = np.full((n, mY), np.nan)
    e_raw  = np.full((n, mY), np.nan)

    for t in range(lags + 1, n - 1):
        if window is None:
            s0 = lags+1
        else:
            s0 = max(lags+1, t - 1 - window + 1)
        T = (t - 1) - s0 + 1
        if T <= 0:
            continue

        X_train = np.zeros((T, m))
        Y_train = np.zeros((T, mY))
        for k, s in enumerate(range(s0, t)):
            regY = np.concatenate([Y[s - i, :] for i in range(1, lags + 1)], axis=0)
            X_train[k, :] = np.concatenate([regY, Z[s-1, :]], axis=0)
            Y_train[k, :] = Y[s, :]

        A = X_train.T @ X_train + ridge * np.eye(m)
        B = X_train.T @ Y_train
        coef = np.linalg.solve(A, B)

        regY_next = np.concatenate([Y[t + 1 - i, :] for i in range(1, lags + 1)], axis=0)
        x_next = np.concatenate([regY_next, Z[t, :]], axis=0)

        yhat = x_next @ coef
        Y_pred[t + 1, :] = yhat
        e_raw[t + 1, :] = Y[t + 1, :] - yhat

    rmse = np.sqrt(np.nanmean(e_raw**2, axis=1))
    return Y_pred, e_raw, rmse

def init_from_varx_rolling(
    Y: np.ndarray,
    Z: np.ndarray,
    lags: int = 2,
    window: int = 40,
    ridge: float = 1e-6,
    eps: float = 1e-8,
):
    """
    Run rolling VARX (fixed window) on the full sample to initialize:
    theta0 (Beta0), Q0, R0, P0.
    Note: this is an initialization step only (no burn-in trimming here).
    """
    n, mY = Y.shape
    mX = Z.shape[1]
    m = lags * mY + mX
    p = m * mY

    theta_hist = []
    resid_hist = []

    for t in range(lags + 1, n):
        s0 = max(lags + 1, t - window)
        T = t - s0
        if T <= 0:
            continue

        X_train = np.zeros((T, m))
        Y_train = np.zeros((T, mY))
        for k, s in enumerate(range(s0, t)):
            regY = np.concatenate([Y[s - i, :] for i in range(1, lags + 1)], axis=0)
            X_train[k, :] = np.concatenate([regY, Z[s - 1, :]], axis=0)
            Y_train[k, :] = Y[s, :]

        A = X_train.T @ X_train + ridge * np.eye(m)
        B = X_train.T @ Y_train
        coef = np.linalg.solve(A, B)  # (m, mY)

        theta_t = coef.T.reshape(-1, order='C')  # (p,)
        theta_hist.append(theta_t)

        x_t = np.concatenate([
            np.concatenate([Y[t - i, :] for i in range(1, lags + 1)], axis=0),
            Z[t - 1, :],
        ])
        y_hat_t = x_t @ coef
        resid_hist.append(Y[t, :] - y_hat_t)

    if len(theta_hist) == 0:
        theta0 = np.zeros((p, 1))
        Q0 = 1e-4 * np.eye(p)
        R0 = np.diag(np.var(Y, axis=0) + 1e-6)
        P0 = 1.0 * np.eye(p)
        return theta0, Q0, R0, P0, m, p

    theta_hist = np.asarray(theta_hist)
    resid_hist = np.asarray(resid_hist)

    theta0 = theta_hist[-1].reshape(-1, 1)

    if theta_hist.shape[0] >= 2:
        dtheta = np.diff(theta_hist, axis=0)
        Q0 = np.cov(dtheta, rowvar=False)
        alpha = 0.01  # tuning range is usually around 0.1~0.3
        Q0_full = np.cov(dtheta, rowvar=False)
        Q0 = np.diag(np.diag(Q0_full))
        Q0 = alpha * Q0
        if Q0.ndim == 0:
            Q0 = np.array([[float(Q0)]])
    else:
        Q0 = 1e-4 * np.eye(p)

    if resid_hist.shape[0] >= 2:
        R0 = 0.1*np.cov(resid_hist, rowvar=False)
        if R0.ndim == 0:
            R0 = np.array([[float(R0)]])
    else:
        R0 = np.diag(np.var(Y, axis=0) + 1e-6)

    if theta_hist.shape[0] >= 2:
        P0 = np.cov(theta_hist, rowvar=False)
        if P0.ndim == 0:
            P0 = np.array([[float(P0)]])
    else:
        P0 = 10.0 * np.eye(p)

    Q0 = 0.5 * (Q0 + Q0.T) + eps * np.eye(p)
    R0 = 0.5 * (R0 + R0.T) + eps * np.eye(mY)
    P0 = 0.5 * (P0 + P0.T) + eps * np.eye(p)

    return theta0, Q0, R0, P0, m, p


# ---------- TVP-VECM Kalman ----------
def kalman_multilag_filter_vecm(
    Y: np.ndarray,                    # ΔY
    Z_all: np.ndarray,                # ΔZ
    Y_level_aligned: np.ndarray,      # levels aligned with ΔY (i.e. original level[1:])
    Q0: np.ndarray,
    R0: np.ndarray,
    P0: np.ndarray,
    theta0: np.ndarray | None = None, # (p, 1) initial coefficients
    dropout0: np.ndarray | None = None,  # dropout mask
    lags: int = 2,
    eps: float = 1e-8,
    Q_R_update = False
):

    global GLOBAL_QR_CACHE
    Q_hist = []
    R_hist = []
    n, mY = Y.shape
    mX = Z_all.shape[1] if Z_all is not None and Z_all.size > 0 else 0

    m = lags * mY + mX 
    p = m * mY

    # Initialization
    theta = theta0.copy() if theta0 is not None else np.zeros((p, 1))
    P = P0.copy()
    R = R0.copy()
    Q = Q0.copy()

    # Dropout mask handling
    if dropout0 is not None:
        dropout_exp = 3
        d0 = np.asarray(dropout0).ravel()[:p]
        idx_dropout = d0 < 1.0
        if np.any(idx_dropout):
            scale = np.maximum(d0[idx_dropout], 1e-3) ** dropout_exp
            P[np.ix_(idx_dropout, idx_dropout)] *= scale[:, None] * scale[None, :]
            Q[np.ix_(idx_dropout, idx_dropout)] *= scale[:, None] * scale[None, :]

    # Storage
    theta_est = np.zeros((n, p))
    Y_pred = np.full((n, mY), np.nan)
    P_hist = np.zeros((p, p, n))
    e_raw = np.full((n, mY), np.nan)
    Q_trace = np.full((n,), np.nan)
    R_trace = np.full((n,), np.nan)

    I_p = np.eye(p)

    rho_R = 0.02
    rho_Q = 0.02

    # Kalman filter loop
    for t in range(lags + 1, n):
        # Build X_t: [ΔY_{t-1}..ΔY_{t-lags}, ΔZ_{t-1}, ECT_{t-1}]
        pieces = []
        for i in range(1, lags + 1):
            pieces.append(Y[t - i, :])
        if mX > 0:
            pieces.append(Z_all[t - 1, :])
        X_t = np.concatenate(pieces)  # (m,)

        # Build H_t equation by equation
        H_t = np.zeros((mY, p))
        for j in range(mY):
            idx = slice(j * m, (j + 1) * m)
            H_t[j, idx] = X_t

        # Standard Kalman filter steps
        # Predict
        theta_pred = theta
        P_pred = P + Q

        # Innovation covariance
        S = H_t @ P_pred @ H_t.T + R
        S = (S + S.T) / 2
        S += eps * np.eye(mY)

        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = P_pred @ H_t.T @ S_inv

        # Update
        y_t = Y[t, :].reshape(-1, 1)
        y_hat = H_t @ theta_pred
        innovation = y_t - y_hat

        theta = theta_pred + K @ innovation
        P = (I_p - K @ H_t) @ P_pred
        P = (P + P.T) / 2
        P += eps * np.eye(p)

        # Save
        theta_est[t, :] = theta.ravel()
        Y_pred[t, :] = y_hat.ravel()
        P_hist[:, :, t] = P
        e_raw[t, :] = innovation.ravel()

        # Record Q/R trace (for diagnostics/visualization)
        Q_trace[t] = np.trace(Q)
        R_trace[t] = np.trace(R)

        if Q_R_update == True:
            # Compute updated R_t
            R_new = (1.0 - rho_R) * R + rho_R * (innovation @ innovation.T)
            # Prevent excessive R blow-up
            if np.trace(R_new) > 5 * np.trace(R0):
                R_new = 5 * np.trace(R0) / np.trace(R_new) * R_new
            R = 0.5 * (R_new + R_new.T) + eps * np.eye(mY)

            # Global scaling for Q
            err2 = float(innovation.T @ innovation) / mY
            tr_Q = np.trace(Q)
            if tr_Q > eps and err2 > 0:
                scale = err2 / tr_Q
                Q = (1.0 - rho_Q) * Q + rho_Q * scale * Q
        
    # Compute error
    valid = ~np.isnan(e_raw).any(axis=1)
    if np.any(valid):
        rmse = np.sqrt(np.nanmean(e_raw[valid] ** 2))
    else:
        rmse = np.nan

    return rmse, e_raw, theta_est, Y_pred, P_hist, Q_trace, R_trace

# ---------- E-step: KF filter with full storage ----------
def kf_e_step_store(
    Y: np.ndarray,
    Z_all: np.ndarray,
    theta0: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    P0: np.ndarray,
    lags: int = 2,
    eps: float = 1e-8,
):
    """
      theta_filt, P_filt, theta_pred, P_pred, H_list, y_list, valid_mask
    """
    n, mY = Y.shape
    mX = Z_all.shape[1] if Z_all is not None and Z_all.size > 0 else 0
    m = lags * mY + mX
    p = m * mY

    theta = theta0.copy().reshape(p, 1)
    P = P0.copy()
    I_p = np.eye(p)

    theta_filt = np.full((n, p), np.nan)
    P_filt = np.zeros((n, p, p))
    theta_pred = np.full((n, p), np.nan)
    P_pred = np.zeros((n, p, p))
    H_list = [None] * n
    y_list = [None] * n
    valid_mask = np.zeros(n, dtype=bool)

    for t in range(lags + 1, n):
        # build regressor
        pieces = [Y[t - i, :] for i in range(1, lags + 1)]
        if mX > 0:
            pieces.append(Z_all[t - 1, :])
        X_t = np.concatenate(pieces)

        H_t = np.zeros((mY, p))
        for j in range(mY):
            idx = slice(j * m, (j + 1) * m)
            H_t[j, idx] = X_t

        # predict
        theta_pr = theta
        P_pr = P + Q

        # update
        y_t = Y[t, :].reshape(-1, 1)
        S = H_t @ P_pr @ H_t.T + R
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

        # store
        theta_pred[t, :] = theta_pr.ravel()
        P_pred[t, :, :] = P_pr
        theta_filt[t, :] = theta.ravel()
        P_filt[t, :, :] = P
        H_list[t] = H_t
        y_list[t] = y_t
        valid_mask[t] = True

    return theta_filt, P_filt, theta_pred, P_pred, H_list, y_list, valid_mask


# ---------- RTS smoother ----------
def rts_smoother(
    theta_filt: np.ndarray,
    P_filt: np.ndarray,
    theta_pred: np.ndarray,
    P_pred: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-8,
):
    """
    RTS smoother for random-walk state model (theta_t = theta_{t-1} + w_t).
    Returns: theta_smooth, P_smooth, J_hist
    """
    n, p = theta_filt.shape
    theta_smooth = theta_filt.copy()
    P_smooth = P_filt.copy()
    J_hist = np.zeros((n, p, p))

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) <= 1:
        return theta_smooth, P_smooth, J_hist

    for k in range(len(valid_idx) - 2, -1, -1):
        t = valid_idx[k]
        t1 = valid_idx[k + 1]

        P_f = P_filt[t]
        P_pr_next = P_pred[t1]
        try:
            inv_Ppr = np.linalg.inv(P_pr_next)
        except np.linalg.LinAlgError:
            inv_Ppr = np.linalg.pinv(P_pr_next)

        J_t = P_f @ inv_Ppr
        J_hist[t] = J_t

        x_f = theta_filt[t].reshape(-1, 1)
        x_pr_next = theta_pred[t1].reshape(-1, 1)
        x_sm_next = theta_smooth[t1].reshape(-1, 1)

        x_sm = x_f + J_t @ (x_sm_next - x_pr_next)
        P_sm = P_f + J_t @ (P_smooth[t1] - P_pr_next) @ J_t.T
        P_sm = 0.5 * (P_sm + P_sm.T) + eps * np.eye(p)

        theta_smooth[t, :] = x_sm.ravel()
        P_smooth[t] = P_sm

    return theta_smooth, P_smooth, J_hist


def compute_kf_smoother_diagnostics(
    R: np.ndarray,
    pack: dict,
    eps: float = 1e-8,
):
    """
    Compute diagnostics over time t using final EM E-step + RTS outputs:
    - innovation_score[t] = v_t' S_t^{-1} v_t, where v_t = y_t - H_t theta_{t|t-1},
      and S_t = H_t P_{t|t-1} H_t' + R
    - coefficient_change[t] = ||beta_smooth,t - beta_smooth,t-1||
    - filter_smoother_gap[t] = ||beta_smooth,t - beta_filt,t||
    Output length matches theta_filt rows; invalid entries are NaN.
    """
    theta_filt = pack["theta_filt"]
    n, _ = theta_filt.shape
    mY = R.shape[0]

    innovation_score = np.full(n, np.nan)
    coefficient_change = np.full(n, np.nan)
    filter_smoother_gap = np.full(n, np.nan)

    theta_smooth = pack["theta_smooth"]
    theta_pred = pack["theta_pred"]
    P_pred = pack["P_pred"]
    H_list = pack["H_list"]
    y_list = pack["y_list"]
    valid_mask = pack["valid_mask"]

    valid_idx = np.where(valid_mask)[0]

    for t in valid_idx:
        H_t = H_list[t]
        y_t = y_list[t]
        if H_t is None or y_t is None:
            continue

        theta_pr = theta_pred[t].reshape(-1, 1)
        P_pr = P_pred[t]
        v = y_t - H_t @ theta_pr
        S = H_t @ P_pr @ H_t.T + R
        S = 0.5 * (S + S.T) + eps * np.eye(mY)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        innovation_score[t] = float((v.T @ S_inv @ v).item())

        xf = theta_filt[t]
        xs = theta_smooth[t]
        filter_smoother_gap[t] = float(np.linalg.norm(xs - xf))

    for k in range(1, len(valid_idx)):
        t = valid_idx[k]
        t0 = valid_idx[k - 1]
        coefficient_change[t] = float(
            np.linalg.norm(theta_smooth[t] - theta_smooth[t0])
        )

    return innovation_score, coefficient_change, filter_smoother_gap


# ---------- M-step ----------
def em_m_step_update(
    Y: np.ndarray,
    H_list: list,
    y_list: list,
    valid_mask: np.ndarray,
    theta_smooth: np.ndarray,
    P_smooth: np.ndarray,
    Q_old: np.ndarray,
    R_old: np.ndarray,
    em_damping: float = 0.7,
    eps: float = 1e-8,
):
    """
    Update Q and R using smoother outputs.
    Practical EM moments used here:
      R <- sample mean of E[(y-Hx)(y-Hx)' + HPH']
      Q <- sample mean of E[(x_t-x_{t-1})(x_t-x_{t-1})' + P_t + P_{t-1}] (approx.)
    """
    n, p = theta_smooth.shape
    mY = Y.shape[1]

    # update R
    R_acc = np.zeros((mY, mY))
    cntR = 0
    for t in np.where(valid_mask)[0]:
        H_t = H_list[t]
        y_t = y_list[t]
        if H_t is None or y_t is None:
            continue
        x_t = theta_smooth[t].reshape(-1, 1)
        resid = y_t - H_t @ x_t
        R_t = resid @ resid.T + H_t @ P_smooth[t] @ H_t.T
        R_acc += R_t
        cntR += 1
    if cntR > 0:
        R_new = R_acc / cntR
    else:
        R_new = R_old.copy()

    # ===== update Q (diagonal, stable version) =====
    valid_idx = np.where(valid_mask)[0]
    Q_acc_diag = np.zeros(p)
    cntQ = 0

    for k in range(1, len(valid_idx)):
        t = valid_idx[k]
        t0 = valid_idx[k - 1]

        x_t = theta_smooth[t]
        x_prev = theta_smooth[t0]

        d = x_t - x_prev

        # More stable approximation to avoid systematic Q overestimation
        q_diag_t = d**2 + 0.5 * (np.diag(P_smooth[t]) + np.diag(P_smooth[t0]))

        Q_acc_diag += q_diag_t
        cntQ += 1

    # Average
    if cntQ > 0:
        Q_new_diag = Q_acc_diag / cntQ
    else:
        Q_new_diag = np.diag(Q_old)

    # ===== damping (in diagonal space) =====
    Q_old_diag = np.diag(Q_old)
    Q_diag = em_damping * Q_old_diag + (1.0 - em_damping) * Q_new_diag

    # ===== anti-explosion guard (dimension-wise clipping) =====
    Q_diag = np.minimum(Q_diag, 10.0 * Q_old_diag)

    # ===== shrinkage (critical to prevent unstable jumps) =====
    Q_diag *= 0.5

    # ===== enforce positive definiteness =====
    Q_diag = np.maximum(Q_diag, eps)

    # ===== restore diagonal matrix =====
    Q = np.diag(Q_diag)

    # damping + stabilize
    R = em_damping * R_old + (1.0 - em_damping) * R_new

    R = 0.5 * (R + R.T) + eps * np.eye(mY)

    # Prevent excessive inflation
    trR_old = max(np.trace(R_old), eps)
    trQ_old = max(np.trace(Q_old), eps)
    if np.trace(R) > 10.0 * trR_old:
        R *= (10.0 * trR_old / max(np.trace(R), eps))
    if np.trace(Q) > 10.0 * trQ_old:
        Q *= (10.0 * trQ_old / max(np.trace(Q), eps))

    return Q, R


# ---------- EM runner (VAR init + E/M iterations) ----------
def run_kf_em(
    Y: np.ndarray,
    Z: np.ndarray,
    lags: int = 2,
    window: int = 40,
    ridge: float = 1e-6,
    max_em_iter: int = 10,
    tol: float = 1e-4,
    em_damping: float = 0.7,
    eps: float = 1e-8,
    verbose: bool = True,
):

    # first init from rolling VARX
    theta0, Q, R, P0, m, p = init_from_varx_rolling(
        Y=Y, Z=Z, lags=lags, window=window, ridge=ridge, eps=eps
    )

    history = {
        "trace_Q": [],
        "trace_R": [],
        "theta0_norm": [],
    }

    last_obj = np.inf
    best_pack = None

    for it in range(max_em_iter):
        # E-step
        theta_filt, P_filt, theta_pred, P_pred, H_list, y_list, valid_mask = kf_e_step_store(
            Y=Y, Z_all=Z, theta0=theta0, Q=Q, R=R, P0=P0, lags=lags, eps=eps
        )

        # smoother
        theta_smooth, P_smooth, J_hist = rts_smoother(
            theta_filt, P_filt, theta_pred, P_pred, valid_mask, eps=eps
        )

        # M-step
        Q_new, R_new = em_m_step_update(
            Y=Y,
            H_list=H_list,
            y_list=y_list,
            valid_mask=valid_mask,
            theta_smooth=theta_smooth,
            P_smooth=P_smooth,
            Q_old=Q,
            R_old=R,
            em_damping=em_damping,
            eps=eps,
        )

        # update theta0 with latest smoothed state
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) > 0:
            theta0 = theta_smooth[valid_idx[0]].reshape(-1, 1)

        Q, R = Q_new, R_new

        # a simple objective proxy (smoothed one-step residual)
        obj = 0.0
        cnt = 0
        for t in valid_idx:
            H_t = H_list[t]
            y_t = y_list[t]
            if H_t is None or y_t is None:
                continue
            e = y_t - H_t @ theta_smooth[t].reshape(-1, 1)
            obj += float((e.T @ e).item())
            cnt += 1
        obj = obj / max(cnt, 1)

        history["trace_Q"].append(float(np.trace(Q)))
        history["trace_R"].append(float(np.trace(R)))
        history["theta0_norm"].append(float(np.linalg.norm(theta0)))

        if verbose:
            print(f"[EM] iter={it+1:02d}, obj={obj:.6f}, trQ={np.trace(Q):.6e}, trR={np.trace(R):.6e}")

        best_pack = {
            "theta_filt": theta_filt,
            "P_filt": P_filt,
            "theta_pred": theta_pred,
            "P_pred": P_pred,
            "theta_smooth": theta_smooth,
            "P_smooth": P_smooth,
            "J_hist": J_hist,
            "valid_mask": valid_mask,
            "H_list": H_list,
            "y_list": y_list,
        }

        if abs(last_obj - obj) < tol:
            if verbose:
                print(f"[EM] converged at iter={it+1}, |Δobj|={abs(last_obj-obj):.3e}")
            break
        last_obj = obj

    # final forward KF run (reuse your original output format)
    rmse, e_raw, theta_est, Y_pred, P_hist, Q_trace, R_trace = kalman_multilag_filter_vecm(
        Y=Y,
        Z_all=Z,
        Y_level_aligned=np.zeros_like(Y),  # level is unused in this KF variant (placeholder)
        Q0=Q,
        R0=R,
        P0=P0,
        theta0=theta0,
        lags=lags,
        eps=eps,
        Q_R_update=False,
    )

    n_y = Y.shape[0]
    innov_score = np.full(n_y, np.nan)
    coef_change = np.full(n_y, np.nan)
    fs_gap = np.full(n_y, np.nan)
    if best_pack is not None:
        innov_score, coef_change, fs_gap = compute_kf_smoother_diagnostics(
            R=R, pack=best_pack, eps=eps
        )

    return {
        "rmse": rmse,
        "e_raw": e_raw,
        "theta_est": theta_est,              # filtered trajectory (original output)
        "Y_pred": Y_pred,
        "P_hist": P_hist,
        "Q_trace": Q_trace,
        "R_trace": R_trace,
        "theta0": theta0,
        "Q": Q,
        "R": R,
        "P0": P0,
        "m": m,
        "p": p,
        "em_history": history,
        "e_step_store": best_pack,           # E-step + smoother trajectories (incl. coefficients)
        "innovation_score": innov_score,
        "coefficient_change": coef_change,
        "filter_smoother_gap": fs_gap,
    }

# ============================================
#              FUNCTION 1
#  load_country_data(): data loading + slicing + year labels
# ============================================

def load_country_data(fname, country, ttls_Y, ttls_X):
    T = pd.read_excel(fname, sheet_name=country)
    T_col = list(T.columns)

    # === moved from main ===
    ttl_Y, idx_TY, idx_Y = intersect_stable(ttls_Y, T_col)
    ttl_X, idx_TX, idx_X = intersect_stable(ttls_X, T_col)
    mY = len(idx_Y)
    mX = len(idx_X)

    Y_lvl_df = T.iloc[:, idx_Y].copy()
    X_lvl_df = T.iloc[:, idx_X].copy()

    Y_levels = Y_lvl_df.to_numpy()
    X_levels = X_lvl_df.to_numpy()

    Y_levels_cut = Y_levels[:, :]
    X_levels_cut = X_levels[:, :]

    # Difference all Y series
    Yd = np.diff(Y_levels_cut, axis=0)

    # X: assume ENSO is the last column
    X_macro = X_levels_cut[:, :-1]   # macro X
    X_enso  = X_levels_cut[:, -1:]   # ENSO (kept in level)

    Xd_macro = np.diff(X_macro, axis=0)
    Xd_enso  = X_enso[1:, :]         # align length after differencing

    # Rebuild Xd: differenced macro X + ENSO level
    Xd = np.hstack([Xd_macro, Xd_enso])

    Y_level_aligned = Y_levels_cut[1:, :]
    n, mY = Yd.shape
    mX = Xd.shape[1]

        # === Year handling (burn-in logic removed) ===
    if len(T) > 0:
        first_col = T.iloc[:, 0]
        try:
            dates = pd.to_datetime(first_col, errors='coerce')

            if dates.notna().sum() > len(dates) / 2:
                # Align with differenced length
                dates_diff = dates[1:].reset_index(drop=True)

                # Use quarterly labels
                year_labels = dates_diff.dt.to_period('Q').astype(str)

            else:
                # Fallback: construct quarterly index (no +40 offset)
                start_year = 1970
                start_quarter = 1

                years = []
                for i in range(len(Yd)):
                    q = (i + start_quarter - 1) % 4 + 1
                    y = start_year + (i + start_quarter - 1) // 4
                    years.append(f"{y}Q{q}")

                year_labels = pd.Series(years)

        except:
            # fallback
            start_year = 1970
            years = []
            for i in range(len(Yd)):
                y = start_year + i // 4
                years.append(str(y))

            year_labels = pd.Series(years)
    else:
        year_labels = pd.Series([str(1970 + i // 4) for i in range(len(Yd))])

    return (
        Yd, Xd, Y_level_aligned, year_labels,
        Y_levels, idx_Y, idx_X, ttl_Y
    )


def trim_start(T0, Yd, Xd, Y_level_aligned, year_labels):
    
    if len(Yd) <= T0:
        raise ValueError("Not enough data after trimming")

    Yd_trim = Yd[T0:, :]
    Xd_trim = Xd[T0:, :]
    Y_level_trim = Y_level_aligned[T0:, :]
    year_trim = np.array(year_labels)[T0:]

    print(f"[DEBUG] Trimmed first {T0} obs")
    print(f"New length = {len(Yd_trim)}")

    return Yd_trim, Xd_trim, Y_level_trim, year_trim

# ============================================
#              FUNCTION 2
#   run_tvpkf(): two-stage run (VARX init + KF)
# ============================================

def run_tvpkf(
    Yd,
    Xd,
    Y_level_aligned,
    lags,
    init_window=40,
    two_stage=True,
):
    # ===== Standardization (ONLY HERE) =====
    Y_mean = Yd.mean(axis=0)
    Y_std  = Yd.std(axis=0) + 1e-8

    X_mean = Xd.mean(axis=0)
    X_std  = Xd.std(axis=0) + 1e-8

    Yd = (Yd - Y_mean) / Y_std
    Xd = (Xd - X_mean) / X_std

    if two_stage:
        # round 1: rolling VARX initialization
        theta0, Q0, R0, P0, m_model, p = init_from_varx_rolling(
            Y=Yd,
            Z=Xd,
            lags=lags,
            window=init_window,
        )
    else:
        # Single-stage fallback: mild default initialization
        mY = Yd.shape[1]
        mX = Xd.shape[1]
        m_model = lags * mY + mX
        p = m_model * mY
        theta0 = np.zeros((p, 1))
        Q0 = 1e-4 * np.eye(p)
        R0 = np.diag(np.var(Yd, axis=0) + 1e-6)
        P0 = 10.0 * np.eye(p)

    # round 2: Kalman main run
    rmse, e_raw, theta_est, Y_pred, P_hist, Q_trace, R_trace = kalman_multilag_filter_vecm(
        Y=Yd,
        Z_all=Xd,
        Y_level_aligned=Y_level_aligned,
        Q0=Q0,
        R0=R0,
        P0=P0,
        theta0=theta0,
        lags=lags,
        Q_R_update=False,
    )

    return {
        "rmse": rmse,
        "e_raw": e_raw,
        "theta_est": theta_est,
        "Y_pred": Y_pred,
        "P_hist": P_hist,
        "Q_trace": Q_trace,
        "R_trace": R_trace,
        "p": p,
        "m_model": m_model,
        "theta0": theta0,
        "Q0": Q0,
        "R0": R0,
        "P0": P0,
        "two_stage": two_stage,
        "init_window": init_window,
        "Yd_used": Yd,
        "Xd_used": Xd,
    }


# ============================================
#              FUNCTION 3
#     plot_core_results(): P trace + RMSE
# ============================================

def plot_core_results(
    country, Yd, Xd, year_labels,
    res, ttls_Y, ttls_X, lags
):
    e_raw = res["e_raw"]
    Y_pred = res["Y_pred"]
    P_hist = res["P_hist"]
    Q_trace = res.get("Q_trace")
    R_trace = res.get("R_trace")
    p = res["p"]

    # === moved from main ===
    n_tmp = e_raw.shape[0]
    P_tr = np.array([np.trace(P_hist[:, :, t]) / p for t in range(n_tmp)])
        
    e_raw_full = Yd - Y_pred
    rmse_raw = np.sqrt(np.nanmean(e_raw_full**2, axis=1))

    # Null model
    Yhat_null = np.full_like(Yd, np.nan)
    Yhat_null[1:, :] = Yd[:-1, :]
    E_null = Yd - Yhat_null
    rmse_null = np.sqrt(np.nanmean(E_null**2, axis=1))

    # VARX benchmark
    Yhat_varx_raw, E_varx_raw, rmse_varx_raw = varx_rolling_predict(Yd, Xd, lags=lags)

    # label spacing
    n_labels = len(year_labels)
    step = max(1, n_labels // 10)
    tick_indices = np.arange(0, n_labels, step)
    tick_labels = [year_labels[i] for i in tick_indices]

    plt.figure(954, figsize=(10, 8))
    plt.clf()

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(np.sqrt(P_tr))
    ax1.set_title('P (sqrt mean coeff var)')
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(rmse_raw, label='KF (TVP-VECM)', alpha=0.9)
    ax2.plot(rmse_varx_raw, label='VARX raw', linestyle='--', alpha=0.9)
    ax2.plot(rmse_null, label='Naive lag-1', linestyle=':', alpha=0.8)
    ax2.set_title('Y−Yhat (rms of prediction error)')
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 2, 3)
    if Q_trace is not None:
        ax3.plot(Q_trace, color='tab:green')
    ax3.set_title('trace(Q_t)')
    ax3.set_xticks(tick_indices)
    ax3.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 2, 4)
    if R_trace is not None:
        ax4.plot(R_trace, color='tab:red')
    ax4.set_title('trace(R_t)')
    ax4.set_xticks(tick_indices)
    ax4.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show(block=False)



# ============================================
#         FUNCTION 4: weighted_corr
#         FUNCTION 5: plot_pred_corr
# ============================================

def weighted_corr(x, y, w):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    w = np.asarray(w).ravel()
    
    n = min(len(x), len(y), len(w))
    x, y, w = x[:n], y[:n], w[:n]
    
    m = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(w)
    if m.sum() < 3:
        return np.nan
    
    x, y, w = x[m], y[m], w[m]
    w = w / np.sum(w)
    
    mx = np.sum(w * x)
    my = np.sum(w * y)
    
    cov_xy = np.sum(w * (x - mx) * (y - my))
    var_x = np.sum(w * (x - mx)**2)
    var_y = np.sum(w * (y - my)**2)
    
    if var_x <= 0 or var_y <= 0:
        return np.nan
    
    return cov_xy / np.sqrt(var_x * var_y)

def plot_pred_corr(Yd, Y_pred, Xd, ttls_Y, ttls_X, country, method, year_labels):

    # === Full reuse of your original function ===

    Y_true = Yd[1:, :]
    Yhat = Y_pred[1:, :]
    Y_lag1 = Yd[:-1, :]
    Z_lag1 = Xd[:-1, :]
    
    year_labels_short = year_labels[1:] if len(year_labels) > len(Y_true) else year_labels[:len(Y_true)]
    n_use = len(year_labels_short)
    step_ts = max(1, n_use // 8)
    tick_indices_ts = np.arange(0, n_use, step_ts)
    tick_labels_ts = [year_labels_short[i] for i in tick_indices_ts]
    
    w_use = np.ones(n_use) / n_use
    
    mY = Y_true.shape[1]
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(2, mY, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)
    
    for j in range(mY):
        ax_ts = fig.add_subplot(gs[0, j])
        ax_ts.plot(Y_true[:, j], color='tab:blue', lw=1.0, label='data')
        ax_ts.plot(Yhat[:, j], color='tab:orange', lw=1.0, ls='--', label='prediction')
        ax_ts.set_title(ttls_Y[j], fontsize=11)
        ax_ts.set_xticks(tick_indices_ts)
        ax_ts.set_xticklabels(tick_labels_ts, rotation=45, ha='right', fontsize=8)
        ax_ts.grid(True, alpha=0.3)
        if j == 0:
            ax_ts.legend(loc='upper right', fontsize=9)
        
        ax_sc = fig.add_subplot(gs[1, j])
        ax_sc.scatter(Y_true[:, j], Yhat[:, j], s=10, color='tab:purple', alpha=0.6)
        lim = np.nanmax(np.abs(np.concatenate([Y_true[:, j], Yhat[:, j]])))
        if np.isfinite(lim) and lim > 0:
            ax_sc.plot([-lim, lim], [-lim, lim], color='gray', lw=1, ls=':')
            ax_sc.set_xlim(-lim, lim)
            ax_sc.set_ylim(-lim, lim)
        ax_sc.set_xlabel('data', fontsize=9)
        ax_sc.set_ylabel('prediction', fontsize=9)
        
        r_pred = weighted_corr(Y_true[:, j], Yhat[:, j], w_use)
        r_lag = weighted_corr(Y_lag1[:, j], Y_true[:, j], w_use)
        r_exog_all = np.array([weighted_corr(Z_lag1[:, k], Y_true[:, j], w_use)
                               for k in range(Z_lag1.shape[1])])
        
        if r_exog_all.size > 0 and np.any(np.isfinite(r_exog_all)):
            k_best = int(np.nanargmax(np.abs(r_exog_all)))
            r_exog = r_exog_all[k_best]
            exog_name = ttls_X[k_best] if k_best < len(ttls_X) else f"X{k_best}"
        else:
            r_exog, exog_name = np.nan, "N/A"
        
        txt = (
            f"corr(Y, Yhat) = {r_pred: .3f}\n"
            f"corr(Y[-1], Y) = {r_lag: .3f}\n"
            f"max corr(Z[-1], Y) = {r_exog: .3f} ({exog_name})"
        )
        ax_sc.text(0.02, -0.25, txt, transform=ax_sc.transAxes, fontsize=9, va='top')
    
    fig.suptitle(f"{country} — {method}", fontsize=14, y=0.98)
    plt.show(block=False)

# ============================================
#              FUNCTION 6
#        plot_dropone_heatmap()
# ============================================

def plot_dropone_heatmap(
    country, Yd, Xd, Y_level_aligned,
    theta0, Q0, R0, P0,
    m_model, r, p, lags, base_rmse,
    ttl_Y, ttls_X
):

    # === Full reuse of your original code ===

    labels_x = []
    mY = Yd.shape[1]

    for j in range(lags * mY):
        lag_id = j // mY + 1
        var_id = j % mY
        labels_x.append(f"L{lag_id}_{ttl_Y[var_id]}")
    for j in range(Xd.shape[1]):
        labels_x.append(ttls_X[j])

    print(f"\n[{country}] computing drop-one heatmap (this may take a while)...")
    E_hists_k = np.full((mY, m_model), np.nan)

    for i_Y in range(mY):
        for j_YX in range(m_model):
            idx_0 = i_Y * m_model + j_YX
            dropout0 = np.ones((p,))
            dropout0[idx_0] = 0.0
            
            rmse_d, *_ = kalman_multilag_filter_vecm(
                Y=Yd, Z_all=Xd, Y_level_aligned=Y_level_aligned,
                Q0=Q0, R0=R0, P0=P0,
                theta0=theta0,
                dropout0=dropout0,
                lags=lags
            )
            E_hists_k[i_Y, j_YX] = rmse_d / base_rmse if base_rmse > 0 else np.nan

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))
    im = axs[0].imshow(E_hists_k, aspect='auto', vmin=0.95, vmax=1.05, cmap="coolwarm")
    axs[0].set_yticks(np.arange(mY))
    axs[0].set_yticklabels(ttl_Y)
    axs[0].set_xticks(np.arange(m_model))
    axs[0].set_xticklabels(labels_x, rotation=45, ha="right")
    axs[0].set_title(f"{country}: mean errors (drop-one coeff)")
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    vals = E_hists_k.ravel()
    axs[1].hist(vals[~np.isnan(vals)], bins=20, edgecolor="black")
    axs[1].set_title(f"{country}: histogram of errors")
    axs[1].set_xlabel("Error value")
    axs[1].set_ylabel("Frequency")
    plt.tight_layout()
    plt.show(block=False)




# if __name__ == "__main__":
    
    print("\n================= LOADING DATA =================\n")

    fname = "/Users/poppy/iCloud Drive (Archive)/Desktop/GVAR/df_country_data_climate.xlsx"

    sheets_9_of_12 = [
        # "India", "Brazil", "Chile", "Indonesia", "Mexico", "Peru",
        "Philippines", "South Africa", "Thailand"
    ]
    CLIMATE_VARS_BY_COUNTRY = {
        "India": {"SOI": 4},           # SOI leads by 2 quarters
        "Brazil": {"NINO3+4": 1},
        "Chile": {"SOI": 3, "copper": 1},
        "Indonesia": {"NINO1+2": 4},
        "Mexico": {"WestenV": 1},
        "Peru": {"SOI": 3},
        "Philippines": {"WestenV": 1},
        "South Africa": {"SOI": 2},
        "Thailand": {"SOI": 2}
    }
    ttls = ['y', 'Dp', 'eq', 'ep', 'r', 'trade', 'ys', 'Dps', 'eqs', 'rs', 'lrs', 'ensos']
    ttls_Y = ttls[:6]
    ttls_X_base = ttls[6:11]    # keep base X names immutable across countries
    lags = 1

    for country in sheets_9_of_12:

        print(f"\n\n====================================")
        print(f"***** Country {country} *****")
        print("====================================\n")

        # =====================================================
        # 1) Load and split country-level data
        # =====================================================
        Yd, Xd, Y_level_aligned, year_labels, Y_levels, idx_Y, idx_X, ttl_Y = \
            load_country_data(fname, country, ttls_Y, ttls_X_base)

        print(f"[DEBUG] len(Y_levels) = {len(Y_levels)}")
        print(f"[DEBUG] len(Yd)       = {len(Yd)}")
        print(f"[DEBUG] len(year_labels) = {len(year_labels)}")


        # =====================================================
        # 2) Build Y_levels_df aligned with Yd
        # =====================================================
        # Yd = diff(Y_levels_cut) -> one row shorter than Y_levels_cut
        # Yd, year_labels, and Y_levels_df must have the same length

        nYd = len(Yd)

        # Y_levels_cut = Y_levels[40:], length = nYd + 1
        # Therefore slice Y_levels[-nYd:] to get nYd rows
        Y_levels_df = pd.DataFrame(
            Y_levels[-nYd:, :],
            columns=ttl_Y
        )

        print(f"[DEBUG] After slicing, len(Y_levels_df) = {len(Y_levels_df)}")

        # Align index
        if len(Y_levels_df) != len(year_labels):
            print("ERROR: Y_levels_df and year_labels length mismatch!")
            print(f"Y_levels_df = {len(Y_levels_df)}, year_labels = {len(year_labels)}")
            print("Stopping execution. Please check date alignment.")
            raise ValueError("Y_levels_df length mismatch.")

        # === Build quarterly date index ===
        # year_labels are strings like "1979", "1980", "1981"
        # convert them to quarter-end timestamps
        Y_levels_df.index = year_labels[-nYd:]
        quarter_index = pd.period_range(start='1979Q1', periods=nYd, freq='Q').to_timestamp('Q')
        # Override Y_levels_df.index with quarterly timestamps
        Y_levels_df.index = quarter_index

        # =====================================================
        # 3) Merge climate data
        # =====================================================
        print("\n[DEBUG] Processing climate variables for", country)
        print("Requested climate variables:", CLIMATE_VARS_BY_COUNTRY.get(country))

        # climate_X = attach_climate_to_X(
        #     Y_levels_df=Y_levels_df,
        #     climate_df=climate_df,
        #     country=country,
        #     climate_vars_dict=CLIMATE_VARS_BY_COUNTRY
        # )

        external_data_sources = [
            pd.read_csv("climate_indice.csv", parse_dates=["date"]),
            pd.read_csv("copper_indices.csv",  parse_dates=["date"]),
        ]
        climate_X = attach_climate_to_X(
            Y_levels_df=Y_levels_df,
            external_dfs=external_data_sources,
            country=country,
            climate_vars_dict=CLIMATE_VARS_BY_COUNTRY
        )

        print(f"[DEBUG] climate_X shape = {climate_X.shape}")
        print(f"[DEBUG] climate_X head():")
        print(climate_X.head(3))
        print(f"[DEBUG] climate_X tail():")
        print(climate_X.tail(3))
        print("\n=== DEBUG: climate data column names ===")
        print(climate_X.columns)

        # =====================================================
        # 4) Merge Xd + climate_X (exogenous variables)
        # =====================================================
        if climate_X.shape[0] != Xd.shape[0]:
            print("WARNING: climate_X row count mismatch, skipping append")
            print(f"Xd rows = {Xd.shape[0]}, climate_X rows = {climate_X.shape[0]}")
        else:
            Xd = np.hstack([Xd, climate_X.to_numpy()])

        # Country-specific exogenous variable names
        ttls_X_country = ttls_X_base + list(climate_X.columns)
        print(f"[DEBUG] ttls_X_country = {ttls_X_country}")

        # =====================================================
        # 4.5) No fixed burn-in; drop rows containing NaN only
        # =====================================================

        print("\n[DEBUG] Dropping rows with NaN ...")

        # 1) Build mask (all variables must be finite)
        mask_valid = (
            ~np.isnan(Yd).any(axis=1)
            & ~np.isnan(Xd).any(axis=1)
            & ~np.isnan(Y_level_aligned).any(axis=1)
        )

        # 2) Print diagnostics
        print(f"[DEBUG] valid rows = {mask_valid.sum()} / {len(mask_valid)}")

        # 3) Apply mask
        Yd = Yd[mask_valid, :]
        Xd = Xd[mask_valid, :]
        Y_level_aligned = Y_level_aligned[mask_valid, :]
        year_labels = np.array(year_labels)[mask_valid]

        print(f"[DEBUG] After drop:")
        print(f"Yd shape = {Yd.shape}")
        print(f"Xd shape = {Xd.shape}")

        # ============================================
        # 4.6) Trim initial period (burn-in)
        # ============================================

        TRIM_T0 = 40   # tune as needed (e.g. 20 / 40)

        Yd, Xd, Y_level_aligned, year_labels = trim_start(
            TRIM_T0,
            Yd, Xd, Y_level_aligned, year_labels
        )

        # =====================================================
        # 5) Kalman Filter
        # =====================================================
        print("\n[DEBUG] Running Kalman filter ...")
        print(f"Xd shape = {Xd.shape}")
        print(f"Yd shape = {Yd.shape}")

        res = run_tvpkf(Yd, Xd, Y_level_aligned, lags)
        print(f"Kalman filter done. RMSE = {res['rmse']:.6f}")


        # =====================================================
        # 7) Plot core results
        # =====================================================
        Yd_plot = res["Yd_used"]
        Xd_plot = res["Xd_used"]

        plot_core_results(country, Yd_plot, Xd_plot, year_labels, res, ttls_Y, ttls_X_country, lags)

        plot_pred_corr(
            Yd_plot, res["Y_pred"], Xd_plot,
            ttls_Y, ttls_X_country, country,
            "Kalman (TVP-VECM)", year_labels
        )

        Yhat_varx_raw, _, _ = varx_rolling_predict(Yd_plot, Xd_plot, lags=lags)
        plot_pred_corr(
            Yd_plot, Yhat_varx_raw, Xd_plot,
            ttls_Y, ttls_X_country, country,
            "VARX", year_labels
        )


        # =====================================================
        # 8) Drop-one stats
        # =====================================================
        plot_dropone_heatmap(
            country, Yd_plot, Xd_plot, Y_level_aligned,
            res["theta0"], res["Q0"], res["R0"], res["P0"],
            res["m_model"], 
            0,
            res["p"],
            lags, res["rmse"],
            ttl_Y, ttls_X_country
        )

# ============================================================
# New data: long panel + fixed exogenous variables (COMMODITY, ENSO)
# This section includes data loading + multi-country four-column heatmap panel_y_yhat_heatmap
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# ---------- Fill these two places ----------
PATH = "gvar_panel_streamlit (3).csv"          # or .xlsx, change path as needed
COUNTRIES = [
"CHL","MEX","BRA","COL", "AUS", "CAN","NOR","PHL","IND", "THA","IDN", "SWE", "CHE", "NZL",  "DNK"
# "AFG", "ALB", "DZA", "AND", "AGO", "ATG", "ARG", "ARM", "AUS", "AUT",
# "AZE", "BHS", "BHR", "BGD", "BRB", "BLR", "BEL", "BLZ", "BEN", "BTN",
# "BOL", "BIH", "BWA", "BRA", "BRN", "BGR", "BFA", "BDI", "CPV", "KHM",
# "CMR", "CAN", "CAF", "TCD", "CHL", "CHN", "COL", "COM", "COG", "COD",
# "CRI", "CIV", "HRV", "CUB", "CYP", "CZE", "DNK", "DJI", "DMA", "DOM",
# "ECU", "EGY", "SLV", "GNQ", "ERI", "EST", "SWZ", "ETH", "FJI", "FIN",
# "FRA", "GAB", "GMB", "GEO", "DEU", "GHA", "GRC", "GRD", "GTM", "GIN",
# "GNB", "GUY", "HTI", "HND", "HUN", "ISL", "IND", "IDN", "IRN", "IRQ",
# "IRL", "ISR", "ITA", "JAM", "JPN", "JOR", "KAZ", "KEN", "KIR", "KWT",
# "KGZ", "LAO", "LVA", "LBN", "LSO", "LBR", "LBY", "LIE", "LTU", "LUX",
# "MDG", "MWI", "MYS", "MDV", "MLI", "MLT", "MHL", "MRT", "MUS", "MEX",
# "FSM", "MDA", "MCO", "MNG", "MNE", "MAR", "MOZ", "MMR", "NAM", "NRU",
# "NPL", "NLD", "NZL", "NIC", "NER", "NGA", "PRK", "MKD", "NOR", "OMN",
# "PAK", "PLW", "PAN", "PNG", "PRY", "PER", "PHL", "POL", "PRT", "QAT",
# "ROU", "RUS", "RWA", "KNA", "LCA", "VCT", "WSM", "SMR", "STP", "SAU",
# "SEN", "SRB", "SYC", "SLE", "SGP", "SVK", "SVN", "SLB", "SOM", "ZAF",
# "KOR", "SSD", "ESP", "LKA", "SDN", "SUR", "SWE", "CHE", "SYR", "TJK",
# "TZA", "THA", "TLS", "TGO", "TON", "TTO", "TUN", "TUR", "TKM", "TUV",
# "UGA", "UKR", "ARE", "GBR", "USA", "URY", "UZB", "VUT", "VEN", "VNM",
# "YEM", "ZMB", "ZWE"
]

countries_to_run = [
"CHL","MEX","BRA","COL", "AUS", "CAN","NOR","PHL","IND", "THA","IDN", "SWE", "CHE", "NZL",  "DNK"
# "AFG", "ALB", "DZA", "AND", "AGO", "ATG", "ARG", "ARM", "AUS", "AUT",
# "AZE", "BHS", "BHR", "BGD", "BRB", "BLR", "BEL", "BLZ", "BEN", "BTN",
# "BOL", "BIH", "BWA", "BRA", "BRN", "BGR", "BFA", "BDI", "CPV", "KHM",
# "CMR", "CAN", "CAF", "TCD", "CHL", "CHN", "COL", "COM", "COG", "COD",
# "CRI", "CIV", "HRV", "CUB", "CYP", "CZE", "DNK", "DJI", "DMA", "DOM",
# "ECU", "EGY", "SLV", "GNQ", "ERI", "EST", "SWZ", "ETH", "FJI", "FIN",
# "FRA", "GAB", "GMB", "GEO", "DEU", "GHA", "GRC", "GRD", "GTM", "GIN",
# "GNB", "GUY", "HTI", "HND", "HUN", "ISL", "IND", "IDN", "IRN", "IRQ",
# "IRL", "ISR", "ITA", "JAM", "JPN", "JOR", "KAZ", "KEN", "KIR", "KWT",
# "KGZ", "LAO", "LVA", "LBN", "LSO", "LBR", "LBY", "LIE", "LTU", "LUX",
# "MDG", "MWI", "MYS", "MDV", "MLI", "MLT", "MHL", "MRT", "MUS", "MEX",
# "FSM", "MDA", "MCO", "MNG", "MNE", "MAR", "MOZ", "MMR", "NAM", "NRU",
# "NPL", "NLD", "NZL", "NIC", "NER", "NGA", "PRK", "MKD", "NOR", "OMN",
# "PAK", "PLW", "PAN", "PNG", "PRY", "PER", "PHL", "POL", "PRT", "QAT",
# "ROU", "RUS", "RWA", "KNA", "LCA", "VCT", "WSM", "SMR", "STP", "SAU",
# "SEN", "SRB", "SYC", "SLE", "SGP", "SVK", "SVN", "SLB", "SOM", "ZAF",
# "KOR", "SSD", "ESP", "LKA", "SDN", "SUR", "SWE", "CHE", "SYR", "TJK",
# "TZA", "THA", "TLS", "TGO", "TON", "TTO", "TUN", "TUR", "TKM", "TUV",
# "UGA", "UKR", "ARE", "GBR", "USA", "URY", "UZB", "VUT", "VEN", "VNM",
# "YEM", "ZMB", "ZWE"
]

ISO3_TO_COUNTRY = {
    "AFG": "Afghanistan", "ALB": "Albania", "DZA": "Algeria", "AND": "Andorra",
    "AGO": "Angola", "ATG": "Antigua and Barbuda", "ARG": "Argentina", "ARM": "Armenia",
    "AUS": "Australia", "AUT": "Austria", "AZE": "Azerbaijan", "BHS": "Bahamas",
    "BHR": "Bahrain", "BGD": "Bangladesh", "BRB": "Barbados", "BLR": "Belarus",
    "BEL": "Belgium", "BLZ": "Belize", "BEN": "Benin", "BTN": "Bhutan",
    "BOL": "Bolivia", "BIH": "Bosnia and Herzegovina", "BWA": "Botswana", "BRA": "Brazil",
    "BRN": "Brunei", "BGR": "Bulgaria", "BFA": "Burkina Faso", "BDI": "Burundi",
    "CPV": "Cabo Verde", "KHM": "Cambodia", "CMR": "Cameroon", "CAN": "Canada",
    "CAF": "Central African Republic", "TCD": "Chad", "CHL": "Chile", "CHN": "China",
    "COL": "Colombia", "COM": "Comoros", "COG": "Republic of the Congo", "COD": "Democratic Republic of the Congo",
    "CRI": "Costa Rica", "CIV": "Cote d'Ivoire", "HRV": "Croatia", "CUB": "Cuba",
    "CYP": "Cyprus", "CZE": "Czech Republic", "DNK": "Denmark", "DJI": "Djibouti",
    "DMA": "Dominica", "DOM": "Dominican Republic", "ECU": "Ecuador", "EGY": "Egypt",
    "SLV": "El Salvador", "GNQ": "Equatorial Guinea", "ERI": "Eritrea", "EST": "Estonia",
    "SWZ": "Eswatini", "ETH": "Ethiopia", "FJI": "Fiji", "FIN": "Finland",
    "FRA": "France", "GAB": "Gabon", "GMB": "Gambia", "GEO": "Georgia",
    "DEU": "Germany", "GHA": "Ghana", "GRC": "Greece", "GRD": "Grenada",
    "GTM": "Guatemala", "GIN": "Guinea", "GNB": "Guinea-Bissau", "GUY": "Guyana",
    "HTI": "Haiti", "HND": "Honduras", "HUN": "Hungary", "ISL": "Iceland",
    "IND": "India", "IDN": "Indonesia", "IRN": "Iran", "IRQ": "Iraq",
    "IRL": "Ireland", "ISR": "Israel", "ITA": "Italy", "JAM": "Jamaica",
    "JPN": "Japan", "JOR": "Jordan", "KAZ": "Kazakhstan", "KEN": "Kenya",
    "KIR": "Kiribati", "KWT": "Kuwait", "KGZ": "Kyrgyzstan", "LAO": "Laos",
    "LVA": "Latvia", "LBN": "Lebanon", "LSO": "Lesotho", "LBR": "Liberia",
    "LBY": "Libya", "LIE": "Liechtenstein", "LTU": "Lithuania", "LUX": "Luxembourg",
    "MDG": "Madagascar", "MWI": "Malawi", "MYS": "Malaysia", "MDV": "Maldives",
    "MLI": "Mali", "MLT": "Malta", "MHL": "Marshall Islands", "MRT": "Mauritania",
    "MUS": "Mauritius", "MEX": "Mexico", "FSM": "Micronesia", "MDA": "Moldova",
    "MCO": "Monaco", "MNG": "Mongolia", "MNE": "Montenegro", "MAR": "Morocco",
    "MOZ": "Mozambique", "MMR": "Myanmar", "NAM": "Namibia", "NRU": "Nauru",
    "NPL": "Nepal", "NLD": "Netherlands", "NZL": "New Zealand", "NIC": "Nicaragua",
    "NER": "Niger", "NGA": "Nigeria", "PRK": "North Korea", "MKD": "North Macedonia",
    "NOR": "Norway", "OMN": "Oman", "PAK": "Pakistan", "PLW": "Palau",
    "PAN": "Panama", "PNG": "Papua New Guinea", "PRY": "Paraguay", "PER": "Peru",
    "PHL": "Philippines", "POL": "Poland", "PRT": "Portugal", "QAT": "Qatar",
    "ROU": "Romania", "RUS": "Russia", "RWA": "Rwanda", "KNA": "Saint Kitts and Nevis",
    "LCA": "Saint Lucia", "VCT": "Saint Vincent and the Grenadines", "WSM": "Samoa",
    "SMR": "San Marino", "STP": "Sao Tome and Principe", "SAU": "Saudi Arabia",
    "SEN": "Senegal", "SRB": "Serbia", "SYC": "Seychelles", "SLE": "Sierra Leone",
    "SGP": "Singapore", "SVK": "Slovakia", "SVN": "Slovenia", "SLB": "Solomon Islands",
    "SOM": "Somalia", "ZAF": "South Africa", "KOR": "South Korea", "SSD": "South Sudan",
    "ESP": "Spain", "LKA": "Sri Lanka", "SDN": "Sudan", "SUR": "Suriname",
    "SWE": "Sweden", "CHE": "Switzerland", "SYR": "Syria", "TJK": "Tajikistan",
    "TZA": "Tanzania", "THA": "Thailand", "TLS": "Timor-Leste", "TGO": "Togo",
    "TON": "Tonga", "TTO": "Trinidad and Tobago", "TUN": "Tunisia", "TUR": "Turkey",
    "TKM": "Turkmenistan", "TUV": "Tuvalu", "UGA": "Uganda", "UKR": "Ukraine",
    "ARE": "United Arab Emirates", "GBR": "United Kingdom", "USA": "United States",
    "URY": "Uruguay", "UZB": "Uzbekistan", "VUT": "Vanuatu", "VEN": "Venezuela",
    "VNM": "Vietnam", "YEM": "Yemen", "ZMB": "Zambia", "ZWE": "Zimbabwe"
}


def country_display_name(iso3: str, iso3_to_country: dict | None = None) -> str:
    """ISO3 -> full country name (for titles, heatmap y-axis, etc.)."""
    if not iso3_to_country:
        return str(iso3)
    return str(iso3_to_country.get(iso3, iso3))


def style_quarter_axis_every_four_quarters(ax) -> None:
    """Set x-axis major ticks/grid every 4 quarters (12 months), label as YYYY-MM."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, which="major", axis="x", alpha=0.35)


def _fig_save_or_show(fig, pdf: PdfPages | None) -> None:
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show(block=False)


# ---------- Column names (edit here if your file uses different names) ----------
COL_COUNTRY = "country"
COL_TIME = "quarter"
ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
EXO = ["COMMODITY_YoY", "ENSO"]

lags = 1

# ---------- Read data ----------
df = pd.read_csv(PATH)
df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
df = df.dropna(subset=[COL_TIME]).sort_values([COL_COUNTRY, COL_TIME])

endo_use = [c for c in ENDO if df[c].notna().any()]
if len(endo_use) < len(ENDO):
    dropped = set(ENDO) - set(endo_use)
    print(f"[INFO] drop all-NaN endogenous columns: {dropped}")
ENDO = endo_use

if len(COUNTRIES) == 0:
    countries_to_run = sorted(df[COL_COUNTRY].dropna().unique())
    print(f"[INFO] COUNTRIES empty → run all: n={len(countries_to_run)}")
else:
    countries_to_run = list(COUNTRIES)

PANEL_HEATMAP_VERSION = "4col-in-cell4-2026-04-22"


def _finite_corr(y: np.ndarray, yhat: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 3:
        return np.nan
    yy, hh = y[m], yhat[m]
    if np.std(yy) < 1e-12 or np.std(hh) < 1e-12:
        return np.nan
    return float(np.corrcoef(yy, hh)[0, 1])


def _finite_rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    m = np.isfinite(y) & np.isfinite(yhat)
    if not np.any(m):
        return np.nan
    return float(np.sqrt(np.mean((y[m] - yhat[m]) ** 2)))


def _metrics_one_country(Yd, Yhat, j):
    y = Yd[1:, j]
    h = Yhat[1:, j]
    return _finite_corr(y, h), _finite_rmse(y, h)


def panel_y_yhat_heatmap(
    PATH,
    COUNTRIES,
    COL_COUNTRY="country",
    COL_TIME="quarter",
    ENDO=None,
    EXO=None,
    lags=1,
    min_T=None,
    countries_per_figure=20,
    iso3_to_country: dict | None = None,
):

    if ENDO is None:
        ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
    if EXO is None:
        EXO = ["COMMODITY_YoY", "ENSO"]
    if min_T is None:
        min_T = lags + 5

    df = pd.read_csv(PATH)
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df = df.dropna(subset=[COL_TIME]).sort_values([COL_COUNTRY, COL_TIME])

    ENDO = [c for c in ENDO if df[c].notna().any()]

    countries_to_run = list(COUNTRIES)
    n_c = len(countries_to_run)
    mY = len(ENDO)

    # ===== 6 columns =====
    mats = {j: np.full((n_c, 6), np.nan) for j in range(mY)}

    for ri, country in enumerate(countries_to_run):

        g = df[df[COL_COUNTRY] == country]
        if g.empty:
            continue

        g = g.sort_values(COL_TIME)

        mask = (
            np.isfinite(g[ENDO].to_numpy(float)).all(axis=1)
            & np.isfinite(g[EXO].to_numpy(float)).all(axis=1)
        )
        g = g.loc[mask].copy().reset_index(drop=True)

        if len(g) < min_T:
            continue

        Yd = g[ENDO].to_numpy(float)
        # Standardize by column
        mu = np.nanmean(Yd, axis=0)
        sigma = np.nanstd(Yd, axis=0) + 1e-8
        Yd = (Yd - mu) / sigma

        Xd_full = g[EXO].to_numpy(float)
        mu_x = np.nanmean(Xd_full, axis=0)
        sigma_x = np.nanstd(Xd_full, axis=0) + 1e-8
        Xd_full = (Xd_full - mu_x) / sigma_x

        T = len(g)

        # ===== no exo (EM) =====
        try:
            Z0 = np.zeros((T, 0))
            r0 = run_kf_em(
                Y=Yd,
                Z=Z0,
                lags=lags,
                window=40,
                max_em_iter=8,
                tol=1e-4,
                em_damping=0.7,
                verbose=False,
            )
            Yh0 = r0["Y_pred"]
        except Exception as e:
            print(f"[WARN] {country} no-exo EM failed: {e}")
            Yh0 = None

        # ===== all exo (EM) =====
        try:
            r_all = run_kf_em(
                Y=Yd,
                Z=Xd_full,
                lags=lags,
                window=40,
                max_em_iter=8,
                tol=1e-4,
                em_damping=0.7,
                verbose=False,
            )
            Yh_all = r_all["Y_pred"]
        except Exception as e:
            print(f"[WARN] {country} +exo EM failed: {e}")
            Yh_all = None

        # ===== null model =====
        Yh_null = np.full_like(Yd, np.nan)
        Yh_null[1:, :] = Yd[:-1, :]

        # ===== metrics =====
        for j in range(mY):

            # no exo
            c0, e0 = (_metrics_one_country(Yd, Yh0, j)
                      if Yh0 is not None else (np.nan, np.nan))

            # +exo
            c1, e1 = (_metrics_one_country(Yd, Yh_all, j)
                      if Yh_all is not None else (np.nan, np.nan))

            # null
            c2, e2 = _metrics_one_country(Yd, Yh_null, j)

            mats[j][ri, :] = [c0, c1, c2, e0, e1, e2]

    # =========================
    #           PLOT
    # =========================

    col_names = [
        "corr\n(no exo)",
        "corr\n(+exo)",
        "corr\n(null)",
        "RMSE\n(no exo)",
        "RMSE\n(+exo)",
        "RMSE\n(null)",
    ]

    cpf = max(1, int(countries_per_figure))
    n_batch = int(np.ceil(n_c / cpf)) if n_c else 1

    for j, name in enumerate(ENDO):

        M = mats[j]

        rmse_block = M[:, 3:6]
        rmse_vmin = 0
        rmse_vmax = 2

        # Guard against all-NaN limits
        if not np.isfinite(rmse_vmin):
            rmse_vmin = 0.0
        if not np.isfinite(rmse_vmax) or rmse_vmax == rmse_vmin:
            rmse_vmax = rmse_vmin + 1.0

        for bi in range(n_batch):
            a = bi * cpf
            b = min(a + cpf, n_c)
            if a >= b:
                continue

            Mb = M[a:b, :]
            rows = Mb.shape[0]
            ylabels_iso = countries_to_run[a:b]
            ylabels = [country_display_name(c, iso3_to_country) for c in ylabels_iso]

            fig = plt.figure(figsize=(16, max(5, rows * 0.3)))
            gs = gridspec.GridSpec(1, 6, wspace=0.4)

            axes = [fig.add_subplot(gs[0, i]) for i in range(6)]

            # ===== corr =====
            for i in range(3):
                im = axes[i].imshow(
                    Mb[:, i:i+1],
                    aspect="auto",
                    vmin=-1,
                    vmax=1,
                    cmap="RdBu_r"
                )
                axes[i].set_xticks([0])
                axes[i].set_xticklabels([col_names[i]], fontsize=8)
                axes[i].set_yticks(range(rows))
                axes[i].set_yticklabels(ylabels if i == 0 else [], fontsize=7)

                for r in range(rows):
                    val = Mb[r, i]
                    if np.isfinite(val):
                        axes[i].text(
                            0, r,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=6
                        )

                plt.colorbar(im, ax=axes[i], fraction=0.046)

            # ===== rmse =====
            for i in range(3, 6):

                im = axes[i].imshow(
                    Mb[:, i:i+1],
                    aspect="auto",
                    vmin=rmse_vmin,   
                    vmax=rmse_vmax,   
                    cmap="viridis"
                )

                axes[i].set_xticks([0])
                axes[i].set_xticklabels([col_names[i]], fontsize=8)
                axes[i].set_yticks(range(rows))
                axes[i].set_yticklabels([])

                # Numeric annotation
                for r in range(rows):
                    val = Mb[r, i]
                    if np.isfinite(val):
                        axes[i].text(
                            0, r,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=6
                        )

                plt.colorbar(im, ax=axes[i], fraction=0.046)

            fig.suptitle(
                f"{name} — Model Comparison (No exo / +exo / Null)",
                fontsize=11,
            )

            plt.tight_layout()
            plt.show(block=False)

    return mats


mats = panel_y_yhat_heatmap(
    PATH,
    countries_to_run,
    COL_COUNTRY=COL_COUNTRY,
    COL_TIME=COL_TIME,
    ENDO=ENDO,
    EXO=EXO,
    lags=lags,
    countries_per_figure=20,
    iso3_to_country=ISO3_TO_COUNTRY,
)


def plot_coeff_trajectories_countries_em(
    PATH,
    COUNTRIES,
    COL_COUNTRY="country",
    COL_TIME="quarter",
    ENDO=None,
    EXO=None,
    lags=1,
    min_T=None,
    max_em_iter=8,
    iso3_to_country: dict | None = None,
    pdf: PdfPages | None = None,
):
    """If pdf=PdfPages(...), write each figure to that file; otherwise call plt.show()."""

    if ENDO is None:
        ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
    if EXO is None:
        EXO = ["COMMODITY_YoY", "ENSO"]
    if min_T is None:
        min_T = lags + 5

    df = pd.read_csv(PATH)
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df = df.dropna(subset=[COL_TIME]).sort_values([COL_COUNTRY, COL_TIME])

    ENDO_use = [c for c in ENDO if c in df.columns and df[c].notna().any()]
    EXO_use = [c for c in EXO if c in df.columns and df[c].notna().any()]


    for country in COUNTRIES:
        g = df[df[COL_COUNTRY] == country]
        cname = country_display_name(country, iso3_to_country)
        if g.empty:
            print(f"[SKIP] {cname}: no rows")
            continue

        g = g.sort_values(COL_TIME)
        mask = (
            np.isfinite(g[ENDO_use].to_numpy(float)).all(axis=1)
            & np.isfinite(g[EXO_use].to_numpy(float)).all(axis=1)
        )
        g = g.loc[mask].copy().reset_index(drop=True)
        if len(g) < min_T:
            print(f"[SKIP] {cname}: T={len(g)} too short")
            continue

        Yd = g[ENDO_use].to_numpy(float)
        mu_y = np.nanmean(Yd, axis=0)
        sd_y = np.nanstd(Yd, axis=0) + 1e-8
        Yd = (Yd - mu_y) / sd_y

        Xd = g[EXO_use].to_numpy(float)
        mu_x = np.nanmean(Xd, axis=0)
        sd_x = np.nanstd(Xd, axis=0) + 1e-8
        Xd = (Xd - mu_x) / sd_x

        try:
            res = run_kf_em(
                Y=Yd,
                Z=Xd,
                lags=lags,
                window=40,
                max_em_iter=max_em_iter,
                tol=1e-4,
                em_damping=0.7,
                verbose=False,
            )
        except Exception as e:
            print(f"[SKIP] {cname}: run_kf_em failed: {e}")
            continue

        theta_est = res["theta_est"]  # (T, p)
        Tn, p = theta_est.shape
        mY = len(ENDO_use)
        mX = len(EXO_use)
        m = lags * mY + mX

        valid = ~np.isnan(theta_est).any(axis=1)
        if valid.sum() <= 2:
            print(f"[SKIP] {cname}: no valid theta trajectory")
            continue

        quarters = pd.to_datetime(g[COL_TIME].values)

        # ----- Plot 1: diagonal own-lag coefficients only (GDP<-L*.GDP, CPI<-L*.CPI, ...) -----
        diag_k, diag_lbl = [], []
        for eq in range(mY):
            for li in range(1, lags + 1):
                off = (li - 1) * mY + eq
                k = eq * m + off
                if k < p:
                    diag_k.append(k)
                    diag_lbl.append(f"{ENDO_use[eq]}<-L{li}.{ENDO_use[eq]}")

        plt.figure(figsize=(12, 5))
        for k, lbl in zip(diag_k, diag_lbl):
            plt.plot(
                quarters[valid],
                theta_est[valid, k],
                linewidth=1.3,
                marker="o",
                markersize=2,
                alpha=0.9,
                label=lbl,
            )
        plt.title(f"{cname} — Diagonal own-variable lag coefficients")
        plt.xlabel("Quarter")
        plt.ylabel("coefficient")
        plt.legend(frameon=False, ncol=2, fontsize=9)
        ax = plt.gca()
        style_quarter_axis_every_four_quarters(ax)
        ax.grid(True, which="major", axis="y", alpha=0.25)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        _fig_save_or_show(plt.gcf(), pdf)

        # ----- Plot 2: ENSO-related coefficients only (one line per endogenous equation) -----
        if "ENSO" not in EXO_use:
            print(f"[INFO] {cname}: no ENSO column in EXO_use, skip ENSO-only plot")
        else:
            idx_enso = EXO_use.index("ENSO")
            enso_k, enso_lbl = [], []
            for eq in range(mY):
                off = lags * mY + idx_enso
                k = eq * m + off
                if k < p:
                    enso_k.append(k)
                    enso_lbl.append(f"{ENDO_use[eq]}<-ENSO")

            plt.figure(figsize=(12, 5))
            for k, lbl in zip(enso_k, enso_lbl):
                plt.plot(
                    quarters[valid],
                    theta_est[valid, k],
                    linewidth=1.3,
                    linestyle="-",
                    marker="o",
                    markersize=2,
                    alpha=0.9,
                    label=lbl,
                )
            plt.title(f"{cname} — ENSO coefficients (one per equation)")
            plt.xlabel("Quarter")
            plt.ylabel("coefficient")
            plt.legend(frameon=False, ncol=2, fontsize=9)
            ax = plt.gca()
            style_quarter_axis_every_four_quarters(ax)
            ax.grid(True, which="major", axis="y", alpha=0.25)
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            _fig_save_or_show(plt.gcf(), pdf)


def plot_em_diagnostics_countries(
    PATH,
    COUNTRIES,
    COL_COUNTRY="country",
    COL_TIME="quarter",
    ENDO=None,
    EXO=None,
    lags=1,
    min_T=None,
    max_em_iter=8,
    iso3_to_country: dict | None = None,
    pdf: PdfPages | None = None,
):
    """
    For each country, plot three diagnostics after training:
    - innovation score: v_t' S_t^{-1} v_t
    - coefficient change: ||beta_smooth,t - beta_smooth,t-1||
    - filter–smoother gap: ||beta_smooth,t - beta_filt,t||
    Data come from run_kf_em outputs:
    innovation_score / coefficient_change / filter_smoother_gap.
    If pdf=PdfPages(...), write each figure to PDF instead of showing.
    """
    if ENDO is None:
        ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
    if EXO is None:
        EXO = ["COMMODITY_YoY", "ENSO"]
    if min_T is None:
        min_T = lags + 5

    df = pd.read_csv(PATH)
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df = df.dropna(subset=[COL_TIME]).sort_values([COL_COUNTRY, COL_TIME])

    ENDO_use = [c for c in ENDO if c in df.columns and df[c].notna().any()]
    EXO_use = [c for c in EXO if c in df.columns and df[c].notna().any()]

    for country in COUNTRIES:
        g = df[df[COL_COUNTRY] == country]
        cname = country_display_name(country, iso3_to_country)
        if g.empty:
            print(f"[SKIP] {cname}: no rows")
            continue

        g = g.sort_values(COL_TIME)
        mask = (
            np.isfinite(g[ENDO_use].to_numpy(float)).all(axis=1)
            & np.isfinite(g[EXO_use].to_numpy(float)).all(axis=1)
        )
        g = g.loc[mask].copy().reset_index(drop=True)
        if len(g) < min_T:
            print(f"[SKIP] {cname}: T={len(g)} too short")
            continue

        Yd = g[ENDO_use].to_numpy(float)
        mu_y = np.nanmean(Yd, axis=0)
        sd_y = np.nanstd(Yd, axis=0) + 1e-8
        Yd = (Yd - mu_y) / sd_y

        Xd = g[EXO_use].to_numpy(float)
        mu_x = np.nanmean(Xd, axis=0)
        sd_x = np.nanstd(Xd, axis=0) + 1e-8
        Xd = (Xd - mu_x) / sd_x

        try:
            res = run_kf_em(
                Y=Yd,
                Z=Xd,
                lags=lags,
                window=40,
                max_em_iter=max_em_iter,
                tol=1e-4,
                em_damping=0.7,
                verbose=False,
            )
        except Exception as e:
            print(f"[SKIP] {cname}: run_kf_em failed: {e}")
            continue

        innov = res.get("innovation_score")
        dcoef = res.get("coefficient_change")
        fsgap = res.get("filter_smoother_gap")
        if innov is None or len(innov) != len(Yd):
            print(f"[SKIP] {cname}: missing diagnostics")
            continue

        quarters = pd.to_datetime(g[COL_TIME].values)
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(quarters, innov, color="C0", linewidth=1.2)
        axes[0].set_ylabel("Innovation score (v' S^{-1} v)")
        axes[0].grid(True, which="major", axis="y", alpha=0.25)
        axes[0].set_title(f"{cname} — EM diagnostics")

        axes[1].plot(quarters, dcoef, color="C1", linewidth=1.2)
        axes[1].set_ylabel("Coefficient change ||b_s,t - b_s,t-1||")
        axes[1].grid(True, which="major", axis="y", alpha=0.25)

        axes[2].plot(quarters, fsgap, color="C2", linewidth=1.2)
        axes[2].set_ylabel("Filter–smoother gap ||b_s,t - b_f,t||")
        axes[2].set_xlabel("Quarter")
        axes[2].grid(True, which="major", axis="y", alpha=0.25)

        style_quarter_axis_every_four_quarters(axes[2])
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        _fig_save_or_show(fig, pdf)


def plot_em_qr_traces_countries(
    PATH,
    COUNTRIES,
    COL_COUNTRY="country",
    COL_TIME="quarter",
    ENDO=None,
    EXO=None,
    lags=1,
    min_T=None,
    max_em_iter=8,
    iso3_to_country: dict | None = None,
    pdf: PdfPages | None = None,
):
    """
    Plot EM iteration traces by country:
    - trace(Q)
    - trace(R)
    If pdf=PdfPages(...), write each figure to PDF instead of showing.
    """
    if ENDO is None:
        ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
    if EXO is None:
        EXO = ["COMMODITY_YoY", "ENSO"]
    if min_T is None:
        min_T = lags + 5

    df = pd.read_csv(PATH)
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df = df.dropna(subset=[COL_TIME]).sort_values([COL_COUNTRY, COL_TIME])

    ENDO_use = [c for c in ENDO if c in df.columns and df[c].notna().any()]
    EXO_use = [c for c in EXO if c in df.columns and df[c].notna().any()]

    for country in COUNTRIES:
        g = df[df[COL_COUNTRY] == country]
        cname = country_display_name(country, iso3_to_country)
        if g.empty:
            print(f"[SKIP] {cname}: no rows")
            continue

        g = g.sort_values(COL_TIME)
        mask = (
            np.isfinite(g[ENDO_use].to_numpy(float)).all(axis=1)
            & np.isfinite(g[EXO_use].to_numpy(float)).all(axis=1)
        )
        g = g.loc[mask].copy().reset_index(drop=True)
        if len(g) < min_T:
            print(f"[SKIP] {cname}: T={len(g)} too short")
            continue

        Yd = g[ENDO_use].to_numpy(float)
        mu_y = np.nanmean(Yd, axis=0)
        sd_y = np.nanstd(Yd, axis=0) + 1e-8
        Yd = (Yd - mu_y) / sd_y

        Xd = g[EXO_use].to_numpy(float)
        mu_x = np.nanmean(Xd, axis=0)
        sd_x = np.nanstd(Xd, axis=0) + 1e-8
        Xd = (Xd - mu_x) / sd_x

        try:
            res = run_kf_em(
                Y=Yd,
                Z=Xd,
                lags=lags,
                window=40,
                max_em_iter=max_em_iter,
                tol=1e-4,
                em_damping=0.7,
                verbose=False,
            )
        except Exception as e:
            print(f"[SKIP] {cname}: run_kf_em failed: {e}")
            continue

        hist = res.get("em_history", {})
        q_tr = np.asarray(hist.get("trace_Q", []), dtype=float)
        r_tr = np.asarray(hist.get("trace_R", []), dtype=float)
        it = np.arange(1, len(q_tr) + 1)

        if len(it) == 0:
            print(f"[SKIP] {cname}: empty EM history")
            continue

        plt.figure(figsize=(9, 4.5))
        plt.plot(it, q_tr, "-o", linewidth=1.4, markersize=3, label="trace(Q)")
        plt.plot(it, r_tr, "-s", linewidth=1.4, markersize=3, label="trace(R)")
        plt.title(f"{cname} - EM iteration traces")
        plt.xlabel("EM iteration")
        plt.ylabel("trace value")
        plt.grid(alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        _fig_save_or_show(plt.gcf(), pdf)


# Country trajectory plots: full countries_to_run list.
# Titles/y-axis labels use full names from ISO3_TO_COUNTRY.
# If non-None, all three plot groups are written into one PDF.
PLOTS_PDF_PATH: str | None = "GVAR_LLM_EM_plots.pdf"
COUNTRIES_FOR_COEF_PLOT = list(countries_to_run)
COUNTRIES_FOR_DIAG_PLOT = list(countries_to_run)
COUNTRIES_FOR_QR_PLOT = list(countries_to_run)
_plot_kw = dict(
    PATH=PATH,
    COL_COUNTRY=COL_COUNTRY,
    COL_TIME=COL_TIME,
    ENDO=ENDO,
    EXO=EXO,
    lags=lags,
    max_em_iter=8,
    iso3_to_country=ISO3_TO_COUNTRY,
)
if PLOTS_PDF_PATH:
    with PdfPages(PLOTS_PDF_PATH) as _plots_pdf:
        plot_coeff_trajectories_countries_em(
            COUNTRIES=COUNTRIES_FOR_COEF_PLOT, pdf=_plots_pdf, **_plot_kw
        )
        plot_em_diagnostics_countries(
            COUNTRIES=COUNTRIES_FOR_DIAG_PLOT, pdf=_plots_pdf, **_plot_kw
        )
        plot_em_qr_traces_countries(
            COUNTRIES=COUNTRIES_FOR_QR_PLOT, pdf=_plots_pdf, **_plot_kw
        )
    print(f"[PDF] Saved: {PLOTS_PDF_PATH}")
else:
    plot_coeff_trajectories_countries_em(
        COUNTRIES=COUNTRIES_FOR_COEF_PLOT, pdf=None, **_plot_kw
    )
    plot_em_diagnostics_countries(
        COUNTRIES=COUNTRIES_FOR_DIAG_PLOT, pdf=None, **_plot_kw
    )
    plot_em_qr_traces_countries(
        COUNTRIES=COUNTRIES_FOR_QR_PLOT, pdf=None, **_plot_kw
    )


def _map_llm_country_to_iso3(llm_df: pd.DataFrame, iso3_to_country: dict) -> pd.DataFrame:
    """Map country names in LLM table (full name / ISO3 mixed) into ISO3 codes."""
    out = llm_df.copy()
    rev = {str(v).strip().lower(): k for k, v in iso3_to_country.items()}

    def _one(v):
        s = str(v).strip()
        su = s.upper()
        if su in iso3_to_country:
            return su
        return rev.get(s.lower(), s)

    out["country"] = out["country"].apply(_one)
    return out


def build_em_break_score_panel(
    PATH,
    COUNTRIES,
    COL_COUNTRY="country",
    COL_TIME="quarter",
    ENDO=None,
    EXO=None,
    lags=1,
    min_T=None,
    max_em_iter=8,
) -> pd.DataFrame:
    """
    Build a quarterly break-score panel from run_kf_em innovation_score.
    Output columns: country, quarter, year, score
    """
    if ENDO is None:
        ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
    if EXO is None:
        EXO = ["COMMODITY_YoY", "ENSO"]
    if min_T is None:
        min_T = lags + 5

    df = pd.read_csv(PATH)
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df = df.dropna(subset=[COL_TIME]).sort_values([COL_COUNTRY, COL_TIME])

    ENDO_use = [c for c in ENDO if c in df.columns and df[c].notna().any()]
    EXO_use = [c for c in EXO if c in df.columns and df[c].notna().any()]
    rows = []

    for country in COUNTRIES:
        g = df[df[COL_COUNTRY] == country].sort_values(COL_TIME)
        if g.empty:
            continue

        mask = (
            np.isfinite(g[ENDO_use].to_numpy(float)).all(axis=1)
            & np.isfinite(g[EXO_use].to_numpy(float)).all(axis=1)
        )
        g = g.loc[mask].copy().reset_index(drop=True)
        if len(g) < min_T:
            continue

        Yd = g[ENDO_use].to_numpy(float)
        mu_y = np.nanmean(Yd, axis=0)
        sd_y = np.nanstd(Yd, axis=0) + 1e-8
        Yd = (Yd - mu_y) / sd_y

        Xd = g[EXO_use].to_numpy(float)
        mu_x = np.nanmean(Xd, axis=0)
        sd_x = np.nanstd(Xd, axis=0) + 1e-8
        Xd = (Xd - mu_x) / sd_x

        try:
            res = run_kf_em(
                Y=Yd,
                Z=Xd,
                lags=lags,
                window=40,
                max_em_iter=max_em_iter,
                tol=1e-4,
                em_damping=0.7,
                verbose=False,
            )
        except Exception as e:
            print(f"[SKIP] {country}: build_em_break_score_panel failed: {e}")
            continue

        innov = np.asarray(res.get("innovation_score"), dtype=float)
        if innov.shape[0] != len(g):
            continue

        q = pd.to_datetime(g[COL_TIME].values)
        for qt, sc in zip(q, innov):
            rows.append(
                {
                    "country": country,
                    "quarter": qt,
                    "year": int(pd.Timestamp(qt).year),
                    "score": float(sc) if np.isfinite(sc) else np.nan,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# LLM integration: break-score overlay + summary charts + time-slice maps
# ============================================================
ENABLE_LLM_INTEGRATION = True
LLM_RESULTS_CSV = "merged_clean_dataset.csv"  # replace with your prepared CSV
SHOW_LLM_BREAK_OVERLAY = True
LLM_OVERLAY_FIG_PATH = "gvar_breakscore_llm_overlay.png"
LLM_STATS_FIG_PATH = "gvar_llm_stats.png"
LLM_MAP_OUTPUT_DIR = "gvar_llm_time_slice_maps"

if ENABLE_LLM_INTEGRATION:
    if not _HAS_LLM_VIS:
        print("[WARN] skip LLM integration: llm_break_visualization not available")
    elif not Path(LLM_RESULTS_CSV).exists():
        print(f"[WARN] skip LLM integration: file not found -> {LLM_RESULTS_CSV}")
    else:
        llm_raw_df = pd.read_csv(LLM_RESULTS_CSV)
        llm_df = _map_llm_country_to_iso3(llm_raw_df, ISO3_TO_COUNTRY)
        llm_df = llm_df[llm_df["country"].isin(list(countries_to_run))].copy()
        print(f"[INFO] LLM matched rows: {len(llm_df)}")

        break_score_df = build_em_break_score_panel(
            PATH=PATH,
            COUNTRIES=list(countries_to_run),
            COL_COUNTRY=COL_COUNTRY,
            COL_TIME=COL_TIME,
            ENDO=ENDO,
            EXO=EXO,
            lags=lags,
            max_em_iter=8,
        )
        break_score_df = break_score_df.dropna(subset=["score"]).copy()
        print(f"[INFO] break_score rows: {len(break_score_df)}")

        # Task 1: overlay plot (toggle + shading + Top5)
        fig_overlay = plot_structural_break_score_with_llm_overlay(
            score_df=break_score_df,
            llm_df=llm_df,
            score_col="score",
            time_col="quarter",
            country_col="country",
            show_llm_overlay=SHOW_LLM_BREAK_OVERLAY,
            top_k=5,
            ncols=3,
        )
        fig_overlay.savefig(LLM_OVERLAY_FIG_PATH, dpi=150, bbox_inches="tight")
        plt.close(fig_overlay)
        print(f"[SAVE] {LLM_OVERLAY_FIG_PATH}")

        # Task 2: two summary charts (backend-safe saving)
        fig_ratio = plot_break_supported_ratio(llm_df)
        fig_type = plot_break_type_distribution(llm_df)
        stats_path = Path(LLM_STATS_FIG_PATH)
        if stats_path.suffix.lower() == ".pdf":
            with PdfPages(str(stats_path)) as stats_pdf:
                stats_pdf.savefig(fig_ratio, dpi=150, bbox_inches="tight")
                stats_pdf.savefig(fig_type, dpi=150, bbox_inches="tight")
            print(f"[SAVE] {LLM_STATS_FIG_PATH} (2 pages)")
        else:
            ratio_path = stats_path.with_name(f"{stats_path.stem}_ratio{stats_path.suffix or '.png'}")
            type_path = stats_path.with_name(f"{stats_path.stem}_type{stats_path.suffix or '.png'}")
            fig_ratio.savefig(str(ratio_path), dpi=150, bbox_inches="tight")
            fig_type.savefig(str(type_path), dpi=150, bbox_inches="tight")
            print(f"[SAVE] {ratio_path}")
            print(f"[SAVE] {type_path}")
        plt.close(fig_ratio)
        plt.close(fig_type)

        # Task 3: yearly maps (1998-2015, marker size = annual mean break score)
        score_year_df = (
            break_score_df.groupby(["country", "year"], as_index=False)["score"]
            .mean()
            .rename(columns={"score": "score"})
        )
        map_files = build_time_slice_maps(
            score_df=score_year_df,
            llm_df=llm_df,
            score_col="score",
            country_col="country",
            year_col="year",
            start_year=1998,
            end_year=2015,
            output_dir=LLM_MAP_OUTPUT_DIR,
        )
        print(f"[SAVE] maps generated: {len(map_files)} files in {LLM_MAP_OUTPUT_DIR}")


def plot_breakscore_countries_em(
    PATH,
    COUNTRIES,
    COL_COUNTRY="country",
    COL_TIME="quarter",
    ENDO=None,
    EXO=None,
    lags=1,
    max_em_iter=8,
    drop_quarters_dict: dict[str, list[pd.Timestamp]] | None = None,
    iso3_to_country: dict | None = None,
    pdf: PdfPages | None = None,
    save_dir: str | None = None,
):
    """Plot per-country break score trajectories from EM innovation score.
    If drop_quarters_dict is provided, those quarters are removed before plotting.
    """
    bs = build_em_break_score_panel(
        PATH=PATH,
        COUNTRIES=COUNTRIES,
        COL_COUNTRY=COL_COUNTRY,
        COL_TIME=COL_TIME,
        ENDO=ENDO,
        EXO=EXO,
        lags=lags,
        max_em_iter=max_em_iter,
    ).dropna(subset=["score"])
    if bs.empty:
        print("[WARN] break score panel empty; skip breakscore plot")
        return

    for country in COUNTRIES:
        g = bs[bs["country"] == country].copy()
        if drop_quarters_dict and country in drop_quarters_dict:
            drop_q = set(pd.to_datetime(drop_quarters_dict[country]))
            g = g[~pd.to_datetime(g["quarter"]).isin(drop_q)].copy()
        if g.empty:
            continue
        cname = country_display_name(country, iso3_to_country)
        quarters = pd.to_datetime(g["quarter"].values)
        vals = g["score"].to_numpy(float)
        plt.figure(figsize=(12, 4.8))
        plt.plot(quarters, vals, color="black", linewidth=1.2)
        plt.title(f"{cname} — Break score (innovation)")
        plt.xlabel("Quarter")
        plt.ylabel("score")
        ax = plt.gca()
        style_quarter_axis_every_four_quarters(ax)
        ax.grid(True, which="major", axis="y", alpha=0.25)
        plt.tight_layout()
        fig = plt.gcf()
        if save_dir:
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"breakscore_{country}.jpg", dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            _fig_save_or_show(fig, pdf)


def build_drop_quarters_dict_from_llm_points(
    base_path: str,
    llm_csv: str,
    countries: list[str],
    near_break_years: int = 2,
    top_score_k: int = 5,
) -> tuple[dict[str, list[pd.Timestamp]], pd.DataFrame]:
    """Build country->quarters-to-drop dictionary from top-k score + LLM break-year rule."""
    if not Path(llm_csv).exists():
        print(f"[WARN] skip refit filter: file not found -> {llm_csv}")
        return {}, pd.DataFrame(columns=["country", "quarter", "year", "score"])

    llm_raw = pd.read_csv(llm_csv)
    llm = _map_llm_country_to_iso3(llm_raw, ISO3_TO_COUNTRY)
    llm = llm[llm["country"].isin(countries)].copy()
    if "break_supported" not in llm.columns or "break_year" not in llm.columns:
        print("[WARN] skip refit filter: missing break_supported/break_year")
        return {}, pd.DataFrame(columns=["country", "quarter", "year", "score"])

    llm["break_year"] = pd.to_numeric(llm["break_year"], errors="coerce")
    llm = llm.dropna(subset=["break_year"]).copy()
    llm["break_year"] = llm["break_year"].astype(int)
    llm["break_supported_bin"] = (
        llm["break_supported"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"1": 1, "true": 1, "yes": 1, "0": 0, "false": 0, "no": 0})
        .fillna(0)
        .astype(int)
    )

    score_df = build_em_break_score_panel(
        PATH=base_path,
        COUNTRIES=countries,
        COL_COUNTRY=COL_COUNTRY,
        COL_TIME=COL_TIME,
        ENDO=ENDO,
        EXO=EXO,
        lags=lags,
        max_em_iter=8,
    ).dropna(subset=["score"])
    if score_df.empty:
        print("[WARN] skip refit filter: break score panel empty")
        return {}, pd.DataFrame(columns=["country", "quarter", "year", "score"])

    topk = (
        score_df.groupby("country", group_keys=False)
        .apply(lambda g: g.nlargest(top_score_k, "score"))
        .reset_index(drop=True)
    )
    supported_years = (
        llm[llm["break_supported_bin"] == 1]
        .groupby("country")["break_year"]
        .apply(list)
        .to_dict()
    )

    drop_rows = []
    for r in topk.itertuples(index=False):
        years = supported_years.get(r.country, [])
        if any(abs(int(r.year) - int(y)) <= near_break_years for y in years):
            drop_rows.append(
                {
                    "country": r.country,
                    "quarter": pd.Timestamp(r.quarter),
                    "year": int(r.year),
                    "score": float(r.score),
                }
            )
    if not drop_rows:
        print("[INFO] no points matched condition; keep original panel")
        return {}, pd.DataFrame(columns=["country", "quarter", "year", "score"])

    drop_df = pd.DataFrame(drop_rows).drop_duplicates(subset=["country", "quarter"]).copy()
    drop_df["quarter"] = pd.to_datetime(drop_df["quarter"])

    # Aggressive expansion: drop the whole year (all 4 quarters) for matched points.
    expanded_rows = []
    for r in drop_df.itertuples(index=False):
        y = int(r.year)
        for qn in (1, 2, 3, 4):
            q = pd.Period(f"{y}Q{qn}", freq="Q").to_timestamp("Q")
            expanded_rows.append(
                {
                    "country": r.country,
                    "quarter": q,
                    "year": y,
                    "score": float(r.score),
                }
            )
    drop_df = pd.DataFrame(expanded_rows).drop_duplicates(subset=["country", "quarter"]).copy()

    drop_quarters_dict = (
        drop_df.groupby("country")["quarter"]
        .apply(lambda s: sorted(set(pd.to_datetime(s).tolist())))
        .to_dict()
    )
    return drop_quarters_dict, drop_df


def build_filtered_panel_from_llm_points(
    base_path: str,
    out_filtered_csv: str,
    drop_quarters_dict: dict[str, list[pd.Timestamp]] | None = None,
    exclude_years: list[int] | None = None,
) -> str:
    """Drop rows by precomputed country->quarter dict and return filtered panel path.
    Note: actual filtering is done by (country, year) to avoid timestamp alignment issues.
    """
    if not drop_quarters_dict:
        return base_path

    src = pd.read_csv(base_path)
    src[COL_TIME] = pd.to_datetime(src[COL_TIME], errors="coerce")
    src["_drop_year"] = src[COL_TIME].dt.year
    key_rows = []
    for c, q_list in drop_quarters_dict.items():
        for q in q_list:
            key_rows.append({COL_COUNTRY: c, "_drop_year": int(pd.Timestamp(q).year)})
    key = pd.DataFrame(key_rows).drop_duplicates()
    if key.empty:
        return base_path

    out = src.merge(key.assign(_drop=1), on=[COL_COUNTRY, "_drop_year"], how="left")
    out = out[out["_drop"].isna()].copy()
    if exclude_years:
        ex_years = set(int(y) for y in exclude_years)
        out = out[~out["_drop_year"].isin(ex_years)].copy()
    out = out.drop(columns=["_drop", "_drop_year"])
    out.to_csv(out_filtered_csv, index=False)
    print(f"[SAVE] {out_filtered_csv} (dropped={len(src)-len(out)})")
    return out_filtered_csv


# Extra pass only: drop points then rerun KF/EM and export coeff + breakscore PDF.
ENABLE_LLM_REFIT_PDF = False
PLOTS_REFIT_PDF_PATH: str | None = "GVAR_LLM_EM_plots_refit.pdf"
DROP_POINTS_LIST_CSV = "llm_top5_near_break_points_to_drop.csv"
REFIT_PANEL_CSV = "gvar_panel_refit_filtered.csv"
TOP_SCORE_K = 5
NEAR_BREAK_YEARS = 2
REFIT_EXCLUDE_YEARS = [2020, 2021, 2022,2023,2024,2025,2026]
BREAKSCORE_OUTPUT_MODE = "pdf"   # "pdf" | "plot" | "jpg"
BREAKSCORE_JPG_DIR = "breakscore_jpg"

if ENABLE_LLM_REFIT_PDF:
    drop_quarters_dict, drop_df = build_drop_quarters_dict_from_llm_points(
        base_path=PATH,
        llm_csv=LLM_RESULTS_CSV,
        countries=list(countries_to_run),
        near_break_years=NEAR_BREAK_YEARS,
        top_score_k=TOP_SCORE_K,
    )
    if not drop_df.empty:
        drop_df.to_csv(DROP_POINTS_LIST_CSV, index=False)
        print(f"[SAVE] {DROP_POINTS_LIST_CSV} (n={len(drop_df)})")
    refit_path = build_filtered_panel_from_llm_points(
        base_path=PATH,
        out_filtered_csv=REFIT_PANEL_CSV,
        drop_quarters_dict=drop_quarters_dict,
        exclude_years=REFIT_EXCLUDE_YEARS,
    )
    refit_kw = dict(_plot_kw)
    refit_kw["PATH"] = refit_path
    if BREAKSCORE_OUTPUT_MODE == "pdf":
        if not PLOTS_REFIT_PDF_PATH:
            raise ValueError("PLOTS_REFIT_PDF_PATH is required when BREAKSCORE_OUTPUT_MODE='pdf'")
        with PdfPages(PLOTS_REFIT_PDF_PATH) as _pdf_refit:
            plot_coeff_trajectories_countries_em(
                COUNTRIES=COUNTRIES_FOR_COEF_PLOT, pdf=_pdf_refit, **refit_kw
            )
            plot_breakscore_countries_em(
                PATH=refit_path,
                COUNTRIES=COUNTRIES_FOR_COEF_PLOT,
                COL_COUNTRY=COL_COUNTRY,
                COL_TIME=COL_TIME,
                ENDO=ENDO,
                EXO=EXO,
                lags=lags,
                max_em_iter=8,
                drop_quarters_dict=drop_quarters_dict,
                iso3_to_country=ISO3_TO_COUNTRY,
                pdf=_pdf_refit,
                save_dir=None,
            )
        print(f"[PDF] Saved: {PLOTS_REFIT_PDF_PATH}")
    elif BREAKSCORE_OUTPUT_MODE == "plot":
        plot_coeff_trajectories_countries_em(
            COUNTRIES=COUNTRIES_FOR_COEF_PLOT, pdf=None, **refit_kw
        )
        plot_breakscore_countries_em(
            PATH=refit_path,
            COUNTRIES=COUNTRIES_FOR_COEF_PLOT,
            COL_COUNTRY=COL_COUNTRY,
            COL_TIME=COL_TIME,
            ENDO=ENDO,
            EXO=EXO,
            lags=lags,
            max_em_iter=8,
            drop_quarters_dict=drop_quarters_dict,
            iso3_to_country=ISO3_TO_COUNTRY,
            pdf=None,
            save_dir=None,
        )
    elif BREAKSCORE_OUTPUT_MODE == "jpg":
        plot_breakscore_countries_em(
            PATH=refit_path,
            COUNTRIES=COUNTRIES_FOR_COEF_PLOT,
            COL_COUNTRY=COL_COUNTRY,
            COL_TIME=COL_TIME,
            ENDO=ENDO,
            EXO=EXO,
            lags=lags,
            max_em_iter=8,
            drop_quarters_dict=drop_quarters_dict,
            iso3_to_country=ISO3_TO_COUNTRY,
            pdf=None,
            save_dir=BREAKSCORE_JPG_DIR,
        )
        print(f"[JPG] Saved breakscore figures to: {BREAKSCORE_JPG_DIR}")
    else:
        raise ValueError("BREAKSCORE_OUTPUT_MODE must be one of: 'pdf', 'plot', 'jpg'")

