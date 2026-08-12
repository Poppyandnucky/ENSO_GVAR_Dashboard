import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


class QScalerNN(nn.Module):
    """Neural scaler for time-varying Q."""

    def __init__(self, input_dim=4, hidden_dim=8, max_log_scale=1.2):
        super().__init__()
        self.max_log_scale = max_log_scale
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        raw = self.net(x)
        log_scale = self.max_log_scale * torch.tanh(raw)
        return torch.exp(log_scale)


class BlockQScalerLinear(nn.Module):
    """Interpretable log-linear scaler for block-wise time-varying Q."""

    def __init__(self, input_dim=4, n_blocks=3, max_log_scale=1.2):
        super().__init__()
        self.max_log_scale = max_log_scale
        self.linear = nn.Linear(input_dim, n_blocks)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        raw = self.linear(x)
        log_scale = self.max_log_scale * torch.tanh(raw)
        return torch.exp(log_scale)

    def raw_contributions(self, x):
        return x.unsqueeze(-1) * self.linear.weight.T.unsqueeze(0)


SIGNAL_NAMES = ["innovation", "coef_change", "filter_smoother_gap", "break_flag"]


def build_q_block_masks(mY, mX, lags=1, exo_names=None):
    """Create disjoint state-index masks for endo-lag and each exogenous coefficient block."""
    if exo_names is None:
        exo_names = [f"exo_{j}" for j in range(mX)]
    block_names = ["endo_lag"] + [str(x) for x in exo_names]
    m = lags * mY + mX
    p = mY * m
    masks = []

    endo_mask = np.zeros(p, dtype=bool)
    for eq in range(mY):
        base = eq * m
        endo_mask[base : base + lags * mY] = True
    masks.append(endo_mask)

    for j in range(mX):
        mask = np.zeros(p, dtype=bool)
        off = lags * mY + j
        for eq in range(mY):
            mask[eq * m + off] = True
        masks.append(mask)
    return block_names, masks


def _scale_Q_by_blocks_torch(Q0, block_scales, block_masks):
    """Apply variance scales with D Q D so the covariance stays PSD."""
    p = Q0.shape[0]
    scale_vec = torch.ones(p, dtype=Q0.dtype, device=Q0.device)
    for b, mask in enumerate(block_masks):
        mask_t = torch.tensor(mask, dtype=torch.bool, device=Q0.device)
        scale_vec = torch.where(mask_t, block_scales[b].expand_as(scale_vec), scale_vec)
    sqrt_scale = torch.sqrt(torch.clamp(scale_vec, min=1e-8))
    return Q0 * sqrt_scale[:, None] * sqrt_scale[None, :]


def varx_rolling_predict(Y, Z, lags=1, window=None, ridge=1e-6):
    n, mY = Y.shape
    mX = Z.shape[1]
    m = lags * mY + mX
    Y_pred = np.full((n, mY), np.nan)
    e_raw = np.full((n, mY), np.nan)

    for t in range(lags + 1, n - 1):
        s0 = lags + 1 if window is None else max(lags + 1, t - 1 - window + 1)
        T = (t - 1) - s0 + 1
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
        try:
            beta_hat = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            beta_hat = np.linalg.pinv(A) @ B

        reg_now = np.concatenate([np.concatenate([Y[t - i, :] for i in range(1, lags + 1)]), Z[t - 1, :]])
        yhat = reg_now @ beta_hat
        Y_pred[t, :] = yhat
        e_raw[t, :] = Y[t, :] - yhat
    return Y_pred, e_raw, None


def init_from_varx_rolling(Y, Z, lags=1, window=40):
    Yhat, _, _ = varx_rolling_predict(Y, Z, lags=lags, window=window)
    n, mY = Y.shape
    mX = Z.shape[1]
    m = lags * mY + mX
    p = mY * m

    theta0 = np.zeros((p, 1))
    P0 = np.eye(p) * 10.0
    R = np.eye(mY) * 0.2
    valid = np.isfinite(Yhat).all(axis=1)
    if valid.sum() > 3:
        E = Y[valid, :] - Yhat[valid, :]
        R = np.cov(E.T) + 1e-6 * np.eye(mY)
    Q = np.eye(p) * 0.02
    return theta0, P0, Q, R


def kalman_multilag_filter(Y, Z, theta0, Q, R, P0, lags=1):
    n, mY = Y.shape
    mX = Z.shape[1]
    m = lags * mY + mX
    p = mY * m

    theta_f = np.zeros((n, p, 1))
    P_f = np.zeros((n, p, p))
    theta_pred = np.zeros((n, p, 1))
    P_pred = np.zeros((n, p, p))
    Y_pred = np.full((n, mY), np.nan)
    theta_prev = theta0.copy()
    P_prev = P0.copy()
    I = np.eye(p)

    for t in range(n):
        theta_t_pred = theta_prev
        P_t_pred = P_prev + Q
        theta_pred[t], P_pred[t] = theta_t_pred, P_t_pred

        if t <= lags:
            theta_f[t], P_f[t] = theta_t_pred, P_t_pred
            theta_prev, P_prev = theta_t_pred, P_t_pred
            continue

        x_t = np.concatenate([np.concatenate([Y[t - i, :] for i in range(1, lags + 1)]), Z[t - 1, :]])
        H_t = np.zeros((mY, p))
        for eq in range(mY):
            H_t[eq, eq * m : (eq + 1) * m] = x_t

        yhat_t = (H_t @ theta_t_pred).ravel()
        Y_pred[t, :] = yhat_t
        v_t = (Y[t, :] - yhat_t).reshape(-1, 1)

        S_t = H_t @ P_t_pred @ H_t.T + R
        S_t = (S_t + S_t.T) / 2 + 1e-9 * np.eye(mY)
        K_t = P_t_pred @ H_t.T @ np.linalg.pinv(S_t)
        theta_t = theta_t_pred + K_t @ v_t
        P_t = (I - K_t @ H_t) @ P_t_pred @ (I - K_t @ H_t).T + K_t @ R @ K_t.T
        P_t = (P_t + P_t.T) / 2

        theta_f[t], P_f[t] = theta_t, P_t
        theta_prev, P_prev = theta_t, P_t

    return theta_f, P_f, Y_pred, theta_pred, P_pred


def kf_e_step_store(Y, Z, theta0, Q, R, P0, lags=1):
    n, mY = Y.shape
    mX = Z.shape[1]
    m = lags * mY + mX
    p = mY * m

    theta_f = np.zeros((n, p, 1))
    P_f = np.zeros((n, p, p))
    theta_pred = np.zeros((n, p, 1))
    P_pred = np.zeros((n, p, p))
    H_list, y_list = [], []
    theta_prev = theta0.copy()
    P_prev = P0.copy()
    I = np.eye(p)

    for t in range(n):
        theta_t_pred = theta_prev
        P_t_pred = P_prev + Q
        theta_pred[t], P_pred[t] = theta_t_pred, P_t_pred

        if t <= lags:
            theta_f[t], P_f[t] = theta_t_pred, P_t_pred
            H_list.append(None)
            y_list.append(None)
            theta_prev, P_prev = theta_t_pred, P_t_pred
            continue

        x_t = np.concatenate([np.concatenate([Y[t - i, :] for i in range(1, lags + 1)]), Z[t - 1, :]])
        H_t = np.zeros((mY, p))
        for eq in range(mY):
            H_t[eq, eq * m : (eq + 1) * m] = x_t
        y_t = Y[t, :].reshape(-1, 1)

        v_t = y_t - H_t @ theta_t_pred
        S_t = H_t @ P_t_pred @ H_t.T + R
        S_t = (S_t + S_t.T) / 2 + 1e-9 * np.eye(mY)
        K_t = P_t_pred @ H_t.T @ np.linalg.pinv(S_t)
        theta_t = theta_t_pred + K_t @ v_t
        P_t = (I - K_t @ H_t) @ P_t_pred @ (I - K_t @ H_t).T + K_t @ R @ K_t.T
        P_t = (P_t + P_t.T) / 2

        theta_f[t], P_f[t] = theta_t, P_t
        H_list.append(H_t)
        y_list.append(y_t)
        theta_prev, P_prev = theta_t, P_t

    return {"theta_f": theta_f, "P_f": P_f, "theta_pred": theta_pred, "P_pred": P_pred, "H_list": H_list, "y_list": y_list}


def rts_smoother(theta_f, P_f, theta_pred, P_pred):
    n, p, _ = theta_f.shape
    theta_s = np.copy(theta_f)
    P_s = np.copy(P_f)
    P_lag = np.zeros((n, p, p))
    I = np.eye(p)
    for t in range(n - 2, -1, -1):
        C_t = P_f[t] @ np.linalg.pinv((P_pred[t + 1] + P_pred[t + 1].T) / 2 + 1e-9 * I)
        theta_s[t] = theta_f[t] + C_t @ (theta_s[t + 1] - theta_pred[t + 1])
        P_s[t] = P_f[t] + C_t @ (P_s[t + 1] - P_pred[t + 1]) @ C_t.T
        P_lag[t + 1] = C_t @ P_s[t + 1]
    return theta_s, P_s, P_lag


def compute_kf_smoother_diagnostics(Y, e_step_store, theta_s, R):
    n, mY = Y.shape
    innovation_score = np.full(n, np.nan)
    coefficient_change = np.full(n, np.nan)
    filter_smoother_gap = np.full(n, np.nan)

    H_list = e_step_store["H_list"]
    theta_pred = e_step_store["theta_pred"]
    P_pred = e_step_store["P_pred"]
    theta_f = e_step_store["theta_f"]

    for t in range(n):
        H_t = H_list[t]
        if H_t is None:
            continue
        v = Y[t, :].reshape(-1, 1) - H_t @ theta_pred[t]
        S = H_t @ P_pred[t] @ H_t.T + R
        S = (S + S.T) / 2 + 1e-9 * np.eye(mY)
        innovation_score[t] = float((v.T @ np.linalg.pinv(S) @ v).item())

    for t in range(1, n):
        coefficient_change[t] = float(np.linalg.norm(theta_s[t] - theta_s[t - 1]))
    for t in range(n):
        filter_smoother_gap[t] = float(np.linalg.norm(theta_s[t] - theta_f[t]))
    return innovation_score, coefficient_change, filter_smoother_gap


def em_m_step_update(theta_s, P_s, P_lag, H_list, y_list, Q_floor=1e-8, R_floor=1e-8):
    n, p, _ = theta_s.shape
    mY = y_list[next(i for i, y in enumerate(y_list) if y is not None)].shape[0]

    Q_num = np.zeros((p, p))
    for t in range(1, n):
        Exx_t = P_s[t] + theta_s[t] @ theta_s[t].T
        Exx_tm1 = P_s[t - 1] + theta_s[t - 1] @ theta_s[t - 1].T
        Exx_lag = P_lag[t] + theta_s[t] @ theta_s[t - 1].T
        Q_num += Exx_t - Exx_lag - Exx_lag.T + Exx_tm1
    Q = (Q_num / max(1, n - 1) + (Q_num / max(1, n - 1)).T) / 2
    ev = np.linalg.eigvalsh(Q)
    if ev.min() < Q_floor:
        Q += (Q_floor - ev.min() + 1e-10) * np.eye(p)

    R_num = np.zeros((mY, mY))
    r_cnt = 0
    for t in range(n):
        H_t = H_list[t]
        y_t = y_list[t]
        if H_t is None:
            continue
        Eyy = y_t @ y_t.T
        Eyx = y_t @ theta_s[t].T
        Exx = P_s[t] + theta_s[t] @ theta_s[t].T
        R_num += Eyy - Eyx @ H_t.T - H_t @ Eyx.T + H_t @ Exx @ H_t.T
        r_cnt += 1
    R = (R_num / max(1, r_cnt) + (R_num / max(1, r_cnt)).T) / 2
    ev = np.linalg.eigvalsh(R)
    if ev.min() < R_floor:
        R += (R_floor - ev.min() + 1e-10) * np.eye(mY)
    return Q, R


def run_kf_em(Y, Z, lags=1, window=40, max_em_iter=8, tol=1e-4, em_damping=0.7, verbose=True):
    theta0, P0, Q, R = init_from_varx_rolling(Y, Z, lags=lags, window=window)
    best_pack = None
    best_obj = np.inf

    for it in range(1, max_em_iter + 1):
        estore = kf_e_step_store(Y, Z, theta0, Q, R, P0, lags=lags)
        theta_s, P_s, P_lag = rts_smoother(estore["theta_f"], estore["P_f"], estore["theta_pred"], estore["P_pred"])
        Q_new, R_new = em_m_step_update(theta_s, P_s, P_lag, estore["H_list"], estore["y_list"])
        Q = em_damping * Q + (1 - em_damping) * Q_new
        R = em_damping * R + (1 - em_damping) * R_new

        theta_f, P_f, Y_pred, theta_pred, P_pred = kalman_multilag_filter(Y, Z, theta0, Q, R, P0, lags=lags)
        obj = np.nanmean((Y[lags + 1 :, :] - Y_pred[lags + 1 :, :]) ** 2)
        if obj < best_obj:
            best_obj = obj
            best_pack = {
                "theta_f": theta_f,
                "P_f": P_f,
                "theta_pred": theta_pred,
                "P_pred": P_pred,
                "theta_s": theta_s,
                "Q": Q.copy(),
                "R": R.copy(),
                "Y_pred": Y_pred,
                "H_list": estore["H_list"],
            }
        if verbose:
            print(f"[EM {it:02d}] obj={obj:.6f}")
        if it > 1 and abs(obj - best_obj) < tol:
            break

    innovation_score, coefficient_change, filter_smoother_gap = compute_kf_smoother_diagnostics(
        Y,
        {
            "H_list": best_pack["H_list"],
            "theta_pred": best_pack["theta_pred"],
            "P_pred": best_pack["P_pred"],
            "theta_f": best_pack["theta_f"],
        },
        best_pack["theta_s"],
        best_pack["R"],
    )
    return {
        "theta_est": best_pack["theta_s"][:, :, 0],
        "Y_pred": best_pack["Y_pred"],
        "Q": best_pack["Q"],
        "R": best_pack["R"],
        "innovation_score": innovation_score,
        "coefficient_change": coefficient_change,
        "filter_smoother_gap": filter_smoother_gap,
    }


def kalman_filter_torch(Y, Z, signals, theta0, Q0, R, P0, model, lags=1):
    device = torch.device("cpu")
    Y = torch.tensor(Y, dtype=torch.float32, device=device)
    Z = torch.tensor(Z, dtype=torch.float32, device=device)
    signals = torch.tensor(signals, dtype=torch.float32, device=device)
    theta_prev = torch.tensor(theta0, dtype=torch.float32, device=device)
    P_prev = torch.tensor(P0, dtype=torch.float32, device=device)
    Q0 = torch.tensor(Q0, dtype=torch.float32, device=device)
    R = torch.tensor(R, dtype=torch.float32, device=device)

    n, mY = Y.shape
    mX = Z.shape[1]
    m = lags * mY + mX
    p = mY * m
    I = torch.eye(p, device=device)

    loss = torch.tensor(0.0, device=device)
    y_pred_list = []
    for t in range(n):
        if t > 0:
            a_t = model(signals[t]).squeeze()
            Q_t = a_t * Q0
        else:
            Q_t = Q0

        theta_pred = theta_prev
        P_pred = P_prev + Q_t

        if t <= lags:
            theta_prev, P_prev = theta_pred, P_pred
            y_pred_list.append(torch.zeros(mY, device=device))
            continue

        x_t = torch.cat([torch.cat([Y[t - i] for i in range(1, lags + 1)]), Z[t - 1]])
        H_t = torch.zeros((mY, p), device=device)
        for eq in range(mY):
            H_t[eq, eq * m : (eq + 1) * m] = x_t

        yhat = (H_t @ theta_pred).squeeze()
        y_pred_list.append(yhat)

        v = (Y[t] - yhat).unsqueeze(1)
        S = H_t @ P_pred @ H_t.T + R
        S = (S + S.T) / 2 + 1e-6 * torch.eye(mY, device=device)
        loss = loss + torch.logdet(S) + (v.T @ torch.linalg.pinv(S) @ v).squeeze()

        K = P_pred @ H_t.T @ torch.linalg.pinv(S)
        theta = theta_pred + K @ v
        P = (I - K @ H_t) @ P_pred
        theta_prev, P_prev = theta, P
    return loss / n, torch.stack(y_pred_list)


def kalman_filter_torch_block(Y, Z, signals, theta0, Q0, R, P0, model, block_masks, lags=1):
    device = torch.device("cpu")
    Y = torch.tensor(Y, dtype=torch.float32, device=device)
    Z = torch.tensor(Z, dtype=torch.float32, device=device)
    signals = torch.tensor(signals, dtype=torch.float32, device=device)
    theta_prev = torch.tensor(theta0, dtype=torch.float32, device=device)
    P_prev = torch.tensor(P0, dtype=torch.float32, device=device)
    Q0 = torch.tensor(Q0, dtype=torch.float32, device=device)
    R = torch.tensor(R, dtype=torch.float32, device=device)

    n, mY = Y.shape
    mX = Z.shape[1]
    m = lags * mY + mX
    p = mY * m
    I = torch.eye(p, device=device)

    loss = torch.tensor(0.0, device=device)
    valid_count = 0
    y_pred_list = []
    scale_list = []

    for t in range(n):
        if t > 0:
            block_scales = model(signals[t]).squeeze()
        else:
            block_scales = torch.ones(len(block_masks), dtype=torch.float32, device=device)
        scale_list.append(block_scales)
        Q_t = _scale_Q_by_blocks_torch(Q0, block_scales, block_masks)

        theta_pred = theta_prev
        P_pred = P_prev + Q_t

        if t <= lags:
            theta_prev, P_prev = theta_pred, P_pred
            y_pred_list.append(torch.zeros(mY, device=device))
            continue

        x_t = torch.cat([torch.cat([Y[t - i] for i in range(1, lags + 1)]), Z[t - 1]])
        H_t = torch.zeros((mY, p), device=device)
        for eq in range(mY):
            H_t[eq, eq * m : (eq + 1) * m] = x_t

        yhat = (H_t @ theta_pred).squeeze()
        y_pred_list.append(yhat)

        v = (Y[t] - yhat).unsqueeze(1)
        S = H_t @ P_pred @ H_t.T + R
        S = (S + S.T) / 2 + 1e-6 * torch.eye(mY, device=device)
        loss = loss + torch.logdet(S) + (v.T @ torch.linalg.pinv(S) @ v).squeeze()
        valid_count += 1

        K = P_pred @ H_t.T @ torch.linalg.pinv(S)
        theta = theta_pred + K @ v
        P = (I - K @ H_t) @ P_pred @ (I - K @ H_t).T + K @ R @ K.T
        P = (P + P.T) / 2 + 1e-8 * I
        theta_prev, P_prev = theta, P

    denom = max(valid_count, 1)
    return loss / denom, torch.stack(y_pred_list), torch.stack(scale_list)


def evaluate_block_q_scaler(Y, Z, signals, Q0, R, lags, model, block_masks):
    theta0, P0, _, _ = init_from_varx_rolling(Y, Z, lags=lags)
    signals = np.asarray(signals, dtype=float)
    signals_norm = (signals - signals.mean(0)) / (signals.std(0) + 1e-8)
    with torch.no_grad():
        loss, y_pred, scales = kalman_filter_torch_block(
            Y, Z, signals_norm, theta0, Q0, R, P0, model, block_masks, lags=lags
        )
    return loss.item(), y_pred.cpu().numpy(), scales.cpu().numpy(), signals_norm


def build_break_indicator_from_years(quarters: np.ndarray, break_years: list[int] | None):
    years = pd.to_datetime(quarters).year.to_numpy()
    if not break_years:
        return np.zeros_like(years, dtype=float)
    bset = set(int(y) for y in break_years)
    return np.array([1.0 if int(y) in bset else 0.0 for y in years], dtype=float)


def train_q_scaler(Y, Z, signals, Q0, R, lags=1, epochs=50, lr=1e-2):
    theta0, P0, _, _ = init_from_varx_rolling(Y, Z, lags=lags)
    signals = np.asarray(signals, dtype=float)
    signals = (signals - signals.mean(0)) / (signals.std(0) + 1e-8)

    model = QScalerNN(input_dim=signals.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        optimizer.zero_grad()
        loss, _ = kalman_filter_torch(Y, Z, signals, theta0, Q0, R, P0, model, lags=lags)
        loss.backward()
        optimizer.step()
        if ep % 5 == 0:
            print(f"[Epoch {ep}] loss={loss.item():.6f}")
    return model


def train_block_q_scaler(
    Y,
    Z,
    signals,
    Q0,
    R,
    block_masks,
    lags=1,
    epochs=80,
    lr=5e-3,
    weight_decay=1e-4,
    verbose=True,
):
    theta0, P0, _, _ = init_from_varx_rolling(Y, Z, lags=lags)
    signals = np.asarray(signals, dtype=float)
    signals_norm = (signals - signals.mean(0)) / (signals.std(0) + 1e-8)

    model = BlockQScalerLinear(input_dim=signals_norm.shape[1], n_blocks=len(block_masks))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_history = []

    for ep in range(epochs):
        optimizer.zero_grad()
        loss, _, _ = kalman_filter_torch_block(
            Y, Z, signals_norm, theta0, Q0, R, P0, model, block_masks, lags=lags
        )
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))
        if verbose and ep % 10 == 0:
            print(f"[Block Epoch {ep}] loss={loss.item():.6f}")
    return model, np.asarray(loss_history, dtype=float)


def load_panel(path: str | Path, time_col: str = "quarter"):
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df.dropna(subset=[time_col]).copy()


def train_q_scaler_by_country(
    path: str,
    countries: list[str],
    break_years_by_country: dict[str, list[int]] | None = None,
    col_country: str = "country",
    col_time: str = "quarter",
    endo: list[str] | None = None,
    exo: list[str] | None = None,
    lags: int = 1,
    em_iter: int = 8,
    epochs: int = 50,
    lr: float = 1e-2,
):
    if endo is None:
        endo = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
    if exo is None:
        exo = ["ENSO", "OIL_YoY"]
    if break_years_by_country is None:
        break_years_by_country = {}

    df = load_panel(path, time_col=col_time)
    out = {}
    for country in countries:
        g = df[df[col_country] == country].sort_values(col_time).copy()
        if g.empty:
            print(f"[SKIP] {country}: no rows")
            continue

        mask = np.isfinite(g[endo].to_numpy(float)).all(axis=1) & np.isfinite(g[exo].to_numpy(float)).all(axis=1)
        g = g.loc[mask].reset_index(drop=True)
        if len(g) < lags + 8:
            print(f"[SKIP] {country}: too short")
            continue

        Y = g[endo].to_numpy(float)
        Z = g[exo].to_numpy(float)
        Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8)
        Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-8)

        res = run_kf_em(Y, Z, lags=lags, max_em_iter=em_iter, verbose=False)
        break_flag = build_break_indicator_from_years(g[col_time].to_numpy(), break_years_by_country.get(country, []))
        signals = np.vstack([
            res["innovation_score"],
            res["coefficient_change"],
            res["filter_smoother_gap"],
            break_flag,
        ]).T
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

        model = train_q_scaler(
            Y,
            Z,
            signals,
            Q0=res["Q"],
            R=res["R"],
            lags=lags,
            epochs=epochs,
            lr=lr,
        )
        out[country] = {"model": model, "em_result": res, "signals": signals, "quarters": g[col_time].to_numpy()}
        print(f"[OK] {country}: q-scaler trained")
    return out


def train_block_q_scaler_by_country(
    path: str,
    countries: list[str],
    break_years_by_country: dict[str, list[int]] | None = None,
    col_country: str = "country",
    col_time: str = "quarter",
    endo: list[str] | None = None,
    exo: list[str] | None = None,
    lags: int = 1,
    em_iter: int = 8,
    epochs: int = 80,
    lr: float = 5e-3,
    weight_decay: float = 1e-4,
):
    if endo is None:
        endo = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
    if exo is None:
        exo = ["ENSO", "OIL_YoY"]
    if break_years_by_country is None:
        break_years_by_country = {}

    df = load_panel(path, time_col=col_time)
    out = {}
    for country in countries:
        g = df[df[col_country] == country].sort_values(col_time).copy()
        if g.empty:
            print(f"[SKIP] {country}: no rows")
            continue

        mask = np.isfinite(g[endo].to_numpy(float)).all(axis=1) & np.isfinite(g[exo].to_numpy(float)).all(axis=1)
        g = g.loc[mask].reset_index(drop=True)
        if len(g) < lags + 8:
            print(f"[SKIP] {country}: too short")
            continue

        Y = g[endo].to_numpy(float)
        Z = g[exo].to_numpy(float)
        Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8)
        Z = (Z - Z.mean(axis=0)) / (Z.std(axis=0) + 1e-8)

        res = run_kf_em(Y, Z, lags=lags, max_em_iter=em_iter, verbose=False)
        break_flag = build_break_indicator_from_years(g[col_time].to_numpy(), break_years_by_country.get(country, []))
        signals = np.vstack([
            res["innovation_score"],
            res["coefficient_change"],
            res["filter_smoother_gap"],
            break_flag,
        ]).T
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

        block_names, block_masks = build_q_block_masks(
            mY=len(endo), mX=len(exo), lags=lags, exo_names=exo
        )
        model, loss_history = train_block_q_scaler(
            Y,
            Z,
            signals,
            Q0=res["Q"],
            R=res["R"],
            block_masks=block_masks,
            lags=lags,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            verbose=False,
        )
        block_loss, block_y_pred, block_scales, signals_norm = evaluate_block_q_scaler(
            Y, Z, signals, res["Q"], res["R"], lags, model, block_masks
        )
        baseline_mse = float(np.nanmean((Y[lags + 1 :] - res["Y_pred"][lags + 1 :]) ** 2))
        block_mse = float(np.nanmean((Y[lags + 1 :] - block_y_pred[lags + 1 :]) ** 2))

        out[country] = {
            "model": model,
            "em_result": res,
            "signals": signals,
            "signals_norm": signals_norm,
            "signal_names": list(SIGNAL_NAMES),
            "quarters": g[col_time].to_numpy(),
            "Y": Y,
            "Z": Z,
            "endo": list(endo),
            "exo": list(exo),
            "block_names": block_names,
            "block_masks": block_masks,
            "block_scales": block_scales,
            "block_y_pred": block_y_pred,
            "block_loss": block_loss,
            "loss_history": loss_history,
            "baseline_mse": baseline_mse,
            "block_mse": block_mse,
            "break_years": list(break_years_by_country.get(country, [])),
        }
        print(
            f"[OK] {country}: block q-scaler trained; "
            f"baseline_mse={baseline_mse:.4f}, block_mse={block_mse:.4f}"
        )
    return out


def run_q_scaler_diagnostics(
    trained: dict,
    output_dir: str = "q_scaler_diagnostics",
    show_plots: bool = True,
    block: bool = False,
):
    """
    Print summary stats and save one a_t trajectory plot per country.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for country, pack in trained.items():
        model = pack["model"]
        signals = np.asarray(pack["signals"], dtype=float)
        quarters = pd.to_datetime(pack["quarters"])

        # Use the same normalization rule as training.
        signals_norm = (signals - signals.mean(0)) / (signals.std(0) + 1e-8)
        with torch.no_grad():
            a_t = model(torch.tensor(signals_norm, dtype=torch.float32)).squeeze().cpu().numpy()
        a_t = np.clip(a_t, 0.1, 10.0)

        print(f"[{country}] a_t mean={a_t.mean():.6f}, std={a_t.std():.6f}")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(quarters, a_t, color="tab:blue", linewidth=1.2)
        ax.set_title(f"{country} - Learned Q scaling (a_t)")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("a_t")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"a_t_{country}.png", dpi=140, bbox_inches="tight")
        if show_plots:
            plt.show(block=block)
            if not block:
                plt.pause(0.001)
        else:
            plt.close(fig)


def _shade_break_years(ax, break_years):
    for year in break_years or []:
        start = pd.Timestamp(int(year), 1, 1)
        end = pd.Timestamp(int(year) + 1, 1, 1)
        ax.axvspan(start, end, color="tab:orange", alpha=0.12, linewidth=0)


def _safe_corr_matrix(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    out = np.zeros((X.shape[1], Y.shape[1]), dtype=float)
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            xi, yj = X[:, i], Y[:, j]
            ok = np.isfinite(xi) & np.isfinite(yj)
            if ok.sum() < 3 or np.nanstd(xi[ok]) < 1e-12 or np.nanstd(yj[ok]) < 1e-12:
                out[i, j] = 0.0
            else:
                out[i, j] = float(np.corrcoef(xi[ok], yj[ok])[0, 1])
    return out


def _plot_heatmap(values, row_labels, col_labels, title, path, cmap="coolwarm", vlim=None):
    values = np.asarray(values, dtype=float)
    if vlim is None:
        vmax = max(float(np.nanmax(np.abs(values))), 1e-8)
    else:
        vmax = float(vlim)
    fig_w = max(6, 1.3 * len(col_labels))
    fig_h = max(3.5, 0.55 * len(row_labels) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_block_q_scaler_diagnostics(
    trained: dict,
    output_dir: str = "q_block_scaler_diagnostics",
    show_plots: bool = False,
    block: bool = False,
):
    """
    Save sanity-check plots for the block log-linear Q-scaler.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for country, pack in trained.items():
        country_dir = out_dir / country
        country_dir.mkdir(parents=True, exist_ok=True)

        quarters = pd.to_datetime(pack["quarters"])
        block_names = pack["block_names"]
        signal_names = pack.get("signal_names", SIGNAL_NAMES)
        break_years = pack.get("break_years", [])
        scales = np.asarray(pack["block_scales"], dtype=float)
        log_scales = np.log(np.clip(scales, 1e-8, None))
        signals_norm = np.asarray(pack["signals_norm"], dtype=float)
        Y = np.asarray(pack["Y"], dtype=float)
        baseline_pred = np.asarray(pack["em_result"]["Y_pred"], dtype=float)
        block_pred = np.asarray(pack["block_y_pred"], dtype=float)
        endo = pack["endo"]

        print(
            f"[{country}] block_loss={pack['block_loss']:.6f}, "
            f"baseline_mse={pack['baseline_mse']:.6f}, block_mse={pack['block_mse']:.6f}"
        )

        fig, ax = plt.subplots(figsize=(11, 4.5))
        _shade_break_years(ax, break_years)
        for j, name in enumerate(block_names):
            ax.plot(quarters, scales[:, j], linewidth=1.35, label=name)
        ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.55)
        ax.set_title(f"{country} - Block Q scales")
        ax.set_xlabel("Quarter")
        ax.set_ylabel("scale")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, ncol=min(3, len(block_names)))
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(country_dir / f"{country}_block_scales.png", dpi=140, bbox_inches="tight")
        if show_plots:
            plt.show(block=block)
            if not block:
                plt.pause(0.001)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(np.arange(len(pack["loss_history"])), pack["loss_history"], color="tab:green", linewidth=1.4)
        ax.set_title(f"{country} - Training loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Kalman NLL")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(country_dir / f"{country}_loss.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

        weights = pack["model"].linear.weight.detach().cpu().numpy()
        _plot_heatmap(
            weights.T,
            signal_names,
            block_names,
            f"{country} - Linear signal weights",
            country_dir / f"{country}_signal_weights.png",
        )

        corr = _safe_corr_matrix(signals_norm, log_scales)
        _plot_heatmap(
            corr,
            signal_names,
            block_names,
            f"{country} - corr(signal, log scale)",
            country_dir / f"{country}_signal_scale_corr.png",
            vlim=1.0,
        )

        n = len(endo)
        fig, axes = plt.subplots(n, 1, figsize=(11, max(3.0 * n, 5.5)), sharex=True)
        axes = np.atleast_1d(axes)
        for j, ax in enumerate(axes):
            _shade_break_years(ax, break_years)
            ax.plot(quarters, Y[:, j], color="black", linewidth=1.4, label="actual" if j == 0 else None)
            ax.plot(quarters, baseline_pred[:, j], color="C0", linewidth=1.1, alpha=0.9, label="baseline EM" if j == 0 else None)
            ax.plot(quarters, block_pred[:, j], color="C3", linewidth=1.1, alpha=0.9, label="block SciML" if j == 0 else None)
            ax.set_ylabel(endo[j])
            ax.grid(alpha=0.25)
        axes[0].legend(loc="upper left", fontsize=8, ncol=3)
        axes[0].set_title(f"{country} - 1-step prediction sanity check")
        axes[-1].set_xlabel("Quarter")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(country_dir / f"{country}_prediction_compare.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

        mean_log_scale = log_scales.mean(axis=1)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
        axes = axes.flatten()
        colors = np.asarray(pack["signals"], dtype=float)[:, -1]
        for k, ax in enumerate(axes):
            ax.scatter(signals_norm[:, k], mean_log_scale, c=colors, cmap="coolwarm", s=18, alpha=0.75)
            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
            ax.set_xlabel(signal_names[k])
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("mean log scale")
        axes[2].set_ylabel("mean log scale")
        fig.suptitle(f"{country} - Signal vs average block scale")
        fig.tight_layout()
        fig.savefig(country_dir / f"{country}_signal_scatter.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    PATH = SCRIPT_DIR / "gvar_panel_streamlit (8 + EGY + PER).csv"
    COUNTRIES = [
        "BRA",
        "CHL",
        "COL",
        "MEX",
        "KEN",
        "ZAF",
        "IND",
        "IDN",
        "THA",
        "PER",
        "PHL",
        "EGY",
    ]
    ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
    EXO = ["ENSO", "OIL_YoY"]

    # Break-year signal source: all quarters in listed years are set to 1.
    BREAK_COUNTRIES = {
        "Chile": [2004, 2005, 2008, 2013, 2014],
        "Mexico": [1998, 2008, 2009, 2010],
        "Brazil": [2003, 2007, 2008, 2012, 2014, 2016],
        "Colombia": [2012, 2014, 2015],
        "Kenya": [],
        "South Africa": [],
        "Philippines": [2014],
        "India": [2002, 2008, 1999, 2012, 2013],
        "Thailand": [2013],
        "Indonesia": [1998, 1999, 2001, 2008, 2011],
        "Peru": [],
        "Egypt": [],
    }
    NAME_TO_ISO3 = {
        "Chile": "CHL",
        "Mexico": "MEX",
        "Brazil": "BRA",
        "Colombia": "COL",
        "Kenya": "KEN",
        "South Africa": "ZAF",
        "Philippines": "PHL",
        "India": "IND",
        "Thailand": "THA",
        "Indonesia": "IDN",
        "Peru": "PER",
        "Egypt": "EGY",
    }
    BREAK_YEARS = {NAME_TO_ISO3[k]: v for k, v in BREAK_COUNTRIES.items() if k in NAME_TO_ISO3}

    trained = train_block_q_scaler_by_country(
        path=PATH,
        countries=COUNTRIES,
        break_years_by_country=BREAK_YEARS,
        endo=ENDO,
        exo=EXO,
        lags=1,
        em_iter=8,
        epochs=80,
        lr=5e-3,
        weight_decay=1e-4,
    )
    print(f"[DONE] trained countries: {len(trained)}")
    run_block_q_scaler_diagnostics(
        trained,
        output_dir=SCRIPT_DIR / "q_block_scaler_diagnostics",
        show_plots=False,
        block=False,
    )
