#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === statsmodels: 协整检验（Johansen） ===
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# ======================================
#        CELL 1: Climate data
# ======================================

# 1) 读取月度气候数据（你可以改成自己的文件路径）
climate_df = pd.read_csv(
    "data/climate_indices.csv",
    parse_dates=["date"],           # 强制按日期解析
    dayfirst=False,                 # 月/日/年
)

# 如果是 Excel：
# climate_df = pd.read_excel("data/climate.xlsx")


# 2) 每个国家要使用哪些气候变量（随意增减）
CLIMATE_VARS_BY_COUNTRY = {
    "India": {"SOI": 2},           # SOI 提前 2 个季度
    "Brazil": {"SOI": 1},
    "Chile": {"SOI": 3},
    "Indonesia": {"SOI": 2},
    "Mexico": {"WestenV": 1},
    "Peru": {"SOI": 1},
    "Philippines": {"SOI": 1},
    "South Africa": {"SOI": 2},
    "Thailand": {"SOI": 2}
}


# ===============================
# 气候数据预处理与合并（不对气候做差分）
# ===============================
import pandas as pd
import numpy as np


def prepare_climate_quarterly(df_climate: pd.DataFrame,
                              date_col: str = 'date',
                              climate_vars: list[str] | None = None,
                              agg='mean',
                              quarter_label='right',
                              normalize: bool = False) -> pd.DataFrame:
    """
    输入: 月度气候数据，多列变量 + 一个日期列（索引或列，日期格式）。
    步骤: 
      1) 若同月有多条记录，先按月取均值；
      2) 转为季度频率（对每季度内各月取均值）；
      3) 对季度序列取滞后1个季度（等价于月度 lag 3）；
      4) （可选）对季度数据做z-score归一化。
    输出: 列包含所选气候变量的季度滞后值，索引为季度末日期。

    参数:
        df_climate: 月度气候DataFrame（索引为日期或包含日期列）
        date_col: 日期列名（如果日期是列的话），如果日期是索引则为None
        climate_vars: 要选择的气候变量列表，None表示使用所有非日期列
        agg: 聚合方式（'mean'或'sum'等）
        quarter_label: 季度标签（'right'表示季度末，也可以是'left'或None）
        normalize: 是否对季度数据做z-score归一化（默认False，假设已标准化）
    """
    df = df_climate.copy()

    # 处理日期索引或日期列
    if date_col and date_col in df.columns:
        df.index = pd.to_datetime(df[date_col], errors='coerce')
        df = df.drop(columns=[date_col])
    elif not isinstance(df.index, pd.DatetimeIndex):
        # 尝试找日期列
        for c in df.columns:
            if 'Date' in str(c).lower():
                df.index = pd.to_datetime(df[c], errors='coerce')
                df = df.drop(columns=[c])
                break

    # 确保索引是DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')

    # 对齐到月末（月度数据）
    df.index = df.index.to_period('M').to_timestamp('M')

    # 选择气候变量
    if climate_vars is None:
        value_cols = list(df.columns)
    else:
        value_cols = [c for c in climate_vars if c in df.columns]

    if not value_cols:
        raise ValueError("没有找到有效的气候变量列")

    df = df[value_cols]

    # 月内聚合（如果有重复月份）
    if agg == 'mean':
        df_m = df.groupby(df.index).mean()
    elif agg == 'sum':
        df_m = df.groupby(df.index).sum()
    else:
        df_m = df.groupby(df.index).agg(agg)

    # 转季度（均值）
    df_q = df_m.resample('QE', label='right', closed='right').mean()

    # 滞后一个季度（= 月度滞后3）
    df_q_lag1 = df_q.shift(1)

    # （可选）z-score归一化
    if normalize:
        df_q_lag1 = (df_q_lag1 - df_q_lag1.mean()) / df_q_lag1.std()

    return df_q_lag1


def remove_enso_from_Y(Y_df: pd.DataFrame) -> pd.DataFrame:
    """
    从Y DataFrame中删除 'ensos' 列（如果存在）。
    这是针对每个国家的Y表，删除其中的ensos列，而不是从变量名列表删除。
    """
    Y = Y_df.copy()
    if 'ensos' in Y.columns:
        Y = Y.drop(columns=['ensos'])
    return Y


def merge_Y_with_climate_q(Y_levels_df: pd.DataFrame,
                           Y_date_col: str | None = None,
                           climate_q_df: pd.DataFrame | None = None,
                           how: str = 'inner') -> pd.DataFrame:
    """
    将 Y（日期索引或日期列）与季度气候数据（已滞后）按日期对齐合并。
    注意：仅合并，不对气候变量做差分；后续对 Y 做 diff 时，气候保持不变。

    参数:
        Y_levels_df: Y的DataFrame（可以是日期索引或包含日期列）
        Y_date_col: Y的日期列名（如果日期是列），None表示日期是索引
        climate_q_df: 季度气候DataFrame（日期索引，已滞后），None表示不使用气候数据
        how: 合并方式（'inner'/'left'等）
    """
    Y = Y_levels_df.copy()

    # 处理Y的日期
    if Y_date_col and Y_date_col in Y.columns:
        Y.index = pd.to_datetime(Y[Y_date_col], errors='coerce')
        Y = Y.drop(columns=[Y_date_col], errors='ignore')
    elif not isinstance(Y.index, pd.DatetimeIndex):
        Y.index = pd.to_datetime(Y.index, errors='coerce')

    # 对齐到季度末
    Y.index = Y.index.to_period('Q').to_timestamp('Q')
    Y = Y.sort_index()

    # 合并气候数据
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

# def attach_climate_to_X(Y_levels_df, climate_df, country, climate_vars_dict):
#     """
#     给某个国家把气候变量合并进去：
#     - climate_df: 月度原始气候数据
#     - climate_vars_dict: 字典 {country: [var1, var2]}
#     - 先转季度、滞后 1、再 merge 到 Y
#     """
#     vars_use = climate_vars_dict.get(country, [])

#     # 没有气候变量：直接返回空 DataFrame
#     if len(vars_use) == 0:
#         return pd.DataFrame(index=Y_levels_df.index)

#     # 1) 提取该国所需变量
#     df_c = climate_df[["date"] + vars_use].copy() if "date" in climate_df.columns else climate_df[vars_use].copy()

#     # 2) 月度 → 季度 → lag1
#     c_q = prepare_climate_quarterly(df_c, climate_vars=vars_use, normalize=False)

#     # 3) 与 Y 合并
#     merged = merge_Y_with_climate_q(Y_levels_df, climate_q_df=c_q)

#     # 只取气候变量那几列（新的外生）
#     climate_block = merged[vars_use].copy()

#     return climate_block

def attach_climate_to_X(Y_levels_df, climate_df, country, climate_vars_dict):
    """
    把 climate 数据合并到国家数据：
    - 支持每个国家每个变量单独设置 lag（季度）
    - 返回的 climate_X 会和 Y_levels_df 对齐
    """
    var_lag_map = climate_vars_dict.get(country, {})

    # 如果没有气候变量
    if len(var_lag_map) == 0:
        return pd.DataFrame(index=Y_levels_df.index)

    # 创建一个空表（行数对齐）
    climate_X_all = pd.DataFrame(index=Y_levels_df.index)

    for var_name, lag_q in var_lag_map.items():

        # 取出该 climate 的原始数据
        df_c = climate_df[["date", var_name]].copy()

        # 先生成季度平均并 shift 1 个季度（你的定义）
        c_q = prepare_climate_quarterly(
            df_c,
            climate_vars=[var_name],
            normalize=False
        )

        # 再根据 **用户设定的 lag_q** 向下 shift
        # lag_q=2 表示 climate 领先 2 季度（即往后 shift 2）
        c_q_lagged = c_q.shift(lag_q)

        # 合并到 Y
        merged = merge_Y_with_climate_q(Y_levels_df, climate_q_df=c_q_lagged)

        # 提取该变量列（有对齐后的 NaN）
        climate_X_all[var_name] = merged[var_name]

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


# ---------- 简单的OLS初始化（参考简洁版）----------
def ols_vecm_init(Y_diff, Z_diff, Y_level_aligned, lags=2, beta_coint=None, eps=1e-8):
    """
    简单的OLS初始化，不使用ridge正则
    """
    n, mY = Y_diff.shape
    mX = Z_diff.shape[1] if Z_diff is not None and Z_diff.size > 0 else 0
    r = 0 if beta_coint is None else beta_coint.shape[1]

    # 准备ECT
    if r > 0:
        ECT_all = Y_level_aligned @ beta_coint  # (n, r)
    else:
        ECT_all = None

    # 构造设计矩阵
    t0 = max(lags + 1, 2)
    T = n - t0
    if T <= 5:
        raise ValueError(f"数据太短，T={T}")

    m = lags * mY + mX + r
    X_design = np.zeros((T, m))
    Y_target = np.zeros((T, mY))

    for k, t in enumerate(range(t0, n)):
        pieces = []
        for i in range(1, lags + 1):
            pieces.append(Y_diff[t - i, :])
        if mX > 0:
            pieces.append(Z_diff[t - 1, :])
        if r > 0:
            pieces.append(ECT_all[t - 1, :])
        X_design[k, :] = np.concatenate(pieces)
        Y_target[k, :] = Y_diff[t, :]

    # OLS解
    coef, _, _, _ = np.linalg.lstsq(X_design, Y_target, rcond=None)  # (m, mY)
    resid = Y_target - X_design @ coef

    # R0
    R0 = resid.T @ resid / max(T - m, 1)
    R0 = (R0 + R0.T) / 2
    R0 += eps * np.eye(mY)

    # Q0: 对角矩阵，基于R的均值缩放
    scale = np.mean(np.diag(R0))
    q_alpha = 2e-3 * scale
    q_gamma = 2e-3 * scale
    q_B = 2e-3 * scale

    q_vec = []
    for j in range(mY):
        # 每个方程的系数顺序：滞后项, 外生项, ECT
        q_vec.extend([q_gamma] * (lags * mY))
        if mX > 0:
            q_vec.extend([q_B] * mX)
        if r > 0:
            q_vec.extend([q_alpha] * r)
    Q0 = np.diag(q_vec)

    # theta0
    theta0 = coef.T.reshape(-1, 1, order='C')

    return theta0, R0, Q0, r, m

# ---------- Johansen 协整：只用内生变量（levels），窗口内 ----------
def johansen_beta(Y_level_window: np.ndarray, det_order: int = 0, k_ar_diff: int = 1, rank: int | None = None):
    """
    返回 beta_coint: (mY x r)。若 rank=None，用 5% 显著性挑秩（trace test）。
    det_order: 0 表示常数项在协整关系内（与文献常见设置匹配）
    """
    mY = Y_level_window.shape[1]
    cj = coint_johansen(Y_level_window, det_order, k_ar_diff)
    # 选择秩
    if rank is None:
        # trace_stat 与 临界值比较，找到最大的 r 使得 trace_stat > crit_5%
        r = 0
        for i in range(mY):
            if cj.lr1[i] > cj.cvt[i, 1]:  # 1: 5% 临界值
                r = i + 1
        rank = r
    if rank == 0:
        return None  # 无协整
    beta = cj.evec[:, :rank]  # (mY, r)
    return beta




# ---------- 你的 VARX benchmark（保留） ----------
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


# In[ ]:


# ---------- 标准的TVP-VECM Kalman Filter（参考简洁版）----------
def kalman_multilag_filter_vecm(
    Y: np.ndarray,                    # ΔY
    Z_all: np.ndarray,                # ΔZ
    Y_level_aligned: np.ndarray,      # 与 ΔY 对齐的 level（即原 level[1:]）
    Q0: np.ndarray,
    R0: np.ndarray,
    P0: np.ndarray,
    beta_coint: np.ndarray | None,    # (mY, r)，可为 None
    theta0: np.ndarray | None = None, # (p, 1) 初始系数
    dropout0: np.ndarray | None = None,  # dropout mask
    lags: int = 2,
    eps: float = 1e-8,
    Q_R_update = True
):


    """
    标准Kalman Filter，不使用标准化和岭回归
    模型：ΔY_t = Σ Γ_i ΔY_{t-i} + B ΔZ_{t-1} + α β' Y_{t-1} + ε_t
    """
    global GLOBAL_QR_CACHE
    Q_hist = []
    R_hist = []
    n, mY = Y.shape
    mX = Z_all.shape[1] if Z_all is not None and Z_all.size > 0 else 0
    r = 0 if beta_coint is None else beta_coint.shape[1]

    m = lags * mY + mX + r
    p = m * mY

    # 初始化
    theta = theta0.copy() if theta0 is not None else np.zeros((p, 1))
    P = P0.copy()
    R = R0.copy()
    Q = Q0.copy()

    # dropout mask处理
    if dropout0 is not None:
        dropout_exp = 3
        d0 = np.asarray(dropout0).ravel()[:p]
        idx_dropout = d0 < 1.0
        if np.any(idx_dropout):
            scale = np.maximum(d0[idx_dropout], 1e-3) ** dropout_exp
            P[np.ix_(idx_dropout, idx_dropout)] *= scale[:, None] * scale[None, :]
            Q[np.ix_(idx_dropout, idx_dropout)] *= scale[:, None] * scale[None, :]

    # 存储
    theta_est = np.zeros((n, p))
    Y_pred = np.full((n, mY), np.nan)
    P_hist = np.zeros((p, p, n))
    e_raw = np.full((n, mY), np.nan)

    I_p = np.eye(p)

    rho_R = 0.02
    rho_Q = 0.02

    # Kalman Filter循环
    for t in range(lags + 1, n):
        # 构造X_t: [ΔY_{t-1}..ΔY_{t-lags}, ΔZ_{t-1}, ECT_{t-1}]
        pieces = []
        for i in range(1, lags + 1):
            pieces.append(Y[t - i, :])
        if mX > 0:
            pieces.append(Z_all[t - 1, :])
        if r > 0:
            ECT_prev = Y_level_aligned[t - 1, :] @ beta_coint  # (r,)
            pieces.append(ECT_prev)
        X_t = np.concatenate(pieces)  # (m,)

        # 构造H_t: 对每个方程
        H_t = np.zeros((mY, p))
        for j in range(mY):
            idx = slice(j * m, (j + 1) * m)
            H_t[j, idx] = X_t

        # 标准Kalman Filter步骤
        # 预测
        theta_pred = theta
        P_pred = P + Q

        # 创新协方差
        S = H_t @ P_pred @ H_t.T + R
        S = (S + S.T) / 2
        S += eps * np.eye(mY)

        # 卡尔曼增益
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = P_pred @ H_t.T @ S_inv

        # 更新
        y_t = Y[t, :].reshape(-1, 1)
        y_hat = H_t @ theta_pred
        innovation = y_t - y_hat

        theta = theta_pred + K @ innovation
        P = (I_p - K @ H_t) @ P_pred
        P = (P + P.T) / 2
        P += eps * np.eye(p)

        # 保存
        theta_est[t, :] = theta.ravel()
        Y_pred[t, :] = y_hat.ravel()
        P_hist[:, :, t] = P
        e_raw[t, :] = innovation.ravel()

        if Q_R_update == True:
            # 计算新的 R_t
            R_new = (1.0 - rho_R) * R + rho_R * (innovation @ innovation.T)
            # 限制 R 不要爆掉太多
            if np.trace(R_new) > 5 * np.trace(R0):
                R_new = 5 * np.trace(R0) / np.trace(R_new) * R_new
            R = 0.5 * (R_new + R_new.T) + eps * np.eye(mY)

            # Q 整体缩放
            err2 = (innovation.T @ innovation).item() / mY
            tr_Q = np.trace(Q)
            if tr_Q > eps and err2 > 0:
                scale = err2 / tr_Q
                Q = (1.0 - rho_Q) * Q + rho_Q * scale * Q

    # 计算误差
    valid = ~np.isnan(e_raw).any(axis=1)
    if np.any(valid):
        rmse = np.sqrt(np.nanmean(e_raw[valid] ** 2))
    else:
        rmse = np.nan

    return rmse, e_raw, theta_est, Y_pred, P_hist


# In[ ]:


# def kalman_multilag_filter_vecm(
#     Y: np.ndarray,                    # ΔY
#     Z_all: np.ndarray,                # ΔZ
#     Y_level_aligned: np.ndarray,      # 与 ΔY 对齐的 level（即原 level[1:]）
#     Q0: np.ndarray,
#     R0: np.ndarray,
#     P0: np.ndarray,
#     beta_coint: np.ndarray | None,    # (mY, r)，可为 None
#     theta0: np.ndarray | None = None, # (p, 1) 初始系数
#     dropout0: np.ndarray | None = None,  # dropout mask
#     lags: int = 2,
#     eps: float = 1e-8,
#     Q_R_update: bool = True
# ):
#     """
#     标准 Kalman Filter + 因果标准化（只标准化回归向量 X_t，目标 y_t 保持原始尺度）
#     模型：ΔY_t = Σ Γ_i ΔY_{t-i} + B ΔZ_{t-1} + α β' Y_{t-1} + ε_t

#     标准化策略（完全因果）：
#       - 对 ΔY 的滞后项 和 ΔZ，维护指数滑动均值/方差（只用 t 之前的数据）。
#       - 在时点 t，构造 X_t 时用「当前存的均值/方差」标准化滞后项：
#             X_Y_lag_i = (ΔY_{t-i} - μ_Y) / σ_Y
#             X_Z       = (ΔZ_{t-1} - μ_Z) / σ_Z
#         这些 μ/σ 只包含历史信息（<= t-1），不会用到当期 ΔY_t。
#       - ECT 不做标准化（仍然是 β'Y_{t-1} 的原始尺度）。
#     """
#     global GLOBAL_QR_CACHE
#     Q_hist = []
#     R_hist = []

#     n, mY = Y.shape
#     mX = Z_all.shape[1] if Z_all is not None and Z_all.size > 0 else 0
#     r = 0 if beta_coint is None else beta_coint.shape[1]

#     m = lags * mY + mX + r
#     p = m * mY

#     # ========== 初始化状态与协方差 ==========
#     theta = theta0.copy() if theta0 is not None else np.zeros((p, 1))
#     P = P0.copy()
#     R = R0.copy()
#     Q = Q0.copy()

#     # dropout mask 处理（对先验 P/Q 做不确定性放大）
#     if dropout0 is not None:
#         dropout_exp = 3
#         d0 = np.asarray(dropout0).ravel()[:p]
#         idx_dropout = d0 < 1.0
#         if np.any(idx_dropout):
#             scale = np.maximum(d0[idx_dropout], 1e-3) ** dropout_exp
#             P[np.ix_(idx_dropout, idx_dropout)] *= scale[:, None] * scale[None, :]
#             Q[np.ix_(idx_dropout, idx_dropout)] *= scale[:, None] * scale[None, :]

#     # ========== 存储 ==========
#     theta_est = np.zeros((n, p))
#     Y_pred = np.full((n, mY), np.nan)   # 这里是原始尺度的 ΔY 预测
#     P_hist = np.zeros((p, p, n))
#     e_raw = np.full((n, mY), np.nan)    # 原始尺度的残差

#     I_p = np.eye(p)

#     # Q/R 更新的步长（保持原来的设置）
#     rho_R = 0.02
#     rho_Q = 0.02

#     # ========== 因果标准化：对 ΔY 和 ΔZ 维护滑动均值/方差 ==========
#     # 指数平滑系数（越接近 1 越“长记忆”）
#     lambda_norm = 0.99

#     # 初始化时，只用最早的观测做 μ 的起点，var 先设为 1，后面会很快调整
#     mu_Y = Y[0, :].copy()
#     var_Y = np.ones(mY)

#     if mX > 0:
#         mu_Z = Z_all[0, :].copy()
#         var_Z = np.ones(mX)
#     else:
#         mu_Z = None
#         var_Z = None

#     # ECT 不做标准化，如果之后你想加，可以再扩展一个 μ_ECT / var_ECT

#     # ========== Kalman Filter 循环 ==========
#     for t in range(lags + 1, n):
#         # ---------- 1. 构造标准化后的滞后项 & ΔZ ----------
#         # 当前用于标准化的刻度：只包含 <= t-1 的历史信息
#         std_Y = np.sqrt(np.maximum(var_Y, eps))

#         # 滞后 ΔY：shape (mY,) -> 用共享的 μ_Y/std_Y 标准化（对每个维度）
#         Y_lag_list = []
#         for i in range(1, lags + 1):
#             Y_lag_raw = Y[t - i, :]
#             Y_lag_std = (Y_lag_raw - mu_Y) / std_Y
#             Y_lag_list.append(Y_lag_std)
#         if len(Y_lag_list) > 0:
#             Y_lags_std = np.concatenate(Y_lag_list, axis=0)  # (lags*mY,)
#         else:
#             Y_lags_std = np.zeros((0,))

#         # ΔZ_{t-1} 标准化
#         if mX > 0:
#             std_Z = np.sqrt(np.maximum(var_Z, eps))
#             Z_prev_raw = Z_all[t - 1, :]
#             Z_prev_std = (Z_prev_raw - mu_Z) / std_Z
#         else:
#             Z_prev_std = np.zeros((0,))

#         # ECT_{t-1} 不标准化
#         if r > 0 and beta_coint is not None:
#             ECT_prev = Y_level_aligned[t - 1, :] @ beta_coint  # (r,)
#         else:
#             ECT_prev = np.zeros((0,))

#         # 回归向量 X_t: [标准化的滞后ΔY, 标准化的ΔZ, 原始ECT]
#         pieces = [Y_lags_std]
#         if mX > 0:
#             pieces.append(Z_prev_std)
#         if r > 0:
#             pieces.append(ECT_prev)
#         X_t = np.concatenate(pieces)  # (m,)

#         # ---------- 2. 构造观测矩阵 H_t ----------
#         # 每个方程 j 用同一个 X_t，但对应自己那一块系数
#         H_t = np.zeros((mY, p))
#         for j in range(mY):
#             idx = slice(j * m, (j + 1) * m)
#             H_t[j, idx] = X_t

#         # ---------- 3. Kalman 预测步 ----------
#         theta_pred = theta              # 随机游走：θ_t = θ_{t-1} + u_t，E[u_t]=0
#         P_pred = P + Q                  # P_pred = P + Q

#         # 创新协方差
#         S = H_t @ P_pred @ H_t.T + R
#         S = (S + S.T) / 2
#         S += eps * np.eye(mY)

#         try:
#             S_inv = np.linalg.inv(S)
#         except np.linalg.LinAlgError:
#             S_inv = np.linalg.pinv(S)

#         K = P_pred @ H_t.T @ S_inv

#         # ---------- 4. 更新步 ----------
#         y_t = Y[t, :].reshape(-1, 1)   # 观测是原始尺度的 ΔY_t
#         y_hat = H_t @ theta_pred       # 预测值也是原始尺度（因为 y 没标准化）
#         innovation = y_t - y_hat       # 残差（原始尺度）

#         theta = theta_pred + K @ innovation
#         P = (I_p - K @ H_t) @ P_pred
#         P = (P + P.T) / 2
#         P += eps * np.eye(p)

#         # ---------- 5. 保存 ----------
#         theta_est[t, :] = theta.ravel()
#         Y_pred[t, :] = y_hat.ravel()
#         P_hist[:, :, t] = P
#         e_raw[t, :] = innovation.ravel()

#         # ---------- 6. 在线更新 Q / R（保持原有写法） ----------
#         if Q_R_update:
#             # 更新观测噪声 R（基于残差）
#             R_new = (1.0 - rho_R) * R + rho_R * (innovation @ innovation.T)
#             if np.trace(R_new) > 5 * np.trace(R0):
#                 R_new = 5 * np.trace(R0) / np.trace(R_new) * R_new
#             R = 0.5 * (R_new + R_new.T) + eps * np.eye(mY)

#             # Q 整体缩放（还是原来的整体 scale 逻辑）
#             err2 = float(innovation.T @ innovation) / mY
#             tr_Q = np.trace(Q)
#             if tr_Q > eps and err2 > 0:
#                 scale = err2 / tr_Q
#                 Q = (1.0 - rho_Q) * Q + rho_Q * scale * Q

#         # ---------- 7. 因果更新标准化刻度（只用“已经看到”的数据） ----------
#         # 这里更新的是「下一期」要用的 μ/var：
#         #   - 对 ΔY：用当前的 Y[t]
#         #   - 对 ΔZ：用当前的 Z_all[t] （下一期会用到 Z_{t} 作为滞后）
#         # 注意：这些更新在本期预测之后进行，不会影响当前的 y_hat / X_t，因而不泄漏未来信息。
#         # ΔY 统计
#         mu_Y_old = mu_Y.copy()
#         mu_Y = lambda_norm * mu_Y + (1.0 - lambda_norm) * Y[t, :]
#         var_Y = lambda_norm * var_Y + (1.0 - lambda_norm) * (Y[t, :] - mu_Y_old) ** 2

#         # ΔZ 统计
#         if mX > 0 and t < n:
#             # 这里使用 Z_all[t]，这样在下一期 t+1 构造 ΔZ_{t} 的标准化时，
#             # μ_Z / var_Z 已经包含了 Z_{0..t}，完全因果。
#             z_new = Z_all[t, :]
#             mu_Z_old = mu_Z.copy()
#             mu_Z = lambda_norm * mu_Z + (1.0 - lambda_norm) * z_new
#             var_Z = lambda_norm * var_Z + (1.0 - lambda_norm) * (z_new - mu_Z_old) ** 2

#     # ========== 8. 计算整体 RMSE ==========
#     valid = ~np.isnan(e_raw).any(axis=1)
#     if np.any(valid):
#         rmse = np.sqrt(np.nanmean(e_raw[valid] ** 2))
#     else:
#         rmse = np.nan

#     return rmse, e_raw, theta_est, Y_pred, P_hist


# In[ ]:


# ============================================
#              FUNCTION 1
#  load_country_data(): 数据读取 + 切片 + 年份标签
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

    Y_levels_cut = Y_levels[40:, :]
    X_levels_cut = X_levels[40:, :]
    Yd = np.diff(Y_levels_cut, axis=0)
    Xd = np.diff(X_levels_cut, axis=0)
    Y_level_aligned = Y_levels_cut[1:, :]
    n, mY = Yd.shape
    mX = Xd.shape[1]

    # === moved from main: 年份处理 ===
    if len(T) > 0:
        first_col = T.iloc[:, 0]
        try:
            dates = pd.to_datetime(first_col, errors='coerce')
            if dates.notna().sum() > n/2:
                dates_cut = dates.iloc[40:].reset_index(drop=True)
                dates_diff = dates_cut.iloc[1:].reset_index(drop=True)
                year_labels = dates_diff.dt.year.astype(str)
            else:
                start_year = 1970
                start_quarter = 1
                total_quarters = 40 + len(Yd)
                years = []
                for i in range(len(Yd)):
                    q = (40 + i + start_quarter - 1) % 4 + 1
                    y = start_year + (40 + i + start_quarter - 1) // 4
                    years.append(f"{y}Q{q}")
                year_labels = pd.Series(years)
        except:
            start_year = 1970
            years = []
            for i in range(len(Yd)):
                y = start_year + (40 + i) // 4
                years.append(str(y))
            year_labels = pd.Series(years)
    else:
        year_labels = pd.Series([str(1970 + i//4) for i in range(len(Yd))])

    return (
        Yd, Xd, Y_level_aligned, year_labels,
        Y_levels, idx_Y, idx_X, ttl_Y
    )



# ============================================
#              FUNCTION 2
#      run_tvpkf(): 协整 + 初始化 + KF
# ============================================

def run_tvpkf(Yd, Xd, Y_level_aligned, lags, beta_coint):

    # === moved from main ===
    r = 0 if beta_coint is None else beta_coint.shape[1]

    theta0, R0, Q0, r_ck, m_model = ols_vecm_init(
        Y_diff=Yd, Z_diff=Xd, Y_level_aligned=Y_level_aligned,
        lags=lags, beta_coint=beta_coint
    )
    assert r_ck == r
    p = m_model * Yd.shape[1]
    P0 = 10.0 * np.eye(p)

    rmse, e_raw, theta_est, Y_pred, P_hist = kalman_multilag_filter_vecm(
            Y=Yd, Z_all=Xd, Y_level_aligned=Y_level_aligned,
            Q0=Q0, R0=R0, P0=P0,
            beta_coint=beta_coint, theta0=theta0,
            lags=lags
        )

    return {
        "rmse": rmse,
        "e_raw": e_raw,
        "theta_est": theta_est,
        "Y_pred": Y_pred,
        "P_hist": P_hist,
        "p": p,
        "m_model": m_model,
        "theta0": theta0,
        "Q0": Q0,
        "R0": R0,
        "P0": P0,
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
    tick_labels = [year_labels.iloc[i] for i in tick_indices]

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

    plt.tight_layout()
    plt.show()



# ============================================
#         FUNCTION 4: weighted_corr
#         FUNCTION 5: plot_pred_corr
# ============================================

# 这两个保持原样，完全从 main 剪出来！


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

    # === 完整引用你原来的函数 ===

    Y_true = Yd[1:, :]
    Yhat = Y_pred[1:, :]
    Y_lag1 = Yd[:-1, :]
    Z_lag1 = Xd[:-1, :]

    year_labels_short = year_labels.iloc[1:] if len(year_labels) > len(Y_true) else year_labels[:len(Y_true)]
    n_use = len(year_labels_short)
    step_ts = max(1, n_use // 8)
    tick_indices_ts = np.arange(0, n_use, step_ts)
    tick_labels_ts = [year_labels_short.iloc[i] for i in tick_indices_ts]

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
    plt.show()



# ============================================
#              FUNCTION 6
#        plot_dropone_heatmap()
# ============================================

def plot_dropone_heatmap(
    country, Yd, Xd, Y_level_aligned,
    beta_coint, theta0, Q0, R0, P0,
    m_model, r, p, lags, base_rmse,
    ttl_Y, ttls_X
):

    # === 完整引用你原来的代码 ===

    labels_x = []
    mY = Yd.shape[1]

    for j in range(lags * mY):
        lag_id = j // mY + 1
        var_id = j % mY
        labels_x.append(f"L{lag_id}_{ttl_Y[var_id]}")
    for j in range(Xd.shape[1]):
        labels_x.append(ttls_X[j])
    for j in range(r):
        labels_x.append(f"ECT{j+1}")

    print(f"\n[{country}] 计算drop-one heatmap（这可能需要一些时间）...")
    E_hists_k = np.full((mY, m_model), np.nan)

    for i_Y in range(mY):
        for j_YX in range(m_model):
            idx_0 = i_Y * m_model + j_YX
            dropout0 = np.ones((p,))
            dropout0[idx_0] = 0.0

            rmse_d, *_ = kalman_multilag_filter_vecm(
                Y=Yd, Z_all=Xd, Y_level_aligned=Y_level_aligned,
                Q0=Q0, R0=R0, P0=P0,
                beta_coint=beta_coint, theta0=theta0,
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
    plt.show()



# ============================================
#              FUNCTION 7
#           plot_ect_stats()
# ============================================

def plot_ect_stats(country, Y_level_aligned, beta_coint,
                   theta_est, year_labels, lags):

    if beta_coint is None:
        return

    r = beta_coint.shape[1]
    window = 6
    n_ect = len(Y_level_aligned)
    ECT_all = Y_level_aligned @ beta_coint

    ect_rolling_mean = np.full_like(ECT_all, np.nan)
    ect_rolling_std = np.full_like(ECT_all, np.nan)

    for t in range(window - 1, n_ect):
        window_data = ECT_all[t - window + 1:t + 1, :]
        ect_rolling_mean[t, :] = np.mean(window_data, axis=0)
        ect_rolling_std[t, :] = np.std(window_data, axis=0, ddof=1)

    theta_start = lags + 1
    ect_plot_start = theta_start - 1

    fig, axes = plt.subplots(r, 1, figsize=(12, 4*r))
    if r == 1:
        axes = [axes]

    for r_idx in range(r):
        ax = axes[r_idx]
        valid_idx = np.arange(
            ect_plot_start, min(n_ect, ect_plot_start + len(theta_est))
        )
        x_plot = np.arange(len(valid_idx))

        ect_vals = ECT_all[valid_idx, r_idx]
        mean_vals = ect_rolling_mean[valid_idx, r_idx]
        std_vals = ect_rolling_std[valid_idx, r_idx]

        year_labels_ect = (
            year_labels[:len(valid_idx)]
            if len(year_labels) >= len(valid_idx)
            else year_labels
        )
        step_ect = max(1, len(valid_idx) // 10)
        tick_indices_ect = np.arange(0, len(valid_idx), step_ect)
        tick_labels_ect = [
            year_labels_ect.iloc[i] if i < len(year_labels_ect) else str(i)
            for i in tick_indices_ect
        ]

        ax.plot(x_plot, ect_vals, color='tab:blue', alpha=0.3, label=f'ECT{r_idx+1} (raw)', lw=0.8)
        ax.plot(x_plot, mean_vals, color='tab:orange', lw=2,
                 label=f'ECT{r_idx+1} rolling mean (w={window})', linestyle='--')
        ax.fill_between(
            x_plot, 
            mean_vals - std_vals, 
            mean_vals + std_vals,
            alpha=0.2, color='tab:orange'
        )

        ax.set_title(f"{country} — ECT{r_idx+1} (rolling window={window})", fontsize=11)
        ax.set_xticks(tick_indices_ect)
        ax.set_xticklabels(tick_labels_ect, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('ECT value', fontsize=9)

    plt.suptitle(f"{country} — ECT Rolling Statistics (window={window})", fontsize=12, y=0.995)
    plt.tight_layout()
    plt.show()


# In[ ]:


# ============================================
#              CLEAN MAIN SCRIPT (DEBUG)
# ============================================

if __name__ == "__main__":

    print("\n================= LOADING DATA =================\n")

    fname = "data/df_country_data_climate.xlsx"

    sheets_9_of_12 = [
        "India", "Brazil", "Chile", "Indonesia", "Mexico", "Peru",
        "Philippines", "South Africa", "Thailand"
    ]

    ttls = ['y', 'Dp', 'eq', 'ep', 'r', 'ys', 'Dps', 'eqs', 'rs', 'lrs', 'ensos']
    ttls_Y = ttls[:5]
    ttls_X_base = ttls[5:10]    # 注意：不要直接改 ttls_X（避免污染下一个国家）
    lags = 2


    print("\n=== DEBUG: 气候数据列名 ===")
    print(climate_df.columns)


    for country in sheets_9_of_12:

        print(f"\n\n====================================")
        print(f"***** Country {country} *****")
        print("====================================\n")

        # =====================================================
        # 1) 读取并切分国家数据
        # =====================================================
        Yd, Xd, Y_level_aligned, year_labels, Y_levels, idx_Y, idx_X, ttl_Y = \
            load_country_data(fname, country, ttls_Y, ttls_X_base)

        print(f"[DEBUG] len(Y_levels) = {len(Y_levels)}")
        print(f"[DEBUG] len(Yd)       = {len(Yd)}")
        print(f"[DEBUG] len(year_labels) = {len(year_labels)}")


        # =====================================================
        # 2) 构造与 Yd 对齐的 Y_levels_df
        # =====================================================
        # Yd = diff(Y_levels_cut) → 长度比 Y_levels_cut 少 1
        # Yd, year_labels, Y_levels_df 三者必须一致长度

        nYd = len(Yd)

        # Y_levels_cut = Y_levels[40:]，长度 = nYd + 1
        # 所以从 Y_levels[-(nYd+1)+1:] = Y_levels[-nYd:] 取 nYd 行
        Y_levels_df = pd.DataFrame(
            Y_levels[-nYd:, :],
            columns=ttl_Y
        )

        print(f"[DEBUG] After slicing, len(Y_levels_df) = {len(Y_levels_df)}")

        # 对齐索引
        if len(Y_levels_df) != len(year_labels):
            print("❌ ERROR: Y_levels_df 与 year_labels 长度不一致!")
            print(f"Y_levels_df = {len(Y_levels_df)}, year_labels = {len(year_labels)}")
            print("程序停止。请检查日期处理。")
            raise ValueError("Y_levels_df length mismatch.")

        # === 创建季度日期索引 ===
        # year_labels 是字符串，例如 "1979", "1980", "1981"
        # 我们把它变成季度末日期
        quarter_index = pd.period_range(start='1979Q1', periods=nYd, freq='Q').to_timestamp('Q')

        Y_levels_df.index = year_labels[-nYd:]

        # 用季度日期覆盖 Y_levels_df.index
        Y_levels_df.index = quarter_index


        # =====================================================
        # 3) 合并气候数据
        # =====================================================
        print("\n[DEBUG] Processing climate variables for", country)
        print("需要的气候变量:", CLIMATE_VARS_BY_COUNTRY.get(country))

        climate_X = attach_climate_to_X(
            Y_levels_df=Y_levels_df,
            climate_df=climate_df,
            country=country,
            climate_vars_dict=CLIMATE_VARS_BY_COUNTRY
        )

        print(f"[DEBUG] climate_X rows = {climate_X.shape[0]}")
        print(f"[DEBUG] climate_X columns = {list(climate_X.columns)}")


        # =====================================================
        # 4) 合并 Xd + climate_X（外生变量）
        # =====================================================
        if climate_X.shape[0] != Xd.shape[0]:
            print("⚠️ WARNING: climate_X 行数不匹配，跳过追加")
            print(f"Xd rows = {Xd.shape[0]}, climate_X rows = {climate_X.shape[0]}")
        else:
            Xd = np.hstack([Xd, climate_X.to_numpy()])

        # 每个国家的外生变量名
        ttls_X_country = ttls_X_base + list(climate_X.columns)
        print(f"[DEBUG] ttls_X_country = {ttls_X_country}")


        # =====================================================
        # 5) Johansen 协整检验
        # =====================================================
        beta_coint = None
        try:
            if Y_levels.shape[1] >= 2:
                beta_coint = johansen_beta(Y_levels, det_order=0, k_ar_diff=1, rank=None)
                if beta_coint is not None:
                    print(f"检测到 {beta_coint.shape[1]} 个协整关系")
                else:
                    print("无协整")
        except Exception as e:
            print("Johansen failed:", e)


        # =====================================================
        # 6) Kalman Filter
        # =====================================================
        print("\n[DEBUG] Running Kalman filter ...")
        print(f"Xd shape = {Xd.shape}")
        print(f"Yd shape = {Yd.shape}")

        res = run_tvpkf(Yd, Xd, Y_level_aligned, lags, beta_coint)
        print(f"Kalman Filter 完成. RMSE = {res['rmse']:.6f}")


        # =====================================================
        # 7) Plot 核心结果
        # =====================================================

        plot_core_results(
            country, Yd, Xd, year_labels,
            res, ttls_Y, ttls_X_country, lags
        )

        plot_pred_corr(
            Yd, res["Y_pred"], Xd,
            ttls_Y, ttls_X_country, country,
            "Kalman (tvp-VECM)", year_labels
        )

        # VARX
        Yhat_varx_raw, _, _ = varx_rolling_predict(Yd, Xd, lags=lags)
        plot_pred_corr(
            Yd, Yhat_varx_raw, Xd,
            ttls_Y, ttls_X_country, country,
            "VARX", year_labels
        )


        # =====================================================
        # 8) Drop-one + ECT stats
        # =====================================================
        plot_dropone_heatmap(
            country, Yd, Xd, Y_level_aligned,
            beta_coint, res["theta0"], res["Q0"], res["R0"], res["P0"],
            res["m_model"], 
            0 if beta_coint is None else beta_coint.shape[1],
            res["p"],
            lags, res["rmse"],
            ttl_Y, ttls_X_country
        )

        plot_ect_stats(
            country, Y_level_aligned, beta_coint,
            res["theta_est"], year_labels, lags
        )


# 

# In[ ]:


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # === statsmodels: 协整检验（Johansen） ===
# from statsmodels.tsa.vector_ar.vecm import coint_johansen

# # ---------- helpers ----------
# def intersect_stable(a_list, b_list):
#     b_index = {v: i for i, v in enumerate(b_list)}
#     ia, ib, c_vals = [], [], []
#     for i, v in enumerate(a_list):
#         if v in b_index:
#             ia.append(i)
#             ib.append(b_index[v])
#             c_vals.append(v)
#     return c_vals, np.array(ia, dtype=int), np.array(ib, dtype=int)


# # ---------- 组装 X_t 的小工具：∇Y 滞后 + ∇Z + ECT ----------
# def _build_X_row(Y_diff, Z_diff, ECT_prev, t, lags, mY, mX, r):
#     """X_t = [ΔY_{t-1} ... ΔY_{t-lags}, ΔZ_{t-1}, ECT_{t-1}]"""
#     pieces = []
#     for i in range(1, lags + 1):
#         pieces.append(Y_diff[t - i, :])      # ΔY 滞后
#     if mX > 0:
#         pieces.append(Z_diff[t - 1, :])      # ΔZ_{t-1}
#     if r > 0:
#         pieces.append(ECT_prev.reshape(-1))  # β'Y_{t-1} （或其标准化）
#     return np.concatenate(pieces, axis=0)
# # --------- 预训练Q R ---------------
# def kalman_em_pretrain(Y, Z, lags=2, max_iter=20, alpha=0.9, beta=0.9, tol=1e-4):
#     """
#     简易 EM 预训练，仅用于估计 Q、R 初始尺度，不包含协整项。
#     输入: ΔY, ΔZ
#     输出: Q_em, R_em
#     """
#     n, mY = Y.shape
#     mX = Z.shape[1]
#     m = lags * mY + mX
#     p = m * mY

#     # 初始值
#     Q = np.eye(p)
#     R = np.eye(mY)
#     P = np.eye(p)
#     theta = np.zeros((p, 1))

#     merr_prev = np.inf
#     for it in range(max_iter):
#         theta_est = np.zeros((n, p))
#         e_std = np.full((n, mY), np.nan)

#         for t in range(lags, n):
#             # 构造 H_t
#             regY = np.concatenate([Y[t - i - 1, :] for i in range(lags)], axis=0)
#             regZ = Z[t - 1, :] if mX > 0 else np.zeros((0,))
#             X_t = np.concatenate([regY, regZ], axis=0)
#             H_t = np.zeros((mY, p))
#             for j in range(mY):
#                 idx = slice(j * m, (j + 1) * m)
#                 H_t[j, idx] = X_t

#             # KF 更新
#             P_pred = P / 0.99 + Q
#             theta_pred = theta
#             R_inv = np.linalg.inv(R)
#             P_inv = np.linalg.inv(P_pred)

#             A_ = H_t.T @ R_inv @ H_t + P_inv
#             b_ = H_t.T @ R_inv @ Y[t, :].reshape(-1, 1) + P_inv @ theta_pred
#             theta = np.linalg.solve(A_, b_)
#             P = np.linalg.inv(A_)

#             e_t = Y[t, :].reshape(-1, 1) - H_t @ theta_pred
#             e_std[t, :] = e_t.ravel()
#             theta_est[t, :] = theta.ravel()

#         # 更新 Q,R
#         valid_e = ~np.isnan(e_std).any(axis=1)
#         R_new = np.cov(e_std[valid_e, :].T) + 1e-6*np.eye(mY)

#         dtheta = np.diff(theta_est, axis=0)
#         Q_new = np.cov(dtheta.T) + 1e-6*np.eye(p)

#         # 平滑更新
#         R = alpha * R + (1 - alpha) * R_new
#         Q = beta * Q + (1 - beta) * Q_new

#         merr = np.nanmean(np.sqrt(np.mean(e_std[valid_e] ** 2, axis=1)))
#         if abs(merr - merr_prev) < tol:
#             print(f"[EM pretrain] converged at {it}, merr={merr:.5f}")
#             break
#         merr_prev = merr

#     return Q, R

# # ---------- 窗口内 ridge-VARX / VECM 初始化 ----------
# def ridge_vecm_init(Y_diff, Z_diff, Y_level_aligned, lags=2, ridge=1e-6,
#                     beta_coint=None, T_init=80, q_scale=0.02, q_alpha_scale=1.0):
#     """
#     用前 T_init 期初始化：
#       1) 若给定 beta_coint (mY x r)，则构造 VECM 的 ECT 列；
#       2) ridge 正则的多元回归，得到初值 coef (m x mY)；
#       3) R0 = 残差协方差；Q0 ~ q_scale^2 * blockdiag(σ_j^2*(X'X+λI)^{-1})
#     返回：theta0 (p x 1), R0, Q0, r(秩), m(单方程回归维度)
#     """
#     n, mY = Y_diff.shape
#     mX = Z_diff.shape[1] if Z_diff is not None and Z_diff.size > 0 else 0
#     r = 0 if beta_coint is None else beta_coint.shape[1]

#     # 训练起止：至少要满足 lags
#     t0 = max(lags + 1, 2)
#     t1 = min(n - 1, T_init)  # 开区间右端在下面 for 中用到
#     T = t1 - t0
#     if T <= 5:
#         raise ValueError("初始化窗口太短，无法做 ridge 初始化。")

#     # 先准备 ECT(t-1)
#     if r > 0:
#         # Y_level_aligned: shape (n, mY)，与 Y_diff 对齐（Y_level[1:]）
#         ECT_all = (Y_level_aligned @ beta_coint)  # (n, r)
#     else:
#         ECT_all = None

#     # 单方程回归维度 m = lags*mY + mX + r
#     m = lags * mY + mX + r
#     X_design = np.zeros((T, m))
#     Y_target = np.zeros((T, mY))

#     # 构造设计矩阵
#     for k, t in enumerate(range(t0, t1)):
#         if r > 0:
#             ECT_prev = ECT_all[t - 1, :]
#         else:
#             ECT_prev = np.zeros((0,))
#         X_design[k, :] = _build_X_row(Y_diff, Z_diff, ECT_prev, t, lags, mY, mX, r)
#         Y_target[k, :] = Y_diff[t, :]

#     # ridge 闭式解
#     A = X_design.T @ X_design + ridge * np.eye(m)
#     coef = np.linalg.solve(A, X_design.T @ Y_target)  # (m, mY)

#     # 残差 & R0
#     E = Y_target - X_design @ coef                   # (T, mY)
#     R0 = (E.T @ E) / max(T - m, 1)                   # (mY, mY)

#     # 近似系数方差：σ_j^2 * (X'X + λI)^(-1)
#     A_inv = np.linalg.inv(A)
#     Q0_blocks = []
#     for j in range(mY):
#         sigma2_j = max(R0[j, j], 1e-8)
#         # 对 ECT 的系数可以放大/缩小 process noise（鼓励/抑制 α 的漂移）
#         scale_mat = np.eye(m)
#         if r > 0:
#             ect_slice = slice(lags * mY + mX, lags * mY + mX + r)
#             scale_mat[ect_slice, ect_slice] *= float(q_alpha_scale)
#         Q0_blocks.append(sigma2_j * A_inv @ scale_mat)
#     # blockdiag
#     Q0 = np.zeros((m * mY, m * mY))
#     for j in range(mY):
#         Q0[j*m:(j+1)*m, j*m:(j+1)*m] = Q0_blocks[j]
#     Q0 = (q_scale ** 2) * Q0

#     # θ0：把 (m, mY) 的系数，堆成 (p, 1)
#     theta0 = coef.T.reshape(-1, 1, order='C')  # 方程优先：eq1的m个→eq2的m个→...

#     return theta0, R0, Q0, r, m

# # ---------- Johansen 协整：只用内生变量（levels），窗口内 ----------
# def johansen_beta(Y_level_window: np.ndarray, det_order: int = 0, k_ar_diff: int = 1, rank: int | None = None):
#     """
#     返回 beta_coint: (mY x r)。若 rank=None，用 5% 显著性挑秩（trace test）。
#     det_order: 0 表示常数项在协整关系内（与文献常见设置匹配）
#     """
#     mY = Y_level_window.shape[1]
#     cj = coint_johansen(Y_level_window, det_order, k_ar_diff)
#     # 选择秩
#     if rank is None:
#         # trace_stat 与 临界值比较，找到最大的 r 使得 trace_stat > crit_5%
#         r = 0
#         for i in range(mY):
#             if cj.lr1[i] > cj.cvt[i, 1]:  # 1: 5% 临界值
#                 r = i + 1
#         rank = r
#     if rank == 0:
#         return None  # 无协整
#     beta = cj.evec[:, :rank]  # (mY, r)
#     return beta

# # ---------- 带 ECT 的多滞后 VARX 的 Kalman（tvp-VECM 版） ----------
# def kalman_multilag_filter_vecm(
#     Y: np.ndarray,                    # 这里传 ΔY
#     Z_all: np.ndarray,                # 这里传 ΔZ
#     Y_level_aligned: np.ndarray,      # 与 ΔY 对齐的 level（即原 level[1:]）
#     Q0: np.ndarray,
#     R0: np.ndarray,
#     P0: np.ndarray,
#     lambda_: float,
#     lambda_R: float,
#     lambda_Q: float,
#     beta_coint: np.ndarray | None,    # (mY, r)，可为 None
#     theta0: np.ndarray | None = None, # (p, 1) 初始系数
#     weights: np.ndarray | None = None,
#     std_ew_hist_IN: np.ndarray | float | None = None,
#     dropout0: np.ndarray | None = None,
#     idx_0: int | None = None,
#     lags: int = 2,
#     standardize: bool = True,
#     lambda_norm: float | None = None,
#     eps: float = 1e-8,
#     standardize_ect: bool = True
# ):
#     """
#     模型：ΔY_t = Σ Γ_i ΔY_{t-i} + B ΔZ_{t-1} + α β' Y_{t-1} + ε_t
#     回归向量 X_t = [ΔY_{t-1}..ΔY_{t-lags}, ΔZ_{t-1}, ECT_{t-1}], 维度 m = lags*mY + mX + r
#     状态 θ 维度：p = m*mY（按方程拼接）。
#     """
#     n, mY = Y.shape
#     mX = Z_all.shape[1] if Z_all is not None and Z_all.size > 0 else 0
#     r = 0 if beta_coint is None else beta_coint.shape[1]

#     m = lags * mY + mX + r
#     p = m * mY

#     P = P0.copy()
#     R = R0.copy()
#     Q = Q0.copy()

#     # 可选：固定/放大某些系数的先验不确定性
#     try:
#         if dropout0 is not None:
#             dropout_exp = 3
#             d0 = np.asarray(dropout0).ravel()[:p]
#             idx_dropout = d0 < 1.0
#             if np.any(idx_dropout):
#                 scale = np.maximum(d0[idx_dropout], 1e-3) ** dropout_exp
#                 P[np.ix_(idx_dropout, idx_dropout)] *= scale[:, None] * scale[None, :]
#                 Q[np.ix_(idx_dropout, idx_dropout)] *= scale[:, None] * scale[None, :]
#     except Exception:
#         pass

#     theta = np.zeros((p, 1)) if theta0 is None else theta0.copy()

#     theta_est  = np.zeros((n, p))
#     Y_pred_std = np.full((n, mY), np.nan)
#     Y_pred_raw = np.full((n, mY), np.nan)
#     P_hist     = np.zeros((p, p, n))
#     R_hist     = np.zeros((mY, mY, n))
#     Q_hist     = np.zeros((p, p, n))
#     e_std      = np.full((n, mY), np.nan)
#     e_raw      = np.full((n, mY), np.nan)

#     I_p = np.eye(p)

#     # ---------- 因果标准化的缓冲 ----------
#     if standardize:
#         if lambda_norm is None:
#             lambda_norm = lambda_R
#         eps_floor = eps if 'eps' in locals() else 1e-8

#         mu_Z       = Z_all[1, :].copy() if mX > 0 else None
#         mu_Y       = Y[1, :].copy()
#         mu_Y_prev1 = Y[1, :].copy()
#         mu_Y_prev2 = Y[0, :].copy()
#         var_Z       = np.ones(mX) if mX > 0 else None
#         var_Y       = np.ones(mY)
#         var_Y_prev1 = np.ones(mY)
#         var_Y_prev2 = np.ones(mY)

#         # ECT 的因果标准化
#         if r > 0 and standardize_ect:
#             mu_ECT = (Y_level_aligned[1, :] @ beta_coint).copy()
#             var_ECT = np.ones(r)
#         else:
#             mu_ECT = None
#             var_ECT = None

#         mu_Y_hist  = np.zeros((n, mY))
#         std_Y_hist = np.ones((n, mY))

#     for t in range(2, n):
#         # ----- 计算 ECT_{t-1} -----
#         if r > 0:
#             ECT_prev_raw = (Y_level_aligned[t - 1, :] @ beta_coint)  # (r,)
#             if standardize and standardize_ect:
#                 mu_ECT_old  = mu_ECT.copy()
#                 std_ECT_old = np.sqrt(np.maximum(var_ECT, eps_floor)).copy()
#                 # 先用旧刻度做标准化
#                 ECT_prev = (ECT_prev_raw - mu_ECT_old) / std_ECT_old
#                 # 再更新方差/均值
#                 var_ECT = lambda_norm * var_ECT + (1 - lambda_norm) * (ECT_prev_raw - mu_ECT) ** 2
#                 mu_ECT  = lambda_norm * mu_ECT  + (1 - lambda_norm) * ECT_prev_raw
#             else:
#                 ECT_prev = ECT_prev_raw
#         else:
#             ECT_prev = np.zeros((0,))

#         if standardize:
#             # 因果反标准化的刻度（用上一期的）
#             mu_Y_inv  = mu_Y.copy()
#             std_Y_inv = np.sqrt(np.maximum(var_Y, eps_floor)).copy()

#             # 取当期原始值（都是 Δ 量）
#             Y_raw_t     = Y[t, :].copy()
#             Y_prev1_raw = Y[t - 1, :].copy()
#             Y_prev2_raw = Y[t - 2, :].copy()
#             Z_raw       = Z_all[t - 1, :].copy() if mX > 0 else None

#             # 先更新方差（基于旧均值）
#             var_Y       = lambda_norm * var_Y       + (1 - lambda_norm) * (Y_raw_t     - mu_Y)       ** 2
#             var_Y_prev1 = lambda_norm * var_Y_prev1 + (1 - lambda_norm) * (Y_prev1_raw - mu_Y_prev1) ** 2
#             var_Y_prev2 = lambda_norm * var_Y_prev2 + (1 - lambda_norm) * (Y_prev2_raw - mu_Y_prev2) ** 2
#             if mX > 0:
#                 var_Z = lambda_norm * var_Z + (1 - lambda_norm) * (Z_raw - mu_Z) ** 2

#             # 再更新均值
#             mu_Y       = lambda_norm * mu_Y       + (1 - lambda_norm) * Y_raw_t
#             mu_Y_prev1 = lambda_norm * mu_Y_prev1 + (1 - lambda_norm) * Y_prev1_raw
#             mu_Y_prev2 = lambda_norm * mu_Y_prev2 + (1 - lambda_norm) * Y_prev2_raw
#             if mX > 0:
#                 mu_Z = lambda_norm * mu_Z + (1 - lambda_norm) * Z_raw

#             # 标准差
#             std_Y       = np.sqrt(np.maximum(var_Y, eps_floor))
#             std_Y_prev1 = np.sqrt(np.maximum(var_Y_prev1, eps_floor))
#             std_Y_prev2 = np.sqrt(np.maximum(var_Y_prev2, eps_floor))
#             std_Z       = np.sqrt(np.maximum(var_Z, eps_floor)) if mX > 0 else None

#             # 标准化
#             Y_t_std = (Y_raw_t     - mu_Y_inv)   / std_Y_inv
#             Y_prev1 = (Y_prev1_raw - mu_Y_prev1) / std_Y_prev1
#             Y_prev2 = (Y_prev2_raw - mu_Y_prev2) / std_Y_prev2
#             Z_t     = (Z_raw - mu_Z) / std_Z if mX > 0 else np.zeros((0,))

#             mu_Y_hist[t, :]  = mu_Y_inv
#             std_Y_hist[t, :] = std_Y_inv

#             Y_t = Y_t_std.reshape(-1, 1)
#         else:
#             Y_t     = Y[t, :].reshape(-1, 1)
#             Y_prev1 = Y[t - 1, :]
#             Y_prev2 = Y[t - 2, :]
#             Z_t     = Z_all[t - 1, :] if mX > 0 else np.zeros((0,))

#         # ---------- 构造观测矩阵 H_t ----------
#         X_t = np.concatenate([Y_prev1, Y_prev2], axis=0)
#         if mX > 0:
#             X_t = np.concatenate([X_t, Z_t], axis=0)
#         if r > 0:
#             X_t = np.concatenate([X_t, np.asarray(ECT_prev).reshape(-1)], axis=0)

#         H_t = np.zeros((mY, p))
#         for j in range(mY):
#             idx = slice(j * m, (j + 1) * m)
#             H_t[j, idx] = X_t

#         # 可选约束
#         try:
#             if idx_0 is not None:
#                 H_t[:, idx_0] = 0.0
#                 theta[idx_0, 0] = 0.0
#         except Exception:
#             pass

#         # ---------- 卡尔曼一步（带 ridge 正则） ----------
#         P_pred     = P / lambda_ + Q
#         theta_pred = theta

#         R_inv = np.linalg.inv(R)
#         P_inv = np.linalg.inv(P_pred)

#         # c1, c2, D 使用全局变量（在 main 里设置）
#         A_ = H_t.T @ R_inv @ H_t + c1 * P_inv + c2 * D
#         b_ = H_t.T @ R_inv @ Y_t + c1 * P_inv @ theta_pred

#         theta = np.linalg.solve(A_, b_)
#         P     = np.linalg.inv(A_)

#         # 残差 & Q/R 在线更新
#         e_t = Y_t - H_t @ theta_pred
#         delta_theta = theta - theta_pred

#         R = lambda_R * R + (1.0 - lambda_R) * (e_t @ e_t.T)
#         Q = lambda_Q * Q + (1.0 - lambda_Q) * (delta_theta @ delta_theta.T)

#         # ---------- 保存 ----------
#         theta_est[t, :]  = theta.ravel()
#         Y_pred_std[t, :] = (H_t @ theta_pred).ravel()
#         P_hist[:, :, t]  = P
#         R_hist[:, :, t]  = R
#         Q_hist[:, :, t]  = Q
#         e_std[t, :]      = e_t.ravel()

#         if standardize:
#             yhat_raw         = Y_pred_std[t, :] * std_Y_hist[t, :] + mu_Y_hist[t, :]
#             Y_pred_raw[t, :] = yhat_raw
#             e_raw[t, :]      = (Y[t, :] - yhat_raw)
#         else:
#             Y_pred_raw[t, :] = Y_pred_std[t, :]
#             e_raw[t, :]      = (Y[t, :] - Y_pred_raw[t, :])

#     # 评估指标（与你原版一致）
#     try:
#         if weights is None:
#             weights = np.ones((n, 1))
#         w = np.asarray(weights).reshape(-1, 1)
#         w = w / np.nansum(w)
#         std_ew_hist = np.sqrt(np.nansum((e_std ** 2) * w, axis=0))
#         denom = std_ew_hist_IN
#         if np.isscalar(denom):
#             denom = float(denom)
#         merr = np.nanmean(std_ew_hist / denom)
#     except Exception:
#         merr = np.nan

#     # === 提取 alpha (调整速度) ===
#     alpha_hist = []
#     if r > 0:
#         for t in range(max(2, n-10), n):
#             # 每期最后 r 个系数块对应 α
#             alpha_t = []
#             for j in range(mY):
#                 start = j * m + (lags * mY + mX)
#                 end   = start + r
#                 alpha_t.append(theta_est[t, start:end])
#             alpha_hist.append(np.vstack(alpha_t))
#         # print("\n=== Last 10 α (adjustment speed) matrices ===")


#     # if beta_coint is not None:
#         # print("\n=== β (cointegration vectors) ===")
#         # print(beta_coint)


#     return merr, e_std, e_raw, theta_est, Y_pred_std, Y_pred_raw, P_hist, R_hist, Q_hist

# def kalman_em_pretrain_vecm_using_kf(
#     Y_diff, Z_diff, Y_level_aligned, beta_coint,
#     theta0, Q0, R0, P0,
#     lags=2, max_iter=10, lambda_=0.99, lambda_R=0.99, lambda_Q=0.98,
#     alpha=0.9, beta=0.9, tol=1e-4):
#     """
#     用现有 kalman_multilag_filter_vecm 做 E-M 预训练：
#       - E-step: 调用 kalman_multilag_filter_vecm()
#       - M-step: 用 e_std, Δθ 更新 Q, R
#     """
#     n, mY = Y_diff.shape
#     mX = Z_diff.shape[1]
#     r = 0 if beta_coint is None else beta_coint.shape[1]
#     m = lags * mY + mX + r
#     p = m * mY

#     Q = Q0.copy()
#     R = R0.copy()
#     merr_prev = np.inf

#     for it in range(max_iter):
#         # === E-step: 调用现有KF ===
#         merr, e_std, e_raw, theta_est, *_ = kalman_multilag_filter_vecm(
#             Y=Y_diff, Z_all=Z_diff, Y_level_aligned=Y_level_aligned,
#             Q0=Q, R0=R, P0=P0,
#             lambda_=lambda_, lambda_R=lambda_R, lambda_Q=lambda_Q,
#             beta_coint=beta_coint, theta0=theta0,
#             lags=lags, standardize=False,  # 训练阶段关闭标准化
#             standardize_ect=False
#         )

#         # === M-step: 更新Q,R ===
#         valid_e = ~np.isnan(e_std).any(axis=1)
#         if np.any(valid_e):
#             R_new = np.cov(e_std[valid_e, :].T) + 1e-6*np.eye(mY)
#         else:
#             R_new = R

#         dtheta = np.diff(theta_est, axis=0)
#         if dtheta.shape[0] > 5:
#             Q_new = np.cov(dtheta.T) + 1e-6*np.eye(p)
#         else:
#             Q_new = Q

#         # 平滑更新
#         R = alpha * R + (1 - alpha) * R_new
#         Q = beta  * Q + (1 - beta)  * Q_new

#         print(f"[EM–KF-VECM] iter {it+1:02d}: merr={merr:.6f}")

#         if abs(merr - merr_prev) < tol:
#             print(f"[EM–KF-VECM] converged at iter {it+1}, merr={merr:.6f}")
#             break
#         merr_prev = merr

#     return Q, R


# # ---------- 你的 VARX benchmark（保留） ----------
# def varx_rolling_predict(Y: np.ndarray,
#                          Z: np.ndarray,
#                          lags: int = 2,
#                          window: int | None = None,
#                          ridge: float = 1e-6):
#     n, mY = Y.shape
#     mX = Z.shape[1]
#     m = lags * mY + mX
#     Y_pred = np.full((n, mY), np.nan)
#     e_raw  = np.full((n, mY), np.nan)

#     for t in range(lags + 1, n - 1):
#         if window is None:
#             s0 = lags+1
#         else:
#             s0 = max(lags+1, t - 1 - window + 1)
#         T = (t - 1) - s0 + 1
#         if T <= 0:
#             continue

#         X_train = np.zeros((T, m))
#         Y_train = np.zeros((T, mY))
#         for k, s in enumerate(range(s0, t)):
#             regY = np.concatenate([Y[s - i, :] for i in range(1, lags + 1)], axis=0)
#             X_train[k, :] = np.concatenate([regY, Z[s-1, :]], axis=0)
#             Y_train[k, :] = Y[s, :]

#         A = X_train.T @ X_train + ridge * np.eye(m)
#         B = X_train.T @ Y_train
#         coef = np.linalg.solve(A, B)

#         regY_next = np.concatenate([Y[t + 1 - i, :] for i in range(1, lags + 1)], axis=0)
#         x_next = np.concatenate([regY_next, Z[t, :]], axis=0)

#         yhat = x_next @ coef
#         Y_pred[t + 1, :] = yhat
#         e_raw[t + 1, :] = Y[t + 1, :] - yhat

#     rmse = np.sqrt(np.nanmean(e_raw**2, axis=1))
#     return Y_pred, e_raw, rmse


# # ---------- main script ----------
# c1 = 1.02   # temporal smoothing weight（越接近1越弱）
# c2 = 0.6    # shrinkage weight（L2 收缩）
# # D 在进入 KF 前根据 p 的维度再设（I_p）

# if __name__ == "__main__":
#     fname = "data/df_country_data_climate.xlsx"
#     xls = pd.ExcelFile(fname)
#     sheets = xls.sheet_names

#     sheets_9_of_12 = [
#         "India", "Brazil", "Chile", "Indonesia", "Mexico", "Peru",
#         "Philippines", "South Africa", "Thailand"
#     ]

#     ttls = ['y', 'Dp', 'eq', 'ep', 'r', 'ys', 'Dps', 'eqs', 'rs', 'lrs', 'ensos']
#     ttls_Y = [ttls[i] for i in range(0, 5)]
#     ttls_X = [ttls[i] for i in range(5, 11)]
#     lags = 2
#     mY0 = len(ttls_Y)
#     mX0 = len(ttls_X)
#     # 注意：m0 仅用于画图布局，不含 ECT；模型里的 m 会含 ECT
#     m0 = lags * mY0 + mX0

#     n_country = len(sheets_9_of_12)
#     # E_hists 的第二维在下方根据实际 m(含ECT)动态处理

#     # KF 参数
#     lambda_  = 0.99
#     lambda_R = 0.95
#     lambda_Q = 0.95
#     lambda_e = 1 - 1 / 20

#     # 初始化窗口、ridge、Q 缩放
#     T_init = 60
#     ridge_init = 1e-4
#     q_scale = 0.03      # 初始化 Q0 放大倍数
#     q_alpha_scale = 0.5 # α(ECT系数) 的 process noise 再缩小（更平滑）

#     for k_country, country in enumerate(sheets_9_of_12):
#         print(f"***** Country {country} *****")
#         T = pd.read_excel(fname, sheet_name=country)
#         T_col = list(T.columns)

#         ttl_Y, idx_TY, idx_Y = intersect_stable(ttls_Y, T_col)
#         ttl_X, idx_TX, idx_X = intersect_stable(ttls_X, T_col)
#         mY = len(idx_Y)
#         mX = len(idx_X)

#         # === 读入 levels（可按需启用 STL）===
#         Y_lvl_df = T.iloc[:, idx_Y].copy()
#         X_lvl_df = T.iloc[:, idx_X].copy()

#         Y_levels = Y_lvl_df.to_numpy()
#         X_levels = X_lvl_df.to_numpy()

#         Y_levels_cut = Y_levels[40:, :]
#         X_levels_cut = X_levels[40:, :]
#         Yd = np.diff(Y_levels_cut, axis=0)
#         Xd = np.diff(X_levels_cut, axis=0)
#         Y_level_aligned = Y_levels_cut[1:, :]
#         n, mY = Yd.shape
#         mX = Xd.shape[1]

#         # === 协整：用全序列估计 beta ===
#         beta_coint = None
#         try:
#             if mY >= 2:
#                 beta_coint = johansen_beta(Y_levels, det_order=0, k_ar_diff=1, rank=None)
#         except Exception as e:
#             print("Johansen failed:", e)
#             beta_coint = None


#         r = 0 if beta_coint is None else beta_coint.shape[1]
#         m_model = lags * mY + mX + r  # 单方程回归维度（含 ECT）
#         p = m_model * mY
#         D = np.eye(p)  # ridge 的 D

#         # === 用前 T_init 做 ridge-VECM 初始化（得到 θ0, R0, Q0）===
#         theta0, R0, Q0, r_ck, m_ck = ridge_vecm_init(
#             Y_diff=Yd, Z_diff=Xd, Y_level_aligned=Y_level_aligned,
#             lags=lags, ridge=ridge_init, beta_coint=beta_coint,
#             T_init=n-1, q_scale=q_scale, q_alpha_scale=q_alpha_scale
#         )
#         assert r_ck == r 
#         m_model = m_ck
#         p = m_model * mY

#         # === EM 预训练 Q,R（不含ECT，不更新θ）===
#         print(f"Pretraining Q,R via EM–KF–VECM on full sequence for {country}...")
#         Q_em, R_em = kalman_em_pretrain_vecm_using_kf(
#             Y_diff=Yd, Z_diff=Xd, Y_level_aligned=Y_level_aligned,
#             beta_coint=beta_coint,
#             theta0=theta0, Q0=Q0, R0=R0, P0=np.eye(p),
#             lags=lags, max_iter=10
#         )
#         print(f"EM–KF–VECM pretraining done. Using Q_em, R_em as initialization.")



#         # 先验 P0（与旧版一致）
#         P0 = (1.0) * np.eye(p)

#         # 加权
#         weights = np.exp(-(1 - lambda_e) * np.arange(n, 0, -1)).reshape(-1, 1)
#         weights = weights / np.sum(weights)

#         print(f"\n=== {country}: tvp-VECM Kalman with VARX/VECM init ===")
#         print(f"coint rank r = {r}, m per eq = {m_model}, p = {p}")

#         merr, e_std, e_raw, theta_est, Y_pred_std, Y_pred_raw, \
#         P_hist, R_hist, Q_hist = kalman_multilag_filter_vecm(
#             Y=Yd, Z_all=Xd, Y_level_aligned=Y_level_aligned,
#             Q0=Q_em, R0=R_em, P0=P0,
#             lambda_=lambda_, lambda_R=0.99, lambda_Q=0.98,
#             beta_coint=beta_coint, theta0=theta0,
#             weights=weights, std_ew_hist_IN=None,
#             lags=lags, standardize=True, lambda_norm=lambda_R,
#             standardize_ect=True
#         )

#         print(f"Kalman done. merr={merr:.6f}")

#         ml = np.isfinite(e_std).any(axis=1)
#         w_eff = weights[ml] / np.sum(weights[ml])
#         std_ew_hist = np.sqrt(np.sum((e_std[ml] ** 2) * w_eff, axis=0))


#         # ==== 你的可视化保持不变/少量改动 ====
#         n_tmp = e_std.shape[0]
#         R_tr = np.array([np.trace(R_hist[:, :, t]) / mY for t in range(n_tmp)])
#         P_tr = np.array([np.trace(P_hist[:, :, t]) / p for t in range(n_tmp)])
#         Q_tr = np.array([np.trace(Q_hist[:, :, t]) / p for t in range(n_tmp)])

#         e_raw_full = Yd - Y_pred_raw
#         rmse_raw   = np.sqrt(np.nanmean(e_raw_full**2, axis=1))

#         # ---- Null model: Yhat(t) = Yd(t-1) ----
#         Yhat_null = np.full_like(Yd, np.nan); Yhat_null[1:, :] = Yd[:-1, :]
#         E_null = Yd - Yhat_null
#         rmse_null = np.sqrt(np.nanmean(E_null**2, axis=1))

#         # ---- VARX benchmark ----
#         Yhat_varx_raw, E_varx_raw, rmse_varx_raw = varx_rolling_predict(Yd, Xd, lags=lags)

#         plt.figure(954, figsize=(10, 8))
#         plt.clf()
#         plt.subplot(2, 2, 1); plt.plot(np.sqrt(Q_tr)); plt.title('Q (rms of coeff update noise)')
#         plt.subplot(2, 2, 2); plt.plot(np.sqrt(R_tr)); plt.title('R (rms of obs noise)')
#         plt.subplot(2, 2, 3); plt.plot(np.sqrt(P_tr)); plt.title('P (sqrt mean coeff var)')
#         ax = plt.subplot(2, 2, 4)
#         ax.plot(rmse_raw,        label='KF (tvp-VECM)', alpha=0.9)
#         ax.plot(rmse_varx_raw,   label='VARX raw', linestyle='--', alpha=0.9)
#         ax.plot(rmse_null,       label='Naive lag-1', linestyle=':', alpha=0.8)
#         ax.set_title('Y−Yhat (rms of prediction error)')
#         ax.legend()
#         plt.tight_layout()
#         plt.show()

#         # === 预测-真实相关（沿用你原函数思想，这里简化演示）===

#         def weighted_corr(x, y, w):
#             x = np.asarray(x).ravel()
#             y = np.asarray(y).ravel()
#             w = np.asarray(w).ravel()

#             # --- 自动对齐长度 ---
#             n = min(len(x), len(y), len(w))
#             x, y, w = x[:n], y[:n], w[:n]

#             # --- 过滤 NaN ---
#             m = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(w)
#             if m.sum() < 3:
#                 return np.nan

#             x, y, w = x[m], y[m], w[m]
#             w = w / np.sum(w)

#             # --- 加权均值 ---
#             mx = np.sum(w * x)
#             my = np.sum(w * y)

#             # --- 加权协方差 & 方差 ---
#             cov_xy = np.sum(w * (x - mx) * (y - my))
#             var_x  = np.sum(w * (x - mx) ** 2)
#             var_y  = np.sum(w * (y - my) ** 2)

#             if var_x <= 0 or var_y <= 0:
#                 return np.nan

#             return cov_xy / np.sqrt(var_x * var_y)

#         def _safe_corr(a, b):
#             a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
#             m = ~np.isnan(a) & ~np.isnan(b)
#             if m.sum() < 3: return np.nan
#             return np.corrcoef(a[m], b[m])[0, 1]


#         def plot_pred_corr(Yd, Y_pred, Xd, ttls_Y, ttls_X, country, method):
#             # 统一长度 n-1
#             Y_true = Yd[1:, :]
#             Yhat   = Y_pred[1:, :]
#             Y_lag1 = Yd[:-1, :]
#             Z_lag1 = Xd[:-1, :]

#             # 权重严格对齐
#             w_use = w_eff[-Y_true.shape[0]:]

#             mY = Y_true.shape[1]
#             fig = plt.figure(figsize=(16, 6))
#             gs = fig.add_gridspec(2, mY, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

#             for j in range(mY):
#                 ax_ts = fig.add_subplot(gs[0, j])
#                 ax_ts.plot(Y_true[:, j], color='tab:blue', lw=1.0, label='data')
#                 ax_ts.plot(Yhat[:, j],  color='tab:orange', lw=1.0, ls='--', label='prediction')
#                 ax_ts.set_title(ttls_Y[j], fontsize=11)
#                 if j == 0: ax_ts.legend(loc='upper right', fontsize=9)

#                 ax_sc = fig.add_subplot(gs[1, j])
#                 ax_sc.scatter(Y_true[:, j], Yhat[:, j], s=10, color='tab:purple', alpha=0.6)
#                 lim = np.nanmax(np.abs(np.concatenate([Y_true[:, j], Yhat[:, j]])))
#                 if np.isfinite(lim) and lim > 0:
#                     ax_sc.plot([-lim, lim], [-lim, lim], color='gray', lw=1, ls=':')
#                     ax_sc.set_xlim(-lim, lim); ax_sc.set_ylim(-lim, lim)
#                 ax_sc.set_xlabel('data', fontsize=9)
#                 ax_sc.set_ylabel('prediction', fontsize=9)

#                 # ---- 加权相关 ----
#                 r_pred = weighted_corr(Y_true[:, j], Yhat[:, j], w_use)
#                 r_lag  = weighted_corr(Y_lag1[:, j], Y_true[:, j], w_use)
#                 r_exog_all = np.array([weighted_corr(Z_lag1[:, k], Y_true[:, j], w_use) 
#                                        for k in range(Z_lag1.shape[1])])

#                 if r_exog_all.size > 0 and np.any(np.isfinite(r_exog_all)):
#                     k_best = int(np.nanargmax(np.abs(r_exog_all)))
#                     r_exog = r_exog_all[k_best]
#                     exog_name = ttls_X[k_best] if k_best < len(ttls_X) else f"X{k_best}"
#                 else:
#                     r_exog, exog_name = np.nan, "N/A"

#                 txt = (
#                     f"corr(Y, Yhat) = {r_pred: .3f}\n"
#                     f"corr(Y[-1], Y) = {r_lag: .3f}\n"
#                     f"max corr(Z[-1], Y) = {r_exog: .3f} ({exog_name})"
#                 )
#                 ax_sc.text(0.02, -0.25, txt, transform=ax_sc.transAxes, fontsize=9, va='top')

#             fig.suptitle(f"{country} — {method}", fontsize=14, y=0.98)
#             plt.show()

#         plot_pred_corr(Yd, Y_pred_raw, Xd, ttl_Y, ttls_X, country, "Kalman (tvp-VECM)")
#         Yhat_varx_raw, _, _ = varx_rolling_predict(Yd, Xd, lags=lags)
#         plot_pred_corr(Yd, Yhat_varx_raw, Xd, ttl_Y, ttls_X, country, "VARX")

#         # === drop-one heatmap（含 ECT）===
#         # 横轴标签：
#         labels_x = []
#         for j in range(lags * mY):
#             lag_id = j // mY + 1
#             var_id = j %  mY
#             labels_x.append(f"L{lag_id}_{ttl_Y[var_id]}")
#         for j in range(mX):
#             labels_x.append(ttls_X[j])
#         for j in range(r):
#             labels_x.append(f"ECT{j+1}")

#         # 逐系数 dropout 误差（与旧逻辑一致）
#         E_hists_k = np.full((mY, m_model), np.nan)
#         # 生成权重基准
#         ml = np.isfinite(e_std).any(axis=1)
#         w_eff = weights[ml] / np.sum(weights[ml])
#         std_ew_hist = np.sqrt(np.sum((e_std[ml] ** 2) * w_eff, axis=0))

#         for i_Y in range(mY):
#             for j_YX in range(m_model):
#                 idx_0 = i_Y * m_model + j_YX
#                 dropout0 = np.ones((p,))
#                 dropout0[idx_0] = 0.0
#                 merr_d, *_ = kalman_multilag_filter_vecm(
#                     Y=Yd, Z_all=Xd, Y_level_aligned=Y_level_aligned,
#                     Q0=Q0, R0=R0, P0=P0,
#                     lambda_=lambda_, lambda_R=lambda_R, lambda_Q=lambda_Q,
#                     beta_coint=beta_coint, theta0=theta0,
#                     weights=weights, std_ew_hist_IN=std_ew_hist, dropout0=dropout0,
#                     lags=lags, standardize=True, lambda_norm=lambda_R,
#                     standardize_ect=True
#                 )
#                 E_hists_k[i_Y, j_YX] = merr_d

#         fig, axs = plt.subplots(2, 1, figsize=(12, 6))
#         im = axs[0].imshow(E_hists_k, aspect='auto', vmin=0.95, vmax=1.05, cmap="coolwarm")
#         axs[0].set_yticks(np.arange(mY)); axs[0].set_yticklabels(ttl_Y)
#         axs[0].set_xticks(np.arange(m_model)); axs[0].set_xticklabels(labels_x, rotation=45, ha="right")
#         axs[0].set_title(f"{country}: mean errors (drop-one coeff)")
#         fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

#         vals = E_hists_k.ravel()
#         axs[1].hist(vals[~np.isnan(vals)], bins=20, edgecolor="black")
#         axs[1].set_title(f"{country}: histogram of errors")
#         axs[1].set_xlabel("Error value"); axs[1].set_ylabel("Frequency")
#         plt.tight_layout(); plt.show()

#         # print(f"\n[{country}] --- α / β diagnostics ---")
#         # if beta_coint is not None:
#         #     print("β (co-integration matrix):")
#         #     print(beta_coint)
#         # else:
#         #     print("No cointegration detected.")



# In[ ]:


import seaborn as sns


# In[ ]:


def run_kalman_and_score(Yd, Xd, Y_level_aligned, lags, beta_coint, metric_start=30):
    """复用你当前的 run_tvpkf 输出分数"""
    res = run_tvpkf(Yd, Xd, Y_level_aligned, lags, beta_coint)
    Y_pred = res["Y_pred"]

    corr_list = []
    rmse_list = []

    for j in range(Yd.shape[1]):
        yt = Yd[:, j]
        yp = Y_pred[:, j]
        mask = np.isfinite(yt) & np.isfinite(yp)
        mask[:metric_start] = False
        if mask.sum() > 10:
            corr_list.append(np.corrcoef(yt[mask], yp[mask])[0, 1])
            rmse_list.append(np.sqrt(np.mean((yt[mask] - yp[mask])**2)))
        else:
            corr_list.append(np.nan)
            rmse_list.append(np.nan)

    return corr_list, rmse_list

def evaluate_climate_lags_for_country(
    fname,
    country,
    ttls_Y,
    ttls_X_base,
    climate_df,
    climate_vars,           # e.g. ["PDO","SOI","NINO1+2"]
    baseline_var="NINO3+4", # ⭐ baseline = NINO3+4
    lags_to_test=[1,2,3,4],
    main_lags=2,
    metric_start=30,
):
    """
    baseline = 使用 NINO3+4(lag k)
    对 var × lag 做 Δcorr / Δrmse 评估
    """

    # -------------------------------
    # 1) load country data
    # -------------------------------
    Yd, Xd_macro, Y_level_aligned, year_labels, Y_levels, *_ , ttl_Y = \
        load_country_data(fname, country, ttls_Y, ttls_X_base)

    idx_y = ttl_Y.index("y")
    idx_dp = ttl_Y.index("Dp")

    # -------------------------------
    # 2) 构造 Y_levels_df 对齐日期
    # -------------------------------
    nYd = len(Yd)
    Y_levels_df = pd.DataFrame(Y_levels[-nYd:, :], columns=ttl_Y)

    Y_levels_df.index = year_labels[-nYd:]
    Y_levels_df.index = quarter_index

    # ======================================================
    #             BASELINE（NINO3+4 × lag）
    # ======================================================
    baseline_results = {}

    # 先生成月→季→lag1 的 baseline 气候序列
    df_base_q = prepare_climate_quarterly(climate_df, climate_vars=[baseline_var], normalize=False)

    for lag in lags_to_test:

        # shift lag（季度）
        df_base_lag = df_base_q.shift(lag)

        # merge 到 Y
        merged = merge_Y_with_climate_q(Y_levels_df, climate_q_df=df_base_lag)
        if baseline_var not in merged.columns:
            continue

        base_clim = merged[[baseline_var]].to_numpy()

        # 对齐检查
        if base_clim.shape[0] != Xd_macro.shape[0]:
            continue

        Xd_base = np.hstack([Xd_macro, base_clim])

        # baseline 跑 kalman
        corr_b, rmse_b = run_kalman_and_score(
            Yd, Xd_base, Y_level_aligned, main_lags,
            beta_coint=None,
            metric_start=metric_start
        )

        baseline_results[lag] = (corr_b, rmse_b)

    # ---------------------------------------------------------------
    #      3) 对每个 candidate var × lag 计算 Δcorr 和 Δrmse
    # ---------------------------------------------------------------
    rows = []
    data = []

    for var in climate_vars:

        # 先月→季→lag1
        df_q = prepare_climate_quarterly(climate_df, climate_vars=[var], normalize=False)

        for lag in lags_to_test:

            if lag not in baseline_results:
                continue

            # shift lag
            df_q_lag = df_q.shift(lag)

            # merge
            merged = merge_Y_with_climate_q(Y_levels_df, climate_q_df=df_q_lag)
            if var not in merged.columns:
                continue

            clim_arr = merged[[var]].to_numpy()
            if clim_arr.shape[0] != Xd_macro.shape[0]:
                continue

            Xd_new = np.hstack([Xd_macro, clim_arr])

            # 跑 Kalman
            corr_new, rmse_new = run_kalman_and_score(
                Yd, Xd_new, Y_level_aligned, main_lags,
                beta_coint=None,
                metric_start=metric_start
            )

            # baseline
            corr_b, rmse_b = baseline_results[lag]

            # Δ
            Δcorr_y  = corr_new[idx_y] - corr_b[idx_y]
            Δcorr_dp = corr_new[idx_dp] - corr_b[idx_dp]
            Δrmse_y  = rmse_new[idx_y] - rmse_b[idx_y]
            Δrmse_dp = rmse_new[idx_dp] - rmse_b[idx_dp]

            rows.append(f"{var}, lag{lag}")
            data.append([Δcorr_y, Δcorr_dp, Δrmse_y, Δrmse_dp])

    df_out = pd.DataFrame(
        data,
        index=rows,
        columns=["Δcorr_y", "Δcorr_Dp", "ΔRMSE_y", "ΔRMSE_Dp"]
    )

    return df_out

def plot_climate_compare(df_compare, country="India", baseline="NINO3+4"):
    """
    输入 evaluate_climate_lags_for_country() 返回的 df_compare
    输出右图风格的 Δcorr / ΔRMSE 热力图
    """

    # 颜色风格
    cmap = sns.color_palette("YlGnBu_r", as_cmap=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # -------------------- Δcorr --------------------
    sns.heatmap(
        df_compare[["Δcorr_y", "Δcorr_Dp"]],
        ax=axes[0],
        annot=True, fmt=".3f",
        cmap=cmap,
        center=0,       # ⭐ 关键：使 0 居中，正负对比更清晰
        cbar=False
    )
    axes[0].set_title(f"{country} — Δcorr vs {baseline}")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")


    # -------------------- ΔRMSE --------------------
    sns.heatmap(
        df_compare[["ΔRMSE_y", "ΔRMSE_Dp"]],
        ax=axes[1],
        annot=True, fmt=".3f",
        cmap=cmap,
        center=0,
        cbar=False
    )
    axes[1].set_title(f"{country} — ΔRMSE vs {baseline}")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    plt.show()

for country in sheets_9_of_12:
    df_compare = evaluate_climate_lags_for_country(
        fname=fname,
        country=country,
        ttls_Y=ttls_Y,
        ttls_X_base=ttls_X_base,
        climate_df=climate_df,
        climate_vars=["PDO","SOI","NINO1+2","WestenV"],
        baseline_var="NINO3+4",
        lags_to_test=[1,2,3,4],
    )
    plot_climate_compare(df_compare, country=country, baseline="NINO3+4")



# In[ ]:


def climate_choice_summary(
    fname,
    countries,
    ttls_Y,
    ttls_X_base,
    climate_df,
    choice_dict,      # {country: {"var": "SOI", "lag": 2}, ...}
    main_lags=2,
    metric_start=30,
    baseline_var="NINO3+4"
):
    """
    choice_dict 示例：
    {
        "India": {"var": "SOI", "lag": 2},
        "Chile": {"var": "PDO", "lag": 1},
        "Indonesia": {"var": "NINO1+2", "lag": 3},
    }
    """

    results = []

    for country in countries:

        if country not in choice_dict:
            continue

        var = choice_dict[country]["var"]
        lag = choice_dict[country]["lag"]

        # ===== 载入国家数据 =====
        Yd, Xd_macro, Y_level_aligned, year_labels, Y_levels, *_ , ttl_Y = \
            load_country_data(fname, country, ttls_Y, ttls_X_base)

        idx_y = ttl_Y.index("y")
        idx_dp = ttl_Y.index("Dp")

        # --- baseline NINO3+4 (lag=1 默认) ---
        df_base_q = prepare_climate_quarterly(climate_df, climate_vars=[baseline_var])
        df_base_lag = df_base_q.shift(1)
        merged_base = merge_Y_with_climate_q(
            pd.DataFrame(Y_levels[-len(Yd):], columns=ttl_Y, index=year_labels),
            climate_q_df=df_base_lag
        )
        base_arr = merged_base[[baseline_var]].to_numpy()
        if base_arr.shape[0] != Xd_macro.shape[0]:
            continue

        Xd_base = np.hstack([Xd_macro, base_arr])
        corr_b, rmse_b = run_kalman_and_score(
            Yd, Xd_base, Y_level_aligned, main_lags,
            beta_coint=None, metric_start=metric_start
        )

        # ===== candidate variable =====
        df_q = prepare_climate_quarterly(climate_df, climate_vars=[var])
        df_q_lag = df_q.shift(lag)
        merged_new = merge_Y_with_climate_q(
            pd.DataFrame(Y_levels[-len(Yd):], columns=ttl_Y, index=year_labels),
            climate_q_df=df_q_lag
        )
        if var not in merged_new:
            continue

        clim_arr = merged_new[[var]].to_numpy()
        if clim_arr.shape[0] != Xd_macro.shape[0]:
            continue

        Xd_new = np.hstack([Xd_macro, clim_arr])

        corr_new, rmse_new = run_kalman_and_score(
            Yd, Xd_new, Y_level_aligned, main_lags,
            beta_coint=None,
            metric_start=metric_start
        )

        # ===== Δ 分数 =====
        Δcorr_y  = corr_new[idx_y] - corr_b[idx_y]
        Δcorr_dp = corr_new[idx_dp] - corr_b[idx_dp]
        Δrmse_y  = rmse_new[idx_y] - rmse_b[idx_y]
        Δrmse_dp = rmse_new[idx_dp] - rmse_b[idx_dp]

        results.append([
            country,
            Δcorr_y, Δcorr_dp,
            Δrmse_y, Δrmse_dp
        ])

    df_final = pd.DataFrame(
        results,
        columns=["country", "Δcorr_y", "Δcorr_Dp", "ΔRMSE_y", "ΔRMSE_Dp"]
    ).set_index("country")

    return df_final


def plot_climate_choice_summary(df, title="Climate Choice Summary"):
    """画一张 纵轴 = 国家，横轴 = 指标 的 heatmap"""

    cmap = sns.color_palette("YlGnBu_r", as_cmap=True)

    plt.figure(figsize=(8, max(3, 0.5 * len(df))))
    sns.heatmap(
        df,
        annot=True, fmt=".3f",
        cmap=cmap,
        center=0,
        cbar=False
    )
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.show()

# choice = {
# "India": {"var": "SOI", "lag": 4},
# "Chile": {"var": "SOI", "lag": 4},
# "Indonesia": {"var": "NINO1+2", "lag": 3},
# "Brazil": {"var": "SOI", "lag": 4},
# "Mexico": {"var": "SOI", "lag": 1},
# "Thailand": {"var": "SOI", "lag": 1},
# "Peru": {"var": "SOI", "lag": 1},
# "Philippines": {"var": "SOI", "lag": 1},
# "South Africa": {"var": "SOI", "lag": 1},
# }

choice = {
"India": {"var": "NINO1+2", "lag": 4},
"Indonesia": {"var": "NINO1+2", "lag": 3},
"Brazil": {"var": "NINO1+2", "lag": 2},
"Thailand": {"var": "NINO1+2", "lag": 2},
"South Africa": {"var": "NINO1+2", "lag": 3},
}


df_choice = climate_choice_summary(
    fname=fname,
    countries=choice.keys(),
    ttls_Y=ttls_Y,
    ttls_X_base=ttls_X_base,
    climate_df=climate_df,
    choice_dict=choice,
    baseline_var="NINO3+4",
)

plot_climate_choice_summary(df_choice, "Best Climate Choice Summary")


# In[ ]:


def run_kalman_and_score(Yd, Xd, Y_level_aligned, lags, beta_coint, metric_start=30):
    """复用你当前的 run_tvpkf 输出分数"""
    res = run_tvpkf(Yd, Xd, Y_level_aligned, lags, beta_coint)
    Y_pred = res["Y_pred"]

    corr_list = []
    rmse_list = []

    for j in range(Yd.shape[1]):
        yt = Yd[:, j]
        yp = Y_pred[:, j]
        mask = np.isfinite(yt) & np.isfinite(yp)
        mask[:metric_start] = False
        if mask.sum() > 10:
            corr_list.append(np.corrcoef(yt[mask], yp[mask])[0, 1])
            rmse_list.append(np.sqrt(np.mean((yt[mask] - yp[mask])**2)))
        else:
            corr_list.append(np.nan)
            rmse_list.append(np.nan)

    return corr_list, rmse_list

def evaluate_climate_lags_for_country(
    fname,
    country,
    ttls_Y,
    ttls_X_base,
    climate_df,
    climate_vars,           # e.g. ["PDO","SOI","NINO1+2"]
    baseline_var="NINO3+4", # ⭐ baseline = NINO3+4
    lags_to_test=[1,2,3,4],
    main_lags=2,
    metric_start=30,
):
    """
    baseline = 使用 NINO3+4(lag k)
    对 var × lag 做 Δcorr / Δrmse 评估
    """

    # -------------------------------
    # 1) load country data
    # -------------------------------
    Yd, Xd_macro, Y_level_aligned, year_labels, Y_levels, *_ , ttl_Y = \
        load_country_data(fname, country, ttls_Y, ttls_X_base)

    idx_y = ttl_Y.index("y")
    idx_dp = ttl_Y.index("Dp")

    # -------------------------------
    # 2) 构造 Y_levels_df 对齐日期
    # -------------------------------
    nYd = len(Yd)
    Y_levels_df = pd.DataFrame(Y_levels[-nYd:, :], columns=ttl_Y)

    quarter_index = pd.period_range(start='1979Q1', periods=nYd, freq='Q').to_timestamp('Q')
    Y_levels_df.index = quarter_index

    # ======================================================
    #             BASELINE（NINO3+4 × lag）
    # ======================================================
    baseline_results = {}

    # 先生成月→季→lag1 的 baseline 气候序列
    df_base_q = prepare_climate_quarterly(climate_df, climate_vars=[baseline_var], normalize=False)

    for lag in lags_to_test:

        # shift lag（季度）
        df_base_lag = df_base_q.shift(lag)

        # merge 到 Y
        merged = merge_Y_with_climate_q(Y_levels_df, climate_q_df=df_base_lag)
        if baseline_var not in merged.columns:
            continue

        base_clim = merged[[baseline_var]].to_numpy()

        # 对齐检查
        if base_clim.shape[0] != Xd_macro.shape[0]:
            continue

        Xd_base = np.hstack([Xd_macro, base_clim])

        # baseline 跑 kalman
        corr_b, rmse_b = run_kalman_and_score(
            Yd, Xd_base, Y_level_aligned, main_lags,
            beta_coint=None,
            metric_start=metric_start
        )

        baseline_results[lag] = (corr_b, rmse_b)

    # ---------------------------------------------------------------
    #      3) 对每个 candidate var × lag 计算 Δcorr 和 Δrmse
    # ---------------------------------------------------------------
    rows = []
    data = []

    for var in climate_vars:

        # 先月→季→lag1
        df_q = prepare_climate_quarterly(climate_df, climate_vars=[var], normalize=False)

        for lag in lags_to_test:

            if lag not in baseline_results:
                continue

            # shift lag
            df_q_lag = df_q.shift(lag)

            # merge
            merged = merge_Y_with_climate_q(Y_levels_df, climate_q_df=df_q_lag)
            if var not in merged.columns:
                continue

            clim_arr = merged[[var]].to_numpy()
            if clim_arr.shape[0] != Xd_macro.shape[0]:
                continue

            Xd_new = np.hstack([Xd_macro, clim_arr])

            # 跑 Kalman
            corr_new, rmse_new = run_kalman_and_score(
                Yd, Xd_new, Y_level_aligned, main_lags,
                beta_coint=None,
                metric_start=metric_start
            )

            # baseline
            corr_b, rmse_b = baseline_results[lag]

            # Δ
            Δcorr_y  = corr_new[idx_y] - corr_b[idx_y]
            Δcorr_dp = corr_new[idx_dp] - corr_b[idx_dp]
            Δrmse_y  = rmse_new[idx_y] - rmse_b[idx_y]
            Δrmse_dp = rmse_new[idx_dp] - rmse_b[idx_dp]

            rows.append(f"{var}, lag{lag}")
            data.append([Δcorr_y, Δcorr_dp, Δrmse_y, Δrmse_dp])

    df_out = pd.DataFrame(
        data,
        index=rows,
        columns=["Δcorr_y", "Δcorr_Dp", "ΔRMSE_y", "ΔRMSE_Dp"]
    )

    return df_out


# In[ ]:







