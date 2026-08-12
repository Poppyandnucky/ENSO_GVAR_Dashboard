"""Shared Kalman-filter / EM core used by the structural-break GVAR pipeline.

This module is the single source of truth for:
  - VAR-based burn-in initialization (``init_from_varx_rolling``)
  - the online multi-lag Kalman filter used for the final, post-EM forward
    pass and for diagnostics such as the drop-one-regressor heatmap
    (``kalman_multilag_filter``)
  - the EM E-step (``kf_e_step_store``), RTS smoother (``rts_smoother``),
    and EM M-step (``em_m_step_update``)
  - the EM orchestrator (``run_kf_em``)

The underlying regressor is a plain multi-lag VARX: ``X_t = [Y_{t-1..t-lags},
Z_{t-1}]``. There is no error-correction term anywhere in this module, so
none of the function names use "vecm" -- an earlier, unused implementation
with a real error-correction term (``beta_coint`` / ECT) still exists in
``main.py`` but was dropped when this pipeline was copied out of it.

Covariance updates always use the full (numerically robust) Joseph form:

    P_t = (I - K_t H_t) P_pred (I - K_t H_t)^T + K_t R_t K_t^T

``kalman_multilag_filter`` supports three mutually exclusive covariance
modes selected via ``covariance_mode``:
  - "fixed" (default): use the supplied Q/R as-is, no online updates.
  - "adaptive_q": R stays fixed; Q is updated online each step using
    ``rho_q``.
  - "adaptive_r": Q stays fixed; R is updated online each step using
    ``rho_r``.
Simultaneous online adaptation of both Q and R is intentionally not
supported.
"""

from __future__ import annotations

import warnings

import numpy as np

COVARIANCE_MODES = ("fixed", "adaptive_q", "adaptive_r")


class NumericalInstabilityError(RuntimeError):
    """Raised when a required Kalman/EM matrix operation is numerically invalid."""

    def __init__(
        self,
        *,
        operation: str,
        matrix: str,
        reason: str,
        em_iteration: int | None = None,
        time_index: int | None = None,
        quarter=None,
    ):
        self.operation = operation
        self.matrix = matrix
        self.reason = reason
        self.em_iteration = em_iteration
        self.time_index = time_index
        self.quarter = quarter
        parts = [operation, matrix, reason]
        if em_iteration is not None:
            parts.append(f"EM={em_iteration}")
        if quarter is not None:
            parts.append(str(quarter))
        elif time_index is not None:
            parts.append(f"t={time_index}")
        super().__init__(", ".join(parts))

    def __str__(self) -> str:
        parts = [self.operation, self.matrix, self.reason]
        if self.em_iteration is not None:
            parts.append(f"EM={self.em_iteration}")
        if self.quarter is not None:
            parts.append(str(self.quarter))
        elif self.time_index is not None:
            parts.append(f"t={self.time_index}")
        return ", ".join(parts)


def _validate_covariance_mode(covariance_mode: str) -> None:
    if covariance_mode not in COVARIANCE_MODES:
        raise ValueError(
            f"Unsupported covariance_mode={covariance_mode!r}; "
            f"expected one of {COVARIANCE_MODES}."
        )


def _check_finite(
    value: np.ndarray | float,
    *,
    operation: str,
    matrix: str,
    em_iteration: int | None = None,
    time_index: int | None = None,
    quarter=None,
) -> None:
    arr = np.asarray(value, dtype=float)
    if not np.isfinite(arr).all():
        raise NumericalInstabilityError(
            operation=operation,
            matrix=matrix,
            reason="non-finite",
            em_iteration=em_iteration,
            time_index=time_index,
            quarter=quarter,
        )


def _compute_allowing_fp_warnings(func):
    """Evaluate direct matrix algebra; validity is decided by finite checks after."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return func()


def _kalman_gain(
    P_pred: np.ndarray,
    H_t: np.ndarray,
    S: np.ndarray,
    *,
    em_iteration: int | None = None,
    time_index: int | None = None,
) -> np.ndarray:
    """Compute P H' S^-1 without explicitly inverting S."""
    try:
        K = np.linalg.solve(S, H_t @ P_pred.T).T
    except np.linalg.LinAlgError as exc:
        raise NumericalInstabilityError(
            operation="Kalman gain solve",
            matrix="S",
            reason="singular",
            em_iteration=em_iteration,
            time_index=time_index,
        ) from exc
    _check_finite(K, operation="Kalman gain solve", matrix="K", em_iteration=em_iteration, time_index=time_index)
    return K


# ---------------------------------------------------------------------------
# VAR-based burn-in (runs once, before any EM iterations)
# ---------------------------------------------------------------------------
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
    Note: this is a one-time initialization step (no burn-in trimming here).
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

        theta_t = coef.T.reshape(-1, order="C")  # (p,)
        theta_hist.append(theta_t)

        x_t = np.concatenate(
            [
                np.concatenate([Y[t - i, :] for i in range(1, lags + 1)], axis=0),
                Z[t - 1, :],
            ]
        )
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
        alpha = 0.01  # tuning range is usually around 0.1~0.3
        Q0_full = np.cov(dtheta, rowvar=False)
        Q0 = np.diag(np.diag(Q0_full))
        Q0 = alpha * Q0
        if Q0.ndim == 0:
            Q0 = np.array([[float(Q0)]])
    else:
        Q0 = 1e-4 * np.eye(p)

    if resid_hist.shape[0] >= 2:
        R0 = 0.1 * np.cov(resid_hist, rowvar=False)
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

    _check_finite(Q0, operation="VARX initialization", matrix="Q0")
    _check_finite(R0, operation="VARX initialization", matrix="R0")
    _check_finite(P0, operation="VARX initialization", matrix="P0")
    return theta0, Q0, R0, P0, m, p


# ---------------------------------------------------------------------------
# Final / standalone forward Kalman filter pass (post-EM, or diagnostic use)
# ---------------------------------------------------------------------------
def kalman_multilag_filter(
    Y: np.ndarray,
    Z_all: np.ndarray,
    Q0: np.ndarray,
    R0: np.ndarray,
    P0: np.ndarray,
    theta0: np.ndarray | None = None,  # (p, 1) initial coefficients
    dropout0: np.ndarray | None = None,  # dropout mask (drop-one diagnostics)
    lags: int = 2,
    eps: float = 1e-8,
    covariance_mode: str = "fixed",
    rho_q: float = 0.02,
    rho_r: float = 0.02,
):
    """Plain multi-lag VARX Kalman filter (no error-correction term).

    Regressor: X_t = [Y_{t-1},...,Y_{t-lags}, Z_{t-1}]. Covariance updates
    always use the full Joseph form. ``covariance_mode`` controls whether Q
    and/or R are adapted online during this pass (see module docstring);
    the EM baseline Q/R supplied via Q0/R0 are otherwise used unchanged.
    """
    _validate_covariance_mode(covariance_mode)

    n, mY = Y.shape
    mX = Z_all.shape[1] if Z_all is not None and Z_all.size > 0 else 0

    m = lags * mY + mX
    p = m * mY

    theta = theta0.copy() if theta0 is not None else np.zeros((p, 1))
    P = P0.copy()
    R = R0.copy()
    Q = Q0.copy()

    if dropout0 is not None:
        dropout_exp = 3
        d0 = np.asarray(dropout0).ravel()[:p]
        idx_dropout = d0 < 1.0
        if np.any(idx_dropout):
            scale = np.maximum(d0[idx_dropout], 1e-3) ** dropout_exp
            P[np.ix_(idx_dropout, idx_dropout)] *= scale[:, None] * scale[None, :]
            Q[np.ix_(idx_dropout, idx_dropout)] *= scale[:, None] * scale[None, :]

    theta_est = np.full((n, p), np.nan)
    Y_pred = np.full((n, mY), np.nan)
    P_hist = np.full((p, p, n), np.nan)
    e_raw = np.full((n, mY), np.nan)
    Q_trace = np.full((n,), np.nan)
    R_trace = np.full((n,), np.nan)

    I_p = np.eye(p)

    for t in range(lags + 1, n):
        pieces = [Y[t - i, :] for i in range(1, lags + 1)]
        if mX > 0:
            pieces.append(Z_all[t - 1, :])
        X_t = np.concatenate(pieces)

        H_t = np.zeros((mY, p))
        for j in range(mY):
            idx = slice(j * m, (j + 1) * m)
            H_t[j, idx] = X_t

        theta_pred = theta
        P_pred = P + Q

        S = _compute_allowing_fp_warnings(lambda: H_t @ P_pred @ H_t.T + R)
        _check_finite(S, operation="Kalman innovation covariance", matrix="S", time_index=t)
        K = _kalman_gain(P_pred, H_t, S, time_index=t)

        y_t = Y[t, :].reshape(-1, 1)
        y_hat = H_t @ theta_pred
        innovation = y_t - y_hat

        theta = theta_pred + K @ innovation
        _check_finite(theta, operation="Kalman state update", matrix="theta", time_index=t)

        # Joseph-form covariance update (always).
        IKH = _compute_allowing_fp_warnings(lambda: I_p - K @ H_t)
        P = _compute_allowing_fp_warnings(lambda: IKH @ P_pred @ IKH.T + K @ R @ K.T)
        _check_finite(P, operation="Joseph covariance update", matrix="P", time_index=t)

        theta_est[t, :] = theta.ravel()
        Y_pred[t, :] = y_hat.ravel()
        P_hist[:, :, t] = P
        e_raw[t, :] = innovation.ravel()

        Q_trace[t] = np.trace(Q)
        R_trace[t] = np.trace(R)

        if covariance_mode == "adaptive_r":
            R_new = (1.0 - rho_r) * R + rho_r * (innovation @ innovation.T)
            _check_finite(R_new, operation="adaptive R update", matrix="R", time_index=t)
            R = R_new
        elif covariance_mode == "adaptive_q":
            err2 = float(innovation.T @ innovation) / mY
            tr_Q = np.trace(Q)
            if tr_Q > eps and err2 > 0:
                scale = err2 / tr_Q
                Q = (1.0 - rho_q) * Q + rho_q * scale * Q
        # "fixed": no online update; Q and R stay at their input values.

    valid = ~np.isnan(e_raw).any(axis=1)
    if np.any(valid):
        rmse = np.sqrt(np.nanmean(e_raw[valid] ** 2))
    else:
        rmse = np.nan

    return rmse, e_raw, theta_est, Y_pred, P_hist, Q_trace, R_trace


# ---------------------------------------------------------------------------
# EM E-step: KF filter with full storage (always "fixed" Q/R -- the EM
# baseline is never adapted online; adaptation is a final-pass-only option).
# ---------------------------------------------------------------------------
def kf_e_step_store(
    Y: np.ndarray,
    Z_all: np.ndarray,
    theta0: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    P0: np.ndarray,
    lags: int = 2,
    eps: float = 1e-8,
    em_iteration: int | None = None,
):
    """
    Returns: theta_filt, P_filt, theta_pred, P_pred, H_list, y_list, valid_mask
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
        pieces = [Y[t - i, :] for i in range(1, lags + 1)]
        if mX > 0:
            pieces.append(Z_all[t - 1, :])
        X_t = np.concatenate(pieces)

        H_t = np.zeros((mY, p))
        for j in range(mY):
            idx = slice(j * m, (j + 1) * m)
            H_t[j, idx] = X_t

        theta_pr = theta
        P_pr = P + Q

        y_t = Y[t, :].reshape(-1, 1)
        S = _compute_allowing_fp_warnings(lambda: H_t @ P_pr @ H_t.T + R)
        _check_finite(S, operation="E-step innovation covariance", matrix="S", em_iteration=em_iteration, time_index=t)
        K = _kalman_gain(P_pr, H_t, S, em_iteration=em_iteration, time_index=t)

        innovation = y_t - H_t @ theta_pr
        theta = theta_pr + K @ innovation
        _check_finite(theta, operation="E-step state update", matrix="theta", em_iteration=em_iteration, time_index=t)

        # Joseph-form covariance update (always).
        IKH = _compute_allowing_fp_warnings(lambda: I_p - K @ H_t)
        P = _compute_allowing_fp_warnings(lambda: IKH @ P_pr @ IKH.T + K @ R @ K.T)
        _check_finite(P, operation="E-step Joseph covariance update", matrix="P", em_iteration=em_iteration, time_index=t)

        theta_pred[t, :] = theta_pr.ravel()
        P_pred[t, :, :] = P_pr
        theta_filt[t, :] = theta.ravel()
        P_filt[t, :, :] = P
        H_list[t] = H_t
        y_list[t] = y_t
        valid_mask[t] = True

    return theta_filt, P_filt, theta_pred, P_pred, H_list, y_list, valid_mask


# ---------------------------------------------------------------------------
# RTS smoother
# ---------------------------------------------------------------------------
def rts_smoother(
    theta_filt: np.ndarray,
    P_filt: np.ndarray,
    theta_pred: np.ndarray,
    P_pred: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-8,
    em_iteration: int | None = None,
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
            J_t = np.linalg.solve(P_pr_next.T, P_f.T).T
        except np.linalg.LinAlgError as exc:
            raise NumericalInstabilityError(
                operation="RTS smoother gain solve",
                matrix="P_pr_next",
                reason="singular",
                em_iteration=em_iteration,
                time_index=t,
            ) from exc
        _check_finite(J_t, operation="RTS smoother gain solve", matrix="J_t", em_iteration=em_iteration, time_index=t)
        J_hist[t] = J_t

        x_f = theta_filt[t].reshape(-1, 1)
        x_pr_next = theta_pred[t1].reshape(-1, 1)
        x_sm_next = theta_smooth[t1].reshape(-1, 1)

        x_sm = _compute_allowing_fp_warnings(lambda: x_f + J_t @ (x_sm_next - x_pr_next))
        P_sm = _compute_allowing_fp_warnings(lambda: P_f + J_t @ (P_smooth[t1] - P_pr_next) @ J_t.T)
        _check_finite(x_sm, operation="RTS smoother state update", matrix="theta_smooth", em_iteration=em_iteration, time_index=t)
        _check_finite(P_sm, operation="RTS smoother covariance update", matrix="P_sm", em_iteration=em_iteration, time_index=t)

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
        _check_finite(S, operation="diagnostic innovation covariance", matrix="S", time_index=t)
        try:
            S_inv_v = np.linalg.solve(S, v)
        except np.linalg.LinAlgError as exc:
            raise NumericalInstabilityError(
                operation="diagnostic innovation solve",
                matrix="S",
                reason="singular",
                time_index=t,
            ) from exc
        innovation_score[t] = float((v.T @ S_inv_v).item())

        xf = theta_filt[t]
        xs = theta_smooth[t]
        filter_smoother_gap[t] = float(np.linalg.norm(xs - xf))

    for k in range(1, len(valid_idx)):
        t = valid_idx[k]
        t0 = valid_idx[k - 1]
        coefficient_change[t] = float(np.linalg.norm(theta_smooth[t] - theta_smooth[t0]))

    return innovation_score, coefficient_change, filter_smoother_gap


# ---------------------------------------------------------------------------
# EM M-step: standard smoother-based Q/R update (Shumway-Stoffer style).
# ---------------------------------------------------------------------------
def em_m_step_update(
    Y: np.ndarray,
    H_list: list,
    y_list: list,
    valid_mask: np.ndarray,
    theta_smooth: np.ndarray,
    P_smooth: np.ndarray,
    J_hist: np.ndarray,
    Q_old: np.ndarray,
    R_old: np.ndarray,
    em_damping: float = 0.0,
    eps: float = 1e-8,
    em_iteration: int | None = None,
):
    """
    Update Q and R using smoother outputs.
    EM moments used here:
      R <- sample mean of E[(y-Hx)(y-Hx)' + HPH']
      Q <- sample mean of E[(x_t-x_{t-1})(x_t-x_{t-1})']
           using the RTS lag-one smoothed covariance.
    """
    n, p = theta_smooth.shape
    mY = Y.shape[1]

    R_acc = np.zeros((mY, mY))
    cntR = 0
    for t in np.where(valid_mask)[0]:
        H_t = H_list[t]
        y_t = y_list[t]
        if H_t is None or y_t is None:
            continue
        x_t = theta_smooth[t].reshape(-1, 1)
        resid = y_t - H_t @ x_t
        R_t = _compute_allowing_fp_warnings(lambda: resid @ resid.T + H_t @ P_smooth[t] @ H_t.T)
        _check_finite(R_t, operation="EM R update term", matrix="R_t", em_iteration=em_iteration, time_index=t)
        R_acc += R_t
        cntR += 1
    if cntR > 0:
        R_new = R_acc / cntR
    else:
        raise NumericalInstabilityError(
            operation="EM R average",
            matrix="R",
            reason="no valid terms",
            em_iteration=em_iteration,
        )

    valid_idx = np.where(valid_mask)[0]
    Q_acc = np.zeros((p, p))
    cntQ = 0

    for k in range(1, len(valid_idx)):
        t = valid_idx[k]
        t0 = valid_idx[k - 1]

        x_t = theta_smooth[t]
        x_prev = theta_smooth[t0]

        d = (x_t - x_prev).reshape(-1, 1)
        P_t_t0 = _compute_allowing_fp_warnings(lambda: P_smooth[t] @ J_hist[t0].T)
        Q_t = _compute_allowing_fp_warnings(lambda: d @ d.T + P_smooth[t] + P_smooth[t0] - P_t_t0 - P_t_t0.T)
        _check_finite(Q_t, operation="EM Q update term", matrix="Q_t", em_iteration=em_iteration, time_index=t)
        Q_acc += Q_t
        cntQ += 1

    if cntQ > 0:
        Q_new = Q_acc / cntQ
    else:
        raise NumericalInstabilityError(
            operation="EM Q average",
            matrix="Q",
            reason="no valid terms",
            em_iteration=em_iteration,
        )

    Q = em_damping * Q_old + (1.0 - em_damping) * Q_new
    _check_finite(Q, operation="EM Q average", matrix="Q", em_iteration=em_iteration)

    R = em_damping * R_old + (1.0 - em_damping) * R_new
    _check_finite(R, operation="EM R average", matrix="R", em_iteration=em_iteration)

    return Q, R


def _kf_log_likelihood(H_list, y_list, valid_mask, theta_pred, P_pred, R, eps=1e-8):
    """Gaussian one-step-ahead log-likelihood, summed over valid t (EM convergence diagnostic)."""
    mY = R.shape[0]
    ll = 0.0
    cnt = 0
    for t in np.where(valid_mask)[0]:
        H_t = H_list[t]
        y_t = y_list[t]
        if H_t is None or y_t is None:
            continue
        theta_pr = theta_pred[t].reshape(-1, 1)
        v = y_t - H_t @ theta_pr
        S = H_t @ P_pred[t] @ H_t.T + R
        _check_finite(S, operation="log-likelihood covariance", matrix="S", time_index=t)
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            raise NumericalInstabilityError(
                operation="log-likelihood covariance",
                matrix="S",
                reason="singular",
                time_index=t,
            )
        try:
            S_inv_v = np.linalg.solve(S, v)
        except np.linalg.LinAlgError as exc:
            raise NumericalInstabilityError(
                operation="log-likelihood solve",
                matrix="S",
                reason="singular",
                time_index=t,
            ) from exc
        ll += -0.5 * (logdet + float((v.T @ S_inv_v).item()) + mY * np.log(2 * np.pi))
        cnt += 1
    return ll if cnt > 0 else np.nan


# ---------------------------------------------------------------------------
# EM orchestrator: VAR init once, then iterate E-step / smoother / M-step.
# ---------------------------------------------------------------------------
def run_kf_em(
    Y: np.ndarray,
    Z: np.ndarray,
    lags: int = 2,
    window: int = 40,
    ridge: float = 1e-6,
    max_em_iter: int = 10,
    tol: float = 1e-4,
    em_damping: float = 0.0,
    eps: float = 1e-8,
    verbose: bool = True,
    update_P0: bool = False,
):
    """Run the EM loop and a final forward pass.

    ``update_P0`` refreshes P0 from the first valid smoothed covariance without
    clipping, flooring, jitter, or fallback. theta0 is always refreshed from
    the first valid smoothed state.
    """
    # One-time VAR-based burn-in.
    theta0, Q, R, P0, m, p = init_from_varx_rolling(
        Y=Y, Z=Z, lags=lags, window=window, ridge=ridge, eps=eps
    )
    P0_init = P0.copy()

    history = {
        "trace_Q": [],
        "trace_R": [],
        "theta0_norm": [],
        "trace_P0": [],
        "eig_min_Q": [],
        "eig_max_Q": [],
        "eig_min_R": [],
        "eig_max_R": [],
        "eig_min_P0": [],
        "eig_max_P0": [],
        "theta0_change_norm": [],
        "theta0_change_std": [],  # ||delta theta0|| relative to sqrt(diag(P_smooth[0]))
        "log_likelihood": [],
        "obj": [],
    }

    last_obj = np.inf
    best_pack = None

    for it in range(max_em_iter):
        theta_filt, P_filt, theta_pred, P_pred, H_list, y_list, valid_mask = kf_e_step_store(
            Y=Y,
            Z_all=Z,
            theta0=theta0,
            Q=Q,
            R=R,
            P0=P0,
            lags=lags,
            eps=eps,
            em_iteration=it + 1,
        )

        theta_smooth, P_smooth, J_hist = rts_smoother(
            theta_filt,
            P_filt,
            theta_pred,
            P_pred,
            valid_mask,
            eps=eps,
            em_iteration=it + 1,
        )

        Q_new, R_new = em_m_step_update(
            Y=Y,
            H_list=H_list,
            y_list=y_list,
            valid_mask=valid_mask,
            theta_smooth=theta_smooth,
            P_smooth=P_smooth,
            J_hist=J_hist,
            Q_old=Q,
            R_old=R,
            em_damping=em_damping,
            eps=eps,
            em_iteration=it + 1,
        )

        valid_idx = np.where(valid_mask)[0]
        theta0_prev = theta0
        if len(valid_idx) > 0:
            theta0 = theta_smooth[valid_idx[0]].reshape(-1, 1)
            if update_P0:
                P0 = P_smooth[valid_idx[0]]
                _check_finite(P0, operation="P0 refresh", matrix="P0", em_iteration=it + 1, time_index=int(valid_idx[0]))

        Q, R = Q_new, R_new

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

        ll = _kf_log_likelihood(H_list, y_list, valid_mask, theta_pred, P_pred, R, eps=eps)

        theta0_change_norm = float(np.linalg.norm(theta0 - theta0_prev))
        if len(valid_idx) > 0:
            diag0 = np.diag(P_smooth[valid_idx[0]])
            if np.all(diag0 > 0):
                sd0 = np.sqrt(diag0)
                theta0_change_std = float(np.linalg.norm((theta0 - theta0_prev).ravel() / sd0))
            else:
                theta0_change_std = np.nan
        else:
            theta0_change_std = np.nan

        ev_q = np.linalg.eigvalsh(0.5 * (Q + Q.T))
        ev_r = np.linalg.eigvalsh(0.5 * (R + R.T))
        ev_p0 = np.linalg.eigvalsh(0.5 * (P0 + P0.T))

        history["trace_Q"].append(float(np.trace(Q)))
        history["trace_R"].append(float(np.trace(R)))
        history["theta0_norm"].append(float(np.linalg.norm(theta0)))
        history["trace_P0"].append(float(np.trace(P0)))
        history["eig_min_Q"].append(float(ev_q.min()))
        history["eig_max_Q"].append(float(ev_q.max()))
        history["eig_min_R"].append(float(ev_r.min()))
        history["eig_max_R"].append(float(ev_r.max()))
        history["eig_min_P0"].append(float(ev_p0.min()))
        history["eig_max_P0"].append(float(ev_p0.max()))
        history["theta0_change_norm"].append(theta0_change_norm)
        history["theta0_change_std"].append(theta0_change_std)
        history["log_likelihood"].append(ll)
        history["obj"].append(obj)

        if verbose:
            print(
                f"[EM] iter={it + 1:02d}, obj={obj:.6f}, ll={ll:.3f}, "
                f"trQ={np.trace(Q):.6e}, trR={np.trace(R):.6e}"
            )

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
                print(f"[EM] converged at iter={it + 1}, |delta obj|={abs(last_obj - obj):.3e}")
            break
        last_obj = obj

    # Final forward pass, using the EM-converged theta0/Q/R/P0, "fixed" mode
    # (no online adaptation) -- this is what the dashboard/forecast consume.
    rmse, e_raw, theta_est, Y_pred, P_hist, Q_trace, R_trace = kalman_multilag_filter(
        Y=Y,
        Z_all=Z,
        Q0=Q,
        R0=R,
        P0=P0,
        theta0=theta0,
        lags=lags,
        eps=eps,
        covariance_mode="fixed",
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
        "theta_est": theta_est,  # full filtered trajectory of the final pass
        "theta_trace": theta_est,  # explicit alias, kept for forecast reconstruction/plots
        "Y_pred": Y_pred,
        "P_hist": P_hist,
        "Q_trace": Q_trace,
        "R_trace": R_trace,
        "theta0": theta0,
        "Q": Q,
        "R": R,
        "P0": P0,
        "P0_init": P0_init,
        "update_P0": update_P0,
        "m": m,
        "p": p,
        "em_history": history,
        "e_step_store": best_pack,
        "innovation_score": innov_score,
        "coefficient_change": coef_change,
        "filter_smoother_gap": fs_gap,
    }
