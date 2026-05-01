# models/kalman_var.py
#
# Numerical implementation of TVP-VAR(X) estimation using a Kalman filter.
#
# Responsibilities:
#   - State-space formulation of VAR coefficients
#   - Kalman prediction and update steps
#   - Stable covariance updates
#
# This module performs NO data preparation and NO plotting.
# All inputs are assumed to be numpy arrays.
import numpy as np
from scipy.linalg import block_diag
from statsmodels.tsa.vector_ar.var_model import VAR

def make_var_design(y, xexo=None, p=1, include_const=False):
    """
    Construct VAR(p) design matrix using numpy arrays.

    Parameters
    ----------
    y : (T, k) ndarray
        Endogenous variables
    xexo : (T, q) ndarray or None
        Exogenous variables
    p : int
        VAR lag order

    Returns
    -------
    Z : (T-p, m) ndarray
        Design matrix [y_{t-1}, ..., y_{t-p}, x_t]
    Yt : (T-p, k) ndarray
        Dependent variables
    """
    y = np.asarray(y)
    T, k = y.shape

    if xexo is not None:
        xexo = np.asarray(xexo)
        q = xexo.shape[1]
    else:
        q = 0

    Z = []
    Yt = []

    for t in range(p, T):
        z_t = []

        # lagged endogenous variables
        for i in range(1, p + 1):
            z_t.append(y[t - i])   # (k,)

        # contemporaneous exogenous variables
        if xexo is not None:
            z_t.append(xexo[t])    # (q,)

        if include_const:
            z_t.append(np.array([1.0]))  # ADD INTERCEPT

        Z.append(np.hstack(z_t))
        Yt.append(y[t])
        # after building z_t (lags + exog)

    Z = np.asarray(Z)
    Yt = np.asarray(Yt)

    return Z, Yt

def tvp_var_with_exog( Y, Xexo, p, lam=1.0, R=1.0, ridge=1.0, beta0=None, P0=None,
                       include_const=False ):
    """
    tvp-VARX via Kalman filter / discounted RLS.
    ----------
    Y : (T, k) array       Endogenous variables
    Xexo : (T, q) array    Exogenous variables
    p : int                VAR lag order
    lam : float            Forgetting factor (1.0 = no forgetting)
    R : float or (k,k)     Measurement noise scale
    ridge : float          Prior strength
    -------
    beta_path : (T_eff, m, k) Time path of coefficients
    Z : (T_eff, m)            Design matrix
    """
    Y = np.asarray(Y)
    Xexo = np.asarray(Xexo)
    Z, Yt = make_var_design(Y, Xexo, p, include_const=include_const)
    T_eff, k = Yt.shape
    m = Z.shape[1]
    n = m * k
    # --- initialize state at zero (or override upstream)
    if beta0 is None:
        beta_vec = np.zeros((n, 1))
    else:
        beta_vec = beta0.reshape(-1, 1, order="F") # matrix -> col vector MATLAB/FORTRAN order
    # DEBUG: check if P0 is zero
    if P0 is None:
        P = np.eye(n) / ridge
    else:
        P = P0.copy()
    # --- measurement noise
    if np.isscalar(R):
        R_mat = np.eye(k) * R
    else:
        R_mat = np.asarray(R)
    I_n = np.eye(n)
    beta_path = np.zeros((T_eff, m, k))

    for t in range(T_eff):
        z = Z[t:t+1, :]
        y = Yt[t:t+1, :].T
        H = np.kron(np.eye(k), z)
        P = P / lam # prediction
        nu = y - H @ beta_vec # innovation
        S = H @ P @ H.T + R_mat
        K = P @ H.T @ np.linalg.solve(S, np.eye(k))
        beta_vec = beta_vec + K @ nu # update
        KH = K @ H
        P = (I_n - KH) @ P @ (I_n - KH).T + K @ R_mat @ K.T
        beta_path[t] = beta_vec.reshape(m, k, order="F")

    return beta_path, Z

def tvp_var_with_exog_OLD(Y, Xexo, p, lam=0.98, R=1.0, ridge=1e-2):
    """
    Y     : (T, k) endogenous
    Xexo  : (T, q) exogenous
    p     : VAR lag order
    lam   : forgetting factor
    """
    q = Xexo.shape[1]
    Z, Yt = make_var_design(Y, Xexo, p)
    T_eff, k = Yt.shape
    m = Z.shape[1]           # regressors per equation

    # initialize
    beta = np.zeros((m, k))

    # OVERRIDE
    # ridge = 1e-0
    lam   = 0.995

    # Estimate R
    res_var = VAR(endog=Yt, exog=Z[:,-q+1:]).fit(p)

    # CHECK STABILITY
    # A = res_var.coefs  # (p, k, k)
    # p, k, _ = A.shape
    #
    # # companion matrix
    # F = np.zeros((p * k, p * k))
    # F[:k, :] = np.hstack(A)
    # if p > 1:
    #     F[k:, :-k] = np.eye((p - 1) * k)
    #
    # eigmax = np.max(np.abs(np.linalg.eigvals(F)))
    # print("Fixed VAR max |eig| =", eigmax)

    beta_var = res_var.params  # shape (m, k)
    cov_beta = res_var.cov_params()  # (m*k, m*k)
    Sigma_eps = np.cov(res_var.resid.T)  # k × k
    R_mat = R * Sigma_eps * 100
    # var_y = np.var(Yt, axis=0)
    # R_mat = R * np.diag(var_y) * 100
    # R_mat = np.eye(k) * R

    # Estimate P
    # make sure it is not singular
    eps = 1e-6 * np.mean(np.diag(cov_beta))
    P0 = cov_beta + eps * np.eye(cov_beta.shape[0])
    shrink = 0.01  # or 0.05, 0.01
    P0 = shrink * P0

    beta = beta_var.copy()
    # use 0.001 to increase prior strength
    # prior_var = 0.01 * np.var(beta)
    # Strong prior around VAR coefficients
    # P = np.eye(m * k) * prior_var
    # P = np.eye(m * k) / ridge
    P = P0.copy()

    beta_path = []

    for t in range(T_eff):
        z = Z[t:t+1]
        z_rep = np.repeat(z, k, axis=0) # (k, m)

        # Unpack columns as separate arguments using the * operator
        z_mat = block_diag(*z_rep) # (k, m*k)
        y = Yt[t:t+1] # (1, k)

        # forgetting
        P = P / lam

        # Kalman gain
        S = z_mat @ P @ z_mat.T + R_mat
        # S(k,k) = z_mat(k,m*k) * P(m*k,m*k) * z_mat.T(m*k,k) + R_mat(k,k)
        # K = P @ z.T / S
        K = P @ z_mat.T @ np.linalg.inv(S)
        # K(m*k,k)  = P(m*k,m*k) * z_mat.T(m*k,k) * inv(S)(k,k)

        # update
        err = y - z @ beta
        # err(1,k) = y(1,k) - z(1,m) * beta(m,k)
        beta_vec_del = K @ err.T
        # beta_vec_del(m*k,1) = K(m*k,k) * err.T(k,1)
        beta = beta + np.reshape(beta_vec_del, (m,k))
        P = (np.eye(m*k) - K @ z_mat) @ P
        # P(m*k,m*k) = (I(m*k,m*k) - K(m*k,k) * z_mat(k,m*k)) * P(m*k,m*k)
        # beta = beta + K @ err
        # P = (np.eye(m) - K @ z) @ P

        beta_path.append(beta.copy())

    return np.array(beta_path), Z
