# analysis/validation/simulate_data.py
#
# Synthetic data generators for validating VAR / TVP-VAR / GVAR logic.
# These functions generate data with known coefficients so that
# estimated beta paths can be compared to truth.

import numpy as np
import pandas as pd

from models.config import DOMESTIC_VARS, FOREIGN_VARS, EXTERNAL_VARS

def simulated_panel_full_system( T=3000, seed=123 ):
    rng = np.random.default_rng(seed)
    all_vars = DOMESTIC_VARS + FOREIGN_VARS + EXTERNAL_VARS
    k = len(all_vars)
    # --- Stable VAR(1) coefficient matrix
    A = 0.4 * np.eye(k)
    A += 0.05 * rng.standard_normal((k, k))
    # ensure stability
    eigvals = np.linalg.eigvals(A)
    A /= max(1.2 * np.max(np.abs(eigvals)), 1.0)
    # --- Full-rank noise covariance
    Sigma = 0.1 * np.eye(k) + 0.02 * np.ones((k, k))
    # --- Simulate
    X = np.zeros((T, k))
    eps = rng.multivariate_normal(np.zeros(k), Sigma, size=T)

    for t in range(1, T):
        X[t] = A @ X[t - 1] + eps[t]

    df = pd.DataFrame(X, columns=all_vars)
    df["country"] = "SIM"
    df["quarter"] = np.arange(T)

    return df

def simulated_panel(df_sim, country="SIM"):
    df = df_sim.copy()
    df["country"] = country
    df["quarter"] = np.arange(len(df))
    # Rename simulated core variables
    df = df.rename(columns={ "y1": "GDP_YoY", "y2": "CPI_YoY", "x1": "ENSO" })

    # --- Need noise so that the covariance matrix of reduced-form residuals is positive definite
    eps = 1e-4

    for col in DOMESTIC_VARS + FOREIGN_VARS + EXTERNAL_VARS:
        if col not in df.columns:
            df[col] = eps * np.random.randn(len(df))

    return df

def simulate_var2_with_break( T=2000, break_frac=0.5, seed=123 ):
    """
    Simulate a stable VAR(2) with one exogenous variable and a break in A1.
    A2 and B remain fixed.

    Returns
    -------
    df : DataFrame with columns y1, y2, x1
    true_params : dict with A1_pre, A1_post, A2, B, break_index
    """
    rng = np.random.default_rng(seed)

    k = 2
    q = 1
    p = 2

    # --- True coefficients (pre-break)
    A1_pre = np.array([[0.5, 0.1],
                       [0.0, 0.4]])
    A2 = np.array([[-0.2, 0.0],
                   [0.0, -0.1]])

    # --- Post-break: change A1 slightly
    A1_post = A1_pre + np.array([[0.15, -0.05],
                                 [0.05, -0.10]])

    B = np.array([[0.3],
                  [0.1]])

    Sigma_eps = np.array([[0.5, 0.1],
                          [0.1, 0.3]])

    Y = np.zeros((T, k))
    X = rng.normal(size=(T, q))
    eps = rng.multivariate_normal(np.zeros(k), Sigma_eps, size=T)

    t_break = int(T * break_frac)

    for t in range(2, T):
        A1 = A1_pre if t < t_break else A1_post
        Y[t] = (
            A1 @ Y[t - 1]
            + A2 @ Y[t - 2]
            + (B @ X[t].reshape(-1, 1)).ravel()
            + eps[t]
        )

    df = pd.DataFrame(np.hstack([Y, X]), columns=["y1", "y2", "x1"])

    true_params = {
        "A1_pre": A1_pre,
        "A1_post": A1_post,
        "A2": A2,
        "B": B,
        "break_index": t_break,
    }

    return df, true_params

def simulate_var2_fixed( T=1000, seed=123 ):
    """
    Simulate a stable VAR(2) with fixed coefficients and one exogenous variable.

    Returns
    -------
    df : pandas.DataFrame
        Columns: y1, y2, x1
    true_params : dict
        True coefficient matrices
    """
    rng = np.random.default_rng(seed)

    k = 2
    q = 1
    p = 2

    A1 = np.array([[0.5, 0.1],
                   [0.0, 0.4]])
    A2 = np.array([[-0.2, 0.0],
                   [0.0, -0.1]])

    B = np.array([[0.3],
                  [0.1]])

    Sigma_eps = np.array([[0.5, 0.1],
                           [0.1, 0.3]])

    Y = np.zeros((T, k))
    X = rng.normal(size=(T, q))
    eps = rng.multivariate_normal(np.zeros(k), Sigma_eps, size=T)

    for t in range(2, T):
        Y[t] = (
            A1 @ Y[t - 1]
            + A2 @ Y[t - 2]
            + (B @ X[t].reshape(-1, 1)).ravel()
            + eps[t]
        )

    df = pd.DataFrame(
        np.hstack([Y, X]),
        columns=["y1", "y2", "x1"]
    )

    true_params = {
        "A1": A1,
        "A2": A2,
        "B": B,
    }

    return df, true_params