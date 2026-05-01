# analysis/estimate_var.py
#
# Country-level VAR and TVP-VAR estimation routines.
#
# Responsibilities:
#   - Select variables and clean data
#   - Fit fixed VAR models (statsmodels)
#   - Call tvp.kalman_var for TVP-VAR estimation
#
# This module defines how models are estimated,
# not how coefficients evolve numerically.

import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from models.config import DOMESTIC_VARS, FOREIGN_VARS, EXTERNAL_VARS
from analysis.results import TVPVARResult
from models.kalman_var import tvp_var_with_exog
from analysis.var_spec import VARSpec

def standardize_data(Y, X):
    """
    Standardize Y and X columnwise.
    Returns standardized data and scaling matrices.
    """
    sigma_y = Y.std(axis=0, ddof=1)
    sigma_x = X.std(axis=0, ddof=1) if X is not None else None

    Dy = np.diag(sigma_y)
    Dx = np.diag(sigma_x) if sigma_x is not None else None

    Y_std = Y / sigma_y
    X_std = X / sigma_x if X is not None else None

    return Y_std, X_std, Dy, Dx

def build_kalman_prior_cov(res_var, Xexo, p, eps_scale):
    """
    Build Kalman-compatible prior covariance P0 from statsmodels VAR results.
    """
    k = res_var.neqs
    q = Xexo.shape[1] if Xexo is not None else 0

    cov_sm = res_var.cov_params()  # statsmodels ordering

    blocks = []

    # --- lag blocks
    for lag in range(p):
        # statsmodels stores lag blocks per equation
        block = np.zeros((k, k))

        for i in range(k):       # equation
            for j in range(k):   # variable
                idx = (
                    i * (p * k + q)      # equation offset
                    + lag * k
                    + j
                )
                block[j, i] = cov_sm[idx, idx]

        blocks.append(block)

    # --- exogenous block
    if q > 0:
        B_block = np.zeros((q, k))
        for i in range(k):
            for j in range(q):
                idx = (
                    i * (p * k + q)
                    + p * k
                    + j
                )
                B_block[j, i] = cov_sm[idx, idx]
        blocks.append(B_block)

    # stack into Kalman ordering
    P0_diag = np.hstack([b.flatten(order="F") for b in blocks])
    P0 = np.diag(P0_diag)

    eps = eps_scale * np.mean(P0_diag)
    P0 += eps * np.eye(P0.shape[0])

    return P0

def tvp_var_with_exog_and_var_prior( Y, Xexo, p, lam, R, shrink=1e-3, eps_scale=1e-8,
    prior_var_perturb_scale = 0, prior_var_n_data = 100, include_const = False ):
    """
    Run TVP-VAR anchored to a fixed VAR prior.
      add intercept
    """
    res_var = VAR(Y[:prior_var_n_data], exog=Xexo[:prior_var_n_data]).fit(p, trend="n")
    k = res_var.neqs
    q = Xexo.shape[1]
    A_blocks = res_var.coefs  # (p, k, k)
    beta_lags = np.vstack(A_blocks)  # (p*k, k)

    params = res_var.params  # (p*k + q, k)
    print(f"q, k: {q}, {k}") # q=3, k=4, p=2 (p*k=8 endogenous, q=3 exogenous, k=4 equations)
    print(f"params.shape: {params.shape}") # (11,4)
    print(f"beta_lags.shape: {beta_lags.shape}") # (8,4)

    if q > 0:
        # B = params[p * k: p * k + q, :]  # (q, k)
        B = params[:q, :]  # (q, k), params = [B A]'
        # beta0 = np.vstack([beta_lags.T, B])  # (p*k + q, k) [(4,8), (3,4)]
        A = params[q:,:]
        beta0 = np.vstack([A,B])
    else:
        # beta0 = beta_lags
        beta0 = params

    # ADD INTERCEPT constant term (one per equation)
    if include_const:
        const_row = np.zeros((1, k))  # or VAR-estimated intercepts if you want
        beta0 = np.vstack([beta0, const_row])

    # eps_scale = addition to the diagonal elements
    # shrink = scale for P0
    P0 = build_kalman_prior_cov( res_var, Xexo=Xexo, p=p, eps_scale=eps_scale )

    # ADD INTERCEPT
    if include_const:
        old_n = (p * k + q) * k
        new_n = (p * k + q + 1) * k
        P0_new = np.zeros((new_n, new_n))
        P0_new[:old_n, :old_n] = P0
        # small variance for intercept
        intercept_var = 1.0
        for eq in range(k):
            idx = old_n + eq
            P0_new[idx, idx] = intercept_var
        P0 = P0_new

    P0 *= shrink # reduce magnitude of P0

    rng = np.random.default_rng(0)
    beta0_vec = beta0.reshape(-1, 1, order="F")
    delta = rng.multivariate_normal( mean=np.zeros(beta0_vec.shape[0]), cov=P0 ).reshape(-1, 1)
    beta0 = beta0_vec + delta * prior_var_perturb_scale
    beta_path, Z = tvp_var_with_exog(Y, Xexo, p, lam=lam, R=R, beta0=beta0, P0=P0,
                                     include_const=include_const)

    return beta_path, Z

def fit_country_varx_TVP(panel, country, spec: VARSpec,
                         R=1.0, lam=0.98, prior_var=False):

    df = panel[panel["country"] == country].set_index("quarter")

    y = df[spec.endog_vars]
    X = df[spec.exog_vars]

    df2 = pd.concat([y, X], axis=1)
    df2 = df2.dropna(subset=spec.endog_vars + spec.exog_vars)
    df2.index.freq = "QS-OCT"

    if len(df2) < 40:
        print(f"{country}: insufficient data ({len(df2)} rows)")
        return None

    Y = df2[spec.endog_vars].values
    X = df2[spec.exog_vars].values

    # --- save for MATLAB
    from scipy.io import savemat
    savemat( "var_input_data.mat",
        {
            "Y": Y.values if hasattr(Y, "values") else Y,
            "X": X.values if hasattr(X, "values") else X,
            "Y_names": spec.endog_vars,
            "X_names": spec.exog_vars,
            "p": spec.p },
        do_compression=True,
    )

    # --------------------------------------------------
    # 🔑 EXACTLY AS IN WORKING VERSION
    # --------------------------------------------------
    Y_std, X_std, Dy, Dx = standardize_data(Y, X)

    # --- fixed VAR (sigma_u only)
    res_var = VAR(endog=Y_std, exog=X_std).fit(spec.p, trend="n")
    sigma_u = res_var.sigma_u.copy()

    # --- measurement noise (scalar, NOT scaled)
    R_mat = R

    # --- TVP-VAR
    if prior_var:
        # Y, Xexo, p, lam, R, shrink=1e-3, eps_scale=1e-8,
        #     prior_var_perturb_scale = 0, prior_var_n_data = 100,
        #     include_const = False
        beta_path, Z = tvp_var_with_exog_and_var_prior(
            Y_std, X_std, spec.p, lam, R_mat, include_const=spec.include_const )
    else:
        # Y, Xexo, p, lam=1.0, R=1.0, ridge=1.0, beta0=None, P0=None,
        #     include_const=False
        beta_path, Z = tvp_var_with_exog( Y=Y_std, Xexo=X_std, p=spec.p,
            lam=lam, R=R_mat, ridge=1e-4 )
    # --------------------------------------------------
    # Result container (unchanged semantics)
    # --------------------------------------------------
    res = TVPVARResult(
        beta_path=beta_path,
        dom_vars=spec.endog_vars,
        endog=df2[spec.endog_vars],
        exog=df2[spec.exog_vars],
        exog_columns=spec.exog_vars,
        p=spec.p,
        R=R,
        lam=lam
    )
    # attach scaling + metadata (CRITICAL)
    res.Dy = Dy
    res.Dx = Dx
    res.sigma_u = sigma_u
    res.var_spec = spec

    return res

def fit_country_varx_TVP_WORKING(panel, country, p=2, R=1.0, lam=0.98, prior_var=False):
    df = panel[panel["country"] == country].set_index("quarter")

    y = df[DOMESTIC_VARS]
    X = df[FOREIGN_VARS + EXTERNAL_VARS]

    df2 = pd.concat([y, X], axis=1)

    # Require domestic + exog variables used in estimation
    df2 = df2.dropna(subset=DOMESTIC_VARS + FOREIGN_VARS + EXTERNAL_VARS)
    df2.index.freq = "QS-OCT"

    if len(df2) < 40:
        print(f"{country}: insufficient data ({len(df2)} rows)")
        return None

    y2 = df2[DOMESTIC_VARS]
    X2 = df2[FOREIGN_VARS + EXTERNAL_VARS]

    Y = y2.values
    X = X2.values
    Y_std, X_std, Dy, Dx = standardize_data(Y, X)

    # --- fixed VAR (for sigma_u only)
    # res_var = VAR(endog=y2, exog=X2).fit(p, trend="n")    # res = model.fit(p)
    # fit_country_varx_VAR does not use trend="n"
    res_var = VAR(endog=Y_std, exog=X_std).fit(p, trend="n")    # res = model.fit(p)
    # res_var = VAR(endog=Y_std, exog=X_std).fit(p)    # res = model.fit(p)
    sigma_u = res_var.sigma_u.copy()

    # R_mat = R * sigma_u
    R_mat = R  # scalar multiplies unity

    # --- save for MATLAB
    # from scipy.io import savemat
    # savemat( "var_input_data.mat",
    #     {
    #         "Y": y2.values if hasattr(y2, "values") else y2,
    #         "X": X2.values if hasattr(X2, "values") else X2,
    #         "Y_names": list(y2.columns),
    #         "X_names": list(X2.columns),
    #         "p": p },
    #     do_compression=True,
    # )

    # --- tvp-VAR with forgetting (replacement for model.fit)
    if prior_var: # ( Y, Xexo, p, lam, R, shrink=1e-3, eps_scale=1e-8, ...)
        # beta_path, Z = tvp_var_with_exog_and_var_prior(y2.to_numpy(), X2.to_numpy(), p, lam,
        beta_path, Z = tvp_var_with_exog_and_var_prior(Y_std, X_std, p, lam,
                                                       R_mat)  # numerical stabilization
    else:
        # beta_path, Z = tvp_var_with_exog( Y=y2, Xexo=X2, p=p, R=R_mat, # measurement noise
        beta_path, Z = tvp_var_with_exog( Y=Y_std, Xexo=X_std, p=p, R=R_mat, # measurement noise
            lam=lam,  # forgetting factor (tune if needed)
            ridge=1e-4 )  # numerical stabilization

    # 🔑 CRITICAL: store actual exog column order
    # res._exog_columns = list(X2.columns)

    # res container to be compatible with previous version of the code
    t0 = -1  # last available quarter
    beta_t = beta_path[t0]
    k = len(DOMESTIC_VARS)
    q = len(FOREIGN_VARS + EXTERNAL_VARS)
    # VAR lag matrices
    A_blocks = beta_t[:p * k].reshape(p, k, k)
    # Exogenous coefficients
    B_block = beta_t[p * k:, :]

    # build res container using class TVPVARResult
    res = TVPVARResult( beta_path=beta_path, dom_vars=DOMESTIC_VARS, p=p, R=R, lam=lam,
        endog=y2, exog=X2, exog_columns=FOREIGN_VARS + EXTERNAL_VARS )
    # make it non-singular
    res.Dy = Dy
    res.Dx = Dx
    res.sigma_u = sigma_u

    return res

# slightly different from WORKING, seems to fit better
def fit_country_varx_VAR(panel, country,
                         spec: VARSpec):
    df = panel[panel["country"] == country].set_index("quarter")

    y = df[spec.endog_vars]
    X = df[spec.exog_vars]

    df2 = pd.concat([y, X], axis=1)
    df2 = df2.dropna(subset=spec.endog_vars + spec.exog_vars)
    df2.index.freq = "QS-OCT"

    if len(df2) < 40:
        print(f"{country}: insufficient data ({len(df2)} rows)")
        return None

    Y = df2[spec.endog_vars].values
    X = df2[spec.exog_vars].values

    # --- save for MATLAB
    from scipy.io import savemat
    savemat( "var_input_data1.mat",
        {
            "Y": Y.values if hasattr(Y, "values") else Y,
            "X": X.values if hasattr(X, "values") else X,
            "Y_names": spec.endog_vars,
            "X_names": spec.exog_vars,
            "p": spec.p },
        do_compression=True,
    )

    # Standardize (as before)
    Y_std, X_std, Dy, Dx = standardize_data(Y, X)

    # Fit VAR WITHOUT implicit constant
    res = VAR(endog=Y_std, exog=X_std).fit(spec.p, trend="n")
    # res = VAR(endog=Y, exog=X).fit(spec.p, trend="n")

    # ---- Restore metadata (OLD behavior, but cleaner)
    res.var_spec = spec
    res.Dy = Dy
    res.Dx = Dx
    res._exog_columns = list(spec.exog_vars)   # optional, for backward compatibility

    return res

def fit_country_varx_VAR_WORKING(panel, country, p=2):
    # p = lag order

    df = panel[panel["country"] == country].set_index("quarter")

    y = df[DOMESTIC_VARS]
    X = df[FOREIGN_VARS + EXTERNAL_VARS]

    df2 = pd.concat([y, X], axis=1)

    # Require domestic + exog variables used in estimation
    df2 = df2.dropna(subset=DOMESTIC_VARS + FOREIGN_VARS + EXTERNAL_VARS)
    df2.index.freq = "QS-OCT"

    if len(df2) < 40:
        print(f"{country}: insufficient data ({len(df2)} rows)")
        return None

    y2 = df2[DOMESTIC_VARS]
    X2 = df2[FOREIGN_VARS + EXTERNAL_VARS]

    Y_std, X_std, Dy, Dx = standardize_data(y2, X2)

    model = VAR(endog=Y_std, exog=X_std)
    res = model.fit(p, trend="n")

    # 🔑 CRITICAL: store actual exog column order
    res._exog_columns = list(X2.columns)
    res.Dy = Dy

    return res
