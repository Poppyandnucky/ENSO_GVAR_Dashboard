# analysis/prediction.py
import numpy as np
import pandas as pd

from analysis.scenario_irf import scenario_irf_exog

def build_prediction_df(panel, results, country, response_var, shock_var=None, H=8,
                        shock_percentile=84, shock_duration_q=1):
    """
    Returns DataFrame with observed, baseline forecast,
    and optional scenario forecast.
    """
    res = results[country]

    # Observed series
    df_obs = (
        panel[panel["country"] == country]
        .set_index("quarter")[response_var]
        .dropna()
    )
    df_obs.index.freq = "QS-OCT"

    last_q = df_obs.index[-1]

    # Forecast index
    fc_index = pd.date_range(
        start=last_q + pd.offsets.QuarterBegin(),
        periods=H,
        freq="QS"
    )

    # Baseline forecast
    fc_base = varx_forecast(res, H)
    resp_idx = list(res.names).index(response_var)

    out = pd.DataFrame(
        {"baseline": fc_base[:, resp_idx]},
        index=fc_index
    )

    # Scenario forecast (optional)
    if shock_var is not None:
        diff = scenario_irf_exog(res, shock_var, H=H, shock_percentile=shock_percentile,
                                 shock_duration_q=shock_duration_q)
        out["scenario"] = out["baseline"] + diff[:, resp_idx]

    # Combine with observed
    observed = df_obs.iloc[-H:].to_frame("observed")
    return observed.join(out, how="outer")

def varx_forecast(res, H=8, t0=-1):
    """
    Baseline VARX forecast (no shock).
    Returns H x n_endog array.
    """
    p = res.k_ar
    y_last = res.endog[-p:]
    X_last = np.asarray(res.exog[-p:])

    # Hold exogenous vars constant at last observed value
    X_future = np.tile(X_last[-1, :], (H, 1))

    # ADD CONSTANT
    # approximate fixed point
    from analysis.validation.compare_betas import extract_A_blocks_from_beta

    if hasattr(res, "beta_path"):  # TVP
        beta_t = res.beta_path[t0]
        A = extract_A_blocks_from_beta(beta_t, p=res.k_ar, k=len(res.names))[0]
        B = beta_t[res.k_ar * len(res.names):, :].T
    else:  # VAR
        A = res.coefs[0]
        B = res.coefs_exog.T
    c = B[-1]  # constant row
    x_star = X_future[0]
    I = np.eye(A.shape[0])
    x_star_aug = np.hstack([x_star, 1.0])  # (15,)
    print("B @ x_star_aug shape:", (B @ x_star_aug).shape)
    y_star = np.linalg.solve(I - A, B.T @ x_star_aug)
    print("Implied steady state:", y_star)
    print("A shape:", A.shape)
    print("B shape:", B.shape)
    print("x_star_aug shape:", x_star_aug.shape)

    if "t0" in res.forecast.__code__.co_varnames:
        fc = res.forecast(y=y_last, steps=H, exog_future=X_future, t0=t0)
    else:
        fc = res.forecast(y=y_last, steps=H, exog_future=X_future)

    return fc
