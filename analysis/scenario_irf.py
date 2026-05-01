# tvp/scenario_irf.py
#
# Impulse response function (IRF) utilities for VAR(p) models.
#
# Responsibilities:
#   - Companion-form construction
#   - IRF computation given VAR coefficient matrices
#
# This module contains no estimation logic and no model state.
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt

def scenario_irf_exog( res, shock_var, H=12, shock_percentile=84, shock_duration_q=1, t0=-1,
    KalmanConfig = None ):
    """
    Scenario-based GIRF with:
    - percentile-scaled shock
    - multi-quarter persistence
    """
    # print("USING scenario GIRF")
    # --- Endogenous state
    p = res.k_ar
    y_last = np.asarray(res.endog[-p:])

    # --- Exogenous state
    X_last = np.asarray(res.exog[-p:])
    # exog_cols = res._exog_columns
    exog_cols = res.var_spec.exog_vars

    if shock_var not in exog_cols:
        raise ValueError(f"{shock_var} not in exog columns")

    shock_idx = exog_cols.index(shock_var)

    # --- Shock magnitude from percentile
    if hasattr(res.exog, "iloc"):  # pandas DataFrame
        shock_series = res.exog.iloc[:, shock_idx].values
    else:  # numpy ndarray
        shock_series = res.exog[:, shock_idx]

    # ----- Shock scaling
    K_shock_scaling = KalmanConfig["shock_scaling"] if KalmanConfig else None
    if K_shock_scaling == "std":
        shock_size = np.nanstd(shock_series)
    elif K_shock_scaling == "percentile":
        shock_size = np.nanpercentile(
            shock_series,
            KalmanConfig["shock_percentile"]
        )
    else:
        shock_size = np.nanpercentile(shock_series, shock_percentile)

    # --- Baseline exog path
    X_base = np.tile(X_last[-1, :], (H, 1))

    # --- Shocked exog path
    X_shock = X_base.copy()
    X_shock[:shock_duration_q, shock_idx] += shock_size

    # --- Forecasts in standardized space
    # --- baseline forecast
    if "t0" in res.forecast.__code__.co_varnames:
        fc_base = res.forecast(y=y_last, steps=H, exog_future=X_base, t0=t0)
    else:
        fc_base = res.forecast(y=y_last, steps=H, exog_future=X_base)

    # --- shocked forecast
    if "t0" in res.forecast.__code__.co_varnames:
        fc_shock = res.forecast(y=y_last, steps=H, exog_future=X_shock, t0=t0)
    else:
        fc_shock = res.forecast(y=y_last, steps=H, exog_future=X_shock)

    # fc_base, fc_shock are STANDARDIZED
    diff_std = fc_shock - fc_base

    # un-standardize once, does both VAR and TVP
    Dy = res.Dy
    diff = diff_std @ Dy.T  # shape (H, k)
    # diff = diff_std

    print("fc_base_std mean:", fc_base.mean())
    print("fc_shock_std mean:", fc_shock.mean())
    print("diff_std mean:", diff_std.mean())
    print("diff mean (orig units):", diff.mean())
    print("exog last row:", X_last[-1, :])
    print("exog last element:", X_last[-1, -1])

    return diff
    # return fc_shock - fc_base

def plot_exog_irf( results_tvp, results_no_tvp, country, shock_var, response_var, H=12 ):
    # --- tvp-VAR result
    res_tvp = results_tvp[country]
    diff_tvp = scenario_irf_exog(res_tvp, shock_var, H=H)

    # --- Fixed VAR result
    res_no = results_no_tvp[country]
    diff_no = scenario_irf_exog(res_no, shock_var, H=H)

    # --- response index (same ordering by construction)
    resp_idx = list(res_tvp.names).index(response_var)

    plt.figure(figsize=(6, 4))

    plt.plot(
        diff_no[:, resp_idx],
        marker="o",
        linestyle="--",
        color="black",
        label="Fixed VAR",
    )

    plt.plot(
        diff_tvp[:, resp_idx],
        marker="o",
        linestyle="-",
        color="tab:red",
        label="tvp-VAR",
    )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Quarters after shock")
    plt.ylabel(f"Δ forecast {response_var}")
    plt.title(f"{country}: {response_var} response to {shock_var} shock")
    plt.legend()
    plt.tight_layout()
    plt.show()

def compare_countries( results, results_no_TVP, countries, shock_var, response_var, H=12 ):
    """
    Cross-country comparison of scenario GIRFs.
    Solid line  = tvp-VAR
    Dashed line = fixed VAR (no_TVP)
    Same color  = same country
    """
    print('***** estimate_GVAR > compare_countries *****')

    plt.figure(figsize=(7, 5))

    # --- fixed color per country
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for c, color in zip(countries, color_cycle):

        # --- tvp-VAR
        if c not in results:
            print(f"[SKIP] {c} (tvp)")
            continue

        res_tvp = results[c]

        if response_var not in res_tvp.names:
            print(f"[SKIP] {c}: {response_var} not in tvp model")
            continue

        diff_tvp = scenario_irf_exog(res_tvp, shock_var, H=H)
        resp_idx = res_tvp.names.index(response_var)

        plt.plot(
            diff_tvp[:, resp_idx],
            color=color,
            linestyle="-",
            linewidth=2,
            label=f"{c} (tvp)"
        )

        # --- fixed VAR (no_TVP)
        if c in results_no_TVP:
            res_no = results_no_TVP[c]

            if response_var in res_no.names:
                diff_no = scenario_irf_exog(res_no, shock_var, H=H)

                plt.plot(
                    diff_no[:, resp_idx],
                    color=color,
                    linestyle="--",
                    linewidth=2,
                    label=f"{c} (fixed)"
                )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Quarters after shock")
    plt.ylabel(f"Δ forecast {response_var}")
    plt.title(f"{response_var} response to 1σ {shock_var} shock")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()
