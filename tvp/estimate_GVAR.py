import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from analysis.estimate_var import fit_country_varx_TVP, fit_country_varx_VAR
from analysis.scenario_irf import plot_exog_irf, compare_countries
from models.kalman_var import make_var_design
from models.config import EXTERNAL_VARS, DOMESTIC_VARS, FOREIGN_VARS

# from tvp.data_CDS import prithvi_moisture_q

flag_run_GVAR = False

# ----- 1 Load and merge panel + external drivers
panel = pd.read_csv("data/gvar_macro_panel_cached.csv", parse_dates=["quarter"])
external = pd.read_csv("data/external_global_drivers.csv", parse_dates=["quarter"])
prithvi_heat_q = pd.read_csv("data/prithvi_heat_anom.csv", parse_dates=["quarter"])
prithvi_heat_x = pd.read_csv("data/prithvi_heat_extent.csv", parse_dates=["quarter"])
prithvi_moisture_q = pd.read_csv("data/prithvi_moisture_anom.csv", parse_dates=["quarter"])
prithvi_moisture_x = pd.read_csv("data/prithvi_moisture_extent.csv", parse_dates=["quarter"])

# Slow-moving regime controls (CPI, FX)
panel["year"] = panel["quarter"].dt.year

# ----- Construct annual CPI / FX regime controls
annual_regimes = (
    panel
    .groupby(["country", "year"])
    .agg({
        "CPI_YoY": "mean",
        "FX_YoY": "mean",
    })
    .reset_index()
    .rename(columns={
        "CPI_YoY": "CPI_YoY_annual",
        "FX_YoY": "FX_YoY_annual",
    })
)

panel = panel.merge(
    annual_regimes,
    on=["country", "year"],
    how="left"
)

# Merge external vars into panel
panel = panel.merge(external, on="quarter", how="left")
panel = panel.merge(
    prithvi_heat_q[["country", "quarter", "PRITHVI_HEAT_STD"]], on=["country", "quarter"], how="left"
)
panel = panel.merge(
    prithvi_moisture_q[["country", "quarter", "PRITHVI_MOISTURE_STD"]],
    on=["country", "quarter"], how="left"
)
panel = panel.merge(
    prithvi_moisture_x[["country", "quarter", "PRITHVI_MOISTURE_EXTENT"]],
    on=["country", "quarter"], how="left"
)
panel = panel.merge(
    prithvi_heat_x[["country", "quarter", "PRITHVI_HEAT_EXTENT"]],
    on=["country", "quarter"], how="left"
)

# THIS IS OLD CODE, START & END ARE NOT DEFINED
# panel = panel[(panel["quarter"] >= START) & (panel["quarter"] <= END)]

# ----- 3 Build a weights matrix W
countries = sorted(panel["country"].unique())
W = pd.DataFrame(0.0, index=countries, columns=countries)

for i in countries:
    others = [j for j in countries if j != i]
    W.loc[i, others] = 1.0 / len(others)

# ----- 4 Construct foreign variables y^* (within your 12)
def add_foreign_vars(panel, W, vars_):
    panel = panel.copy()
    panel = panel.sort_values(["country", "quarter"])

    # pivot each var to quarter x country
    for v in vars_:
        pivot = panel.pivot(index="quarter", columns="country", values=v)

        # compute foreign weighted average for each i
        foreign = {}
        for i in W.index:
            w = W.loc[i].copy()
            foreign[i] = pivot.dot(w)  # dot across countries

        foreign_df = pd.DataFrame(foreign)  # quarter x country
        foreign_df = foreign_df.stack().rename(f"{v}_star").reset_index()
        foreign_df.columns = ["quarter", "country", f"{v}_star"]

        panel = panel.merge(foreign_df, on=["quarter", "country"], how="left")

    return panel

panel = add_foreign_vars(panel, W, DOMESTIC_VARS)
panel.to_csv("data/gvar_panel_streamlit.csv", index=False)
print("estimate_GVAR > panel.columns:")
print(panel.columns.tolist())

print("Panel shape:", panel.shape)
print("Unique countries:", panel["country"].unique())
print("Quarter range:", panel["quarter"].min(), panel["quarter"].max())

panel[panel.country=="BRA"][
    ["quarter", "CPI_YoY_annual", "FX_YoY_annual"]
].head(8)

# ----- Kalman Filter helper functions
def tvp_var_with_exog_scalar_R(Y, Xexo, p, lam=0.98, R=1.0, ridge=1e-4):
    """
    Y     : (T, k) endogenous
    Xexo  : (T, q) exogenous
    p     : VAR lag order
    lam   : forgetting factor
    """

    Z, Yt = make_var_design(Y, Xexo, p)
    T_eff, k = Yt.shape
    m = Z.shape[1]           # regressors per equation

    # initialize
    beta = np.zeros((m, k))
    P = np.eye(m) / ridge
    # R = np.eye(k)

    beta_path = []

    for t in range(T_eff):
        z = Z[t:t+1]         # (1, m)
        y = Yt[t:t+1]        # (1, k)

        # forgetting
        P = P / lam

        # Kalman gain
        S = z @ P @ z.T + R
        K = P @ z.T / S
        # K = P @ z.T @ np.linalg.inv(S)

        # update
        err = y - z @ beta
        beta = beta + K @ err
        P = (np.eye(m) - K @ z) @ P

        beta_path.append(beta.copy())

    return np.array(beta_path), Z

print("DOMESTIC_VARS:", DOMESTIC_VARS)
print("FOREIGN_VARS:", FOREIGN_VARS)
print("EXTERNAL_VARS:", EXTERNAL_VARS)

if __name__ == "__main__" and flag_run_GVAR:
    # ORDER
    p = 1
    results = {}
    results_no_TVP = {}
    for c in countries:
        res = fit_country_varx_TVP(panel, c, p=p)
        res_no_TVP = fit_country_varx_VAR(panel, c, p=p)
        if res is not None:
            results[c] = res
        if res_no_TVP is not None:
            results_no_TVP[c] = res_no_TVP

    print("Estimated countries:", list(results.keys()))

    # ----- 6 First GIRF plots (TRP-ready)
    # Example 1: CPI response to an ENSO shock (India)
    c = "MEX"
    res = results[c]
    res_no_TVP = results_no_TVP[c]

    # ----- IRF for impulses and responses for all endogenous variables
    H = 12
    irf_obj = res.irf(H=H, t0=-1)
    irf_obj.plot( impulse="CPI_YoY", response="GDP_YoY", orth=False )
    plt.show()

    irf_obj_no_TVP = res_no_TVP.irf(H)
    irf_obj_no_TVP.plot( impulse="CPI_YoY", response="GDP_YoY", orth=False )
    # response CPI to CPI innovation (baseline)
    # irf.plot(impulse="CPI_YoY", response="CPI_YoY", orth=False)
    # response GDP to CA innovation (baseline)
    plt.show()

    # plot_exog_irf(results, results_no_TVP, "BRA", "COMMODITY_YoY", "CPI_YoY", H=H)
    plot_exog_irf(results, results_no_TVP, c, "ENSO", "GDP_YoY", H=H)
    # plot_exog_irf(results, results_no_TVP, "IND", "PRITHVI_HEAT_STD", "GDP_YoY", H=H)

    compare_countries(results, results_no_TVP, ["IND", "BRA", "ZAF", "THA", "MEX"], "ENSO", "GDP_YoY", H=H)
    # compare_countries(results, results_no_TVP, ["IND","BRA","ZAF","THA","MEX"], "PRITHVI_HEAT_STD", "GDP_YoY", H=H)

