# analysis/rolling_forecast.py
#
# Integration and diagnostic script for rolling GVAR forecasts.
#
# This script:
#   - Runs end-to-end GVAR / VAR / TVP-VAR logic
#   - Exercises estimation, forecasting, and scenario code together
#   - Is used for validation and debugging after refactoring
#
# This is NOT a numerical core module and NOT a Streamlit app.
# It is intended to be run manually (e.g., via `python rolling_forecast.py`)
# and may change frequently during development.

import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

def load_panel():
    return pd.read_csv(
        "../../data/gvar_panel_streamlit.csv",
        parse_dates=["quarter"]
    )

def get_country_df(panel, country, vars_):
    df = panel[panel["country"] == country].copy()
    df = df.set_index("quarter").sort_index()
    df = df.asfreq("QS")  # quarterly, quarter-start
    return df[vars_].dropna()

def rolling_forecast_var(
    panel,
    country,
    vars_,
    target,
    p=2,
    min_obs=40
):
    """
    Rolling 1-quarter-ahead forecast using VAR.
    """

    df = get_country_df(panel, country, vars_)

    y_true = []
    y_pred = []
    dates  = []

    for t in range(min_obs, len(df) - 1):

        train = df.iloc[:t]

        # Fit VAR on training window only
        model = VAR(train)
        res = model.fit(p)

        # Forecast one step ahead
        fc = res.forecast(
            y=train.values[-p:],
            steps=1
        )

        target_idx = df.columns.get_loc(target)

        y_true.append(df.iloc[t+1, target_idx])
        y_pred.append(fc[0, target_idx])
        dates.append(df.index[t+1])

    return (
        np.array(y_true),
        np.array(y_pred),
        pd.Index(dates, name="quarter")
    )

def forecast_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(np.mean((y_true - y_pred)**2)),
        "MAE":  np.mean(np.abs(y_true - y_pred)),
        "Bias": np.mean(y_pred - y_true),
    }

# ----- RUN COMPARISON
def main():
    panel = load_panel()
    BASE_VARS = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
    CLIMATE_VARS = BASE_VARS + ["ENSO", "PRITHVI_HEAT_EXTENT", "PRITHVI_MOISTURE_EXTENT"]
    # CLIMATE_VARS = BASE_VARS + ["PRITHVI_HEAT_EXTENT"]
    country = "COL"

    y0, yhat0, dates0 = rolling_forecast_var(
        panel, country, BASE_VARS, "GDP_YoY"
    )
    y1, yhat1, dates1 = rolling_forecast_var(
        panel, country, CLIMATE_VARS, "GDP_YoY"
    )
    n0 = dates0.shape[0]
    n1 = dates1.shape[0]
    y0 = y0[:n1]
    yhat0 = yhat0[:n1]
    dates0 = dates0[:n1]

    m0 = forecast_metrics(y0, yhat0)
    m1 = forecast_metrics(y1, yhat1)

    print("Baseline VAR (no climate):")
    for k, v in m0.items():
        print(f"  {k}: {v:.3f}")

    print("\nClimate-augmented VAR:")
    for k, v in m1.items():
        print(f"  {k}: {v:.3f}")

    stress = (
        panel.query(f"country == '{country}'")
        .set_index("quarter")
        .loc[dates0, "ENSO"]
        >= 0.5
    )
    rmse_base_stress = np.sqrt(np.mean((y0[stress] - yhat0[stress])**2))
    rmse_clim_stress = np.sqrt(np.mean((y1[stress] - yhat1[stress])**2))
    print("Stress-period RMSE:")
    print("  Baseline:", rmse_base_stress)
    print("  Climate :", rmse_clim_stress)

    dir_base = np.mean(np.sign(y0[stress]) == np.sign(yhat0[stress]))
    dir_clim = np.mean(np.sign(y0[stress]) == np.sign(yhat1[stress]))
    print("Directional accuracy (stress periods):")
    print("Baseline:", dir_base)
    print("Climate :", dir_clim)

    q = 0.2  # bottom 20% of realizations
    thresh = np.quantile(y0, q)
    tail = y0 <= thresh
    rmse_base_tail = np.sqrt(np.mean((y0[tail] - yhat0[tail])**2))
    rmse_clim_tail = np.sqrt(np.mean((y0[tail] - yhat1[tail])**2))
    print("Tail accuracy (bottom 20%):")
    print("Baseline:", rmse_base_tail)
    print("Climate :", rmse_clim_tail)

if __name__ == "__main__":
    main()