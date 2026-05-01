# heat_df columns: country, quarter, PRITHVI_HEAT_STD
# moisture_df columns: country, quarter, PRITHVI_MOISTURE_STD
import pandas as pd

# ----- PARAMETERS USED FOR PROBABILITIES
add_extreme_flag_quantile = 0.9 # used in add_extreme_flag()

# ----- Read CSV files
prithvi_heat_q = pd.read_csv(
    "../data/prithvi_heat_anom.csv",
    parse_dates=["quarter"]
)
prithvi_moisture_q = pd.read_csv(
    "../data/prithvi_moisture_anom.csv",
    parse_dates=["quarter"]
)

heat_df = prithvi_heat_q[[
    "country",
    "quarter",
    "PRITHVI_HEAT_STD"
]].copy()
heat_df.index.freq = "QS-OCT"

moist_df = prithvi_moisture_q[[
    "country",
    "quarter",
    "PRITHVI_MOISTURE_STD"
]].copy()
moist_df.index.freq = "QS-OCT"

enso = pd.read_csv(
    "../data/enso_quarterly.csv",
    parse_dates=["quarter"]
)

heat_df = heat_df.merge(enso, on="quarter", how="left")
moist_df = moist_df.merge(enso, on="quarter", how="left")

# Lag ENSO by one quarter (predict next-quarter stress)
heat_df["ENSO_lag"] = heat_df.groupby("country")["ENSO"].shift(1)
moist_df["ENSO_lag"] = moist_df.groupby("country")["ENSO"].shift(1)

# Calculate probabilities
def add_extreme_flag(df, var):
    thresh = (
        df.groupby("country")[var]
        .quantile(add_extreme_flag_quantile)
        .rename("threshold")
    )
    df = df.merge(thresh, on="country", how="left")
    df["extreme"] = (df[var] > df["threshold"]).astype(int)
    df.index.freq = "QS-OCT"

    return df

from sklearn.linear_model import LogisticRegression
import numpy as np

def estimate_probabilities(df, var):
    out = []

    for c in df["country"].unique():
        sub = df[df["country"] == c].dropna(subset=["ENSO_lag", "extreme"])

        if sub["extreme"].sum() < 5:
            # not enough extremes → fallback to historical frequency
            p = sub["extreme"].mean()
        else:
            X = sub[["ENSO_lag"]].values
            y = sub["extreme"].values

            logit = LogisticRegression(solver="lbfgs")
            logit.fit(X, y)

            # current ENSO value
            enso_now = sub["ENSO"].iloc[-1]
            p = logit.predict_proba([[enso_now]])[0, 1]

        out.append({"country": c, var: p})

    return pd.DataFrame(out)

heat_probs = estimate_probabilities(
    add_extreme_flag(heat_df, "PRITHVI_HEAT_STD"),
    "P_HEAT_NEXT_Q"
)

moist_probs = estimate_probabilities(
    add_extreme_flag(moist_df, "PRITHVI_MOISTURE_STD"),
    "P_MOISTURE_NEXT_Q"
)

prob_df = heat_probs.merge(moist_probs, on="country", how="outer")

prob_df.to_csv(
    "../data/prithvi_stressor_probabilities.csv",
    index=False
)
