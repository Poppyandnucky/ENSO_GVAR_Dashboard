# data_external.py
from fredapi import Fred
import pandasdmx as sdmx
import pandas as pd
# --- SSL certificate fix (macOS / PyCharm safe)
import ssl
import certifi

# ----- Install Python certificate
def _certifi_context():
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = _certifi_context

# ----- Helpers
def ensure_datetime_index(df, name):
    if not isinstance(df.index, pd.DatetimeIndex):
        print(f"[FIX] Converting index of {name} to DatetimeIndex")
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df

# ----- US GDP (Quarterly, Real)
# FRED – Federal Reserve Economic Data
# 	•	Series: Real Gross Domestic Product
# 	•	Code: GDPC1
# 	•	Frequency: Quarterly
# 	•	Units: Billions of chained 2017 dollars
def get_us_gdp_yoy():
    us = pd.read_csv("../data/us_gdp.csv")

    us["quarter"] = pd.to_datetime(
        us["observation_date"],
        errors="coerce",
        infer_datetime_format=True
    )

    # Sanity check
    assert us["quarter"].notna().all(), "Some dates failed to parse"

    us = us.set_index("quarter").sort_index()

    us["US_GDP_YoY"] = us["GDPC1"].pct_change(4, fill_method=None) * 100

    us_gdp = us[["US_GDP_YoY"]].dropna()
    us_gdp.index.name = "quarter"

    print(us_gdp.head())
    print(us_gdp.tail())

    return us_gdp

# ----- China GDP (Quarterly), also contains US GDP (already included in FRED)
# OECD Quarterly National Accounts (QNA)
# 	•	Dataset: QNA
# 	•	Variable: GDP, volume, seasonally adjusted
# 	•	Coverage: China quarterly GDP (consistent, reliable)
def get_china_gdp_yoy_from_oecd():
    oecd = pd.read_csv("../data/oecd_qna_gdp_quarterly.csv")

    # --- Filter to China GDP YoY series ---
    chn = oecd[
        (oecd["REF_AREA"] == "CHN") &
        (oecd["FREQ"] == "Q") &
        (oecd["TRANSACTION"] == "B1GQ") &   # GDP
        (oecd["TRANSFORMATION"] == "GY") &  # Year-over-year growth
        (oecd["UNIT_MEASURE"] == "PC")      # Percent
    ].copy()

    print("China GDP YoY rows found:", len(chn))

    if chn.empty:
        raise RuntimeError("China GDP YoY series not found in OECD file.")

    # --- Parse quarter like '1993-Q1' ---
    chn["quarter"] = pd.PeriodIndex(
        chn["TIME_PERIOD"],
        freq="Q"
    ).to_timestamp()

    chn = chn.set_index("quarter").sort_index()

    # --- Use OBS_VALUE directly ---
    chn["CHN_GDP_YoY"] = pd.to_numeric(
        chn["OBS_VALUE"],
        errors="coerce"
    )

    chn_gdp = chn[["CHN_GDP_YoY"]].dropna()
    chn_gdp.index.name = "quarter"

    return chn_gdp

# ----- Global Commodity Price Index
# ----- Load World Bank Pink Sheet Monthly Data -----
def get_commodity_yoy(col_name="TOTAL_INDEX"):
    pink_path = "../data/CMO-Historical-Data-Monthly.xlsx"

    # --- Read the Monthly Indices sheet, skipping descriptive header rows
    # Data starts at the row with '1960M01' (typically row 9)
    pink_df = pd.read_excel(
        pink_path,
        sheet_name="Monthly Indices",
        skiprows=9,
        header=None
    )

    # --- Assign column names (first column = month, second = Total Index)
    pink_df.columns = [
        "month",
        "TOTAL_INDEX",
        "ENERGY",
        "NON_ENERGY",
        "AGRICULTURE",
        "BEVERAGES",
        "FOOD",
        "OILS_MEALS",
        "GRAINS",
        "OTHER_FOOD",
        "RAW_MATERIALS",
        "TIMBER",
        "OTHER_RAW_MAT",
        "METALS_MINERALS",
        "BASE_METALS",
        "IRON_ORE",
        "PRECIOUS_METALS"
    ][:len(pink_df.columns)]  # safe if fewer columns

    # --- Parse monthly date like '1960M01'
    pink_df["date"] = pd.to_datetime(
        pink_df["month"],
        format="%YM%m",
        errors="coerce"
    )

    # Drop rows where date failed to parse
    pink_df = pink_df.dropna(subset=["date"])

    # --- Use Total Commodity Index
    pink_df = pink_df.set_index("date").sort_index()
    pink_df["COMMODITY_INDEX"] = pd.to_numeric(
        pink_df[col_name],
        # pink_df["TOTAL_INDEX"],
        errors="coerce"
    )

    pink_df = pink_df[["COMMODITY_INDEX"]].dropna()

    # --- Monthly → quarterly mean
    pink_q = pink_df.resample("QE").mean()

    # --- YoY growth
    pink_q["COMMODITY_YoY"] = pink_q["COMMODITY_INDEX"].pct_change(
        4, fill_method=None
    ) * 100

    commodity_q = pink_q[["COMMODITY_YoY"]].dropna()

    # IMPORTANT: index name must be 'quarter'
    commodity_q.index = commodity_q.index.to_period("Q").to_timestamp(how="start")
    commodity_q.index.name = "quarter"

    return commodity_q  # return indexed by quarter (no reset_index here)

# ----- ENSO 3.4
def get_enso_q():
    import pandas as pd

    enso_path = "../data/nina34.csv"

    # Load CSV
    enso = pd.read_csv(enso_path)

    # Strip whitespace from column names
    enso.columns = [c.strip() for c in enso.columns]

    # Identify columns
    date_col = "Date"
    value_col = [c for c in enso.columns if "Nino" in c][0]

    # Parse date WITHOUT forcing format
    enso["date"] = pd.to_datetime(
        enso[date_col],
        errors="coerce",
        infer_datetime_format=True
    )

    # Parse ENSO values
    enso["ENSO"] = pd.to_numeric(enso[value_col], errors="coerce")

    # Remove missing-value flag (-99.99)
    enso.loc[enso["ENSO"] <= -99, "ENSO"] = pd.NA

    # Drop bad rows
    enso = enso.dropna(subset=["date", "ENSO"])

    # Set monthly index
    enso = enso.set_index("date").sort_index()

    # Monthly → quarterly mean
    enso_q = enso["ENSO"].resample("QE").mean().to_frame("ENSO")

    # Align quarter to start (match macro panel)
    enso_q.index = enso_q.index.to_period("Q").to_timestamp(how="start")
    enso_q.index.name = "quarter"

    return enso_q

# ----- Build df_temp
def build_external_variables():
    """
    Returns a quarterly DataFrame with external/global exogenous variables.
    Index: quarter (DatetimeIndex)
    """

    # --- US GDP (YoY)
    us_gdp = get_us_gdp_yoy()          # your existing code
    # columns: ["US_GDP_YoY"], index=quarter

    # --- China GDP (YoY)
    chn_gdp = get_china_gdp_yoy_from_oecd()      # your existing code
    # columns: ["CHN_GDP_YoY"], index=quarter

    # --- Commodity prices (YoY)
    commodity = get_commodity_yoy()    # from Pink Sheet
    commodity_A = get_commodity_yoy("AGRICULTURE")    # from Pink Sheet
    commodity_A = commodity_A.rename(columns={'COMMODITY_YoY': 'COMMODITY_AGR_YoY'})
    # columns: ["COMMODITY_YoY"], index=quarter

    # --- ENSO
    enso = get_enso_q()                # optional, index=quarter

    # --- Force all indices to be DatetimeIndex
    us_gdp = ensure_datetime_index(us_gdp, "US_GDP")
    chn_gdp = ensure_datetime_index(chn_gdp, "CHN_GDP")
    commodity_q = ensure_datetime_index(commodity, "COMMODITY")
    commodity_Aq = ensure_datetime_index(commodity_A, "COMMODITY_AGR")

    # --- Combine
    external = (
        us_gdp
        .join(chn_gdp, how="outer")
        .join(commodity_q, how="outer")
        .join(commodity_Aq, how="outer")
        .join(enso, how="outer")
        .sort_index()
    )

    external.index.name = "quarter"

    return external

# TEST get_commodity_yoy()
# commodity = get_commodity_yoy()
# print(commodity.head())
# print(commodity.tail())
# print(commodity["COMMODITY_YoY"].describe())

# TEST get_china_gdp_yoy_from_oecd()
# chn_gdp = get_china_gdp_yoy_from_oecd(
#     "data/oecd_qna_gdp_quarterly.csv"
# )
# print(chn_gdp.head())
# print(chn_gdp.tail())

# TEST get_enso()
enso = get_enso_q()
print(enso.head())
print(enso.tail())
print(enso.describe())

# RUN AND SAVE
external = build_external_variables()
external.reset_index().to_csv("../data/external_global_drivers.csv", index=False)
