import pandas as pd
import numpy as np
import os
import re
from models.config import ISO3_TO_IMF_NAME, COUNTRY_CODES, START_DATE, END_DATE

# BUILD CSV FILES
# imf_cache/gvar_macro_{c}.csv
# data/raw_macro_panel.csv
# data/gvar_macro_panel_cached.csv

# IMF, DOTS series codes
# IFS_SERIES = {
#     "CPI": "PCPI_IX",                # CPI index (monthly)
#     "GDP_NOM_Q": "NGDP_Q",           # Nominal GDP (quarterly)
#     "FX": "ENDA_XDC_USD_RATE"        # Exchange rate LCU per USD (monthly)
# }
# DOTS_SERIES = {
#     "EXPORTS": "TXG_FOB_USD"         # Exports, USD (monthly)
# }

# START_YEAR = 2000
# END_YEAR = 2024
START_YEAR = 1980
END_YEAR = 2025

# ----- HELPERS
_RE_M = re.compile(r"^\d{4}-M\d{2}$")
_RE_Q = re.compile(r"^\d{4}-Q[1-4]$")

def imf_month_cols(df):
    return [c for c in df.columns if isinstance(c, str) and _RE_M.match(c)]

def imf_quarter_cols(df):
    return [c for c in df.columns if isinstance(c, str) and _RE_Q.match(c)]

def imf_cols_to_datetime(cols, freq):
    # cols like "2025-M05" or "2019-Q4"
    periods = pd.PeriodIndex(cols, freq=freq)
    return periods.to_timestamp()  # quarter end / month end timestamps

def imf_quarter_cols(df):
    return [
        c for c in df.columns
        if isinstance(c, str)
        and "Q" in c
        and any(y in c for y in map(str, range(1950, 2100)))
    ]

def imf_month_cols(df):
    return [
        c for c in df.columns
        if isinstance(c, str)
        and any(y in c for y in map(str, range(1950, 2100)))
        and ("M" in c or "-" in c or ":" in c)
    ]

def imf_month_cols_to_datetime(cols):
    dates = []

    for c in cols:
        c0 = c.replace(":", "").replace("-", "").upper()

        if "M" in c0:
            year = int(c0[:4])
            month = int(c0[-2:])
        else:
            raise ValueError(f"Unrecognized IMF month format: {c}")

        dates.append(pd.Timestamp(year=year, month=month, day=1))

    return pd.DatetimeIndex(dates)

# ----- Helper function for processing a row of data
def select_row(meta,sub,time_cols,var_type='GDP'):
    # --- choose best row (the lists below are row-order preserving)
    meta["SERIES_CODES"] = sub["SERIES_CODE"].tolist()
    nn = sub[time_cols].notna().sum(axis=1)
    meta["nn"] = nn.tolist()
    if nn.max() == 0:
        print(f"[SKIP] {var_type} CSV values missing for {meta['country']}")
        return pd.DataFrame()

    nn_max_list = [np.nan] * len(nn)
    selected_list = [np.nan] * len(nn)
    # nn.idxmax() refers to the indeces of the original df and not the position within nn
    #   index label of nn is inherited from the original df, sub
    #   pandas are label-based, python lists are position-based
    pos_max = nn.index.get_loc(nn.idxmax())
    nn_max_list[pos_max] = nn.max()
    selected_list[pos_max] = True # row that is selected
    meta["nn-max"] = nn_max_list
    meta["selected"] = selected_list

    first_dates = []
    last_dates = []
    transformations = []
    if var_type=='GDP':
        price_types = []
        adjustments = []

    for _, row in sub.iterrows():
        # convert time columns to numeric so NaNs behave properly
        s = pd.to_numeric(row[time_cols], errors="coerce")
        first_dates.append(s.first_valid_index())
        last_dates.append(s.last_valid_index())
        transformations.append(row["TYPE_OF_TRANSFORMATION"])
        if var_type=='GDP':
            price_types.append(row["PRICE_TYPE"])
            adjustments.append(row["S_ADJUSTMENT"])

    meta["first-date"] = first_dates
    meta["last-date"] = last_dates
    if var_type=='GDP':
        meta["price-types"] = price_types
        meta["adjustments"] = adjustments
        meta["transformations"] = transformations

    row = sub.loc[nn.idxmax()]
    s = pd.to_numeric(row[time_cols], errors="coerce")

    return s

# ----- CPI
def process_cpi_from_csv_monthly(country, cpi_df):
    imf_name = ISO3_TO_IMF_NAME.get(country, country)
    sub = cpi_df[cpi_df["COUNTRY"] == imf_name]
    if sub.empty:
        print(f"[SKIP] CPI CSV for {country} (no country match)")
        return pd.DataFrame()

    cols = imf_month_cols(cpi_df)
    if not cols:
        print(f"[SKIP] CPI CSV for {country} (no monthly columns)")
        return pd.DataFrame()

    # pick the row with the most non-missing monthly values
    nn = sub[cols].notna().sum(axis=1)
    if nn.max() == 0:
        print(f"[SKIP] CPI CSV values missing for {country}")
        return pd.DataFrame()
    row = sub.loc[nn.idxmax()]

    s = pd.to_numeric(row[cols], errors="coerce")
    idx = imf_cols_to_datetime(cols, freq="M")
    s.index = idx
    s = s.dropna()

    s = s[(s.index.year >= START_YEAR) & (s.index.year <= END_YEAR)]
    if s.empty:
        print(f"[SKIP] CPI CSV after date filter for {country}")
        return pd.DataFrame()

    q = s.resample("QE").mean().to_frame("CPI")
    q["CPI_YoY"] = q["CPI"].pct_change(4, fill_method=None) * 100
    return q[["CPI_YoY"]]

def process_cpi_from_csv(country, date_str, cpi_df, food=False):
    meta = {'country': country, 'date': date_str}
    imf_name = ISO3_TO_IMF_NAME.get(country, country)
    sub = cpi_df[cpi_df["COUNTRY"] == imf_name]
    if food:
        str_cpi = "F"
    else:
        str_cpi = ""

    if sub.empty:
        print(f"[SKIP] CPI{str_cpi} CSV for {country} (no country match)")
        return pd.DataFrame(), meta

    # Identify quarterly IMF columns (e.g. 2020Q1, 2020-Q1, etc.)
    time_cols = imf_quarter_cols(cpi_df)
    if time_cols:
        freq = "Q"
    else:
        freq = ""
    meta["q-time_cols"] = freq
    meta["time-time_cols-exists"] = bool(time_cols)

    if not time_cols:
        print(f"[SKIP] CPI CSV{str_cpi} for {country} (no quarterly columns)")
        return pd.DataFrame(), meta

    # pick the row with the most non-missing quarterly values
    flag_YoY_percent = True
    if not flag_YoY_percent: # Usually YoY percent is used
        meta["SERIES_CODES"] = sub["SERIES_CODES"].tolist()
        # nn = sub[time_cols].notna().sum(axis=1)
        # meta["nn"] = nn.tolist()
        # if nn.max() == 0:
        #     print(f"[SKIP] CPI{str_cpi} CSV values missing for {country}")
        #     return pd.DataFrame(), meta
        # row = sub.loc[nn.idxmax()]
        s = select_row(meta, sub, time_cols, var_type='CPI')
    else:
        # use sub[...] to get a dataframe based on the logical array in brackets
        # squeeze makes the 1D dataframe into a dataseries
        if food:
            series_code =".CPI.CP01.YOY_PCH_PA_PT.Q"
        else:
            series_code =".CPI._T.YOY_PCH_PA_PT.Q" # total
        sub_YoY = sub[sub["SERIES_CODE"].str.endswith(series_code)]
        row = sub_YoY.squeeze()
        if row.empty:
            print(f"[SKIP] CPI{str_cpi} CSV values missing for {country}")
            return pd.DataFrame(), meta
        # numeric series
        # s = pd.to_numeric(row[time_cols], errors="coerce")
        s = select_row(meta, sub_YoY, time_cols, var_type='CPI')

    # build quarterly datetime index
    idx = imf_cols_to_datetime(time_cols, freq="Q")
    s.index = idx
    s = s.dropna()

    # year filtering
    s = s[(s.index.year >= START_YEAR) & (s.index.year <= END_YEAR)]
    meta["q-exists"] = not s.empty
    if s.empty:
        print(f"[SKIP] CPI{str_cpi} CSV after date filter for {country}")
        return pd.DataFrame(), meta
    # else:
    #     meta["first_date"] = s.first_valid_index()
    #     meta["last_date"] = s.last_valid_index()

    # already quarterly → no resampling
    q = s.to_frame(f"CPI{str_cpi}")
    if flag_YoY_percent:
        q[f"CPI{str_cpi}_YoY"] = q[f"CPI{str_cpi}"]
    else:
        q["fCPI{str_cpi}_YoY"] = q[f"CPI{str_cpi}"].pct_change(4, fill_method=None) * 100

    return q[[f"CPI{str_cpi}_YoY"]], meta

# ----- Nominal GDP → YoY growth (quarterly)
def process_gdp_from_csv_OLD(country, gdp_df):
    """
    Process quarterly GDP from IMF wide-format GDP CSV.
    Selects the GDP series for the country with the most non-missing
    quarterly observations.
    Returns quarterly YoY GDP growth.
    """

    imf_name = ISO3_TO_IMF_NAME.get(country, country)

    # --- Filter to country
    sub = gdp_df[gdp_df["COUNTRY"] == imf_name]

    if sub.empty:
        print(f"[SKIP] GDP CSV for {country} (no country match)")
        return pd.DataFrame()

    # --- Identify quarterly columns
    time_cols = imf_quarter_cols(gdp_df)
    # time_cols = [c for c in gdp_df.columns if isinstance(c, str) and "-Q" in c]

    if not time_cols:
        print(f"[SKIP] GDP CSV for {country} (no quarterly columns)")
        return pd.DataFrame()

    # --- Choose the row with the most non-missing quarterly values
    non_missing_counts = sub[time_cols].notna().sum(axis=1)

    if non_missing_counts.max() == 0:
        print(f"[SKIP] GDP CSV values missing for {country}")
        return pd.DataFrame()

    row = sub.loc[non_missing_counts.idxmax()]

    # --- Extract series
    s = row[time_cols]

    # Convert values
    s = pd.to_numeric(s, errors="coerce")

    # Convert IMF quarter labels -> DatetimeIndex
    s.index = imf_cols_to_datetime(time_cols, freq="Q")
    print(type(s.index), s.index[:3])

    # Drop missing
    s = s.dropna()

    # --- Now filter by year (this will work)
    s = s[
        (s.index.year >= START_YEAR) &
        (s.index.year <= END_YEAR)
        ]

    if s.empty:
        print(f"[SKIP] GDP CSV after date filter for {country}")
        return pd.DataFrame()

    # --- Compute YoY growth
    s = s.sort_index()
    q = s.to_frame("GDP")
    q["GDP_YoY"] = q["GDP"].pct_change(4, fill_method=None) * 100

    return q[["GDP_YoY"]]

def process_gdp_from_csv(country, date_str, gdp_df):
    meta = {'country': country, 'date': date_str}
    imf_name = ISO3_TO_IMF_NAME.get(country, country)
    sub = gdp_df[gdp_df["COUNTRY"] == imf_name]

    if sub.empty:
        print(f"[SKIP] GDP CSV for {country} (no country match)")
        return pd.DataFrame(), meta

    # --- detect quarterly first, e.g. ['1950-Q1','1950-Q2',...]
    q_cols = imf_quarter_cols(gdp_df)

    if q_cols:
        freq = "Q"
        time_cols = q_cols
    else:
        # fallback to monthly
        time_cols = imf_month_cols(gdp_df)
        freq = "M"
    meta["q-cols"] = freq

    meta["time-cols-exists"] = bool(time_cols)
    if not time_cols:
        print(f"[SKIP] GDP CSV for {country} (no time columns)")
        return pd.DataFrame(), meta

    s = select_row(meta,sub,time_cols,var_type='GDP')

    # --- build index
    if freq == "Q":
        s.index = imf_cols_to_datetime(time_cols, freq="Q")
        q = s.to_frame("GDP")
    else:
        s.index = imf_month_cols_to_datetime(time_cols)
        s = s.dropna().sort_index()

        # monthly → quarterly GDP (SUM, not mean)
        q = s.resample("QE").sum().to_frame("GDP")

    # --- filter years
    q = q[
        (q.index.year >= START_YEAR) &
        (q.index.year <= END_YEAR)
    ]

    meta["q-exists"] = not q.empty
    if q.empty:
        print(f"[SKIP] GDP CSV after date filter for {country}")
        return pd.DataFrame(), meta

    # --- YoY growth (4 quarters)
    q["GDP_YoY"] = q["GDP"].pct_change(4, fill_method=None) * 100

    return q[["GDP_YoY"]], meta

# ----- Exchange rate → YoY depreciation
def process_fx_from_csv(country, date_str, fx_df):
    meta = {'country': country, 'date': date_str}
    imf_name = ISO3_TO_IMF_NAME.get(country, country)
    sub = fx_df[fx_df["COUNTRY"] == imf_name]

    if sub.empty:
        print(f"[SKIP] FX CSV for {country} (no country match)")
        return pd.DataFrame(), meta

    # --- Identify quarterly columns (NOT monthly)
    cols = imf_quarter_cols(fx_df)
    if not cols:
        print(f"[SKIP] FX CSV for {country} (no quarterly columns)")
        return pd.DataFrame(), meta

    # --- Pick series with most data
    nn = sub[cols].notna().sum(axis=1)
    if nn.max() == 0:
        print(f"[SKIP] FX CSV values missing for {country}")
        return pd.DataFrame(), meta
    row = sub.loc[nn.idxmax()]

    # --- Extract FX series
    s = pd.to_numeric(row[cols], errors="coerce")
    s.index = imf_cols_to_datetime(cols, freq="Q")
    s = s.dropna().sort_index()

    # --- Filter years
    s = s[
        (s.index.year >= START_YEAR) &
        (s.index.year <= END_YEAR)
    ]

    if s.empty:
        print(f"[SKIP] FX CSV after date filter for {country}")
        return pd.DataFrame(), meta

    q = s.to_frame("FX")

    # --- YoY depreciation (positive = depreciation)
    q["FX_YoY"] = q["FX"].pct_change(4, fill_method=None) * 100
    # ✅ align FX to quarter-start timestamps
    q.index = q.index.to_period("Q").to_timestamp(how="start")

    return q[["FX_YoY"]], meta

# ----- Exports to WORLD (almost identical to FX except it checks for WORLD)
def process_exports_from_csv(country, date_str, ex_df):
    meta = {'country': country, 'date': date_str}
    imf_name = ISO3_TO_IMF_NAME.get(country, country)

    sub = ex_df[
        (ex_df["COUNTRY"] == imf_name) &
        (ex_df["COUNTERPART_COUNTRY"] == "World")
    ]

    if sub.empty:
        print(f"[SKIP] EX CSV for {country} (no World counterpart)")
        return pd.DataFrame(), meta

    # quarterly columns
    cols = imf_quarter_cols(ex_df)
    if not cols:
        print(f"[SKIP] EX CSV for {country} (no quarterly columns)")
        return pd.DataFrame(), meta

    # pick longest series
    nn = sub[cols].notna().sum(axis=1)
    if nn.max() == 0:
        print(f"[SKIP] EX CSV values missing for {country}")
        return pd.DataFrame(), meta

    row = sub.loc[nn.idxmax()]

    # levels
    s = pd.to_numeric(row[cols], errors="coerce")
    s.index = imf_cols_to_datetime(cols, freq="Q")
    s = s.dropna().sort_index()

    # align to quarter-start (QS)
    s.index = s.index.to_period("Q").to_timestamp(how="start")

    # year filter
    s = s[
        (s.index.year >= START_YEAR) &
        (s.index.year <= END_YEAR)
    ]

    if s.empty:
        print(f"[SKIP] EX CSV after date filter for {country}")
        return pd.DataFrame(), meta

    q = s.to_frame("Exports")

    # YoY growth
    q["EX_YoY"] = q["Exports"].pct_change(4, fill_method=None) * 100

    return q[["EX_YoY"]], meta

def build_gvar_macro_panel(
    macro_path="data/gvar_macro_panel_cached.csv",
    external_path="data/external_global_drivers.csv",
    start=START_DATE,
    end=END_DATE,
    keep_vars=None,
):
    """
    Build the final country-quarter panel for GVAR / VARX estimation.

    Parameters
    ----------
    macro_path : str
        Path to cached internal macro panel
        (quarter, country, CPI_YoY, CA_YoY, FX_YoY, ...)

    external_path : str
        Path to cached external global drivers
        (quarter, ENSO, US_GDP_YoY, CHN_GDP_YoY, COMMODITY_YoY)

    start, end : str or Timestamp
        Common estimation window

    keep_vars : list[str] or None
        Optional list of macro variables to keep
        (default keeps all available macro variables)

    Returns
    -------
    panel : pandas.DataFrame
        Tidy panel with columns:
        quarter, country, macro vars, external vars
    """
    # ---- Load internal macro panel ----
    panel = pd.read_csv(macro_path, parse_dates=["quarter"])

    # Basic sanity check
    required_cols = {"quarter", "country"}
    if not required_cols.issubset(panel.columns):
        raise ValueError("Macro panel must contain 'quarter' and 'country' columns.")

    # ---- Optionally restrict macro variables ----
    if keep_vars is not None:
        cols = ["quarter", "country"] + keep_vars
        panel = panel[[c for c in cols if c in panel.columns]]

    # ---- Load external global drivers ----
    external = pd.read_csv(external_path, parse_dates=["quarter"])

    if "quarter" not in external.columns:
        raise ValueError("External drivers file must contain 'quarter' column.")

    # ---- Merge external variables into panel ----
    panel = panel.merge(external, on="quarter", how="left")

    # ---- Restrict to common estimation window ----
    panel = panel[
        (panel["quarter"] >= pd.to_datetime(start)) & # old version is start[0]
        (panel["quarter"] <= pd.to_datetime(end))     #                end[0]
    ].copy()

    # ---- Sort and reset index ----
    panel = panel.sort_values(["country", "quarter"]).reset_index(drop=True)

    # ---- Final sanity prints (safe to comment out later) ----
    print("GVAR panel built:")
    print("Countries:", panel["country"].nunique())
    print("Date range:", panel["quarter"].min(), "→", panel["quarter"].max())
    print("Columns:", panel.columns.tolist())

    return panel

def clean_quarterly_panel(panel, macro_vars=None):
    panel = panel.copy()

    if macro_vars is None:
#       macro_vars = ["GDP_YoY", "CPI_YoY", "FX_YoY", "CA_YoY"]
        macro_vars = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY", "CPIF_YoY"]

    # Canonical quarter
    panel["qtr"] = panel["quarter"].dt.to_period("Q")

    # Collapse to one row per country–quarter
    panel = (
        panel
        .groupby(["country", "qtr"], as_index=False)
        .agg({v: "first" for v in macro_vars})
    )

    # Use quarter start
    panel["quarter"] = panel["qtr"].dt.to_timestamp(how="start")
    panel = panel.drop(columns="qtr")

    # Drop rows with no macro info at all
    panel = panel.dropna(how="all", subset=macro_vars)

    # Final sort
    panel = panel.sort_values(["country", "quarter"]).reset_index(drop=True)

    return panel

if __name__ == "__main__":
    # ----- Wrappers: get_macro_data(country) and extract quarterly time series
    # need to load CSVs (only if this is run)
    flag_build_raw_csv = True  # gvar_macro_{c}.csv
    flag_build_gvar_macro_panel = True # gvar_macro_panel_cached.csv
    flag_save_combined = True  # raw_macro_panel.csv

    if flag_build_raw_csv:
        # BUILD PANELS FOR INDIVIDUAL COUNTIRES
        # use [:1] for one country, [:6] for half, [:] for all
        # [8] India, [7] SA

        COUNTRY_RUN = COUNTRY_CODES[:]
        # COUNTRY_RUN = ['AUS']
        print("Running countries:", COUNTRY_RUN)

        IFS_CPI_DF = pd.read_csv("../data_large/IFS_CPI.csv",
                                 dtype={14:str,15:str,24:str})
        IFS_GDP_DF = pd.read_csv("../data_large/IFS_GDP.csv")
        IFS_FX_DF = pd.read_csv("../data_large/IFS_FX.csv")
        IFS_EX_DF = pd.read_csv("../data_large/IFS_EX.csv")
        #   IFS_BOP_DF = pd.read_csv("data_large/IFS_BOP.csv")

        def get_macro_data(country,meta):
            """
            Retrieve macroeconomic data for one country using CSV-only sources.
            Returns a quarterly DataFrame indexed by quarter.
            """

            dfs = []
            date_str = meta["date"]

            # --- GDP (quarterly, IMF topic CSV)
            # IFS_GDP_DF is a df that was read from a csv file in /data_large
            gdp,meta_gdp = process_gdp_from_csv(country, date_str, IFS_GDP_DF)
            meta["GDP-exists"] = not gdp.empty
            meta["GDP"] = meta_gdp
            if not gdp.empty:
                dfs.append(gdp)
            else:
                print(f"[MISS] GDP CSV for {country}")

            # --- CPI (monthly → quarterly, IMF CPI CSV)
            cpi,meta_cpi = process_cpi_from_csv(country, date_str, IFS_CPI_DF)
            meta["CPI-exists"] = not cpi.empty
            meta["CPI"] = meta_cpi
            if not cpi.empty:
                dfs.append(cpi)
            else:
                print(f"[MISS] CPI CSV for {country}")

            # --- CPI (monthly → quarterly, IMF CPI CSV)
            cpif,meta_cpif = process_cpi_from_csv(country, date_str, IFS_CPI_DF, food=True)
            meta["CPIF-exists"] = not cpif.empty
            meta["CPIF"] = meta_cpif
            if not cpif.empty:
                dfs.append(cpif)
            else:
                print(f"[MISS] CPIF FOOD CSV for {country}")

            # --- FX (monthly → quarterly, IMF FX CSV)
            fx,meta_fx = process_fx_from_csv(country, date_str, IFS_FX_DF)
            meta["FX-exists"] = not fx.empty
            meta["FX"] = meta_fx
            if not fx.empty:
                dfs.append(fx)
            else:
                print(f"[MISS] FX CSV for {country}")

            # --- Exports
            ex,meta_ex = process_exports_from_csv(country, date_str, IFS_EX_DF)
            meta["EX-exists"] = not ex.empty
            meta["EX"] = meta_ex
            if not ex.empty:
                dfs.append(ex)
            else:
                print(f"[MISS] EX CSV for {country}")

            if not dfs:
                print(f"[SKIP] {country}: no macro data available from CSVs")
                return pd.DataFrame(),meta

            # --- Combine
            df = pd.concat(dfs, axis=1)
            df.index.name = "quarter"
            return df,meta

        def extract_imf_wide_series(df, country, freq, start_year, end_year, agg="mean", ):
            """
            Extract a quarterly time series from an IMF wide-format CSV.

            Parameters
            ----------
            df : pandas.DataFrame
                IMF topic CSV (wide format)
            country : str
                IMF country code (e.g. "MEX")
            freq : {"M", "Q"}
                Desired source frequency
            start_year, end_year : int
                Year range
            agg : {"mean", "sum"}
                Aggregation for monthly → quarterly

            Returns
            -------
            pandas.Series
                Quarterly series indexed by Timestamp
            """

            # --- select rows for country + frequency
            imf_name = ISO3_TO_IMF_NAME.get(country, country)
            sub = df[
                (df["COUNTRY"] == imf_name) &
                (df["FREQUENCY"] == freq)
                ]

            if sub.empty:
                return pd.Series(dtype=float)

            row = sub.iloc[0]

            # --- identify time columns
            if freq == "Q":
                time_cols = [c for c in df.columns if isinstance(c, str) and "-Q" in c]
            else:  # freq == "M"
                time_cols = [c for c in df.columns if isinstance(c, str) and "-M" in c]

            if not time_cols:
                return pd.Series(dtype=float)

            s = row[time_cols]
            # s.index = pd.to_datetime(s.index, errors="coerce")
            s = pd.to_numeric(s, errors="coerce").dropna()

            if s.empty:
                return pd.Series(dtype=float)

            # --- restrict years
            s = s[(s.index.year >= start_year) & (s.index.year <= end_year)]

            if s.empty:
                return pd.Series(dtype=float)

            # --- aggregate to quarterly if monthly
            if freq == "M":
                if agg == "mean":
                    s = s.resample("Q").mean()
                elif agg == "sum":
                    s = s.resample("Q").sum()
                else:
                    raise ValueError("agg must be 'mean' or 'sum'")

            return s.sort_index()

        # ----- Function meta -> xlsx with mulitple sheets
        from pathlib import Path

        def write_meta_excel(meta, filename, country_key="country"):
            """
            meta can be a dict (single record) or a list of dicts (multiple records).
            Creates/updates an Excel file with sheets: main, GDP, CPI.
            New rows are prepended if the file exists.
            Blank cells are written instead of NaN/NaT.
            """

            # Allow passing a single dict
            if isinstance(meta, dict):
                meta = [meta]

            xlsx_path = Path(filename)

            # --- main sheet ---
            main_rows = []
            # --- GDP sheet (flatten GDP dict keys into columns) ---
            def add_rows(main_rows,var_type='GDP'):
                gdp_rows = []
                for m in meta:
                    country = m.get(country_key, "")
                    gdp = m.get("GDP") or {}
                    # Expand rows for repeated values, e.g. SERIES_CODES
                    #   need enumerate to get index i
                    for i, row in enumerate(to_rows(gdp)):
                        gdp_rows.append(row)
                        if row['selected']==True:
                            GDP_nn = row['nn']
                            GDP_series = row['SERIES_CODES']
                    main_rows.append({
                        country_key: country,
                        f'{var_type}-nn': GDP_nn,
                        f'{var_type}-SERIES_CODE': GDP_series,
                    })

            gdp_rows = add_rows(main_rows, var_type='GDP')
            df_main_new = pd.DataFrame(main_rows)
            df_gdp_new = pd.DataFrame(gdp_rows)

            # --- CPI sheet ---
            df_cpi_new = pd.DataFrame(
                [{country_key: m.get(country_key, ""), **(m.get("CPI") or {})} for m in meta]
            )

            def order_cols(df):
                if country_key not in df.columns:
                    return df
                cols = [c for c in df.columns if c != country_key]
                return df[[country_key] + sorted(cols)]

            df_gdp_new = order_cols(df_gdp_new)
            df_cpi_new = order_cols(df_cpi_new)

            # --- read old + prepend ---
            if xlsx_path.exists():
                df_main_old = pd.read_excel(xlsx_path, sheet_name="main")
                df_gdp_old = pd.read_excel(xlsx_path, sheet_name="GDP")
                df_cpi_old = pd.read_excel(xlsx_path, sheet_name="CPI")

                df_main = pd.concat([df_main_new, df_main_old], ignore_index=True)
                df_gdp = pd.concat([df_gdp_new, df_gdp_old], ignore_index=True)
                df_cpi = pd.concat([df_cpi_new, df_cpi_old], ignore_index=True)
            else:
                df_main, df_gdp, df_cpi = df_main_new, df_gdp_new, df_cpi_new

            # blanks instead of NaN/NaT
            df_main = df_main.fillna("")
            df_gdp = df_gdp.fillna("")
            df_cpi = df_cpi.fillna("")

            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df_main.to_excel(writer, sheet_name="main", index=False)
                df_gdp.to_excel(writer, sheet_name="GDP", index=False)
                df_cpi.to_excel(writer, sheet_name="CPI", index=False)

        from collections.abc import Sequence

        # ----- Function to created repeated rows when there are multi-valued
        #       dictionary entries
        def to_rows(record: dict) -> list[dict]:
            """
            Expand a dict where some values are multi-valued (list/tuple/set)
            into multiple row dicts.

            Rule:
            - multi-valued fields are zipped together by position
            - scalar fields are repeated on every row
            - all multi-valued fields must have the same length
            """

            def is_multi(v):
                return isinstance(v, Sequence) and not isinstance(v, (str, bytes))

            # Normalize sets to a stable list (optional: sort for determinism)
            normalized = {}
            for k, v in record.items():
                if isinstance(v, set):
                    normalized[k] = sorted(v)
                else:
                    normalized[k] = v

            multi_keys = [k for k, v in normalized.items() if is_multi(v)]
            if not multi_keys:
                return [normalized]

            lengths = {k: len(normalized[k]) for k in multi_keys}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Multi-valued fields must have same length; got {lengths}")

            n = next(iter(lengths.values()))
            rows = []
            for i in range(n):
                row = {}
                for k, v in normalized.items():
                    row[k] = v[i] if k in multi_keys else v
                rows.append(row)
            return rows

        # --- Example ---
        # record = {"A": [1, 3, 6], "B": 2, "C": ["a", "b", "w"]}
        # df = pd.DataFrame(to_rows(record))

        # ----- Function to build the country-quarter panel
        CACHE_DIR = "../data/imf_cache"

        def build_raw_macro_panel(countries, cache_dir=CACHE_DIR, save_combined=True,
                                  combined_path="../data/raw_macro_panel.csv",
                                  rebuild_country_files=False):
            from datetime import datetime
            # REQUIRED_COLS = ["GDP_YoY", "CPI_YoY", "CA_YoY"]
            # REQUIRED_COLS = ["GDP_YoY", "CPI_YoY", "CA_YoY"]
            # REQUIRED_COLS = ["GDP_YoY", "CPI_YoY"]
            # If CPIF_YoY data exists, will include
            REQUIRED_COLS = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]

            os.makedirs(cache_dir, exist_ok=True)
            all_panels = []
            all_metas = []
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for c in countries:
                meta = {'country': c, 'date': date_str}  # initialize dictionary
                cache_path = os.path.join(cache_dir, f"gvar_macro_{c}.csv")

                if os.path.exists(cache_path):
                    print(f"[LOAD] {c}")
                    meta['exists'] = True
                    df_country = pd.read_csv(cache_path, parse_dates=["quarter"])

                    missing = [col for col in REQUIRED_COLS + ["country"] if col not in df_country.columns]
                    if not missing and not rebuild_country_files:
                        all_panels.append(df_country)
                        continue # continue with the next country
                    else:
                        if missing:
                            print(f"[REBUILD] {c}: cache missing {missing}")
                        elif rebuild_country_files:
                            print(f"[REBUILD ALL] {c}")

                print(f"[FETCH] {c}")  # if df_country is NG, then generate
                [ df,meta ] = get_macro_data(c,meta) # get data from IFS_GDP_DF
                if df.empty:           # if no data then next country
                    continue
                # if data exists, then do some checks and then save

                df = df.reset_index()
                # df_temp["quarter"] = pd.to_datetime(df_temp["quarter"])
                df["country"] = c  # ensure it exists for sorting/merging

                missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
                if missing_cols:
                    print(f"[WARN] {c}: missing required columns {missing_cols}")
                    continue

                # df_temp = df_temp.dropna(subset=REQUIRED_COLS, how="any")
                # TEMPORARY: do not drop rows yet
                if df[REQUIRED_COLS].notna().sum().min() == 0:
                    print(f"[WARN] {c}: required variables exist but not aligned yet")

                if df.empty:
                    print(f"[WARN] {c}: no usable data after cleaning")
                    continue
                if df.shape[0] < 40:
                    print(f"[WARN] {c}: short sample ({df.shape[0]} quarters)")

                print(
                    f"[OK] {c}: "
                    f"{df['quarter'].min().date()} → {df['quarter'].max().date()} "
                    f"({len(df)} quarters)"
                )

                df.to_csv(cache_path, index=False) # gvar_macro_{c}.csv
                print(f"[SAVE] {cache_path}")

                all_metas.append(meta)
                all_panels.append(df)

            if not all_panels:
                print("[WARN] No country data loaded — check sources/caches.")
                return pd.DataFrame()
            else:
                df_new = pd.DataFrame(all_metas)
                file_path = "../data/all_metas.xlsx"
                write_meta_excel(all_metas, file_path)

                print(f"[SAVE] {file_path}")

            panel = pd.concat(all_panels, ignore_index=True)
            panel = panel.sort_values(["country", "quarter"])

            if save_combined:
                panel.to_csv(combined_path, index=False)
                print(f"[SAVE] {combined_path}")

            return panel

        # panel is unused
        panel = build_raw_macro_panel(COUNTRY_RUN, save_combined=flag_save_combined,
                                      rebuild_country_files=True)

    if flag_build_gvar_macro_panel:
        # ----- BUILD COMBINED PANEL FOR ALL COUNTRIES
        #       THE FOLLOWING ARE BUILT BY build_raw_macro_panel()
        #       gvar_macro_{c}.csv IF NECESSARY, USING IFS_GDP_DF IN get_macro_data(c)
        #       raw_macro_panel.csv
        #       THE FOLLOWING IS BUILT BELOW
        #       gvar_macro_panel_cached.csv
        # 1. Build raw macro panel
        panel = build_gvar_macro_panel(
            macro_path="../data/raw_macro_panel.csv",   # or however you assemble it
            external_path="../data/external_global_drivers.csv",
        )

        # 2. Clean quarterly structure
        panel = clean_quarterly_panel(panel)

        # 3. Save canonical macro panel
        panel.to_csv("../data/gvar_macro_panel_cached.csv", index=False)
