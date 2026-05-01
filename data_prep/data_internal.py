import pandas as pd
import numpy as np
import os
import re
import pandas as pd

from xarray.computation.ops import NAN_REDUCE_METHODS

from models.config import (ISO3_TO_IMF_NAME, COUNTRY_CODES, COUNTRY_NAMES, START_DATE, END_DATE,
                           ISO3_TO_EURO_RATE, ISO3_TO_EURO_CONVERSION_QTR, ISO3,
                           iso3_to_imf_EX_dict)

# BUILD CSV FILES
# imf_cache/gvar_macro_{c}.csv
# data/raw_macro_panel.csv [UNUSED, SAME AS gvar_macro_panel_cached]
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
def select_row(meta,sub,time_cols,var_type='GDP',fx_df=None):
    # --- choose best row (the lists below are row-order preserving)
    if var_type=='FX':
        country = meta['country']
        if country in ISO3_TO_EURO_RATE:
            sub_euro = fx_df[fx_df["COUNTRY"] == "Euro Area (EA)"]
            s_euro = pd.to_numeric(sub_euro.iloc[0][time_cols])

            for idx, row in sub.iterrows():
                # convert time columns to numeric so NaNs behave properly
                s = pd.to_numeric(row[time_cols], errors="coerce")
                s[s.first_valid_index():s.last_valid_index()] /= ISO3_TO_EURO_RATE[country]
                s_last_valid_position = s.index.get_loc(s.last_valid_index())
                s_first_euro_index = s.index[s_last_valid_position + 1]
                if not s_first_euro_index == ISO3_TO_EURO_CONVERSION_QTR[country]:
                    raise ValueError(f"euro start date not right for {country}")
                s[s_first_euro_index:] = s_euro[s_first_euro_index:]
                sub.loc[idx, time_cols] = s

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
    if var_type=='GDP':
        price_types = []
        adjustments = []
        transformations = []
    elif var_type=='FX':
        indicators = []
        transformations = []
    elif var_type=='EX':
        indicators = []

    for _, row in sub.iterrows():
        # convert time columns to numeric so NaNs behave properly
        s = pd.to_numeric(row[time_cols], errors="coerce")
        if var_type=='GDP':
            price_types.append(row["PRICE_TYPE"])
            adjustments.append(row["S_ADJUSTMENT"])
            transformations.append(row["TYPE_OF_TRANSFORMATION"])
        elif var_type=='FX':
            indicators.append(row["INDICATOR"])
            transformations.append(row["TYPE_OF_TRANSFORMATION"])
        elif var_type=='EX':
            indicators.append(row["INDICATOR"])
        first_dates.append(s.first_valid_index())
        last_dates.append(s.last_valid_index())

    meta["first-date"] = first_dates
    meta["last-date"] = last_dates
    if var_type=='GDP':
        meta["price-types"] = price_types
        meta["adjustments"] = adjustments
        meta["transformations"] = transformations
    elif var_type=='FX':
        meta["indicators"] = indicators
        meta["transformations"] = transformations
    elif var_type=='EX':
        meta["indicators"] = indicators

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
    meta["time_cols-exists"] = bool(time_cols)

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

# -------------------------------------------------
# Core FRED helpers
# -------------------------------------------------
import requests
import time

FRED_BASE = "https://api.stlouisfed.org/fred"
FRED_API_KEY = "9ea7ca4ddf75813ac88c303dd72b748c"

def fred_get(endpoint: str, params: dict, timeout: int = 30) -> dict:
    url = f"{FRED_BASE}/{endpoint.lstrip('/')}"
    params = dict(params)
    params["file_type"] = "json"
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fred_series_search(api_key: str,
                       search_text: str,
                       frequency: str = "Quarterly",
                       limit: int = 1000) -> pd.DataFrame:

    j = fred_get(
        "series/search",
        params={
            "api_key": api_key,
            "search_text": search_text,
            "search_type": 'full_text',
            "filter_variable": "frequency",
            "filter_value": frequency,
            "limit": limit,
            "order_by": "popularity",
            "sort_order": "desc",
        },
    )

    return pd.DataFrame(j.get("seriess", []))

def pick_best_gdp_series(results: pd.DataFrame) -> dict | None:
    """
    Preference order:
    1) Units contain 'Index', '2015', '100'
    2) Seasonal adjustment contains 'Seasonally Adjusted'
    3) Otherwise first result
    """

    if results.empty:
        return None

    df = results.copy()

    for col in ["units", "seasonal_adjustment", "title"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # 1) Index 2015=100
    idx_mask = (
        df["units"].str.contains("index", case=False, na=False)
        & df["units"].str.contains("2015", case=False, na=False)
        & df["units"].str.contains("100", case=False, na=False)
    )

    df_idx = df[idx_mask]
    if not df_idx.empty:
        return df_idx.iloc[0].to_dict()

    # 2) Seasonally Adjusted
    sa_mask = df["seasonal_adjustment"].str.contains("seasonally adjusted", case=False, na=False)
    df_sa = df[sa_mask]
    if not df_sa.empty:
        return df_sa.iloc[0].to_dict()

    # 3) fallback
    return df.iloc[0].to_dict()

def fred_series_observations(api_key: str,
                             series_id: str,
                             observation_start: str | None = None,
                             observation_end: str | None = None) -> pd.DataFrame:

    params = {
        "api_key": api_key,
        "series_id": series_id
    }

    if observation_start:
        params["observation_start"] = observation_start
    if observation_end:
        params["observation_end"] = observation_end

    j = fred_get("series/observations", params=params)

    obs = pd.DataFrame(j.get("observations", []))
    if obs.empty:
        return obs

    obs["date"] = pd.to_datetime(obs["date"])
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")

    return obs[["date", "value"]].sort_values("date").reset_index(drop=True)

# Panel builder
def build_quarterly_gdp_panel(api_key: str, ISO3_list: list,
    observation_start: str | None = None, observation_end: str | None = None,
    sleep_between_calls: float = 0.5):

    # Use ido_NG_list to prevent using too many API calls

    panel_data = []
    metadata = []
    iso_NG_list = ['Afghanistan', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda', 'Armenia',
                   'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belize',
                   'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana',
                   'Brunei Darussalam', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia',
                   'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo, Republic of',
                   'Congo, Democratic Republic of the', "Côte d'Ivoire", 'Cuba', 'Djibouti',
                   'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador',
                   'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Fiji', 'Gabon', 'Gambia',
                   'Ghana', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti',
                   'Honduras', 'Iran', 'Iraq', 'Jamaica', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati',
                   'Kuwait', 'Kyrgyz Republic', 'Lao PDR', 'Lebanon', 'Lesotho', 'Liberia', 'Libya',
                   'Liechtenstein', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali',
                   'Marshall Islands', 'Mauritania', 'Mauritius', 'Micronesia', 'Moldova', 'Monaco',
                   'Mongolia', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal',
                   'Nicaragua', 'Niger', 'Nigeria', 'Korea, North', 'North Macedonia', 'Oman',
                   'Pakistan', 'Palau', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines',
                   'Qatar', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia',
                   'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe',
                   'Senegal', 'Seychelles', 'Sierra Leone', 'Singapore', 'Solomon Islands', 'Somalia',
                   'South Sudan', 'Sri Lanka', 'Sudan', 'Suriname', 'Syria', 'Tajikistan', 'Tanzania',
                   'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia',
                   'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine']

    for iso in [iso for iso in ISO3_list if iso not in iso_NG_list]:
        print(f"Processing {iso}...")

        # Search using ISO3 directly
        search_df = fred_series_search(
            api_key=api_key,
            # search_text=f"{iso} gross domestic product",
            search_text=f"{iso} gross domestic product",
            frequency="Quarterly"
        )

        chosen = pick_best_gdp_series(search_df)

        if chosen is None:
            print(f"  No GDP found for {iso}")
            iso_NG_list.append(iso)
            print(iso_NG_list)
            continue

        try:
            gdp = fred_series_observations(
                api_key,
                chosen["id"],
                observation_start,
                observation_end
            )

            if gdp.empty:
                print(f"  No observations for {iso}")
                continue

            gdp["country"] = iso
            gdp["series_id"] = chosen["id"]
            gdp["units"] = chosen.get("units", "")
            gdp["seasonal_adjustment"] = chosen.get("seasonal_adjustment", "")

            panel_data.append(gdp)

            metadata.append({
                "country": iso,
                "series_id": chosen["id"],
                "title": chosen.get("title", ""),
                "units": chosen.get("units", ""),
                "seasonal_adjustment": chosen.get("seasonal_adjustment", "")
            })

            print(f"  ✓ Using {chosen['id']} ({chosen.get('units','')})")

        except Exception as e:
            print(f"  Error for {iso}: {e}")

        time.sleep(sleep_between_calls)  # avoid rate limiting

    panel_df = pd.concat(panel_data, ignore_index=True) if panel_data else pd.DataFrame()
    meta_df = pd.DataFrame(metadata)

    return panel_df, meta_df

# Wide format (compatible with IMF)
def build_imf_style_gdp_csv(panel_df, meta_df,
                            start_year=1950,
                            end_year=2025):

    df = panel_df.copy()

    # Convert to quarter labels like 1950-Q1
    df["quarter_str"] = (
        df["date"]
        .dt.to_period("Q")
        .astype(str)
        .str.replace("Q", "-Q")
    )

    # Pivot to wide format
    wide = (
        df.pivot(index="country",
                 columns="quarter_str",
                 values="value")
        .sort_index(axis=1)
    )

    # ---- Ensure all quarters exist as columns
    all_quarters = pd.period_range(
        start=f"{start_year}Q1",
        end=f"{end_year}Q4",
        freq="Q"
    )

    all_cols = [str(q).replace("Q", "-Q") for q in all_quarters]

    for col in all_cols:
        if col not in wide.columns:
            wide[col] = np.nan

    wide = wide[all_cols]  # reorder chronologically

    # ---- Merge metadata
    meta_small = meta_df.set_index("country")

    wide = wide.merge(meta_small, left_index=True, right_index=True, how="left")

    # ---- Add IMF-style metadata columns
    wide.insert(0, "DATASET", "FRED")
    wide.insert(1, "SERIES_CODE", wide["series_id"])
    wide.insert(2, "OBS_MEASURE", "OBS_VALUE")
    wide.insert(3, "COUNTRY", wide.index)
    wide.insert(4, "INDICATOR", "Gross domestic product (GDP)")
    wide.insert(5, "PRICE_TYPE", wide["units"])
    wide.insert(6, "S_ADJUSTMENT", wide["seasonal_adjustment"])
    wide.insert(7, "TYPE_OF_TRANSFORMATION", "")
    wide.insert(8, "FREQUENCY", "Quarterly")
    wide.insert(9, "SCALE", "")

    # Drop internal columns
    wide = wide.drop(columns=["series_id", "units", "seasonal_adjustment"],
                     errors="ignore")

    return wide.reset_index(drop=True)

# ----- Exchange rate → YoY depreciation
def process_fx_from_csv(country, date_str, fx_df):
    meta = {'country': country, 'date': date_str}
    imf_name = ISO3_TO_IMF_NAME.get(country, country)
    sub = fx_df[fx_df["COUNTRY"] == imf_name]

    if sub.empty:
        print(f"[SKIP] FX CSV for {country} (no country match)")
        return pd.DataFrame(), meta

    # --- Identify quarterly columns (NOT monthly)
    time_cols = imf_quarter_cols(fx_df)
    if time_cols:
        freq = "Q"
    else:
        freq = ""
    meta["q-time_cols"] = freq
    meta["time_cols-exists"] = bool(time_cols)

    if not time_cols:
        print(f"[SKIP] FX CSV for {country} (no quarterly columns)")
        return pd.DataFrame(), meta

    # --- Pick series with most data, and insert EURO if necessary
    s = select_row(meta, sub, time_cols, var_type='FX', fx_df=fx_df)
    s.index = imf_cols_to_datetime(time_cols, freq="Q")
    s = s.dropna().sort_index()

    # --- Filter years
    s = s[(s.index.year >= START_YEAR) & (s.index.year <= END_YEAR)]
    meta["q-exists"] = not s.empty
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
def process_exports_from_csv(country, date_str, ex_df, ex_str):
    meta = {'country': country, 'date': date_str}
    # imf_name = ISO3_TO_IMF_NAME.get(country, country) # if not found, set imf_name=country
    imf_name = iso3_to_imf_EX_dict.get(country, country)

    sub = ex_df[
        (ex_df["COUNTRY"] == imf_name)
        # & (ex_df["COUNTERPART_COUNTRY"] == "World")
    ]

    if sub.empty:
        print(f"[SKIP] {ex_str} CSV for {country} (not found)")
        return pd.DataFrame(), meta

    # quarterly columns
    time_cols = imf_quarter_cols(ex_df)
    if time_cols:
        freq = "Q"
    else:
        freq = ""
    meta["q-time_cols"] = freq
    meta["time_cols-exists"] = bool(time_cols)

    if not time_cols:
        print(f"[SKIP] {ex_str} CSV for {country} (no quarterly columns)")
        return pd.DataFrame(), meta

    # pick longest series
    s = select_row(meta, sub, time_cols, var_type=ex_str)
    s.index = imf_cols_to_datetime(time_cols, freq="Q")
    s = s.dropna().sort_index()

    # year filter
    s = s[(s.index.year >= START_YEAR) & (s.index.year <= END_YEAR)]
    meta["q-exists"] = not s.empty

    if s.empty:
        print(f"[SKIP] {ex_str} CSV after date filter for {country}")
        return pd.DataFrame(), meta

    q = s.to_frame(ex_str)

    # YoY growth
    q[f"{ex_str}_YoY"] = q[ex_str].pct_change(4, fill_method=None) * 100
    # align to quarter-start (QS)
    s.index = s.index.to_period("Q").to_timestamp(how="start")

    return q[[f"{ex_str}_YoY"]], meta

def build_gvar_macro_panel( macro_path="data/gvar_macro_panel_cached.csv",
                            external_path="data/external_global_drivers.csv",
                            start=START_DATE, end=END_DATE, keep_vars=None, ):
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
        macro_vars = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY", "IM_YoY", "BAL_YoY", "CPIF_YoY"]

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
    # ----- FRED GDP [ONLY FOR INITIAL RUNS]
    flag_fred_gdp = False
    if flag_fred_gdp:
        panel_df, meta_df = build_quarterly_gdp_panel(
            api_key=FRED_API_KEY,
            ISO3_list=COUNTRY_NAMES,
            # ISO3_list=ISO3,
            observation_start=START_DATE,
        )

        print(panel_df.head())
        print(meta_df)
        print(f"Done downloading FRED GDP")

        wide_df = build_imf_style_gdp_csv(panel_df, meta_df)
        wide_df.to_csv("data/fred_gdp_imf_style.csv", index=False)
        print("Saved IMF-style wide GDP file.")

        exit()

    # ----- Wrappers: get_macro_data(country) and extract quarterly time series
    # need to load CSVs (only if this is run)
    flag_build_raw_csv = True  # build gvar_macro_{c}.csv
    flag_save_files = True # save gvar_macro{c}.csv
    flag_build_gvar_macro_panel = True # gvar_macro_panel_cached.csv
    flag_save_combined = True  # raw_macro_panel.csv [UNUSED]
    flag_save_meta = True # all_metas_ALL.xlsx
    file_path = "../data/all_metas_ALL.xlsx" # change as needed

    if flag_build_raw_csv:
        # BUILD PANELS FOR INDIVIDUAL COUNTIRES
        # use [:1] for one country, [:6] for half, [:] for all
        # [8] India, [7] SA

        COUNTRY_RUN = COUNTRY_CODES[:]
        # COUNTRY_RUN = ["China, People's Republic of"]
        print("Running countries:", COUNTRY_RUN)
        if len(COUNTRY_RUN) == 1:
            flag_build_gvar_macro_panel = False

        IFS_CPI_DF = pd.read_csv("../data_large/IFS_CPI.csv",
                                 dtype={14:str,15:str,24:str})
        IFS_GDP_DF = pd.read_csv("../data_large/IFS_GDP.csv")
        IFS_FX_DF = pd.read_csv("../data_large/IFS_FX.csv")
        IFS_EX_DF = pd.read_csv("../data_large/IFS_EX.csv")
        IFS_IM_DF = pd.read_csv("../data_large/IFS_IM.csv")
        IFS_BAL_DF = pd.read_csv("../data_large/IFS_BAL.csv")

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
            from collections.abc import Sequence

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

        def add_rows(meta, main_rows, country_key, var_type='GDP'):
            gdp_rows = []
            for m in meta:
                country = m.get(country_key, "")
                gdp = m.get(var_type) or {}
                found = False
                # Expand rows for repeated values, e.g. SERIES_CODES
                #   need enumerate to get index i
                # Default values, changed only if row['selected'] is found
                GDP_nn = np.nan
                GDP_series = ''

                for i, row in enumerate(to_rows(gdp)):
                    gdp_rows.append(row)
                    if row.get('selected') == True:
                        GDP_nn = row['nn']
                        GDP_series = row['SERIES_CODES']

                for row in main_rows:
                    if row.get("country") == country:
                        row[f'{var_type}-nn'] = GDP_nn
                        row[f'{var_type}-series'] = GDP_series
                        found = True
                        break

                if not found:
                    main_rows.append({
                        "country": country,
                        f'{var_type}-nn': GDP_nn,
                        f'{var_type}-series': GDP_series,
                    })

            return gdp_rows

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
            gdp_rows = add_rows(meta, main_rows, country_key, var_type='GDP')
            df_gdp_new = pd.DataFrame(gdp_rows)

            # --- CPI sheet ---
            cpi_rows = add_rows(meta, main_rows, country_key, var_type='CPI')
            df_cpi_new = pd.DataFrame(cpi_rows)

            # --- CPIF sheet ---
            cpif_rows = add_rows(meta, main_rows, country_key, var_type='CPIF')
            df_cpif_new = pd.DataFrame(cpif_rows)

            # --- FX sheet ---
            fx_rows = add_rows(meta, main_rows, country_key, var_type='FX')
            df_fx_new = pd.DataFrame(fx_rows)

            # --- EX sheet ---
            ex_rows = add_rows(meta, main_rows, country_key, var_type='EX')
            im_rows = add_rows(meta, main_rows, country_key, var_type='IM')
            ba_rows = add_rows(meta, main_rows, country_key, var_type='BAL')
            df_ex_new = pd.DataFrame(ex_rows)
            df_im_new = pd.DataFrame(im_rows)
            df_ba_new = pd.DataFrame(ba_rows)

            df_main_new = pd.DataFrame(main_rows)

            def order_cols(df):
                if country_key not in df.columns:
                    return df
                cols = [c for c in df.columns if c != country_key]
                return df[[country_key] + sorted(cols)]

            df_gdp_new = order_cols(df_gdp_new)
            df_cpi_new = order_cols(df_cpi_new)
            df_cpif_new = order_cols(df_cpif_new)
            df_fx_new = order_cols(df_fx_new)
            df_ex_new = order_cols(df_ex_new)
            df_im_new = order_cols(df_im_new)
            df_ba_new = order_cols(df_ba_new)

            # --- read old + prepend ---
            if xlsx_path.exists():
                df_main_old = pd.read_excel(xlsx_path, sheet_name="main")
                df_gdp_old = pd.read_excel(xlsx_path, sheet_name="GDP")
                df_cpi_old = pd.read_excel(xlsx_path, sheet_name="CPI")
                df_cpif_old = pd.read_excel(xlsx_path, sheet_name="CPIF")
                df_fx_old = pd.read_excel(xlsx_path, sheet_name="FX")
                df_ex_old = pd.read_excel(xlsx_path, sheet_name="EX")
                df_im_old = pd.read_excel(xlsx_path, sheet_name="IM")
                df_ba_old = pd.read_excel(xlsx_path, sheet_name="BAL")

                df_main = pd.concat([df_main_new, df_main_old], ignore_index=True)
                df_gdp = pd.concat([df_gdp_new, df_gdp_old], ignore_index=True)
                df_cpi = pd.concat([df_cpi_new, df_cpi_old], ignore_index=True)
                df_cpif = pd.concat([df_cpif_new, df_cpif_old], ignore_index=True)
                df_fx = pd.concat([df_fx_new, df_fx_old], ignore_index=True)
                df_ex = pd.concat([df_ex_new, df_ex_old], ignore_index=True)
                df_im = pd.concat([df_im_new, df_im_old], ignore_index=True)
                df_ba = pd.concat([df_ba_new, df_ba_old], ignore_index=True)
            else:
                df_main, df_gdp, df_cpi, df_cpif, df_fx, df_ex, df_im, df_ba = (df_main_new,
                    df_gdp_new, df_cpi_new, df_cpif_new, df_fx_new, df_ex_new, df_im_new, df_ba_new)

            # blanks instead of NaN/NaT
            df_main = df_main.fillna("")
            df_gdp = df_gdp.fillna("")
            df_cpi = df_cpi.fillna("")
            df_cpif = df_cpif.fillna("")
            df_fx = df_fx.fillna("")
            df_ex = df_ex.fillna("")
            df_im = df_im.fillna("")
            df_ba = df_ba.fillna("")

            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df_main.to_excel(writer, sheet_name="main", index=False)
                df_gdp.to_excel(writer, sheet_name="GDP", index=False)
                df_cpi.to_excel(writer, sheet_name="CPI", index=False)
                df_cpif.to_excel(writer, sheet_name="CPIF", index=False)
                df_fx.to_excel(writer, sheet_name="FX", index=False)
                df_ex.to_excel(writer, sheet_name="EX", index=False)
                df_im.to_excel(writer, sheet_name="IM", index=False)
                df_ba.to_excel(writer, sheet_name="BAL", index=False)

            # # ----- Function to build the country-quarter panel
            # CACHE_DIR = "../data/imf_cache"
            #
            # # panel is unused
            # panel = build_raw_macro_panel(COUNTRY_RUN, save_combined=flag_save_combined,
            #                               rebuild_country_files=True)

        def get_macro_data(country,meta):
            """
            Retrieve macroeconomic data for one country using CSV-only sources.
            Returns a quarterly DataFrame indexed by quarter.
            """

            def check_if_empty(cpif,meta_cpif,dfs,cpif_str="CPIF"):
                meta[f"{cpif_str}-exists"] = not cpif.empty
                meta[cpif_str] = meta_cpif
                if not cpif.empty:
                    dfs.append(cpif)
                else:
                    print(f"[MISS] {cpif_str} CSV for {country}")

            dfs = []
            date_str = meta["date"]

            # --- GDP (quarterly, IMF topic CSV)
            # IFS_GDP_DF is a df that was read from a csv file in /data_large
            gdp,meta_gdp = process_gdp_from_csv(country, date_str, IFS_GDP_DF)
            check_if_empty(gdp,meta_gdp,dfs,cpif_str="GDP")

            # --- CPI (monthly → quarterly, IMF CPI CSV)
            cpi,meta_cpi = process_cpi_from_csv(country, date_str, IFS_CPI_DF)
            check_if_empty(cpi,meta_cpi,dfs,cpif_str="CPI")

            # --- CPIF (monthly → quarterly, IMF CPIF CSV)
            cpif,meta_cpif = process_cpi_from_csv(country, date_str, IFS_CPI_DF, food=True)
            check_if_empty(cpif,meta_cpif,dfs,cpif_str="CPIF")

            # --- FX (monthly → quarterly, IMF FX CSV)
            fx,meta_fx = process_fx_from_csv(country, date_str, IFS_FX_DF)
            check_if_empty(fx,meta_fx,dfs,cpif_str="FX")

            # --- Exports
            ex,meta_ex = process_exports_from_csv(country, date_str, IFS_EX_DF,  "EX")
            im,meta_im = process_exports_from_csv(country, date_str, IFS_IM_DF,  "IM")
            ba,meta_ba = process_exports_from_csv(country, date_str, IFS_BAL_DF, "BAL")
            check_if_empty(ex,meta_ex,dfs,cpif_str="EX")
            check_if_empty(im,meta_im,dfs,cpif_str="IM")
            check_if_empty(ba,meta_ba,dfs,cpif_str="BAL")

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
        # ----- Function to build the country-quarter panel
        CACHE_DIR = "../data/imf_cache"

        def build_raw_macro_panel(countries, cache_dir=CACHE_DIR, save_combined=True,
                                  combined_path="../data/gvar_macro_panel_cached.csv",
                                  rebuild_country_files=False):
            from datetime import datetime
            # REQUIRED_COLS = ["GDP_YoY", "CPI_YoY", "CA_YoY"]
            # REQUIRED_COLS = ["GDP_YoY", "CPI_YoY"]
            # REQUIRED_COLS = ["EX_YoY"]
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
                        continue  # continue with the next country
                    else:
                        if missing:
                            print(f"[REBUILD] {c}: cache missing {missing}")
                        elif rebuild_country_files:
                            print(f"[REBUILD ALL] {c}")

                print(f"[FETCH] {c}")  # if df_country is NG, then generate
                [df, meta] = get_macro_data(c, meta)  # get data from IFS_GDP_DF
                if df.empty:  # if no data then next country
                    all_metas.append(meta)
                    continue
                # if data exists, then do some checks and then save

                df = df.reset_index()
                # df_temp["quarter"] = pd.to_datetime(df_temp["quarter"])
                df["country"] = c  # ensure it exists for sorting/merging

                missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
                if missing_cols:
                    print(f"[WARN] {c}: missing required columns {missing_cols}")
                    all_metas.append(meta)
                    continue

                # df_temp = df_temp.dropna(subset=REQUIRED_COLS, how="any")
                # TEMPORARY: do not drop rows yet
                if df[REQUIRED_COLS].notna().sum().min() == 0:
                    print(f"[WARN] {c}: required variables exist but not aligned yet")

                if df.empty:
                    print(f"[WARN] {c}: no usable data after cleaning")
                    all_metas.append(meta)
                    continue
                if df.shape[0] < 40:
                    print(f"[WARN] {c}: short sample ({df.shape[0]} quarters)")

                print(
                    f"[OK] {c}: "
                    f"{df['quarter'].min().date()} → {df['quarter'].max().date()} "
                    f"({len(df)} quarters)"
                )

                if flag_save_files:
                    df.to_csv(cache_path, index=False)  # gvar_macro_{c}.csv
                    print(f"[SAVE] {cache_path}")
                else:
                    print(f"[NOT SAVED] {cache_path}")

                all_metas.append(meta)
                all_panels.append(df)

            if not all_panels:
                print("[WARN] No country data loaded — check sources/caches.")
                return pd.DataFrame()
            else:
                df_new = pd.DataFrame(all_metas)
                # ----- GENERATE main, GDP, CPI, ... SHEETS
                write_meta_excel(all_metas, file_path)

                print(f"[SAVE] {file_path}")

            panel = pd.concat(all_panels, ignore_index=True)
            panel = panel.sort_values(["country", "quarter"])

            if save_combined:
                panel.to_csv(combined_path, index=False)
                print(f"[SAVE] {combined_path}")

            return panel

        from collections.abc import Sequence

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
            macro_path="../data/gvar_macro_panel_cached.csv",   # or however you assemble it
            external_path="../data/external_global_drivers.csv",
        )

        # 2. Clean quarterly structure
        panel = clean_quarterly_panel(panel)

        # 3. Save canonical macro panel
        panel.to_csv("../data/gvar_macro_panel_cached.csv", index=False)
