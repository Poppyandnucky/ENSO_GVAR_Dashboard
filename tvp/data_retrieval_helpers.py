def imf_fetch(database, country, series, freq):
    """
    database: 'IFS' or 'DOTS'
    freq: 'M' or 'Q'
    """
    url = (
        f"https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/"
        f"{database}/{freq}.{country}.{series}"
    )
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()

    try:
        obs = data["CompactData"]["DataSet"]["Series"]["Obs"]
    except KeyError:
        return pd.DataFrame(columns=["date", "value"])

    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["@TIME_PERIOD"])
    df["value"] = pd.to_numeric(df["@OBS_VALUE"], errors="coerce")
    return df[["date", "value"]]


def process_cpi(country):
    df = imf_fetch("IFS", country, IFS_SERIES["CPI"], "M")
    df = df[(df["date"].dt.year >= START_YEAR) & (df["date"].dt.year <= END_YEAR)]
    df["quarter"] = df["date"].dt.to_period("Q").dt.to_timestamp()
    q = df.groupby("quarter")["value"].mean().to_frame("CPI")
    q["CPI_YoY"] = q["CPI"].pct_change(4) * 100
    return q[["CPI_YoY"]]


def process_gdp(country):
    df = imf_fetch("IFS", country, IFS_SERIES["GDP_NOM_Q"], "Q")
    df = df[(df["date"].dt.year >= START_YEAR) & (df["date"].dt.year <= END_YEAR)]
    df = df.rename(columns={"date": "quarter", "value": "GDP_NOM"})
    df["GDP_YoY"] = df["GDP_NOM"].pct_change(4) * 100
    return df.set_index("quarter")[["GDP_YoY"]]


def process_fx(country):
    df = imf_fetch("IFS", country, IFS_SERIES["FX"], "M")
    df = df[(df["date"].dt.year >= START_YEAR) & (df["date"].dt.year <= END_YEAR)]
    df["quarter"] = df["date"].dt.to_period("Q").dt.to_timestamp()
    q = df.groupby("quarter")["value"].mean().to_frame("FX")
    q["FX_YoY"] = q["FX"].pct_change(4) * 100
    return q[["FX_YoY"]]


def process_exports(country):
    df = imf_fetch("DOTS", country, DOTS_SERIES["EXPORTS"], "M")
    df = df[(df["date"].dt.year >= START_YEAR) & (df["date"].dt.year <= END_YEAR)]
    df["quarter"] = df["date"].dt.to_period("Q").dt.to_timestamp()
    q = df.groupby("quarter")["value"].sum().to_frame("EXPORTS")
    q["EXPORTS_YoY"] = q["EXPORTS"].pct_change(4) * 100
    return q[["EXPORTS_YoY"]]
