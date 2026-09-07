"""Merge crop-climate and oil exogenous variables into the production panel.

Replaces the earlier crop_weighted_climate.csv / weighted_heat / weighted_dryness
approach (analysis/build_crop_weighted_climate.py, now removed): those columns
are dropped here, and HeatDry / HeatDryYoY from analysis/validation/yield_climate_total.csv
take their place as the two crop-climate exogenous drivers.

analysis/validation/yield_climate_total.csv columns: Country (name), Year, Quarter (1-4),
HeatDry, HeatDryYoY -- one row per country-quarter for the 12 study countries,
already extending through 2027Q4 (i.e. it already carries projected/estimated
values for future quarters, not just historical observations).

analysis/Brent_Crude_Oil_Spot_Price.csv columns: Quarter, Brent Crude Oil Spot
Price dollars per barrel. This script computes OIL_YoY as the quarter-over-same-
quarter-prior-year fractional change, then merges it to every country row by quarter.

analysis/validation/enso_quarterly_merged_with_screenshot.csv columns: quarter,
index. The index column is merged as IOD by quarter.

analysis/validation/ENSO_climate_total.csv or analysis/ENSO_climate_total.csv
columns: date, RONI, RONI_lower_2.5%, RONI_upper_97.5%. RONI is averaged from monthly to quarterly and merged as
the panel ENSO column. Forecast rows with source=forecast_50pct also update
FORECAST_ENSO_MEAN/MIN/MAX in the forecast/dashboard code.

Writes the merged panel back in place:
    analysis/gvar_panel_streamlit (8 + EGY + PER).csv
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "analysis" / "gvar_panel_streamlit (8 + EGY + PER).csv"
YIELD_CLIMATE_PATH = ROOT / "analysis" / "validation" / "yield_climate_total.csv"
OIL_PRICE_PATH = ROOT / "analysis" / "Brent_Crude_Oil_Spot_Price.csv"
IOD_PATH = ROOT / "analysis" / "validation" / "enso_quarterly_merged_with_screenshot.csv"
ENSO_MONTHLY_CANDIDATES = [
    ROOT / "analysis" / "validation" / "ENSO_climate_total.csv",
    ROOT / "analysis" / "ENSO_climate_total.csv",
]
ENSO_FORECAST_TARGETS = [
    ROOT / "analysis" / "Dash_Output" / "gvar_kf_forecast.py",
    ROOT / "apps" / "streamlit_TRP.py",
]
OIL_PRICE_COL = "Brent Crude Oil Spot Price dollars per barrel"
OIL_YOY_COL = "OIL_YoY"
IOD_COL = "IOD"
ENSO_COL = "ENSO"

COUNTRY_TO_ISO3 = {
    "Brazil": "BRA",
    "Chile": "CHL",
    "Colombia": "COL",
    "Egypt": "EGY",
    "India": "IND",
    "Indonesia": "IDN",
    "Kenya": "KEN",
    "Mexico": "MEX",
    "Peru": "PER",
    "Philippines": "PHL",
    "South Africa": "ZAF",
    "Thailand": "THA",
}

QUARTER_TO_MONTH = {1: 1, 2: 4, 3: 7, 4: 10}


def load_yield_climate() -> pd.DataFrame:
    df = pd.read_csv(YIELD_CLIMATE_PATH)
    df = df.rename(columns={"HeatDry2": "HeatDry", "HeatDryYoY2": "HeatDryYoY"})
    df["ISO3"] = df["Country"].map(COUNTRY_TO_ISO3)
    missing = sorted(df.loc[df["ISO3"].isna(), "Country"].unique())
    if missing:
        raise ValueError(f"Add ISO3 mappings for: {missing}")

    df["month"] = df["Quarter"].map(QUARTER_TO_MONTH)
    df["quarter"] = pd.to_datetime(
        dict(year=df["Year"], month=df["month"], day=1)
    )
    return df[["ISO3", "quarter", "HeatDry", "HeatDryYoY"]].rename(columns={"ISO3": "country"})


def enso_monthly_path() -> Path:
    for path in ENSO_MONTHLY_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "ENSO_climate_total.csv not found in analysis/validation/ or analysis/."
    )


def append_future_heatdry_rows(panel: pd.DataFrame, yield_climate: pd.DataFrame) -> pd.DataFrame:
    """Add future country-quarter skeleton rows so HeatDry can merge through source end."""
    panel = panel.copy()
    panel["quarter"] = pd.to_datetime(panel["quarter"]).dt.to_period("Q").dt.to_timestamp()
    yc = yield_climate[["country", "quarter"]].drop_duplicates().copy()
    yc["quarter"] = pd.to_datetime(yc["quarter"]).dt.to_period("Q").dt.to_timestamp()

    existing = set(zip(panel["country"], panel["quarter"]))
    rows = []
    for country, g_yc in yc.groupby("country"):
        g_panel = panel.loc[panel["country"] == country, "quarter"]
        if g_panel.empty:
            continue
        last_panel_q = g_panel.max()
        for q in sorted(g_yc.loc[g_yc["quarter"] > last_panel_q, "quarter"]):
            key = (country, q)
            if key not in existing:
                rows.append({"country": country, "quarter": q})
                existing.add(key)

    if not rows:
        return panel

    extra = pd.DataFrame(rows)
    for col in panel.columns:
        if col not in extra.columns:
            extra[col] = pd.NA
    extra = extra[panel.columns]
    if "year" in extra.columns:
        extra["year"] = pd.to_datetime(extra["quarter"]).dt.year
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
            category=FutureWarning,
        )
        out = pd.concat([panel, extra], ignore_index=True)
    return out.sort_values(["country", "quarter"]).reset_index(drop=True)


def load_oil_yoy() -> pd.DataFrame:
    df = pd.read_csv(OIL_PRICE_PATH)
    q = df["Quarter"].astype(str).str.extract(r"Q([1-4])\s+(\d{4})")
    if q.isna().any().any():
        bad = df.loc[q.isna().any(axis=1), "Quarter"].head(10).tolist()
        raise ValueError(f"Could not parse oil quarters: {bad}")

    df["quarter_num"] = q[0].astype(int)
    df["year"] = q[1].astype(int)
    df["quarter"] = [
        pd.Period(year=int(year), quarter=int(quarter), freq="Q").to_timestamp()
        for year, quarter in zip(df["year"], df["quarter_num"])
    ]
    df["oil_price"] = pd.to_numeric(df[OIL_PRICE_COL], errors="coerce")
    oil = (
        df.dropna(subset=["oil_price"])
        .sort_values("quarter")
        .groupby("quarter", as_index=False)["oil_price"]
        .mean()
    )
    oil[OIL_YOY_COL] = oil["oil_price"].pct_change(4)
    return oil[["quarter", OIL_YOY_COL]]


def load_iod() -> pd.DataFrame:
    df = pd.read_csv(IOD_PATH)
    if "quarter" not in df.columns or "index" not in df.columns:
        raise ValueError(f"{IOD_PATH} must contain quarter and index columns")
    df["quarter"] = df["quarter"].astype(str).str.strip().map(
        lambda q: pd.Period(q, freq="Q").to_timestamp()
    )
    df[IOD_COL] = pd.to_numeric(df["index"], errors="coerce")
    return (
        df.dropna(subset=[IOD_COL])
        .sort_values("quarter")
        .groupby("quarter", as_index=False)[IOD_COL]
        .mean()
    )


def load_quarterly_roni_enso() -> pd.DataFrame:
    path = enso_monthly_path()
    df = pd.read_csv(path, parse_dates=["date"])
    if "date" not in df.columns or "RONI" not in df.columns:
        raise ValueError(f"{path} must contain date and RONI columns")
    df["quarter"] = df["date"].dt.to_period("Q").dt.to_timestamp()
    df[ENSO_COL] = pd.to_numeric(df["RONI"], errors="coerce")
    df["ENSO_lower"] = pd.to_numeric(df.get("RONI_lower_2.5%"), errors="coerce")
    df["ENSO_upper"] = pd.to_numeric(df.get("RONI_upper_97.5%"), errors="coerce")
    keep = df.dropna(subset=[ENSO_COL])
    return (
        keep.sort_values("date")
        .groupby("quarter", as_index=False)[[ENSO_COL, "ENSO_lower", "ENSO_upper"]]
        .mean()
    )


def load_forecast_roni_enso() -> pd.DataFrame:
    path = enso_monthly_path()
    df = pd.read_csv(path, parse_dates=["date"])
    needed = {"date", "source", "RONI", "RONI_lower_2.5%", "RONI_upper_97.5%"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[df["source"].astype(str).eq("forecast_50pct")].copy()
    df["quarter"] = df["date"].dt.to_period("Q").dt.to_timestamp()
    df["mean"] = pd.to_numeric(df["RONI"], errors="coerce")
    df["min"] = pd.to_numeric(df["RONI_lower_2.5%"], errors="coerce")
    df["max"] = pd.to_numeric(df["RONI_upper_97.5%"], errors="coerce")
    out = (
        df.dropna(subset=["mean"])
        .sort_values("date")
        .groupby("quarter", as_index=False)[["mean", "min", "max"]]
        .mean()
    )
    out["quarter_label"] = out["quarter"].dt.to_period("Q").astype(str)
    return out


def _format_enso_dict(values: pd.Series) -> str:
    lines = ["{"]
    for quarter, value in values.items():
        lines.append(f'    "{quarter}": {float(value):.4f},')
    lines.append("}")
    return "\n".join(lines)


def _replace_dict_literal(text: str, name: str, values: pd.Series) -> str:
    text_new, n = re.subn(
        rf"{name}\s*=\s*\{{.*?\n\}}",
        f"{name} = {_format_enso_dict(values)}",
        text,
        count=1,
        flags=re.S,
    )
    if n != 1:
        raise ValueError(f"Could not replace {name}")
    return text_new


def update_enso_forecast_constants() -> None:
    forecast = load_forecast_roni_enso().set_index("quarter_label")
    replacements = {
        "FORECAST_ENSO_MEAN": forecast["mean"],
        "FORECAST_ENSO_MIN": forecast["min"],
        "FORECAST_ENSO_MAX": forecast["max"],
        "ENSO_FORECAST_MEAN": forecast["mean"],
        "ENSO_FORECAST_MIN": forecast["min"],
        "ENSO_FORECAST_MAX": forecast["max"],
    }
    for path in ENSO_FORECAST_TARGETS:
        text = path.read_text()
        for name, values in replacements.items():
            if name in text:
                text = _replace_dict_literal(text, name, values.dropna())
        path.write_text(text)
        print(f"[SAVE] updated ENSO forecast constants in {path}")


def main() -> None:
    panel = pd.read_csv(PANEL_PATH, parse_dates=["quarter"])
    # Drop old weighted_heat/weighted_dryness, plain HeatDry/HeatDryYoY from a
    # prior run of this script (read+write the same file, so this must be
    # idempotent), and any _x/_y suffixed leftovers from a previous run that
    # predates this fix (pandas merge() silently suffixes on a name collision
    # instead of erroring, which is what produced them).
    panel = panel.drop(
        columns=[
            "weighted_heat", "weighted_dryness",
            "HeatDry", "HeatDryYoY",
            "HeatDry_x", "HeatDryYoY_x", "HeatDry_y", "HeatDryYoY_y",
            OIL_YOY_COL, f"{OIL_YOY_COL}_x", f"{OIL_YOY_COL}_y",
            IOD_COL, f"{IOD_COL}_x", f"{IOD_COL}_y",
            f"{ENSO_COL}_x", f"{ENSO_COL}_y",
        ],
        errors="ignore",
    )

    yield_climate = load_yield_climate()
    panel = append_future_heatdry_rows(panel, yield_climate)
    dupes = yield_climate.duplicated(["country", "quarter"], keep=False)
    if dupes.any():
        raise ValueError(
            f"Duplicate country-quarter rows in {YIELD_CLIMATE_PATH}:\n"
            f"{yield_climate.loc[dupes].to_string(index=False)}"
        )

    merged = panel.merge(yield_climate, on=["country", "quarter"], how="left", validate="many_to_one")
    roni_enso = load_quarterly_roni_enso()
    merged = merged.drop(columns=[ENSO_COL], errors="ignore").merge(
        roni_enso[["quarter", ENSO_COL]], on="quarter", how="left", validate="many_to_one"
    )
    oil_yoy = load_oil_yoy()
    merged = merged.merge(oil_yoy, on="quarter", how="left", validate="many_to_one")
    iod = load_iod()
    merged = merged.merge(iod, on="quarter", how="left", validate="many_to_one")
    merged.to_csv(PANEL_PATH, index=False, date_format="%Y-%m-%d")
    update_enso_forecast_constants()

    n_matched = merged["HeatDry"].notna().sum()
    n_enso = merged[ENSO_COL].notna().sum()
    n_oil = merged[OIL_YOY_COL].notna().sum()
    n_iod = merged[IOD_COL].notna().sum()
    print(f"[SAVE] {PANEL_PATH}")
    print(f"HeatDry matched on {n_matched}/{len(merged)} panel rows")
    print(merged.groupby("country")["HeatDry"].apply(lambda s: s.notna().sum()).loc[lambda s: s > 0])
    print(f"{ENSO_COL} from quarterly RONI matched on {n_enso}/{len(merged)} panel rows")
    print(f"{OIL_YOY_COL} matched on {n_oil}/{len(merged)} panel rows")
    print(f"{IOD_COL} matched on {n_iod}/{len(merged)} panel rows")


if __name__ == "__main__":
    main()
