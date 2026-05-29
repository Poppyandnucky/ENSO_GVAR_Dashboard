"""Build Egypt quarterly YoY rows and append them to the GVAR panel.

Inputs are raw files downloaded outside the repo:
  - FX daily levels
  - exports quarterly levels
  - CPI monthly m/m inflation
  - GDP quarterly real growth rate

The script preserves the existing panel schema. It writes:
  - analysis/Dash_Output/egypt_quarterly_yoy.csv
  - analysis/Dash_Output/egypt_panel_preprocessing_summary.csv
  - analysis/gvar_panel_streamlit (7 + EGY).csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PANEL_IN = ROOT / "analysis" / "gvar_panel_streamlit (7).csv"
PANEL_OUT = ROOT / "analysis" / "gvar_panel_streamlit (7 + EGY).csv"
OUT_DIR = ROOT / "analysis" / "Dash_Output"

FX_DAILY_XLSX = Path("/Users/poppy/Downloads/FX_daily.xlsx")
EXPORTS_CSV = Path("/Users/poppy/Downloads/Gross Domestic Product Export.csv")
CPI_XLSX = Path("/Users/poppy/Downloads/Inflations Historical.xlsx")
GDP_CSV = Path("/Users/poppy/Downloads/Gross Domestic Product.csv")

GLOBAL_COLS = [
    "US_GDP_YoY",
    "CHN_GDP_YoY",
    "COMMODITY_YoY",
    "COMMODITY_AGR_YoY",
    "ENSO",
]


def _parse_percent(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace("%", "").replace(",", "")
    return pd.to_numeric(text, errors="coerce")


def _fiscal_quarter_to_calendar_start(fiscal_year: str, quarter: str) -> pd.Timestamp:
    """Map Egypt fiscal-year quarters to calendar quarter starts.

    Example: fiscal year 2026/2025 Q1 -> 2025Q3, Q2 -> 2025Q4,
    Q3 -> 2026Q1, Q4 -> 2026Q2.
    """
    years = [int(x) for x in str(fiscal_year).replace(" ", "").split("/") if x]
    if len(years) != 2:
        return pd.NaT
    fiscal_start_year = min(years)
    q_num = int(str(quarter).strip().upper().replace("Q", ""))
    if q_num not in {1, 2, 3, 4}:
        return pd.NaT
    month = {1: 7, 2: 10, 3: 1, 4: 4}[q_num]
    year = fiscal_start_year if q_num in {1, 2} else fiscal_start_year + 1
    return pd.Timestamp(year=year, month=month, day=1)


def load_gdp_yoy() -> pd.DataFrame:
    df = pd.read_csv(GDP_CSV)
    df["quarter"] = [
        _fiscal_quarter_to_calendar_start(fy, q)
        for fy, q in zip(df["Fiscal Year"], df["Quarter"])
    ]
    df["GDP_YoY"] = pd.to_numeric(df["Real Growth Rate"], errors="coerce")
    return df[["quarter", "GDP_YoY"]].dropna(subset=["quarter"]).sort_values("quarter")


def load_exports_yoy() -> pd.DataFrame:
    df = pd.read_csv(EXPORTS_CSV)
    df["quarter"] = [
        _fiscal_quarter_to_calendar_start(fy, q)
        for fy, q in zip(df["Fiscal Year"], df["Quarter"])
    ]
    df["EX_level"] = pd.to_numeric(df["Exports of goods and services"], errors="coerce")
    out = (
        df[["quarter", "EX_level"]]
        .dropna(subset=["quarter"])
        .sort_values("quarter")
        .drop_duplicates("quarter", keep="last")
    )
    out["EX_YoY"] = out["EX_level"].pct_change(4) * 100.0
    return out[["quarter", "EX_YoY", "EX_level"]]


def load_fx_yoy() -> pd.DataFrame:
    df = pd.read_excel(FX_DAILY_XLSX, sheet_name=0)
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "FX_level"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["FX_level"] = pd.to_numeric(df["FX_level"], errors="coerce")
    df = df.dropna(subset=["date", "FX_level"]).copy()
    df["quarter"] = df["date"].dt.to_period("Q").dt.to_timestamp()
    out = df.groupby("quarter", as_index=False)["FX_level"].mean().sort_values("quarter")
    out["FX_YoY"] = out["FX_level"].pct_change(4) * 100.0
    return out[["quarter", "FX_YoY", "FX_level"]]


def load_cpi_yoy() -> pd.DataFrame:
    df = pd.read_excel(CPI_XLSX, sheet_name=0)
    df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "headline_mom"})
    df["month"] = pd.to_datetime(df["date"], format="%b %Y", errors="coerce")
    df["headline_mom"] = df["headline_mom"].map(_parse_percent)
    df = df.dropna(subset=["month", "headline_mom"]).sort_values("month").copy()
    df["cpi_index"] = (1.0 + df["headline_mom"] / 100.0).cumprod() * 100.0
    df["quarter"] = df["month"].dt.to_period("Q").dt.to_timestamp()
    q = df.groupby("quarter", as_index=False)["cpi_index"].mean().sort_values("quarter")
    q["CPI_YoY"] = q["cpi_index"].pct_change(4) * 100.0
    return q[["quarter", "CPI_YoY", "cpi_index"]]


def build_egypt_rows(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    q_min = pd.to_datetime(panel["quarter"]).min()
    q_max = pd.to_datetime(panel["quarter"]).max()
    quarters = pd.date_range(q_min, q_max, freq="QS")
    egy = pd.DataFrame({"country": "EGY", "quarter": quarters})

    sources = [
        load_gdp_yoy(),
        load_cpi_yoy(),
        load_fx_yoy(),
        load_exports_yoy(),
    ]
    for src in sources:
        egy = egy.merge(src, on="quarter", how="left")

    globals_by_q = (
        panel.assign(quarter=pd.to_datetime(panel["quarter"]))
        .groupby("quarter", as_index=False)[GLOBAL_COLS]
        .first()
    )
    egy = egy.merge(globals_by_q, on="quarter", how="left")
    egy["year"] = egy["quarter"].dt.year

    egy["CPI_YoY_annual"] = egy.groupby("year")["CPI_YoY"].transform("mean")
    egy["FX_YoY_annual"] = egy.groupby("year")["FX_YoY"].transform("mean")

    for col in panel.columns:
        if col not in egy.columns:
            egy[col] = np.nan

    # Keep source levels in the audit file, but not in the final panel.
    audit_cols = [
        "country",
        "quarter",
        "GDP_YoY",
        "CPI_YoY",
        "FX_YoY",
        "EX_YoY",
        "FX_level",
        "EX_level",
        "cpi_index",
    ] + GLOBAL_COLS
    audit = egy[[c for c in audit_cols if c in egy.columns]].copy()
    final_rows = egy[panel.columns].copy()
    return final_rows, audit


def summarize(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]:
        non = audit.dropna(subset=[col])
        rows.append(
            {
                "variable": col,
                "first_nonnull": non["quarter"].min() if not non.empty else pd.NaT,
                "last_nonnull": non["quarter"].max() if not non.empty else pd.NaT,
                "nonnull_quarters": len(non),
                "missing_quarters_in_panel_range": int(audit[col].isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL_IN, parse_dates=["quarter"])
    egy_rows, audit = build_egypt_rows(panel)

    combined = pd.concat(
        [panel[panel["country"].astype(str) != "EGY"], egy_rows],
        ignore_index=True,
    ).sort_values(["country", "quarter"])
    combined.to_csv(PANEL_OUT, index=False)

    audit_path = OUT_DIR / "egypt_quarterly_yoy.csv"
    summary_path = OUT_DIR / "egypt_panel_preprocessing_summary.csv"
    audit.to_csv(audit_path, index=False)
    summarize(audit).to_csv(summary_path, index=False)

    print(f"[SAVE] {PANEL_OUT}")
    print(f"[SAVE] {audit_path}")
    print(f"[SAVE] {summary_path}")
    print(summarize(audit).to_string(index=False))


if __name__ == "__main__":
    main()
