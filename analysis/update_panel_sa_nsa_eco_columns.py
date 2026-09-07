"""Update four economic columns in the main GVAR panel from SA/NSA panels.

Rules:
  - Main panel: analysis/gvar_panel_streamlit (8 + EGY + PER).csv
  - Use SA panel for PHL.
  - Use NSA panel for other countries.
  - Keep EGY exactly as-is.
  - Only update GDP_YoY, CPI_YoY, FX_YoY, EX_YoY.
  - Keep all climate/oil/other columns from the main panel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MAIN_PANEL = ROOT / "analysis" / "gvar_panel_streamlit (8 + EGY + PER).csv"
NSA_PANEL = ROOT / "analysis" / "gvar_panel_streamlit (NSA).csv"
SA_PANEL = ROOT / "analysis" / "gvar_panel_streamlit (SA).csv"

KEY = ["country", "quarter"]
ECO_COLS = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["quarter"])
    df["country"] = df["country"].astype(str)
    df["quarter"] = pd.to_datetime(df["quarter"]).dt.to_period("Q").dt.to_timestamp()
    return df


def main() -> None:
    panel = _load(MAIN_PANEL)
    nsa = _load(NSA_PANEL)
    sa = _load(SA_PANEL)

    egypt_before = panel.loc[panel["country"].eq("EGY")].reset_index(drop=True)

    source = pd.concat(
        [
            nsa.loc[~nsa["country"].isin(["PHL", "EGY"]), KEY + ECO_COLS],
            sa.loc[sa["country"].eq("PHL"), KEY + ECO_COLS],
        ],
        ignore_index=True,
    ).drop_duplicates(KEY, keep="last")

    out = panel.set_index(KEY)
    source_i = source.set_index(KEY)
    common = out.index.intersection(source_i.index)
    common = common[common.get_level_values("country") != "EGY"]

    changed = {}
    for col in ECO_COLS:
        old = out.loc[common, col]
        new = source_i.loc[common, col]
        changed[col] = int(
            (
                old.fillna("__NA__").astype(str).to_numpy()
                != new.fillna("__NA__").astype(str).to_numpy()
            ).sum()
        )
        out.loc[common, col] = new

    out = out.reset_index().sort_values(KEY).reset_index(drop=True)
    out = out[panel.columns]
    egypt_after = out.loc[out["country"].eq("EGY")].reset_index(drop=True)
    if not egypt_before.equals(egypt_after):
        raise RuntimeError("EGY changed unexpectedly; aborting.")
    if out.duplicated(KEY).any():
        raise RuntimeError("Duplicate country-quarter rows created; aborting.")

    out["quarter"] = out["quarter"].dt.strftime("%Y-%m-%d")
    out.to_csv(MAIN_PANEL, index=False)

    print(f"[SAVE] {MAIN_PANEL}")
    print(f"source rows: {len(source)}")
    print(f"updated matching non-EGY rows: {len(common)}")
    print(f"changed cells: {changed}")
    print("EGY unchanged: True")
    print("duplicate country-quarter: 0")


if __name__ == "__main__":
    main()
