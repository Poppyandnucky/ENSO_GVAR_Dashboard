"""Load the pre-built dashboard panel CSV (no in-repo build pipeline)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PANEL_CSV = _REPO_ROOT / "data" / "gvar_panel_streamlit.csv"
DEFAULT_STRESSOR_PROB_CSV = _REPO_ROOT / "data" / "prithvi_stressor_probabilities.csv"


def panel_csv_path() -> Path:
    raw = os.environ.get("TRP_PANEL_CSV", "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else _REPO_ROOT / p
    return DEFAULT_PANEL_CSV


def load_gvar_panel(path: Path | str | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else panel_csv_path()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Panel CSV not found: {csv_path}. "
            "Expected a pre-built `data/gvar_panel_streamlit.csv`."
        )
    return pd.read_csv(csv_path, parse_dates=["quarter"])


def load_stressor_probabilities(
    path: Path | str | None = None,
    *,
    panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    csv_path = Path(
        path
        if path is not None
        else os.environ.get("TRP_STRESSOR_PROB_CSV", DEFAULT_STRESSOR_PROB_CSV)
    )
    if not Path(csv_path).is_absolute():
        csv_path = _REPO_ROOT / csv_path
    if Path(csv_path).is_file():
        return pd.read_csv(csv_path)
    if panel is not None and "country" in panel.columns:
        countries = sorted(panel["country"].dropna().astype(str).unique())
        return pd.DataFrame(
            {
                "country": countries,
                "P_HEAT_NEXT_Q": float("nan"),
                "P_MOISTURE_NEXT_Q": float("nan"),
            }
        )
    return pd.DataFrame(columns=["country", "P_HEAT_NEXT_Q", "P_MOISTURE_NEXT_Q"])


def load_country_dataframe(country: str, path: Path | str | None = None) -> pd.DataFrame:
    df = load_gvar_panel(path)
    return df[df["country"] == country].sort_values("quarter")
