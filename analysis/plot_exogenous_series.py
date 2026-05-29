"""Generate exogenous history + forecast plots used by the Streamlit app."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FORECAST_DIR = ROOT / "analysis" / "Dash_Output"
if str(FORECAST_DIR) not in sys.path:
    sys.path.insert(0, str(FORECAST_DIR))

os.environ.setdefault("GVAR_IMPORT_ONLY", "1")

from gvar_kf_forecast import (  # noqa: E402
    FORECAST_COMMODITY,
    FORECAST_ENSO_MAX,
    FORECAST_ENSO_MEAN,
    FORECAST_ENSO_MIN,
)
from trp.inputs import panel_csv_path  # noqa: E402


PANEL_CSV = panel_csv_path()
OUTPUT_ENSO = ROOT / "analysis" / "Dash_Output" / "enso_series" / "enso_history_forecast.png"
OUTPUT_COMMODITY = ROOT / "analysis" / "Dash_Output" / "commodity_series" / "commodity_history_forecast.png"
PLOT_START = pd.Timestamp("2014-01-01")


def _quarter_ts(q: str) -> pd.Timestamp:
    s = str(q).strip().upper()
    if len(s) == 6 and s[4] == "Q":
        return pd.Period(s, freq="Q").to_timestamp()
    return pd.to_datetime(q).to_period("Q").to_timestamp()


def _quarter_start(ts) -> pd.Timestamp:
    return pd.to_datetime(ts).to_period("Q").to_timestamp()


def _load_column_csv(col: str, path: Path = PANEL_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["quarter", col])
    df["q"] = pd.to_datetime(df["quarter"]).dt.to_period("Q").dt.to_timestamp()
    out = df.groupby("q", as_index=False)[col].first().sort_values("q")
    out = out[out["q"] >= PLOT_START].dropna(subset=[col])
    return out.reset_index(drop=True)


def _dict_quarter_series(value_map: dict[str, float], col: str) -> pd.DataFrame:
    rows = [{"q": _quarter_ts(k), col: float(v)} for k, v in value_map.items()]
    return pd.DataFrame(rows).sort_values("q").reset_index(drop=True)


def _commodity_dict_to_series(commodity_map: dict[str, float]) -> pd.DataFrame:
    rows = [{"q": _quarter_start(k), "COMMODITY_YoY": float(v)} for k, v in commodity_map.items()]
    return pd.DataFrame(rows).sort_values("q").reset_index(drop=True)


def _future_only(forecast: pd.DataFrame, hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    last_hist_q = hist["q"].max()
    future = forecast[forecast["q"] > last_hist_q].reset_index(drop=True)
    start = future["q"].min() if not future.empty else last_hist_q
    return future, start


def plot_enso_history_forecast(
    *,
    panel_csv: Path = PANEL_CSV,
    output_path: Path = OUTPUT_ENSO,
    plot_start: pd.Timestamp = PLOT_START,
) -> Path:
    hist = _load_column_csv("ENSO", panel_csv)
    hist = hist[hist["q"] >= plot_start].reset_index(drop=True)

    fc_mean, fc_start = _future_only(_dict_quarter_series(FORECAST_ENSO_MEAN, "ENSO"), hist)
    fc_min = _dict_quarter_series(FORECAST_ENSO_MIN, "ENSO")
    fc_max = _dict_quarter_series(FORECAST_ENSO_MAX, "ENSO")
    fc_min = fc_min[fc_min["q"].isin(fc_mean["q"])].reset_index(drop=True)
    fc_max = fc_max[fc_max["q"].isin(fc_mean["q"])].reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(
        hist["q"],
        hist["ENSO"],
        color="C0",
        linewidth=1.8,
        marker=".",
        markersize=4,
        label="ENSO (panel CSV)",
        zorder=2,
    )

    if len(hist) and len(fc_mean):
        ax.plot(
            [hist["q"].iloc[-1], fc_mean["q"].iloc[0]],
            [hist["ENSO"].iloc[-1], fc_mean["ENSO"].iloc[0]],
            color="gray",
            linewidth=1.0,
            linestyle=":",
            alpha=0.6,
            zorder=1,
        )
        ax.fill_between(
            fc_mean["q"],
            fc_min["ENSO"],
            fc_max["ENSO"],
            color="C1",
            alpha=0.28,
            label="forecast min-max band",
            zorder=3,
        )
        ax.plot(
            fc_mean["q"],
            fc_mean["ENSO"],
            color="C1",
            linewidth=2.2,
            marker="o",
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label="forecast mean (new)",
            zorder=4,
        )

    ax.axvline(fc_start, color="gray", linestyle="--", linewidth=0.9, alpha=0.75)
    ax.set_ylabel("ENSO")
    ax.set_xlabel("quarter")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(
        f"ENSO from {plot_start.year} - panel CSV + forecast "
        f"({pd.Period(fc_start, freq='Q')} onward highlighted)"
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {output_path}")
    return output_path


def plot_commodity_history_forecast(
    *,
    panel_csv: Path = PANEL_CSV,
    output_path: Path = OUTPUT_COMMODITY,
    plot_start: pd.Timestamp = PLOT_START,
) -> Path:
    hist = _load_column_csv("COMMODITY_YoY", panel_csv)
    hist = hist[hist["q"] >= plot_start].reset_index(drop=True)
    fc, fc_start = _future_only(_commodity_dict_to_series(FORECAST_COMMODITY), hist)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(
        hist["q"],
        hist["COMMODITY_YoY"],
        color="C0",
        linewidth=1.8,
        marker=".",
        markersize=4,
        label="COMMODITY_YoY (panel CSV)",
        zorder=2,
    )

    if len(hist) and len(fc):
        ax.plot(
            [hist["q"].iloc[-1], fc["q"].iloc[0]],
            [hist["COMMODITY_YoY"].iloc[-1], fc["COMMODITY_YoY"].iloc[0]],
            color="gray",
            linewidth=1.0,
            linestyle=":",
            alpha=0.6,
            zorder=1,
        )
        ax.plot(
            fc["q"],
            fc["COMMODITY_YoY"],
            color="C1",
            linewidth=2.2,
            marker="o",
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.4,
            label="forecast (new)",
            zorder=4,
        )

    ax.axvline(fc_start, color="gray", linestyle="--", linewidth=0.9, alpha=0.75)
    ax.set_ylabel("COMMODITY_YoY")
    ax.set_xlabel("quarter")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(
        f"COMMODITY_YoY from {plot_start.year} - panel CSV + forecast "
        f"({pd.Period(fc_start, freq='Q')} onward highlighted)"
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {output_path}")
    return output_path


def main() -> None:
    plot_enso_history_forecast()
    plot_commodity_history_forecast()


if __name__ == "__main__":
    main()
