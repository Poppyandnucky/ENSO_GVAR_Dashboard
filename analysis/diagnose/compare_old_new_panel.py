"""Old-vs-new panel diagnostics for KF outcome instability checks.

Outputs:
  - economic_indicator_paths_old_vs_new.png
  - economic_indicator_paths_by_country_old_vs_new.pdf
  - macro_by_country/*.png
  - enso_old_vs_new_history_forecast.png
  - country_diag_coefficients_old_vs_new.pdf
  - old_new_panel_diff_summary.csv
  - country_diag_coefficient_diff_summary.csv

The coefficient paths reuse the same production KF/EM entry point and default
diagnostic settings used by analysis/diagnose/country_diag.py, while changing
only the input panel path.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("GVAR_IMPORT_ONLY", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "trp_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "trp_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STRUCTURAL_BREAK_DIR = ROOT / "structural_break"
DASH_OUTPUT_DIR = ROOT / "analysis" / "Dash_Output"
for p in (ROOT, STRUCTURAL_BREAK_DIR, DASH_OUTPUT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import GVAR_LLM_pickle as gp  # noqa: E402
import analysis.diagnose.country_diag as cd  # noqa: E402

NEW_PANEL = ROOT / "analysis" / "gvar_panel_streamlit (8 + EGY + PER).csv"
OLD_PANEL = ROOT / "analysis" / "gvar_panel_streamlit_old (8 + EGY + PER).csv"
OUT_DIR = ROOT / "analysis" / "diagnose" / "output" / "old_new_panel"

DASHBOARD_COUNTRIES = [
    "BRA",
    "CHL",
    "COL",
    "MEX",
    "KEN",
    "ZAF",
    "IND",
    "IDN",
    "THA",
    "PER",
    "PHL",
    "EGY",
]
MACRO_VARS = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]

OLD_ENSO_FORECAST_MEAN = {
    "2025Q3": -0.6055,
    "2025Q4": -0.9034,
    "2026Q1": -0.6499,
    "2026Q2": 0.49,
    "2026Q3": 1.7784,
    "2026Q4": 2.2875,
    "2027Q1": 1.8143,
    "2027Q2": 1.61,
}
OLD_ENSO_FORECAST_MIN = {
    "2025Q3": -0.6055,
    "2025Q4": -0.9034,
    "2026Q1": -0.6499,
    "2026Q2": 0.49,
    "2026Q3": 0.9385,
    "2026Q4": 1.4430,
    "2027Q1": 0.8575,
    "2027Q2": 1.3,
}
OLD_ENSO_FORECAST_MAX = {
    "2025Q3": -0.6055,
    "2025Q4": -0.9034,
    "2026Q1": -0.6499,
    "2026Q2": 0.49,
    "2026Q3": 2.0330,
    "2026Q4": 3.0678,
    "2027Q1": 2.2604,
    "2027Q2": 2.3,
}


def _quarter_ts(label: str) -> pd.Timestamp:
    return pd.Period(str(label).strip().upper(), freq="Q").to_timestamp()


def _load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["quarter"])
    df["quarter"] = df["quarter"].dt.to_period("Q").dt.to_timestamp()
    return df


def _series_from_dict(values: dict[str, float]) -> pd.Series:
    return pd.Series({_quarter_ts(k): float(v) for k, v in values.items()}).sort_index()


def save_panel_diff_summary(new: pd.DataFrame, old: pd.DataFrame) -> Path:
    merged = new[["country", "quarter", *MACRO_VARS, "ENSO", "OIL_YoY", "IOD", "HeatDry", "HeatDryF"]].merge(
        old[["country", "quarter", *MACRO_VARS, "ENSO", "OIL_YoY", "IOD", "HeatDry", "HeatDryF"]],
        on=["country", "quarter"],
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    rows = []
    for col in [*MACRO_VARS, "ENSO", "OIL_YoY", "IOD", "HeatDry", "HeatDryF"]:
        d = pd.to_numeric(merged[f"{col}_new"], errors="coerce") - pd.to_numeric(merged[f"{col}_old"], errors="coerce")
        changed = d.abs().fillna(0.0) > 1e-10
        row = {
            "variable": col,
            "changed_rows": int(changed.sum()),
            "max_abs_delta": float(d.abs().max(skipna=True)),
            "mean_abs_delta": float(d.abs().mean(skipna=True)),
        }
        if changed.any():
            idx = d.abs().idxmax()
            row.update(
                {
                    "max_delta_country": merged.loc[idx, "country"],
                    "max_delta_quarter": pd.Timestamp(merged.loc[idx, "quarter"]).to_period("Q").strftime("%YQ%q"),
                    "old_value_at_max": merged.loc[idx, f"{col}_old"],
                    "new_value_at_max": merged.loc[idx, f"{col}_new"],
                }
            )
        rows.append(row)
    path = OUT_DIR / "old_new_panel_diff_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def plot_macro_paths(new: pd.DataFrame, old: pd.DataFrame) -> Path:
    new_d = new[new["country"].isin(DASHBOARD_COUNTRIES)]
    old_d = old[old["country"].isin(DASHBOARD_COUNTRIES)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
    axes = axes.ravel()
    for ax, col in zip(axes, MACRO_VARS):
        n = new_d.groupby("quarter")[col].mean()
        o = old_d.groupby("quarter")[col].mean()
        ax.plot(o.index, o.values, color="#d62728", linewidth=1.8, label="old panel")
        ax.plot(n.index, n.values, color="#2ca02c", linewidth=1.8, label="new panel")
        ax.fill_between(
            n.index,
            new_d.groupby("quarter")[col].quantile(0.25).reindex(n.index),
            new_d.groupby("quarter")[col].quantile(0.75).reindex(n.index),
            color="#2ca02c",
            alpha=0.12,
            linewidth=0,
            label="new IQR" if col == MACRO_VARS[0] else None,
        )
        ax.axvline(pd.Timestamp("2026-07-01"), color="0.35", linestyle=":", linewidth=1.0)
        ax.set_title(f"{col}: dashboard-country mean")
        ax.set_ylabel("YoY")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=3, fontsize=9)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle("gvar_panel_streamlit (8 + EGY + PER): old vs new macro indicator paths")
    fig.tight_layout()
    path = OUT_DIR / "economic_indicator_paths_old_vs_new.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_macro_paths_by_country(new: pd.DataFrame, old: pd.DataFrame) -> tuple[Path, list[Path]]:
    png_dir = OUT_DIR / "macro_by_country"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "economic_indicator_paths_by_country_old_vs_new.pdf"
    png_paths = []

    with PdfPages(pdf_path) as pdf:
        for country in DASHBOARD_COUNTRIES:
            new_c = new[new["country"].eq(country)].sort_values("quarter")
            old_c = old[old["country"].eq(country)].sort_values("quarter")
            fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
            axes = axes.ravel()
            for ax, col in zip(axes, MACRO_VARS):
                ax.plot(old_c["quarter"], old_c[col], color="#d62728", linewidth=1.5, label="old panel")
                ax.plot(new_c["quarter"], new_c[col], color="#2ca02c", linewidth=1.5, label="new panel")
                ax.axvline(pd.Timestamp("2026-07-01"), color="0.35", linestyle=":", linewidth=1.0)
                ax.set_title(col)
                ax.set_ylabel("YoY")
                ax.grid(alpha=0.25)
            axes[0].legend(frameon=False, ncol=2, fontsize=9)
            axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            fig.suptitle(f"{country}: old vs new macro indicator paths")
            fig.tight_layout()
            pdf.savefig(fig, dpi=180)
            png_path = png_dir / f"{country}_macro_old_vs_new.png"
            fig.savefig(png_path, dpi=180)
            png_paths.append(png_path)
            plt.close(fig)

    return pdf_path, png_paths


def plot_enso(new: pd.DataFrame, old: pd.DataFrame) -> Path:
    hist_new = new.groupby("quarter")["ENSO"].first().dropna().sort_index()
    hist_old = old.groupby("quarter")["ENSO"].first().dropna().sort_index()
    new_mean = _series_from_dict(cd.gkf.FORECAST_ENSO_MEAN)
    new_min = _series_from_dict(cd.gkf.FORECAST_ENSO_MIN)
    new_max = _series_from_dict(cd.gkf.FORECAST_ENSO_MAX)
    old_mean = _series_from_dict(OLD_ENSO_FORECAST_MEAN)
    old_min = _series_from_dict(OLD_ENSO_FORECAST_MIN)
    old_max = _series_from_dict(OLD_ENSO_FORECAST_MAX)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(hist_old.index, hist_old.values, color="#d62728", label="old panel ENSO", linewidth=1.3)
    ax.plot(hist_new.index, hist_new.values, color="#2ca02c", label="new panel ENSO", linewidth=1.3)
    ax.plot(old_mean.index, old_mean.values, color="#d62728", marker="o", linestyle="--", label="old forecast mean")
    ax.fill_between(old_mean.index, old_min.reindex(old_mean.index), old_max.reindex(old_mean.index), color="#d62728", alpha=0.12, label="old forecast min-max")
    ax.plot(new_mean.index, new_mean.values, color="#2ca02c", marker="o", linestyle="--", label="new forecast mean")
    ax.fill_between(new_mean.index, new_min.reindex(new_mean.index), new_max.reindex(new_mean.index), color="#2ca02c", alpha=0.14, label="new forecast min-max")
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.axvline(pd.Timestamp("2026-07-01"), color="0.35", linestyle=":", linewidth=1.0)
    ax.set_title("ENSO old vs new, including forecasted values")
    ax.set_ylabel("ENSO / RONI")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    path = OUT_DIR / "enso_old_vs_new_history_forecast.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _country_coefficients(panel_path: Path, country: str) -> tuple[pd.DataFrame, str | None]:
    prep, res, fc = cd._run_forecast_with_settings(
        country,
        list(cd.SELECTED_EXO),
        max_em_iter=cd.MAX_EM_ITER,
        update_p0=cd.UPDATE_P0,
        coeff_method=cd.COEFF_METHOD,
    )
    # cd._run_forecast_with_settings reads gp.PATH internally for non-toggle
    # exogenous history, so callers set gp.PATH before invoking this helper.
    if prep is None or res is None:
        return pd.DataFrame(), "missing result"
    theta = np.asarray(fc["theta_trace"], dtype=float)
    valid = ~np.isnan(theta).any(axis=1)
    q = pd.to_datetime(fc["theta_trace_quarters"])
    endo = list(fc["ENDO_use"])
    exo = list(fc["EXO_use"])
    if "ENSO" not in exo:
        return pd.DataFrame(), "ENSO absent"
    m_y = len(endo)
    m = cd.gp.lags * m_y + len(exo)
    off = cd.gp.lags * m_y + exo.index("ENSO")
    rows = []
    for j, var in enumerate(endo):
        k = j * m + off
        for quarter, value, ok in zip(q, theta[:, k], valid):
            if ok:
                rows.append({"country": country, "quarter": quarter, "variable": var, "coef": float(value)})
    return pd.DataFrame(rows), None


def plot_country_diag_coefficients() -> tuple[Path, Path]:
    summary_rows = []
    old_gp_path = gp.PATH
    old_cd_gp_path = cd.gp.PATH
    old_gkf_gp_path = cd.gkf.gp.PATH
    pdf_path = OUT_DIR / "country_diag_coefficients_old_vs_new.pdf"

    with PdfPages(pdf_path) as pdf:
        for country in DASHBOARD_COUNTRIES:
            coeffs = {}
            errors = {}
            for label, panel_path in [("old", OLD_PANEL), ("new", NEW_PANEL)]:
                gp.PATH = str(panel_path)
                cd.gp.PATH = str(panel_path)
                cd.gkf.gp.PATH = str(panel_path)
                coeffs[label], errors[label] = _country_coefficients(panel_path, country)

            if errors.get("old") or errors.get("new") or coeffs["old"].empty or coeffs["new"].empty:
                summary_rows.append(
                    {
                        "country": country,
                        "status": "skipped",
                        "old_error": errors.get("old"),
                        "new_error": errors.get("new"),
                    }
                )
                continue

            merged = coeffs["new"].merge(
                coeffs["old"],
                on=["country", "quarter", "variable"],
                suffixes=("_new", "_old"),
                validate="one_to_one",
            )
            merged["abs_delta"] = (merged["coef_new"] - merged["coef_old"]).abs()
            for var, g in merged.groupby("variable"):
                idx = g["abs_delta"].idxmax()
                summary_rows.append(
                    {
                        "country": country,
                        "variable": var,
                        "status": "ok",
                        "n_quarters": len(g),
                        "max_abs_delta": float(g["abs_delta"].max()),
                        "mean_abs_delta": float(g["abs_delta"].mean()),
                        "max_delta_quarter": pd.Timestamp(merged.loc[idx, "quarter"]).to_period("Q").strftime("%YQ%q"),
                        "old_coef_at_max": float(merged.loc[idx, "coef_old"]),
                        "new_coef_at_max": float(merged.loc[idx, "coef_new"]),
                    }
                )

            fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), sharex=True)
            axes = axes.ravel()
            for ax, var in zip(axes, MACRO_VARS):
                o = coeffs["old"][coeffs["old"]["variable"].eq(var)]
                n = coeffs["new"][coeffs["new"]["variable"].eq(var)]
                ax.plot(o["quarter"], o["coef"], color="#d62728", linewidth=1.2, label="old data")
                ax.plot(n["quarter"], n["coef"], color="#2ca02c", linewidth=1.2, label="new data")
                ax.axhline(0, color="0.35", linewidth=0.8)
                ax.set_title(f"ENSO -> {var}")
                ax.grid(alpha=0.25)
            axes[0].legend(frameon=False, ncol=2, fontsize=9)
            fig.suptitle(
                f"{country}: country_diag.py coefficient path comparison\n"
                f"EXO={'+'.join(cd.SELECTED_EXO)}, EM={cd.MAX_EM_ITER}, P0={'EM' if cd.UPDATE_P0 else 'init'}, coeff={cd.COEFF_METHOD}"
            )
            fig.tight_layout()
            pdf.savefig(fig, dpi=180)
            plt.close(fig)

    gp.PATH = old_gp_path
    cd.gp.PATH = old_cd_gp_path
    cd.gkf.gp.PATH = old_gkf_gp_path

    summary_path = OUT_DIR / "country_diag_coefficient_diff_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    return pdf_path, summary_path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new = _load_panel(NEW_PANEL)
    old = _load_panel(OLD_PANEL)
    print(f"[CONFIRM] dashboard panel path: {NEW_PANEL}")
    print(f"[CONFIRM] old comparison panel: {OLD_PANEL}")
    print(f"[CONFIRM] country_diag settings: EXO={cd.SELECTED_EXO}, EM={cd.MAX_EM_ITER}, UPDATE_P0={cd.UPDATE_P0}, COEFF={cd.COEFF_METHOD}")
    outputs = [
        save_panel_diff_summary(new, old),
        plot_macro_paths(new, old),
        plot_enso(new, old),
    ]
    macro_pdf, macro_pngs = plot_macro_paths_by_country(new, old)
    outputs.append(macro_pdf)
    outputs.extend(macro_pngs)
    outputs.extend(plot_country_diag_coefficients())
    print("[SAVED]")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
