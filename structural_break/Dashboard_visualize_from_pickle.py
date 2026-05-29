"""
SDR visualization from pickle only (no model refit).

All behavior is controlled by the CONFIG block below. CLI can override the
pickle path; --countries overrides the country subset (replacing USE_ALL_* and
COUNTRY_ISO3_ALLOWLIST).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from llm_break_visualization import (
    build_time_slice_maps,
    plot_break_supported_ratio,
    plot_break_type_distribution,
    plot_structural_break_score_with_llm_overlay,
)

# =============================================================================
# CONFIG — edit here (English). Modes: "plot" | "jpg" | "pdf"
# =============================================================================
# Input pickle (full bundle from SDR_build_pickle / GVAR_LLM_pickle).
PICKLE_PATH: str = "Dash_Input/gvar_pipeline_results.pkl"

# --- Country subset (ISO3) ---
# If True (default): plot every country from pickle["config"]["COUNTRIES_FOR_COEF_PLOT"].
# If False: only countries listed in COUNTRY_ISO3_ALLOWLIST (must be non-empty).
# CLI:  python ... --countries CHL,MEX overrides both (same as a non-empty allowlist).
USE_ALL_COUNTRIES_FROM_PICKLE: bool = True
COUNTRY_ISO3_ALLOWLIST: list[str] = [
    "CHL",
    "MEX",
]

# --- Section toggles ---
ENABLE_CORE_EM: bool = True  # coeff / diagnostics / Q–R traces from offline_plot_data.base
ENABLE_LLM: bool = True  # overlay, stats charts, optional HTML maps
ENABLE_REFIT: bool = False  # refit core + breakscore from offline_plot_data.refit

# Core EM bundle: one mode for the whole block (all figures for selected countries).
CORE_EM_MODE: Literal["plot", "jpg", "pdf"] = "pdf"
# PDF path when CORE_EM_MODE == "pdf". If "", use pickle config PLOTS_PDF_PATH or fallback.
CORE_EM_PDF_PATH: str = ""
# JPG root when CORE_EM_MODE == "jpg" (one file per figure, see naming below).
CORE_EM_JPG_DIR: str = "Dash_Output/SDR_core_em"

# LLM overlay (single large figure).
LLM_OVERLAY_MODE: Literal["plot", "jpg", "pdf"] = "pdf"
# If "", use pickle config LLM_OVERLAY_FIG_PATH or Dash_Input default.
LLM_OVERLAY_FILE: str = ""

# LLM stats (two figures: ratio + type).
LLM_STATS_MODE: Literal["plot", "jpg", "pdf"] = "jpg"
# Base path; for jpg two files get _ratio / _type suffixes. For pdf one two-page PDF.
LLM_STATS_FILE: str = ""

# Plotly time-slice maps: HTML only (not matplotlib). "on" | "off".
LLM_TIME_SLICE_MAPS: Literal["on", "off"] = "on"
# If "", use pickle config LLM_MAP_OUTPUT_DIR.
LLM_MAP_OUTPUT_DIR: str = "Dash_Input/map1998-2024"
# Map year range.
LLM_MAP_START_YEAR: int = 1998
LLM_MAP_END_YEAR: int = 2024

# Refit block (core + breakscore for refit panel).
REFIT_MODE: Literal["plot", "jpg", "pdf"] = "pdf"
REFIT_PDF_PATH: str = ""  # if "", use Dash_Input/GVAR_LLM_EM_plots_refit.pdf
# JPG: breakscore figures go here; refit core EM figures go to REFIT_CORE_JPG_DIR.
REFIT_JPG_DIR: str = "Dash_Input/breakscore_jpg"
REFIT_CORE_JPG_DIR: str = "Dash_Output/SDR_refit_core_em"

# DPI for saved raster figures.
FIG_DPI: int = 150

# =============================================================================


VisMode = Literal["plot", "jpg", "pdf"]


def style_quarter_axis_every_four_quarters(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, which="major", axis="x", alpha=0.35)


def _parse_countries_arg(countries_arg: str | None) -> list[str] | None:
    if not countries_arg:
        return None
    vals = [x.strip() for x in countries_arg.split(",") if x.strip()]
    return vals or None


def _country_selected(country: str, selected: list[str] | None) -> bool:
    if not selected:
        return True
    s = {x.lower() for x in selected}
    return str(country).lower() in s


def _resolve_country_list(
    cfg: dict[str, Any], allowlist: list[str]
) -> list[str]:
    base = list(cfg.get("COUNTRIES_FOR_COEF_PLOT", []))
    if not allowlist:
        return base
    al = {x.strip().upper() for x in allowlist if str(x).strip()}
    return [c for c in base if str(c).strip().upper() in al]


def _finish_figure(
    fig: plt.Figure,
    mode: VisMode,
    pdf: PdfPages | None,
    jpg_path: Path | None,
) -> None:
    if mode == "pdf" and pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    elif mode == "jpg" and jpg_path is not None:
        jpg_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(jpg_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show(block=False)


def plot_offline_core_from_pack(
    offline_pack: dict[str, Any],
    countries: list[str],
    selected_countries: list[str] | None,
    mode: VisMode,
    pdf: PdfPages | None,
    jpg_dir: Path | None,
    jpg_name_prefix: str = "core",
) -> None:
    per_country = offline_pack.get("per_country", {})
    for country in countries:
        if not _country_selected(country, selected_countries):
            continue
        d = per_country.get(country)
        if not d:
            continue
        cname = d.get("country_name", country)
        slug = str(country)
        coeff_quarters = pd.to_datetime(d.get("coeff_quarters", []))
        diag_quarters = pd.to_datetime(d.get("diag_quarters", []))
        diag_coeff_series = d.get("diag_coeff_series", [])
        enso_coeff_series = d.get("enso_coeff_series", [])
        innov = np.asarray(d.get("innovation_score", []), dtype=float)
        dcoef = np.asarray(d.get("coefficient_change", []), dtype=float)
        fsgap = np.asarray(d.get("filter_smoother_gap", []), dtype=float)
        q_tr = np.asarray(d.get("trace_Q", []), dtype=float)
        r_tr = np.asarray(d.get("trace_R", []), dtype=float)

        if len(coeff_quarters) > 0 and diag_coeff_series:
            plt.figure(figsize=(12, 5))
            for s in diag_coeff_series:
                vals = np.asarray(s.get("values", []), dtype=float)
                if len(vals) == len(coeff_quarters):
                    plt.plot(
                        coeff_quarters,
                        vals,
                        linewidth=1.3,
                        marker="o",
                        markersize=2,
                        alpha=0.9,
                        label=s.get("label", ""),
                    )
            plt.title(f"{cname} — Diagonal own-variable lag coefficients")
            plt.xlabel("Quarter")
            plt.ylabel("coefficient")
            plt.legend(frameon=False, ncol=2, fontsize=9)
            ax = plt.gca()
            style_quarter_axis_every_four_quarters(ax)
            ax.grid(True, which="major", axis="y", alpha=0.25)
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            fig = plt.gcf()
            jp = (
                (jpg_dir / f"{jpg_name_prefix}_{slug}_diag_own.jpg")
                if jpg_dir is not None
                else None
            )
            _finish_figure(fig, mode, pdf, jp)

        if len(coeff_quarters) > 0 and enso_coeff_series:
            plt.figure(figsize=(12, 5))
            for s in enso_coeff_series:
                vals = np.asarray(s.get("values", []), dtype=float)
                if len(vals) == len(coeff_quarters):
                    plt.plot(
                        coeff_quarters,
                        vals,
                        linewidth=1.3,
                        linestyle="-",
                        marker="o",
                        markersize=2,
                        alpha=0.9,
                        label=s.get("label", ""),
                    )
            plt.title(f"{cname} — ENSO coefficients (one per equation)")
            plt.xlabel("Quarter")
            plt.ylabel("coefficient")
            plt.legend(frameon=False, ncol=2, fontsize=9)
            ax = plt.gca()
            style_quarter_axis_every_four_quarters(ax)
            ax.grid(True, which="major", axis="y", alpha=0.25)
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            fig = plt.gcf()
            jp = (
                (jpg_dir / f"{jpg_name_prefix}_{slug}_enso.jpg")
                if jpg_dir is not None
                else None
            )
            _finish_figure(fig, mode, pdf, jp)

        if len(diag_quarters) == len(innov) == len(dcoef) == len(fsgap) and len(diag_quarters) > 0:
            fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
            axes[0].plot(diag_quarters, innov, color="C0", linewidth=1.2)
            axes[0].set_ylabel("Innovation score (v' S^{-1} v)")
            axes[0].grid(True, which="major", axis="y", alpha=0.25)
            axes[0].set_title(f"{cname} — EM diagnostics")
            axes[1].plot(diag_quarters, dcoef, color="C1", linewidth=1.2)
            axes[1].set_ylabel("Coefficient change ||b_s,t - b_s,t-1||")
            axes[1].grid(True, which="major", axis="y", alpha=0.25)
            axes[2].plot(diag_quarters, fsgap, color="C2", linewidth=1.2)
            axes[2].set_ylabel("Filter–smoother gap ||b_s,t - b_f,t||")
            axes[2].set_xlabel("Quarter")
            axes[2].grid(True, which="major", axis="y", alpha=0.25)
            style_quarter_axis_every_four_quarters(axes[2])
            plt.gcf().autofmt_xdate()
            plt.tight_layout()
            jp = (
                (jpg_dir / f"{jpg_name_prefix}_{slug}_diagnostics.jpg")
                if jpg_dir is not None
                else None
            )
            _finish_figure(fig, mode, pdf, jp)

        if len(q_tr) > 0 and len(r_tr) > 0:
            it = np.arange(1, min(len(q_tr), len(r_tr)) + 1)
            plt.figure(figsize=(9, 4.5))
            plt.plot(it, q_tr[: len(it)], "-o", linewidth=1.4, markersize=3, label="trace(Q)")
            plt.plot(it, r_tr[: len(it)], "-s", linewidth=1.4, markersize=3, label="trace(R)")
            plt.title(f"{cname} - EM iteration traces")
            plt.xlabel("EM iteration")
            plt.ylabel("trace value")
            plt.grid(alpha=0.3)
            plt.legend(frameon=False)
            plt.tight_layout()
            fig = plt.gcf()
            jp = (
                (jpg_dir / f"{jpg_name_prefix}_{slug}_qr_traces.jpg")
                if jpg_dir is not None
                else None
            )
            _finish_figure(fig, mode, pdf, jp)


def plot_offline_breakscore_from_pack(
    offline_pack: dict[str, Any],
    countries: list[str],
    selected_countries: list[str] | None,
    drop_quarters_dict: dict[str, list] | None,
    mode: VisMode,
    pdf: PdfPages | None,
    jpg_dir: Path | None,
    jpg_prefix: str = "breakscore",
) -> None:
    per_country = offline_pack.get("per_country", {})
    for country in countries:
        if not _country_selected(country, selected_countries):
            continue
        d = per_country.get(country)
        if not d:
            continue
        cname = d.get("country_name", country)
        quarters = pd.to_datetime(d.get("diag_quarters", []))
        vals = np.asarray(d.get("breakscore", []), dtype=float)
        if len(quarters) != len(vals) or len(quarters) == 0:
            continue
        if drop_quarters_dict and country in drop_quarters_dict:
            drop_q = set(pd.to_datetime(drop_quarters_dict[country]))
            keep_mask = ~pd.Series(quarters).isin(drop_q).to_numpy()
            quarters = quarters[keep_mask]
            vals = vals[keep_mask]
        if len(quarters) == 0:
            continue
        plt.figure(figsize=(12, 4.8))
        plt.plot(quarters, vals, color="black", linewidth=1.2)
        plt.title(f"{cname} — Break score (innovation)")
        plt.xlabel("Quarter")
        plt.ylabel("score")
        ax = plt.gca()
        style_quarter_axis_every_four_quarters(ax)
        ax.grid(True, which="major", axis="y", alpha=0.25)
        plt.tight_layout()
        fig = plt.gcf()
        slug = str(country)
        jp = (
            (jpg_dir / f"{jpg_prefix}_{slug}.jpg") if jpg_dir is not None else None
        )
        _finish_figure(fig, mode, pdf, jp)


def visualize_from_pickle(
    pickle_path: str | None,
    cli_countries: list[str] | None,
) -> None:
    path = pickle_path or PICKLE_PATH
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pickle file not found: {p}")

    with p.open("rb") as f:
        vis_pack = pickle.load(f)

    cfg = vis_pack.get("config", {})

    if cli_countries:
        allow: list[str] = cli_countries
    elif USE_ALL_COUNTRIES_FROM_PICKLE:
        allow = []
    else:
        allow = list(COUNTRY_ISO3_ALLOWLIST)
        if not allow:
            raise ValueError(
                "USE_ALL_COUNTRIES_FROM_PICKLE is False but COUNTRY_ISO3_ALLOWLIST is "
                'empty. Set ISO3 codes (e.g. ["CHL", "MEX"]) or set '
                "USE_ALL_COUNTRIES_FROM_PICKLE=True."
            )
    countries = _resolve_country_list(cfg, allow)
    if countries:
        print(f"[SDR_visualize_from_pickle] plotting {len(countries)} countries: {countries}")
    else:
        print(
            "[SDR_visualize_from_pickle] warning: resolved country list is empty "
            "(check ISO3 codes vs pickle COUNTRIES_FOR_COEF_PLOT)."
        )

    # -------- Core EM --------
    offline_base = vis_pack.get("offline_plot_data", {}).get("base")
    if ENABLE_CORE_EM and offline_base:
        core_pdf_out = CORE_EM_PDF_PATH or cfg.get("PLOTS_PDF_PATH") or str(
            Path("Dash_Input") / "GVAR_LLM_EM_plots.pdf"
        )
        core_jdir = Path(CORE_EM_JPG_DIR)

        if CORE_EM_MODE == "pdf":
            Path(core_pdf_out).parent.mkdir(parents=True, exist_ok=True)
            with PdfPages(core_pdf_out) as pdf_doc:
                plot_offline_core_from_pack(
                    offline_base,
                    countries,
                    countries,
                    "pdf",
                    pdf_doc,
                    None,
                    jpg_name_prefix="core",
                )
        elif CORE_EM_MODE == "jpg":
            plot_offline_core_from_pack(
                offline_base,
                countries,
                countries,
                "jpg",
                None,
                core_jdir,
                jpg_name_prefix="core",
            )
        else:
            plot_offline_core_from_pack(
                offline_base,
                countries,
                countries,
                "plot",
                None,
                None,
                jpg_name_prefix="core",
            )

    # -------- LLM --------
    llm_pack = vis_pack.get("llm_integration")
    if ENABLE_LLM and llm_pack:
        llm_df = llm_pack["llm_df"]
        break_score_df = llm_pack["break_score_df"]
        score_year_df = llm_pack["score_year_df"]
        sset = {c.lower() for c in countries} if countries else None
        if sset:
            llm_df = llm_df[llm_df["country"].astype(str).str.lower().isin(sset)].copy()
            break_score_df = break_score_df[
                break_score_df["country"].astype(str).str.lower().isin(sset)
            ].copy()
            score_year_df = score_year_df[
                score_year_df["country"].astype(str).str.lower().isin(sset)
            ].copy()

        overlay_path = Path(
            LLM_OVERLAY_FILE or cfg.get("LLM_OVERLAY_FIG_PATH")
            or "Dash_Input/gvar_breakscore_llm_overlay.png"
        )

        fig_overlay = plot_structural_break_score_with_llm_overlay(
            score_df=break_score_df,
            llm_df=llm_df,
            score_col="score",
            time_col="quarter",
            country_col="country",
            show_llm_overlay=cfg.get("SHOW_LLM_BREAK_OVERLAY", True),
            top_k=5,
            ncols=3,
            iso3_to_country=cfg.get("iso3_to_country"),
        )

        if LLM_OVERLAY_MODE == "pdf":
            overlay_pdf = overlay_path.with_suffix(".pdf")
            overlay_pdf.parent.mkdir(parents=True, exist_ok=True)
            fig_overlay.savefig(overlay_pdf, dpi=FIG_DPI, bbox_inches="tight")
            plt.close(fig_overlay)
        elif LLM_OVERLAY_MODE == "jpg":
            outp = overlay_path.with_suffix(".jpg")
            outp.parent.mkdir(parents=True, exist_ok=True)
            fig_overlay.savefig(outp, dpi=FIG_DPI, bbox_inches="tight")
            plt.close(fig_overlay)
        else:
            plt.figure(fig_overlay.number)
            plt.show(block=False)

        stats_base = Path(
            LLM_STATS_FILE or cfg.get("LLM_STATS_FIG_PATH")
            or "Dash_Input/gvar_llm_stats.png"
        )
        fig_ratio = plot_break_supported_ratio(llm_df)
        fig_type = plot_break_type_distribution(llm_df)

        if LLM_STATS_MODE == "pdf":
            stats_pdf_path = stats_base.with_suffix(".pdf")
            stats_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            with PdfPages(str(stats_pdf_path)) as stats_pdf:
                stats_pdf.savefig(fig_ratio, dpi=FIG_DPI, bbox_inches="tight")
                stats_pdf.savefig(fig_type, dpi=FIG_DPI, bbox_inches="tight")
            plt.close(fig_ratio)
            plt.close(fig_type)
        elif LLM_STATS_MODE == "jpg":
            suf = stats_base.suffix or ".png"
            r_path = stats_base.with_name(f"{stats_base.stem}_ratio{suf}")
            if r_path.suffix.lower() not in (".jpg", ".jpeg"):
                r_path = r_path.with_suffix(".jpg")
            t_path = stats_base.with_name(f"{stats_base.stem}_type{suf}")
            if t_path.suffix.lower() not in (".jpg", ".jpeg"):
                t_path = t_path.with_suffix(".jpg")
            r_path.parent.mkdir(parents=True, exist_ok=True)
            fig_ratio.savefig(r_path, dpi=FIG_DPI, bbox_inches="tight")
            fig_type.savefig(t_path, dpi=FIG_DPI, bbox_inches="tight")
            plt.close(fig_ratio)
            plt.close(fig_type)
        else:
            for _fig in (fig_ratio, fig_type):
                plt.figure(_fig.number)
                plt.show(block=False)

        maps_dir = LLM_MAP_OUTPUT_DIR or cfg.get("LLM_MAP_OUTPUT_DIR") or "Dash_Input/gvar_llm_time_slice_maps"
        if LLM_TIME_SLICE_MAPS == "on":
            build_time_slice_maps(
                score_df=score_year_df,
                llm_df=llm_df,
                score_col="score",
                country_col="country",
                year_col="year",
                start_year=LLM_MAP_START_YEAR,
                end_year=LLM_MAP_END_YEAR,
                output_dir=maps_dir,
                iso3_to_country=cfg.get("iso3_to_country"),
            )

    # -------- Refit --------
    refit_pack = vis_pack.get("refit")
    offline_refit = vis_pack.get("offline_plot_data", {}).get("refit")
    if ENABLE_REFIT and refit_pack and offline_refit:
        drop_quarters_dict = refit_pack.get("drop_quarters_dict")
        ref_pdf = REFIT_PDF_PATH or str(Path("Dash_Input") / "GVAR_LLM_EM_plots_refit.pdf")
        ref_jpg = Path(REFIT_JPG_DIR)

        if REFIT_MODE == "pdf":
            Path(ref_pdf).parent.mkdir(parents=True, exist_ok=True)
            with PdfPages(ref_pdf) as pdf_doc:
                plot_offline_core_from_pack(
                    offline_refit,
                    countries,
                    countries,
                    "pdf",
                    pdf_doc,
                    None,
                    jpg_name_prefix="refit_core",
                )
                plot_offline_breakscore_from_pack(
                    offline_refit,
                    countries,
                    countries,
                    drop_quarters_dict,
                    "pdf",
                    pdf_doc,
                    None,
                    jpg_prefix="refit_breakscore",
                )
        elif REFIT_MODE == "jpg":
            core_j = Path(REFIT_CORE_JPG_DIR)
            core_j.mkdir(parents=True, exist_ok=True)
            ref_jpg.mkdir(parents=True, exist_ok=True)
            plot_offline_core_from_pack(
                offline_refit,
                countries,
                countries,
                "jpg",
                None,
                core_j,
                jpg_name_prefix="refit_core",
            )
            plot_offline_breakscore_from_pack(
                offline_refit,
                countries,
                countries,
                drop_quarters_dict,
                "jpg",
                None,
                ref_jpg,
                jpg_prefix="refit_breakscore",
            )
        else:
            plot_offline_core_from_pack(
                offline_refit,
                countries,
                countries,
                "plot",
                None,
                None,
                jpg_name_prefix="refit_core",
            )
            plot_offline_breakscore_from_pack(
                offline_refit,
                countries,
                countries,
                drop_quarters_dict,
                "plot",
                None,
                None,
                jpg_prefix="refit_breakscore",
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="SDR visualize from pickle (config at top of file).")
    parser.add_argument(
        "--pickle",
        default=None,
        help="Override CONFIG PICKLE_PATH.",
    )
    parser.add_argument(
        "--countries",
        default=None,
        help="Comma-separated ISO3 list; overrides USE_ALL_COUNTRIES_FROM_PICKLE "
        "and COUNTRY_ISO3_ALLOWLIST.",
    )
    args = parser.parse_args()
    cli_sel = _parse_countries_arg(args.countries)
    visualize_from_pickle(args.pickle, cli_sel)


if __name__ == "__main__":
    main()
