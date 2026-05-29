import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px


def _build_country_to_iso_map(iso3_to_country: dict[str, Any] | None = None) -> dict[str, str]:
    """Build country-name -> ISO3 mapping from the pipeline config."""
    out: dict[str, str] = {}
    if iso3_to_country:
        for iso3, country_name in iso3_to_country.items():
            key = str(country_name).strip().lower()
            val = str(iso3).strip().upper()
            if key and len(val) == 3:
                out[key] = val
    return out


def _normalize_country_to_iso3(
    series: pd.Series,
    iso3_to_country: dict[str, Any] | None = None,
) -> pd.Series:
    """Normalize mixed country labels (full names or ISO3) into ISO3 codes."""
    cmap = _build_country_to_iso_map(iso3_to_country)
    raw = series.astype(str).str.strip()
    iso3_direct = raw.str.upper().where(raw.str.len() == 3)
    mapped = raw.str.lower().map(cmap)
    return iso3_direct.fillna(mapped)


def _to_binary(value, default=0):
    if pd.isna(value):
        return default
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return 1
    if s in {"0", "false", "no", "n"}:
        return 0
    try:
        return int(float(s))
    except Exception:
        return default


def _to_confidence(value, default=0.0):
    if pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _normalize_break_type(text):
    if pd.isna(text):
        return "other"
    t = str(text).strip().lower()
    if "financial" in t:
        return "Financial Crisis"
    if "policy" in t:
        return "Policy Change"
    if "external" in t:
        return "External Shock"
    if "trade" in t or "commodity" in t:
        return "Trade/Commodity Shock"
    if "climate" in t:
        return "Climate Shock"
    return "Other"


def preprocess_llm_breaks(
    llm_df: pd.DataFrame,
    iso3_to_country: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Clean LLM break table and build break_strength."""
    need = ["country", "break_year", "break_supported", "break_type", "duration", "climate_related"]
    missing = [c for c in need if c not in llm_df.columns]
    if missing:
        raise ValueError(f"LLM table missing required fields: {missing}")

    out = llm_df.copy()
    out["country"] = _normalize_country_to_iso3(out["country"], iso3_to_country)
    out["break_year"] = pd.to_numeric(out["break_year"], errors="coerce")
    out = out.dropna(subset=["country", "break_year"]).copy()
    out["break_year"] = out["break_year"].astype(int)
    out["break_supported"] = out["break_supported"].apply(_to_binary)
    out["climate_related"] = out["climate_related"].apply(_to_binary)
    out["confidence"] = out["confidence"].apply(_to_confidence) if "confidence" in out.columns else 0.0
    out["break_strength"] = out["break_supported"] * out["confidence"]
    out["break_type_mapped"] = out["break_type"].apply(_normalize_break_type)
    return out


def build_break_structures(
    llm_df: pd.DataFrame,
    iso3_to_country: dict[str, Any] | None = None,
):
    """Build break_dict and climate_flag."""
    df = preprocess_llm_breaks(llm_df, iso3_to_country=iso3_to_country)
    supported = df[df["break_supported"] == 1].copy()

    break_dict = (
        supported.groupby("country")["break_year"]
        .apply(lambda s: sorted(set(s.tolist())))
        .to_dict()
    )
    climate_flag = {
        (r.country, int(r.break_year)): int(r.climate_related)
        for r in supported[["country", "break_year", "climate_related"]].itertuples(index=False)
    }
    return break_dict, climate_flag, supported


def plot_structural_break_score_with_llm_overlay(
    score_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    score_col: str = "score",
    time_col: str = "quarter",
    country_col: str = "country",
    show_llm_overlay: bool = True,
    top_k: int = 5,
    ncols: int = 3,
    iso3_to_country: dict[str, Any] | None = None,
):
    """
    Overlay LLM break information on the structural break score curves:
    - If show_llm_overlay=True, add 4-quarter shading (yellow=climate, blue=non-climate)
    - Mark top_k breaks by break_strength for each country with "*"
    """
    if score_col not in score_df.columns:
        raise ValueError(f"score_df missing column: {score_col}")
    if country_col not in score_df.columns:
        raise ValueError(f"score_df missing column: {country_col}")
    if time_col not in score_df.columns:
        raise ValueError(f"score_df missing column: {time_col}")

    work = score_df.copy()
    work[country_col] = _normalize_country_to_iso3(work[country_col], iso3_to_country)
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=[country_col, time_col, score_col]).copy()

    break_dict, climate_flag, _supported = build_break_structures(
        llm_df,
        iso3_to_country=iso3_to_country,
    )

    countries = sorted(work[country_col].unique().tolist())
    n = len(countries)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), sharex=False)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, c in enumerate(countries):
        ax = axes[i]
        g = work[work[country_col] == c].sort_values(time_col)
        ax.plot(g[time_col], g[score_col], color="black", linewidth=1.4, label="structural break score")
        ax.set_title(c)
        ax.grid(True, alpha=0.25)

        if show_llm_overlay:
            for y in break_dict.get(c, []):
                start = pd.Timestamp(year=int(y), month=1, day=1)
                end = start + pd.DateOffset(months=12)
                is_climate = climate_flag.get((c, int(y)), 0) == 1
                ax.axvspan(
                    start,
                    end,
                    color="#F3D34A" if is_climate else "#5DA5DA",
                    alpha=0.22,
                    lw=0,
                )

        # Mark top-k highest score points for each country.
        c_top = g.nlargest(top_k, score_col)
        for _, r in c_top.iterrows():
            x_star = r[time_col]
            y_star = r[score_col]
            ax.plot([x_star], [y_star], marker="*", markersize=12, color="red", zorder=5)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles = [
        plt.Line2D([0], [0], color="black", lw=1.4, label="score"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#5DA5DA", alpha=0.22, label="LLM break (non-climate)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#F3D34A", alpha=0.22, label="LLM break (climate)"),
        plt.Line2D([0], [0], marker="*", color="red", lw=0, markersize=10, label=f"Top {top_k} score"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_break_supported_ratio(
    llm_df: pd.DataFrame,
    iso3_to_country: dict[str, Any] | None = None,
):
    df = preprocess_llm_breaks(llm_df, iso3_to_country=iso3_to_country)
    ratio = (df["break_supported"] == 1).mean()
    values = [ratio, 1 - ratio]
    labels = ["break_supported = 1", "other"]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Share of Supported Structural Breaks")
    return fig


def plot_break_type_distribution(
    llm_df: pd.DataFrame,
    iso3_to_country: dict[str, Any] | None = None,
):
    df = preprocess_llm_breaks(llm_df, iso3_to_country=iso3_to_country)
    use = df[df["break_supported"] == 1].copy()
    if use.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No samples with break_supported = 1", ha="center", va="center")
        ax.axis("off")
        return fig

    vc = use["break_type_mapped"].value_counts()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.pie(vc.values, labels=vc.index.tolist(), autopct="%1.1f%%", startangle=120)
    ax.set_title("Break Type Distribution (break_supported = 1 only)")
    return fig


def build_time_slice_maps(
    score_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    score_col: str = "score",
    country_col: str = "country",
    year_col: str = "year",
    start_year: int = 1998,
    end_year: int = 2024,
    output_dir: str = "time_slice_maps",
    iso3_to_country: dict[str, Any] | None = None,
):
    """
    Generate yearly HTML maps for the selected year range:
    - Marker size: structural break score
    - Marker color: red if break_supported=1 in that year, else gray
    - Hover: country, break_year, break_type, duration, climate_related, summary
    """
    llm = preprocess_llm_breaks(llm_df, iso3_to_country=iso3_to_country)
    score = score_df.copy()
    score[country_col] = _normalize_country_to_iso3(score[country_col], iso3_to_country)
    score[year_col] = pd.to_numeric(score[year_col], errors="coerce")
    score[score_col] = pd.to_numeric(score[score_col], errors="coerce")
    score = score.dropna(subset=[country_col, year_col, score_col]).copy()
    score[year_col] = score[year_col].astype(int)

    if "summary" not in llm.columns:
        llm["summary"] = ""

    llm_match = llm[
        ["country", "break_year", "break_supported", "break_type", "duration", "climate_related", "summary"]
    ].copy()
    llm_match = llm_match.rename(columns={"break_year": year_col})

    merged = score.merge(llm_match, on=[country_col, year_col], how="left")
    merged["break_supported"] = merged["break_supported"].fillna(0).astype(int)
    # Keep only LLM-supported breaks on the map.
    merged = merged[merged["break_supported"] == 1].copy()
    merged["point_color"] = merged["climate_related"].map({1: "green", 0: "blue"})
    merged["break_type"] = merged["break_type"].fillna("")
    merged["duration"] = merged["duration"].fillna("")
    merged["climate_related"] = merged["climate_related"].fillna(0).astype(int)
    merged["summary"] = merged["summary"].fillna("")
    # country_col is normalized to ISO3; keep an explicit ISO column for plotly.
    merged["iso"] = merged[country_col].astype(str).str.strip().str.upper()
    merged = merged.dropna(subset=["iso"]).copy()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for year in range(start_year, end_year + 1):
        ydf = merged[merged[year_col] == year].copy()
        if ydf.empty:
            continue
        ydf["size_scaled"] = pd.to_numeric(ydf[score_col], errors="coerce") * 100.0
        ydf["size_scaled"] = ydf["size_scaled"].fillna(0.0).clip(lower=0.1)

        fig = px.scatter_geo(
            ydf,
            locations="iso",
            locationmode="ISO-3",
            size="size_scaled",
            size_max=25,
            color="point_color",
            color_discrete_map={
                "green": "green",
                "blue": "blue",
            },
            hover_name=country_col,
            hover_data={
                year_col: True,
                "break_type": True,
                "duration": True,
                "climate_related": True,
                "summary": True,
                score_col: ":.3f",
                "iso": True,
                "size_scaled": ":.1f",
                "point_color": False,
            },
            title=f"Structural Break Map - {year}",
            projection="natural earth",
        )
        path = out / f"map_{year}.html"
        fig.write_html(str(path))
        saved_files.append(str(path))
    return saved_files
