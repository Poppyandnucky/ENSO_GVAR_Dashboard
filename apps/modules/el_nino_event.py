from __future__ import annotations

from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st


MONTH_ORDER = ["6", "7", "8", "9", "10", "11", "12", "1", "2", "3", "4"]
MONTH_LABELS = {
    "6": "2026-06",
    "7": "2026-07",
    "8": "2026-08",
    "9": "2026-09",
    "10": "2026-10",
    "11": "2026-11",
    "12": "2026-12",
    "1": "2027-01",
    "2": "2027-02",
    "3": "2027-03",
    "4": "2027-04",
}

PRODUCTS = {
    "Heat": {
        "file": "temp_NMME.csv",
        "label": "Heat",
        "description": "Predicted monthly temperature anomaly.",
    },
    "Moisture": {
        "file": "precipitation_NMME.csv",
        "label": "Moisture",
        "description": "Predicted monthly moisture anomaly.",
    },
    "H_Maize": {
        "file": "H_Maize.csv",
        "label": "H_Maize",
        "description": "Monthly maize harvested-area weight.",
    },
    "H_Rice": {
        "file": "H_Rice.csv",
        "label": "H_Rice",
        "description": "Monthly rice harvested-area weight.",
    },
    "H_Soya": {
        "file": "H_Soya.csv",
        "label": "H_Soya",
        "description": "Monthly soya harvested-area weight.",
    },
    "H_Wheat": {
        "file": "H_Wheat.csv",
        "label": "H_Wheat",
        "description": "Monthly wheat harvested-area weight.",
    },
}

CLIMATE_PRODUCTS = ["Heat", "Moisture"]
WEIGHT_PRODUCTS = ["Population", "H_Maize", "H_Rice", "H_Soya", "H_Wheat"]
COUNTRY_NAME_TO_ISO = {
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


def _file_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _display_lon(values: pd.Series) -> pd.Series:
    return ((pd.to_numeric(values, errors="coerce") + 180) % 360) - 180


def _month_timestamp(month: str) -> pd.Timestamp:
    return pd.Timestamp(f"{MONTH_LABELS[month]}-01")


@st.cache_data(show_spinner=False)
def _load_geolat_product(path: str, path_mtime: float) -> pd.DataFrame:
    del path_mtime
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    keep_cols = ["lat", "lon"] + [m for m in MONTH_ORDER if m in df.columns]
    df = df[keep_cols].copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    for month in [c for c in df.columns if c not in {"lat", "lon"}]:
        df[month] = pd.to_numeric(df[month], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df["join_lon"] = df["lon"] % 360
    df["plot_lon"] = _display_lon(df["lon"])
    return df


def _product_path(geolat_dir: Path, product_id: str) -> Path:
    return geolat_dir / PRODUCTS[product_id]["file"]


def _load_product_points(geolat_dir: Path, product_id: str, month: str) -> pd.DataFrame:
    path = _product_path(geolat_dir, product_id)
    df = _load_geolat_product(str(path), _file_mtime(path))
    if month not in df.columns:
        return pd.DataFrame(columns=["lat", "lon", "value", "join_lon", "plot_lon"])
    out = df[["lat", "lon", "join_lon", "plot_lon", month]].rename(columns={month: "value"})
    return out.dropna(subset=["value"])


@st.cache_data(show_spinner=False)
def _load_country_boundaries(
    shapefile_path: str,
    shapefile_mtime: float,
    countries: tuple[str, ...],
) -> dict:
    del shapefile_mtime
    world = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    world = world[world["ISO_A3"].isin(countries)].copy()
    return world[["ISO_A3", "NAME", "geometry"]].__geo_interface__


@st.cache_data(show_spinner=False)
def _load_country_masks(geolat_dir: str, masks_mtime: tuple[tuple[str, float], ...]) -> pd.DataFrame:
    del masks_mtime
    root = Path(geolat_dir)
    frames = []
    for path in sorted(root.glob("mask_*.csv")):
        country_name = path.stem.removeprefix("mask_")
        iso3 = COUNTRY_NAME_TO_ISO.get(country_name)
        if not iso3:
            continue
        df = pd.read_csv(path)
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
        df["country_mask"] = pd.to_numeric(df["country_mask"], errors="coerce").fillna(0)
        df = df[df["country_mask"] > 0][["lat", "lon"]].copy()
        df["join_lon"] = df["lon"] % 360
        df["iso_a3"] = iso3
        frames.append(df[["lat", "join_lon", "iso_a3"]])
    if not frames:
        return pd.DataFrame(columns=["lat", "join_lon", "iso_a3"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["lat", "join_lon", "iso_a3"])


def _mask_mtimes(geolat_dir: Path) -> tuple[tuple[str, float], ...]:
    return tuple((p.name, _file_mtime(p)) for p in sorted(geolat_dir.glob("mask_*.csv")))


def _find_population_grid(geolat_dir: Path) -> Path | None:
    candidates = [
        geolat_dir / "Population.csv",
        geolat_dir / "population.csv",
        geolat_dir / "population" / "Population.csv",
        geolat_dir / "population" / "population.csv",
        geolat_dir / "population" / "population_weight_grid.csv",
        geolat_dir / "population" / "nmme_population_weight_grid.csv",
    ]
    return next((path for path in candidates if path.is_file()), None)


@st.cache_data(show_spinner=False)
def _load_population_grid(path: str, path_mtime: float) -> pd.DataFrame:
    del path_mtime
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    value_col = next(
        (
            c
            for c in ["population", "Population", "population_weight", "weight", "value"]
            if c in df.columns
        ),
        None,
    )
    if value_col is None:
        raise ValueError("Population grid must contain a population/weight/value column.")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["weight"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "weight"])
    df["join_lon"] = df["lon"] % 360
    return df[["lat", "join_lon", "weight"]]


def _load_weight_points(geolat_dir: Path, weight_id: str, month: str) -> pd.DataFrame | None:
    if weight_id == "Population":
        path = _find_population_grid(geolat_dir)
        if path is None:
            return None
        return _load_population_grid(str(path), _file_mtime(path))

    weight = _load_product_points(geolat_dir, weight_id, month)
    return weight[["lat", "join_lon", "value"]].rename(columns={"value": "weight"})


def _color_points(df: pd.DataFrame, reverse_colors: bool = False) -> pd.DataFrame:
    out = df.copy()
    values = out["value"].to_numpy(dtype=float)
    max_abs = float(np.nanmax(np.abs(values))) if len(values) else 1.0
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0

    def color(value: float) -> list[int]:
        scaled = max(-1.0, min(1.0, float(value) / max_abs))
        neutral = np.array([246, 246, 246])
        blue = np.array([49, 130, 189])
        red = np.array([202, 0, 32])

        if reverse_colors:
            # For Moisture: negative = red/dry, positive = blue/wet
            if scaled < 0:
                rgb = neutral + abs(scaled) * (red - neutral)
            else:
                rgb = neutral + scaled * (blue - neutral)
        else:
            # For Heat: negative = blue/cool, positive = red/hot
            if scaled < 0:
                rgb = neutral + abs(scaled) * (blue - neutral)
            else:
                rgb = neutral + scaled * (red - neutral)

        return [int(x) for x in rgb] + [185]

    out["fill_color"] = out["value"].map(color)
    return out

def _build_map(
    points: pd.DataFrame,
    boundaries_geojson: dict,
    product_label: str,
    month: str,
    reverse_colors: bool = False,
) -> pdk.Deck:
    colored = _color_points(points, reverse_colors=reverse_colors)
    center_lat = float(colored["lat"].mean()) if not colored.empty else 0.0
    center_lon = float(colored["plot_lon"].mean()) if not colored.empty else 0.0

    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=colored,
        get_position="[plot_lon, lat]",
        get_radius=50000,
        radius_min_pixels=3,
        radius_max_pixels=15,
        get_fill_color="fill_color",
        pickable=True,
        opacity=0.82,
    )
    boundary_layer = pdk.Layer(
        "GeoJsonLayer",
        data=boundaries_geojson,
        stroked=True,
        filled=False,
        get_line_color=[77, 77, 77, 220],
        get_line_width=30000,
        line_width_min_pixels=1,
        pickable=False,
    )
    return pdk.Deck(
        layers=[point_layer, boundary_layer],
        initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=2.2, pitch=0),
        map_style=pdk.map_styles.CARTO_LIGHT_NO_LABELS,
        tooltip={
            "html": (
                f"<b>{product_label}</b><br/>"
                f"Month: {MONTH_LABELS[month]}<br/>"
                "Lat: {lat}<br/>Lon: {plot_lon}<br/>Value: {value}"
            ),
            "style": {"backgroundColor": "#173f73", "color": "white"},
        },
    )


def _weighted_field(
    geolat_dir: Path,
    climate_id: str,
    weight_id: str,
    month: str,
) -> pd.DataFrame | None:
    climate = _load_product_points(geolat_dir, climate_id, month)
    weight = _load_weight_points(geolat_dir, weight_id, month)
    if weight is None:
        return None
    merged = climate.merge(weight, on=["lat", "join_lon"], how="inner")
    merged = merged.dropna(subset=["value", "weight"])
    if weight_id == "Population":
        finite = np.isfinite(merged["weight"]) & (merged["weight"] > 0)
        if not finite.any():
            return merged.assign(value=np.nan).dropna(subset=["value"])
        mean_weight = float(merged.loc[finite, "weight"].mean())
        if not np.isfinite(mean_weight) or mean_weight == 0:
            return merged.assign(value=np.nan).dropna(subset=["value"])
        merged["effective_weight"] = merged["weight"] / mean_weight
    else:
        merged["effective_weight"] = merged["weight"]
    merged = merged.rename(columns={"value": "climate_value"})
    merged["raw_value"] = merged["climate_value"] * merged["effective_weight"]
    merged["value"] = merged["raw_value"]
    merged["plot_lon"] = _display_lon(merged["join_lon"])
    return merged[
        [
            "lat",
            "join_lon",
            "plot_lon",
            "climate_value",
            "raw_value",
            "value",
            "weight",
            "effective_weight",
        ]
    ]


def _normalize_field_by_country_abs_max(
    field: pd.DataFrame,
    masks: pd.DataFrame,
) -> pd.DataFrame:
    joined = field.merge(masks, on=["lat", "join_lon"], how="inner")
    if joined.empty:
        return field.iloc[0:0].copy()

    denominators = (
        joined.assign(abs_raw_value=joined["raw_value"].abs())
        .groupby("iso_a3", as_index=False)["abs_raw_value"]
        .max()
        .rename(columns={"abs_raw_value": "country_abs_max"})
    )
    joined = joined.merge(denominators, on="iso_a3", how="left")
    valid = np.isfinite(joined["country_abs_max"]) & (joined["country_abs_max"] > 0)
    joined = joined[valid].copy()
    joined["value"] = joined["raw_value"] / joined["country_abs_max"]
    return joined[
        [
            "lat",
            "join_lon",
            "plot_lon",
            "value",
            "raw_value",
            "climate_value",
            "weight",
            "effective_weight",
            "iso_a3",
            "country_abs_max",
        ]
    ]


@st.cache_data(show_spinner=False)
def _country_weighted_timeseries(
    geolat_dir: str,
    climate_id: str,
    weight_id: str,
    masks_mtime: tuple[tuple[str, float], ...],
    population_mtime: float,
) -> pd.DataFrame:
    del population_mtime
    root = Path(geolat_dir)
    masks = _load_country_masks(geolat_dir, masks_mtime)
    rows = []
    for month in MONTH_ORDER:
        field = _weighted_field(root, climate_id, weight_id, month)
        if field is None or field.empty:
            continue
        joined = field.merge(masks, on=["lat", "join_lon"], how="inner")
        if joined.empty:
            continue
        valid = (
            np.isfinite(joined["climate_value"])
            & np.isfinite(joined["effective_weight"])
            & (joined["effective_weight"] > 0)
        )
        joined = joined[valid].copy()
        if joined.empty:
            continue
        joined["weighted_sum_part"] = joined["climate_value"] * joined["effective_weight"]
        grouped = joined.groupby("iso_a3", as_index=False).agg(
            weighted_sum=("weighted_sum_part", "sum"),
            weight_sum=("effective_weight", "sum"),
            cells=("climate_value", "size"),
        )
        grouped = grouped[grouped["weight_sum"] > 0].copy()
        grouped["value"] = grouped["weighted_sum"] / grouped["weight_sum"]
        grouped = grouped[["iso_a3", "value", "cells", "weight_sum"]]
        grouped["target_month"] = _month_timestamp(month)
        grouped["month"] = month
        rows.append(grouped)
    if not rows:
        return pd.DataFrame(columns=["iso_a3", "value", "cells", "target_month", "month"])
    return pd.concat(rows, ignore_index=True).sort_values(["iso_a3", "target_month"])


def _country_timeseries_chart(
    df: pd.DataFrame,
    selected_country: str,
    country_label_func: Callable[[str], str],
    title: str,
):
    country_df = df[df["iso_a3"].astype(str).eq(selected_country)].copy()
    if country_df.empty:
        return None
    country_df["month_label"] = country_df["target_month"].dt.strftime("%Y-%m")
    fig = px.line(
        country_df,
        x="target_month",
        y="value",
        markers=True,
        title=f"{title}: {country_label_func(selected_country)}",
        labels={"target_month": "Target month", "value": "Country weighted mean"},
        hover_data={"month_label": True, "cells": True, "weight_sum": True, "target_month": False},
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=55, b=35))
    return fig


def render_el_nino_event_module(
    repo_root: Path,
    dashboard_countries: list[str],
    selected_country: str,
    country_label_func: Callable[[str], str],
) -> None:
    output_root = repo_root / "models" / "output"
    geolat_dir = output_root / "Geo_Lat"
    if not geolat_dir.exists():
        st.warning("No processed climate outputs found under `models/output/Geo_Lat/`.")
        return

    st.markdown(
        "This module reads already processed 2026-2027 El Nino event outputs from "
        "`models/output/Geo_Lat/`. It does not read NetCDF files or perform climate preprocessing."
    )

    shapefile = repo_root / "data" / "natural_earth" / "ne_110m_admin_0_countries.shp"
    boundaries = _load_country_boundaries(
        str(shapefile),
        _file_mtime(shapefile),
        tuple(dashboard_countries),
    )

    st.subheader("Monthly Climate and Crop Fields")
    c1, c2 = st.columns([1.25, 1])
    with c1:
        product_id = st.selectbox(
            "Variable",
            list(PRODUCTS),
            format_func=lambda x: PRODUCTS[x]["label"],
            key="el_nino_product",
        )
    with c2:
        selected_month = st.selectbox(
            "Target month",
            MONTH_ORDER,
            format_func=lambda x: MONTH_LABELS[x],
            key="el_nino_month",
        )

    points = _load_product_points(geolat_dir, product_id, selected_month)
    if points.empty:
        st.warning(f"No data were available for {PRODUCTS[product_id]['label']} in {MONTH_LABELS[selected_month]}.")
    else:
        st.caption(PRODUCTS[product_id]["description"])
        st.pydeck_chart(
            _build_map(
                points,
                boundaries,
                PRODUCTS[product_id]["label"],
                selected_month,
                reverse_colors=(product_id == "Moisture"),
            ),
            width="stretch",
        )

    st.subheader("Weighted Climate Exposure")
    w1, w2, w3 = st.columns([1, 1, 1])
    with w1:
        climate_id = st.selectbox(
            "Climate variable",
            CLIMATE_PRODUCTS,
            format_func=lambda x: PRODUCTS[x]["label"],
            key="weighted_climate",
        )
    with w2:
        weight_id = st.selectbox(
            "Weight",
            WEIGHT_PRODUCTS,
            key="weighted_weight",
        )
    with w3:
        weighted_month = st.selectbox(
            "Weighted field month",
            MONTH_ORDER,
            format_func=lambda x: MONTH_LABELS[x],
            key="weighted_month",
        )

    population_path = _find_population_grid(geolat_dir)
    population_mtime = _file_mtime(population_path) if population_path else 0.0
    if weight_id == "Population" and population_path is None:
        st.warning(
            "Population weighting needs a processed `lat, lon, population` grid CSV under "
            "`models/output/Geo_Lat/population/`. The dashboard will not process the GeoTIFF at runtime."
        )
        return

    weighted = _weighted_field(geolat_dir, climate_id, weight_id, weighted_month)
    if weighted is None or weighted.empty:
        st.warning("No weighted climate field could be computed for the current selection.")
        return

    masks_mtime = _mask_mtimes(geolat_dir)
    masks = _load_country_masks(str(geolat_dir), masks_mtime)
    weighted_for_map = _normalize_field_by_country_abs_max(
        weighted,
        masks,
    )
    if weighted_for_map.empty:
        st.warning(
            "No country has a non-zero maximum absolute weighted exposure for this "
            "climate/weight/month combination."
        )
        return
    if weight_id == "Population":
        st.caption(
            f"Displayed value = [{PRODUCTS[climate_id]['label']} x "
            f"(Population / mean(Population))] / max(abs(value)) within each country "
            f"for {MONTH_LABELS[weighted_month]}."
        )
    else:
        st.caption(
            f"Displayed value = ({PRODUCTS[climate_id]['label']} x {weight_id}) / "
            f"max(abs(value)) within each country for {MONTH_LABELS[weighted_month]}."
        )

    st.pydeck_chart(
        _build_map(
            weighted_for_map,
            boundaries,
            f"Weighted {PRODUCTS[climate_id]['label']} by {weight_id}",
            weighted_month,
            reverse_colors=(climate_id == "Moisture"),
        ),
        width="stretch",
    )
        

    ts = _country_weighted_timeseries(
        str(geolat_dir),
        climate_id,
        weight_id,
        masks_mtime,
        population_mtime,
    )
    if ts.empty:
        st.warning("No country time series could be computed for the current weighted exposure selection.")
        return

    chart = _country_timeseries_chart(
        ts,
        selected_country,
        country_label_func,
        f"{PRODUCTS[climate_id]['label']} weighted by {weight_id}",
    )
    if chart is not None:
        st.plotly_chart(chart, width="stretch")

    latest = ts[ts["target_month"].eq(_month_timestamp(weighted_month))].copy()
    latest["Country"] = latest["iso_a3"].map(country_label_func)
    latest = latest.sort_values("value", ascending=False)
    st.dataframe(
        latest[["Country", "iso_a3", "value", "cells"]],
        column_config={
            "Country": st.column_config.TextColumn("Country"),
            "iso_a3": st.column_config.TextColumn("ISO3"),
            "value": st.column_config.NumberColumn("Country weighted mean", format="%.3f"),
            "cells": st.column_config.NumberColumn("Mask cells"),
        },
        hide_index=True,
        width="stretch",
    )
