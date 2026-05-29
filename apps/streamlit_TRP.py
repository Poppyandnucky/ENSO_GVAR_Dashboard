import sys
from pathlib import Path
import pickle
import re

# Project root must be on sys.path when launching: streamlit run apps/streamlit_TRP.py
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import streamlit.components.v1 as components
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

from models.config import ISO3_TO_IMF_NAME_FULL
import numpy as np
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import warnings
warnings.simplefilter("default", Warning)

STRUCTURAL_BREAK_DIR = _ROOT / "structural_break"
GEMINI_OUTPUT_DIR = STRUCTURAL_BREAK_DIR / "gemini output"
WB_TOP4_PATH = STRUCTURAL_BREAK_DIR / "wb_top4.csv"
PREGENERATED_MAP_DIR = STRUCTURAL_BREAK_DIR / "map1998-2024"
PIPELINE_PICKLE_CANDIDATES = [
    STRUCTURAL_BREAK_DIR / "gvar_pipeline_results.pkl",
    STRUCTURAL_BREAK_DIR / "Dash_Input" / "gvar_pipeline_results.pkl",
    _ROOT / "Dash_Input" / "gvar_pipeline_results.pkl",
]
FORECAST_PICKLE_CANDIDATES = [
    _ROOT / "Dash_Input" / "gvar_forecast_results.pkl",
    _ROOT / "analysis" / "Dash_Input" / "gvar_forecast_results.pkl",
]
SCENARIO_OUTPUT_ROOT = _ROOT / "analysis" / "Dash_Output"

DASHBOARD_COUNTRIES = [
    "BRA",  # Brazil
    "CHL",  # Chile
    "COL",  # Colombia
    "MEX",  # Mexico
    "KEN",  # Kenya
    "ZAF",  # South Africa
    "IND",  # India
    "IDN",  # Indonesia
    "THA",  # Thailand
    "PER",  # Peru
    "PHL",  # Philippines
    "EGY",  # Egypt
]
MACRO_IMPACT_VARS = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]

HELP_TEXT = {
    "country": "Primary country used as the default selection across dashboard tabs.",
    "response": "Macroeconomic response variable used for scenario charts and impact summaries.",
    "enso_forecast": "Forecast ENSO index value for the next quarter. Positive values indicate El Nino-like conditions; negative values indicate La Nina-like conditions.",
    "stress_threshold": "Percentile cutoff used to define an extreme physical stress event. Higher values focus on rarer, more severe heat or moisture outcomes.",
    "baseline_probability": "Unconditional probability implied by the selected stress threshold before applying ENSO information.",
    "enso_probability": "Estimated probability after conditioning on the selected ENSO forecast. The delta compares this value with the stored baseline probability.",
    "risk_summary": "Country-level probabilities generated from the selected ENSO forecast and stress threshold.",
    "scenario_countries": "Countries included in the scenario forecast charts, summary tables, and cumulative impact maps.",
    "qr_country": "Country used for the precomputed QR adaptive Kalman filter comparison charts.",
    "sb_countries": "Countries shown in the structural-break score, document-evidence, Gemini output, and overlap panels.",
    "llm_overlay": "Adds highlighted years where the Gemini/LLM evidence flags a supported structural break.",
    "score_series": "Score diagnostics to plot. Innovation captures forecast surprise; coefficient change captures parameter movement; composite combines available signals.",
    "wb_year": "World Bank document year used to display the top supporting document records.",
    "impact_window": "Number of forward quarters used to average raw macro impact and model-surprise scores.",
    "map_year": "Pre-generated structural-break map year to display.",
}

def get_country_regime_ts(panel, country):
    df = panel[panel["country"] == country].copy()
    df["year"] = df["quarter"].dt.year
    df.index.freq = "QS-OCT"

    regime = (
        df.groupby("year")
        .agg({
            "CPI_YoY_annual": "first",
            "FX_YoY_annual": "first",
        })
        .reset_index()
    )

    return regime

def get_country_regime(panel, country):
    df = panel[panel["country"] == country].copy()
    df.index.freq = "QS-OCT"

    regime = (
        df.groupby(df["quarter"].dt.year)
        .agg({
            "CPI_YoY_annual": "first",
            "FX_YoY_annual": "first",
        })
        .reset_index()
        .rename(columns={"quarter": "year"})
    )
    regime.index.freq("QS-OCT")

    return regime


def _norm_country_name(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


ISO_TO_NAME = {k: v for k, v in ISO3_TO_IMF_NAME_FULL.items()}
NAME_TO_ISO = {_norm_country_name(v): k for k, v in ISO_TO_NAME.items()}
NAME_TO_ISO.update(
    {
        "united states": "USA",
        "united kingdom": "GBR",
        "czech republic": "CZE",
        "slovak republic": "SVK",
        "korea republic of": "KOR",
        "russian federation": "RUS",
    }
)


def country_to_iso3(value):
    s = str(value).strip()
    if len(s) == 3 and s.upper().isalpha():
        return s.upper()
    return NAME_TO_ISO.get(_norm_country_name(s))


def iso3_to_label(iso3):
    return f"{iso3} - {ISO_TO_NAME.get(iso3, iso3)}"


def _filter_dashboard_countries(values):
    allowed = set(DASHBOARD_COUNTRIES)
    return [c for c in DASHBOARD_COUNTRIES if c in set(values) and c in allowed]


def prepare_country_stressor_data(panel, country, stressor_var, stressor_pct=90):
    df = panel[panel["country"] == country].copy().sort_values("quarter")
    if stressor_var not in df.columns:
        return pd.DataFrame()
    thr = df[stressor_var].quantile(stressor_pct / 100)
    df["stressor_event"] = (df[stressor_var] >= thr).astype(int)
    df["stressor_event_next"] = df["stressor_event"].shift(-1)
    return df.dropna(subset=["ENSO", "stressor_event_next"])


def fit_enso_stressor_model(df):
    y = df["stressor_event_next"].astype(float)
    if len(y) < 10 or y.nunique(dropna=False) < 2 or df["ENSO"].nunique() < 2:
        return None
    X = sm.add_constant(df["ENSO"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            return sm.Logit(y, X).fit(disp=0)
        except (np.linalg.LinAlgError, ValueError):
            return None


@st.cache_data
def build_climate_probability_rows(_panel_path, enso_forecast, heat_moist_pct):
    panel_local = load_gvar_panel(_panel_path)
    panel_local = panel_local[panel_local["country"].astype(str).isin(DASHBOARD_COUNTRIES)].copy()
    rows = []
    for c in DASHBOARD_COUNTRIES:
        if c not in set(panel_local["country"].dropna().astype(str)):
            continue
        df_heat = prepare_country_stressor_data(
            panel_local, c, "PRITHVI_HEAT_EXTENT", stressor_pct=heat_moist_pct
        )
        df_moist = prepare_country_stressor_data(
            panel_local, c, "PRITHVI_MOISTURE_EXTENT", stressor_pct=heat_moist_pct
        )

        p_heat = np.nan
        model_heat = fit_enso_stressor_model(df_heat) if len(df_heat) >= 10 else None
        if model_heat is not None:
            p_heat = float(model_heat.predict(pd.DataFrame({"const": [1.0], "ENSO": [enso_forecast]}))[0])

        p_moist = np.nan
        model_moist = fit_enso_stressor_model(df_moist) if len(df_moist) >= 10 else None
        if model_moist is not None:
            p_moist = float(model_moist.predict(pd.DataFrame({"const": [1.0], "ENSO": [enso_forecast]}))[0])

        rows.append(
            {
                "Country": c,
                "Heat probability (%)": p_heat * 100,
                "Moisture probability (%)": p_moist * 100,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def load_wb_top4():
    if not WB_TOP4_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(WB_TOP4_PATH, low_memory=False)
    df["country"] = df["country"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df.get("break_year"), errors="coerce")
    return df


def _extract_field(raw_text, field):
    if pd.isna(raw_text):
        return None
    pattern = rf"{field}\s*:\s*(.+)"
    m = re.search(pattern, str(raw_text), flags=re.IGNORECASE)
    return m.group(1).strip() if m else None


@st.cache_data
def load_gemini_outputs():
    if not GEMINI_OUTPUT_DIR.exists():
        return pd.DataFrame()
    files = sorted(GEMINI_OUTPUT_DIR.glob("*.csv"))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            df["source_file"] = f.name
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["country"] = out["country"].astype(str).str.strip()
    out["year"] = pd.to_numeric(out.get("break_year"), errors="coerce")
    out["break_supported"] = pd.to_numeric(
        out["raw_output"].map(lambda x: _extract_field(x, "break_supported")), errors="coerce"
    )
    out["confidence"] = pd.to_numeric(
        out["raw_output"].map(lambda x: _extract_field(x, "confidence")), errors="coerce"
    )
    out["break_type"] = out["raw_output"].map(lambda x: _extract_field(x, "break_type"))
    out["summary"] = out["raw_output"].map(lambda x: _extract_field(x, "summary"))
    out["llm_joint_score"] = out["break_supported"].fillna(0) * out["confidence"].fillna(0)
    out["iso3"] = out["country"].map(country_to_iso3)
    return out


@st.cache_data
def load_pipeline_break_scores():
    for p in PIPELINE_PICKLE_CANDIDATES:
        if not p.exists():
            continue
        try:
            with open(p, "rb") as f:
                bundle = pickle.load(f)
            llm_pack = bundle.get("llm_integration", {}) if isinstance(bundle, dict) else {}
            offline_plot = bundle.get("offline_plot_data", {}) if isinstance(bundle, dict) else {}
            base_pack = offline_plot.get("base", {}) if isinstance(offline_plot, dict) else {}
            refit_pack = offline_plot.get("refit", {}) if isinstance(offline_plot, dict) else {}
            per_country = base_pack.get("per_country", {}) if isinstance(base_pack, dict) else {}
            refit_per_country = refit_pack.get("per_country", {}) if isinstance(refit_pack, dict) else {}
            break_df = llm_pack.get("break_score_df")
            comp_df = llm_pack.get("composite_break_df")
            score_year_df = llm_pack.get("score_year_df")
            llm_df = llm_pack.get("llm_df")
            config = bundle.get("config", {}) if isinstance(bundle, dict) else {}
            return {
                "path": str(p),
                "break_score_df": break_df if isinstance(break_df, pd.DataFrame) else pd.DataFrame(),
                "composite_break_df": comp_df if isinstance(comp_df, pd.DataFrame) else pd.DataFrame(),
                "score_year_df": score_year_df if isinstance(score_year_df, pd.DataFrame) else pd.DataFrame(),
                "llm_df": llm_df if isinstance(llm_df, pd.DataFrame) else pd.DataFrame(),
                "offline_per_country": per_country if isinstance(per_country, dict) else {},
                "offline_refit_per_country": refit_per_country if isinstance(refit_per_country, dict) else {},
                "config": config if isinstance(config, dict) else {},
            }
        except Exception:
            continue
    return {
        "path": None,
        "break_score_df": pd.DataFrame(),
        "composite_break_df": pd.DataFrame(),
        "score_year_df": pd.DataFrame(),
        "llm_df": pd.DataFrame(),
        "offline_per_country": {},
        "offline_refit_per_country": {},
        "config": {},
    }


@st.cache_data
def load_forecast_bundle():
    for p in FORECAST_PICKLE_CANDIDATES:
        if not p.exists():
            continue
        try:
            with open(p, "rb") as f:
                bundle = pickle.load(f)
            if isinstance(bundle, dict):
                return {"path": str(p), "bundle": bundle}
        except Exception:
            continue
    return {"path": None, "bundle": {}}


def _scenario_from_slider(value):
    if value <= -0.34:
        return "min"
    if value >= 0.34:
        return "max"
    return "mean"


def _scenario_image_path(scenario, kind, country):
    kind_to_dir_prefix = {
        "forecast": ("forecast", "forecast"),
        "kf_track": ("kf_track", "forecast_kf_track"),
        "varx_track": ("varx_track", "forecast_varx_track"),
    }
    subdir, prefix = kind_to_dir_prefix[kind]
    return SCENARIO_OUTPUT_ROOT / f"forecast_enso_{scenario}" / subdir / f"{prefix}_{country}.png"


def show_scenario_image(scenario, kind, country, caption):
    path = _scenario_image_path(scenario, kind, country)
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Missing precomputed chart: `{path}`")


def _forecast_scenarios(bundle):
    if not isinstance(bundle, dict):
        return {}
    if "scenarios" in bundle and isinstance(bundle["scenarios"], dict):
        return bundle["scenarios"]
    if "per_country" in bundle:
        scenario = bundle.get("config", {}).get("enso_scenario", "mean")
        return {scenario: bundle}
    return {}


def _country_y_scale(country_pack, panel_df, country, endo_vars):
    mu = country_pack.get("y_mu")
    sd = country_pack.get("y_sd")
    if mu is not None and sd is not None:
        return np.asarray(mu, dtype=float), np.asarray(sd, dtype=float)
    cdf = panel_df[panel_df["country"] == country].sort_values("quarter")
    vals = cdf[endo_vars].to_numpy(float)
    return np.nanmean(vals, axis=0), np.nanstd(vals, axis=0) + 1e-8


def _to_raw_y(country_pack, arr, panel_df, country):
    arr = np.asarray(arr, dtype=float)
    endo_vars = list(country_pack.get("ENDO_use", MACRO_IMPACT_VARS))
    mu, sd = _country_y_scale(country_pack, panel_df, country, endo_vars)
    return arr * sd.reshape(1, -1) + mu.reshape(1, -1)


def _forecast_country_frame(scenarios, panel_df, country, response_var):
    frames = []
    for scenario_name, scenario_bundle in scenarios.items():
        d = scenario_bundle.get("per_country", {}).get(country)
        if not d or response_var not in d.get("ENDO_use", []):
            continue
        idx = list(d["ENDO_use"]).index(response_var)
        q = pd.to_datetime(d["fc_quarters"])
        period_type = d.get("period_type")
        if not period_type or len(period_type) != len(q):
            period_type = ["Scenario forecast"] * len(q)
        y = _to_raw_y(d, d["y_hat"], panel_df, country)[:, idx]
        base = None
        if d.get("y_hat_enso0") is not None:
            base = _to_raw_y(d, d["y_hat_enso0"], panel_df, country)[:, idx]
        frames.append(
            pd.DataFrame(
                {
                    "country": country,
                    "scenario": scenario_name,
                    "quarter": q,
                    "period_type": period_type,
                    "value": y,
                    "no_enso_value": base if base is not None else np.nan,
                    "impact_vs_no_enso": y - base if base is not None else np.nan,
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_forecast_plot_df(forecast_bundle, panel_df, countries, response_var):
    scenarios = _forecast_scenarios(forecast_bundle)
    frames = [_forecast_country_frame(scenarios, panel_df, c, response_var) for c in countries]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_forecast_ranges(plot_df):
    if plot_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    q_summary = (
        plot_df.groupby(["country", "quarter", "period_type"], as_index=False)
        .agg(
            impact_min=("impact_vs_no_enso", "min"),
            impact_mean=("impact_vs_no_enso", "mean"),
            impact_max=("impact_vs_no_enso", "max"),
            value_min=("value", "min"),
            value_mean=("value", "mean"),
            value_max=("value", "max"),
        )
        .sort_values(["country", "quarter"])
    )
    scenario_df = plot_df[plot_df["period_type"].eq("Scenario forecast")].copy()
    if scenario_df.empty:
        scenario_df = plot_df.copy()
    c_summary = (
        scenario_df.groupby(["country", "scenario"], as_index=False)["impact_vs_no_enso"]
        .sum()
        .groupby("country", as_index=False)
        .agg(
            cumulative_min=("impact_vs_no_enso", "min"),
            cumulative_mean=("impact_vs_no_enso", "mean"),
            cumulative_max=("impact_vs_no_enso", "max"),
        )
    )
    return q_summary, c_summary


def plot_core_forecast(plot_df, response_var):
    if plot_df.empty:
        return None
    fig = go.Figure()
    for c, cdf in plot_df.groupby("country"):
        piv = cdf.pivot_table(index="quarter", columns="scenario", values="value", aggfunc="mean").sort_index()
        if piv.empty:
            continue
        lower = piv.min(axis=1)
        upper = piv.max(axis=1)
        mean = piv["mean"] if "mean" in piv else piv.mean(axis=1)
        fig.add_trace(
            go.Scatter(
                x=list(piv.index) + list(piv.index[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill="toself",
                fillcolor="rgba(31, 119, 180, 0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=piv.index,
                y=mean,
                mode="lines+markers",
                name=c,
                hovertemplate=f"{c}<br>%{{x|%Y-Q%q}}<br>{response_var}: %{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"{response_var}: forecast path (mean with min-max ENSO band)",
        xaxis_title="Quarter",
        yaxis_title=response_var,
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=80, b=110),
        height=560,
    )
    return fig


def plot_metric_impact_map(summary_df, response_var):
    if summary_df.empty:
        return None
    world = gpd.read_file(_ROOT / "data" / "natural_earth" / "ne_110m_admin_0_countries.shp")
    df = world[world["ISO_A3"].isin(DASHBOARD_COUNTRIES)].merge(
        summary_df,
        left_on="ISO_A3",
        right_on="country",
        how="left",
    )
    vmax = float(np.nanmax(np.abs(df["cumulative_mean"]))) if df["cumulative_mean"].notna().any() else 1.0
    vmax = max(vmax, 1e-6)
    fig = px.choropleth(
        df,
        geojson=df.geometry,
        locations=df.index,
        color="cumulative_mean",
        color_continuous_scale="RdBu_r",
        range_color=(-vmax, vmax),
        hover_name="NAME",
        hover_data={
            "country": True,
            "cumulative_min": ":.2f",
            "cumulative_mean": ":.2f",
            "cumulative_max": ":.2f",
        },
        labels={"cumulative_mean": "Mean cumulative impact"},
        title=f"{response_var}: cumulative impact vs no ENSO baseline",
    )
    fig.update_traces(marker_line_color="#4D4D4D", marker_line_width=0.8)
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showcountries=True,
        countrycolor="#B8B8B8",
        showcoastlines=True,
        coastlinecolor="#B8B8B8",
    )
    fig.update_layout(
        height=520,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title="Mean cumulative impact"),
    )
    return fig

def _percent_rank(s):
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() <= 1:
        return pd.Series(np.nan, index=s.index)
    return x.rank(pct=True)


def _forward_rolling_mean(s, window):
    # Event-year view: score at t summarizes impact from t through t+window-1.
    return s.iloc[::-1].rolling(window=window, min_periods=1).mean().iloc[::-1]


def build_raw_macro_impact_yearly(panel, iso3, horizon_q=8, macro_vars=None):
    """
    Raw impact formula:
      RawImpact_q = mean_v percentile_rank_c(|Macro_v,q - Macro_v,q-1|)
      RawImpact_y = mean_{q in year y} forward_avg_horizon(RawImpact_q)
    """
    if macro_vars is None:
        macro_vars = MACRO_IMPACT_VARS
    vars_use = [v for v in macro_vars if v in panel.columns]
    if not vars_use or "country" not in panel or "quarter" not in panel:
        return pd.DataFrame()

    df = panel[panel["country"].astype(str) == iso3].copy()
    if df.empty:
        return pd.DataFrame()
    df["quarter"] = pd.to_datetime(df["quarter"], errors="coerce")
    df = df.dropna(subset=["quarter"]).sort_values("quarter").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    pct_cols = []
    for v in vars_use:
        d = pd.to_numeric(df[v], errors="coerce").diff().abs()
        col = f"{v}_raw_pct"
        df[col] = _percent_rank(d)
        pct_cols.append(col)

    df["raw_impact_score"] = df[pct_cols].mean(axis=1)
    df["raw_impact_score"] = _forward_rolling_mean(df["raw_impact_score"], int(horizon_q))
    df["year"] = df["quarter"].dt.year
    out = (
        df.groupby("year", as_index=False)["raw_impact_score"]
        .mean()
        .dropna(subset=["raw_impact_score"])
    )
    return out


def build_model_surprise_yearly(df_sc, horizon_q=8):
    """
    Model surprise formula:
      Surprise_q = percentile_rank_c(preferred model diagnostic)
      Surprise_y = mean_{q in year y} forward_avg_horizon(Surprise_q)
    Preferred diagnostic: composite_score, then score, then innovation_score.
    """
    if df_sc is None or df_sc.empty:
        return pd.DataFrame()
    score_col = next(
        (c for c in ["composite_score", "score", "innovation_score"] if c in df_sc.columns),
        None,
    )
    if score_col is None:
        return pd.DataFrame()

    df = df_sc.copy()
    if "quarter" in df.columns:
        df["quarter"] = pd.to_datetime(df["quarter"], errors="coerce")
        df = df.dropna(subset=["quarter"]).sort_values("quarter")
        df["year"] = df["quarter"].dt.year
    elif "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["year"]).sort_values("year")
    else:
        return pd.DataFrame()

    df["model_surprise_score"] = _percent_rank(df[score_col])
    if "quarter" in df.columns:
        df["model_surprise_score"] = _forward_rolling_mean(
            df["model_surprise_score"], int(horizon_q)
        )

    out = (
        df.groupby("year", as_index=False)["model_surprise_score"]
        .mean()
        .dropna(subset=["model_surprise_score"])
    )
    out["year"] = out["year"].astype(int)
    return out


def build_break_evidence_years(llm_overlay_df, iso3):
    if llm_overlay_df is None or llm_overlay_df.empty:
        return []
    if "iso3" not in llm_overlay_df or "year" not in llm_overlay_df:
        return []
    df = llm_overlay_df[llm_overlay_df["iso3"] == iso3].copy()
    if df.empty or "break_supported" not in df:
        return []
    supported = pd.to_numeric(df["break_supported"], errors="coerce").fillna(0) == 1
    years = pd.to_numeric(df.loc[supported, "year"], errors="coerce").dropna()
    return sorted(set(years.astype(int).tolist()))


def build_climate_related_break_years(llm_overlay_df, iso3):
    if llm_overlay_df is None or llm_overlay_df.empty:
        return []
    if "iso3" not in llm_overlay_df or "year" not in llm_overlay_df:
        return []
    df = llm_overlay_df[llm_overlay_df["iso3"] == iso3].copy()
    if df.empty or "break_supported" not in df:
        return []
    supported = pd.to_numeric(df["break_supported"], errors="coerce").fillna(0) == 1
    if "climate_related" in df.columns:
        climate = pd.to_numeric(df["climate_related"], errors="coerce").fillna(0) == 1
    else:
        climate = pd.Series(False, index=df.index)
    years = pd.to_numeric(df.loc[supported & climate, "year"], errors="coerce").dropna()
    return sorted(set(years.astype(int).tolist()))


def plot_impact_overlap(raw_yearly, surprise_yearly, info_years, climate_years, iso3):
    fig = go.Figure()

    raw_top = (
        raw_yearly.nlargest(5, "raw_impact_score")["year"].astype(int).tolist()
        if not raw_yearly.empty
        else []
    )
    surprise_top = (
        surprise_yearly.nlargest(5, "model_surprise_score")["year"].astype(int).tolist()
        if not surprise_yearly.empty
        else []
    )

    raw_lookup = (
        raw_yearly.set_index("year")["raw_impact_score"].to_dict()
        if not raw_yearly.empty
        else {}
    )
    surprise_lookup = (
        surprise_yearly.set_index("year")["model_surprise_score"].to_dict()
        if not surprise_yearly.empty
        else {}
    )
    info_set = set(int(y) for y in info_years)
    climate_set = set(int(y) for y in climate_years)
    raw_top_set = set(raw_top)
    surprise_top_set = set(surprise_top)

    if not raw_yearly.empty:
        raw_custom = []
        for y in raw_yearly["year"].astype(int).tolist():
            raw_custom.append(
                [
                    "Yes" if y in info_set else "No",
                    "Yes" if y in climate_set else "No",
                    "Yes" if y in raw_top_set else "No",
                    "Yes" if y in surprise_top_set else "No",
                    surprise_lookup.get(y, np.nan),
                ]
            )
        fig.add_trace(
            go.Scatter(
                x=raw_yearly["year"],
                y=raw_yearly["raw_impact_score"],
                mode="lines+markers",
                name="Raw impact",
                line=dict(color="#d95f02"),
                customdata=np.asarray(raw_custom, dtype=object),
                hovertemplate=(
                    "Country: " + iso3
                    + "<br>Year: %{x}"
                    + "<br>Raw impact score: %{y:.3f}"
                    + "<br>Model surprise score: %{customdata[4]:.3f}"
                    + "<br>Info break year: %{customdata[0]}"
                    + "<br>Climate-related break year: %{customdata[1]}"
                    + "<br>Top-5 raw impact: %{customdata[2]}"
                    + "<br>Top-5 model surprise: %{customdata[3]}"
                    + "<extra></extra>"
                ),
            )
        )

    if not surprise_yearly.empty:
        surprise_custom = []
        for y in surprise_yearly["year"].astype(int).tolist():
            surprise_custom.append(
                [
                    "Yes" if y in info_set else "No",
                    "Yes" if y in climate_set else "No",
                    "Yes" if y in raw_top_set else "No",
                    "Yes" if y in surprise_top_set else "No",
                    raw_lookup.get(y, np.nan),
                ]
            )
        fig.add_trace(
            go.Scatter(
                x=surprise_yearly["year"],
                y=surprise_yearly["model_surprise_score"],
                mode="lines+markers",
                name="Model surprise",
                line=dict(color="#1f77b4"),
                customdata=np.asarray(surprise_custom, dtype=object),
                hovertemplate=(
                    "Country: " + iso3
                    + "<br>Year: %{x}"
                    + "<br>Model surprise score: %{y:.3f}"
                    + "<br>Raw impact score: %{customdata[4]:.3f}"
                    + "<br>Info break year: %{customdata[0]}"
                    + "<br>Climate-related break year: %{customdata[1]}"
                    + "<br>Top-5 raw impact: %{customdata[2]}"
                    + "<br>Top-5 model surprise: %{customdata[3]}"
                    + "<extra></extra>"
                ),
            )
        )

    for yr in info_years:
        fig.add_vrect(
            x0=yr - 0.5,
            x1=yr + 0.5,
            fillcolor="gold",
            opacity=0.14,
            line_width=0,
            layer="below",
        )
    for yr in raw_top:
        fig.add_vrect(
            x0=yr - 0.35,
            x1=yr + 0.35,
            fillcolor="#d95f02",
            opacity=0.10,
            line_width=0,
            layer="below",
        )
    for yr in surprise_top:
        fig.add_vrect(
            x0=yr - 0.20,
            x1=yr + 0.20,
            fillcolor="#1f77b4",
            opacity=0.12,
            line_width=0,
            layer="below",
        )

    fig.update_layout(
        title=f"{iso3}: raw impact vs model surprise",
        xaxis_title="Year",
        yaxis_title="Percentile score, forward-window averaged",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=70, b=40),
    )
    return fig, raw_top, surprise_top


def build_enso_coeff_df_from_offline(country_pack):
    if not isinstance(country_pack, dict):
        return pd.DataFrame()
    coeff_quarters = pd.to_datetime(country_pack.get("coeff_quarters", []), errors="coerce")
    enso_coeff_series = country_pack.get("enso_coeff_series", [])
    if len(coeff_quarters) == 0 or not enso_coeff_series:
        return pd.DataFrame()

    out = pd.DataFrame({"quarter": coeff_quarters})
    for s in enso_coeff_series:
        vals = np.asarray(s.get("values", []), dtype=float)
        if len(vals) == len(coeff_quarters):
            out[s.get("label", "enso")] = vals
    if out.shape[1] <= 1:
        return pd.DataFrame()
    return out


def format_quarter_label(q):
    period = pd.Period(q, freq="Q") if not isinstance(q, pd.Period) else q
    return f"Q{period.quarter} {period.year}"


def build_enso_peak_event_study(
    panel_df,
    forecast_bundle,
    pipeline_pack,
    country,
    value_mode,
    selected_peak_labels=None,
):
    enso = (
        panel_df[["quarter", "ENSO"]]
        .dropna()
        .drop_duplicates("quarter")
        .assign(
            quarter=lambda d: pd.to_datetime(d["quarter"], errors="coerce"),
            ENSO=lambda d: pd.to_numeric(d["ENSO"], errors="coerce"),
        )
        .dropna(subset=["quarter"])
        .sort_values("quarter")
    )
    peaks = (
        enso[(enso["ENSO"] >= enso["ENSO"].shift(1)) & (enso["ENSO"] > enso["ENSO"].shift(-1))]
        .nlargest(5, "ENSO")
        .sort_values("quarter")
    )
    if selected_peak_labels:
        peak_labels = peaks["quarter"].dt.to_period("Q").map(format_quarter_label)
        peaks = peaks[peak_labels.isin(selected_peak_labels)]

    df = panel_df[panel_df["country"].astype(str) == country].copy()
    df["quarter"] = pd.to_datetime(df["quarter"], errors="coerce")
    vars_use = [v for v in MACRO_IMPACT_VARS if v in df.columns]

    if value_mode == "ENSO model contribution":
        scenarios = _forecast_scenarios(forecast_bundle)
        scenario_bundle = scenarios.get("mean") or next(iter(scenarios.values()), {})
        country_pack = scenario_bundle.get("per_country", {}).get(country, {})
        offline_country = pipeline_pack.get("offline_per_country", {}).get(country, {})
        enso_coeff_df = build_enso_coeff_df_from_offline(offline_country)
        if not country_pack or enso_coeff_df.empty or "ENSO" not in df.columns:
            return pd.DataFrame(), peaks

        enso_coeff_df = enso_coeff_df.rename(
            columns={f"{v}<-ENSO": f"{v}_enso_beta" for v in MACRO_IMPACT_VARS}
        )
        df = df.merge(enso_coeff_df, on="quarter", how="inner")
        vars_use = [v for v in vars_use if f"{v}_enso_beta" in df.columns]
        if not vars_use:
            return pd.DataFrame(), peaks

        enso = pd.to_numeric(df["ENSO"], errors="coerce")
        enso_z = (enso - enso.mean()) / (enso.std(ddof=0) + 1e-8)
        y_sd = dict(zip(country_pack.get("ENDO_use", []), np.asarray(country_pack.get("y_sd", []), dtype=float)))
        for v in vars_use:
            df[v] = (
                pd.to_numeric(df[f"{v}_enso_beta"], errors="coerce")
                * enso_z
                * float(y_sd.get(v, 1.0))
            )

    df = df.dropna(subset=["quarter"]).sort_values("quarter")
    if df.empty or not vars_use:
        return pd.DataFrame(), peaks

    df = df.set_index(df["quarter"].dt.to_period("Q"))
    rows = []
    for peak in peaks.itertuples(index=False):
        peak_period = peak.quarter.to_period("Q")
        if peak_period not in df.index:
            continue
        base = df.loc[peak_period, vars_use]
        if isinstance(base, pd.DataFrame):
            base = base.iloc[0]
        for rel_q in range(-4, 13):
            q = peak_period + rel_q
            if q not in df.index:
                continue
            row = df.loc[q, vars_use]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for v in vars_use:
                y = pd.to_numeric(row[v], errors="coerce")
                y0 = 0.0 if value_mode == "ENSO model contribution" else pd.to_numeric(base[v], errors="coerce")
                if pd.notna(y) and pd.notna(y0):
                    rows.append(
                        {
                            "variable": v,
                            "event_label": format_quarter_label(peak.quarter.to_period("Q")),
                            "peak_quarter": peak.quarter,
                            "enso_value": peak.ENSO,
                            "relative_quarter": rel_q,
                            "value": y - y0,
                        }
                    )

    return pd.DataFrame(rows), peaks


def plot_enso_peaks(panel_df, peak_df, selected_peak_labels=None):
    enso = (
        panel_df[["quarter", "ENSO"]]
        .dropna()
        .drop_duplicates("quarter")
        .assign(
            quarter=lambda d: pd.to_datetime(d["quarter"], errors="coerce"),
            ENSO=lambda d: pd.to_numeric(d["ENSO"], errors="coerce"),
        )
        .dropna(subset=["quarter", "ENSO"])
        .sort_values("quarter")
    )
    if enso.empty:
        return None

    peaks = peak_df.copy()
    if not peaks.empty:
        peaks["event_label"] = peaks["quarter"].dt.to_period("Q").map(format_quarter_label)
        selected = set(selected_peak_labels or peaks["event_label"].tolist())
        peaks["selected"] = peaks["event_label"].isin(selected)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=enso["quarter"],
            y=enso["ENSO"],
            customdata=enso["quarter"].dt.to_period("Q").astype(str),
            mode="lines",
            name="ENSO",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Quarter: %{customdata}<br>ENSO: %{y:.2f}<extra></extra>",
        )
    )
    if not peaks.empty:
        for is_selected, label, color, size in [
            (False, "Top ENSO peaks", "#9ca3af", 8),
            (True, "Selected ENSO peaks", "#d62728", 11),
        ]:
            pdf = peaks[peaks["selected"].eq(is_selected)]
            if pdf.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=pdf["quarter"],
                    y=pdf["ENSO"],
                    mode="markers+text",
                    name=label,
                    text=pdf["event_label"],
                    textposition="top center",
                    marker=dict(color=color, size=size, line=dict(color="white", width=1)),
                    hovertemplate="ENSO peak: %{text}<br>ENSO: %{y:.2f}<extra></extra>",
                )
            )

    fig.update_layout(
        title="ENSO index with top peaks labeled",
        xaxis_title="Quarter",
        yaxis_title="ENSO",
        height=380,
        margin=dict(l=30, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

# ----- STREAMLIT SETUP
st.set_page_config(
    page_title="Climate–Macro GVAR Explorer",
    layout="wide"
)
st.title("Climate-Macroeconomic Risk Explorer")

from trp.inputs import load_gvar_panel, load_stressor_probabilities, panel_csv_path

# ----- STREAMLIT CACHED LOADERS (UI-level caching only)
@st.cache_data
def load_panel(_panel_path: str):
    return load_gvar_panel(_panel_path)
@st.cache_data
def load_probabilities(_panel_path: str):
    return load_stressor_probabilities(panel=load_gvar_panel(_panel_path))

_panel_path = str(panel_csv_path())
panel = load_panel(_panel_path)
panel = panel[panel["country"].astype(str).isin(DASHBOARD_COUNTRIES)].copy()
prob_df = load_probabilities(_panel_path)
prob_df = prob_df[prob_df["country"].astype(str).isin(DASHBOARD_COUNTRIES)].copy()

country_options = _filter_dashboard_countries(panel["country"].dropna().astype(str).unique())
if not country_options:
    st.error("No configured dashboard countries are available in the panel.")
    st.stop()

control_cols = st.columns([1.2, 1.2, 2.6])
with control_cols[0]:
    country = st.selectbox(
        "Country",
        country_options,
        format_func=iso3_to_label,
        key="country_select",
        help=HELP_TEXT["country"],
    )
with control_cols[1]:
    response_var = st.selectbox(
        "Response",
        [v for v in MACRO_IMPACT_VARS if v in panel.columns],
        key="response_select",
        help=HELP_TEXT["response"],
    )
with control_cols[2]:
    st.caption("Dashboard uses the offline EM/KF pickle and precomputed Dash_Output charts; no online KF fitting is run.")

# ----- STREAMLIT TABS
tab_climate_risk, tab_scenario, tab_event_study, tab_structural_break = st.tabs(
    ["Climate Risk", "Scenario Impacts", "ENSO Event Study", "Structural Break"]
)

with tab_climate_risk:
    st.header("Climate Early-Warning Chain")

    st.markdown(
        """
        This panel links **ENSO conditions today** to the **probability of localized heat
        and moisture stress next quarter**. Macroeconomic impacts are evaluated separately
        in the offline scenario output.
        """
    )

    ew_cols = st.columns([1, 1, 2])
    with ew_cols[0]:
        enso_forecast = st.slider(
            "ENSO forecast for next quarter",
            min_value=-2.5,
            max_value=2.5,
            value=1.5,
            step=0.1,
            key="enso_forecast",
            help=HELP_TEXT["enso_forecast"],
        )
    with ew_cols[1]:
        heat_moist_pct = st.slider(
            "Heat and moisture stress threshold (percentile)",
            min_value=80,
            max_value=99,
            value=90,
            step=1,
            key="heat_threshold",
            help=HELP_TEXT["stress_threshold"],
        )

    prob_rows_df = build_climate_probability_rows(_panel_path, enso_forecast, heat_moist_pct)
    selected_prob = prob_rows_df[prob_rows_df["Country"] == country]
    p_heat = (
        float(selected_prob["Heat probability (%)"].iloc[0]) / 100
        if not selected_prob.empty and pd.notna(selected_prob["Heat probability (%)"].iloc[0])
        else np.nan
    )
    p_moist = (
        float(selected_prob["Moisture probability (%)"].iloc[0]) / 100
        if not selected_prob.empty and pd.notna(selected_prob["Moisture probability (%)"].iloc[0])
        else np.nan
    )

    st.subheader(f"Results for {country}")
    p_row = prob_df[prob_df["country"] == country]
    c1, c2, c3, c4 = st.columns(4)
    baseline = 1.0 - heat_moist_pct / 100
    with c1:
        st.metric(
            "Baseline probability of extreme heat next quarter",
            f"{baseline:.0%}",
            help=HELP_TEXT["baseline_probability"],
        )
    with c2:
        base_heat = pd.to_numeric(p_row.get("P_HEAT_NEXT_Q"), errors="coerce").iloc[0] if not p_row.empty else np.nan
        delta_heat = p_heat - base_heat if pd.notna(base_heat) and pd.notna(p_heat) else np.nan
        st.metric(
            "ENSO-conditioned probability of extreme heat next quarter",
            "—" if np.isnan(p_heat) else f"{p_heat:.0%}",
            **({ "delta": f"{delta_heat:+.0%}" } if pd.notna(delta_heat) else {}),
            help=HELP_TEXT["enso_probability"],
        )
    with c3:
        st.metric(
            "Probability of moisture stress next quarter",
            f"{baseline:.0%}",
            help=HELP_TEXT["baseline_probability"],
        )
    with c4:
        base_moist = pd.to_numeric(p_row.get("P_MOISTURE_NEXT_Q"), errors="coerce").iloc[0] if not p_row.empty else np.nan
        delta_moist = p_moist - base_moist if pd.notna(base_moist) and pd.notna(p_moist) else np.nan
        st.metric(
            "ENSO-conditioned probability of extreme moisture next quarter",
            "—" if np.isnan(p_moist) else f"{p_moist:.0%}",
            **({ "delta": f"{delta_moist:+.0%}" } if pd.notna(delta_moist) else {}),
            help=HELP_TEXT["enso_probability"],
        )

    st.caption(
        "Probabilities are estimated from historical ENSO → physical stress relationships "
        "and provide context for scenario selection. Macroeconomic impacts are evaluated "
        "separately via scenario analysis."
    )

    st.subheader("Early-warning comparison across countries")
    st.write("#### Country Risk Summary")
    risk_table = prob_rows_df.sort_values("Heat probability (%)", ascending=False)
    st.dataframe(
        risk_table,
        column_config={
            "Country": st.column_config.TextColumn("Country"),
            "Heat probability (%)": st.column_config.NumberColumn(
                "Heat Probability",
                format="%.1f%%",
                help=HELP_TEXT["risk_summary"],
            ),
            "Moisture probability (%)": st.column_config.NumberColumn(
                "Moisture Probability",
                format="%.1f%%",
                help=HELP_TEXT["risk_summary"],
            ),
        },
        hide_index=True,
        width="stretch",
    )

    world = gpd.read_file(_ROOT / "data" / "natural_earth" / "ne_110m_admin_0_countries.shp")
    world_modeled = world[world["ISO_A3"].isin(DASHBOARD_COUNTRIES)].merge(
        prob_rows_df, left_on="ISO_A3", right_on="Country", how="left"
    )

    def choropleth_map(plt_title, colorbar_title, map_color, map_lbl):
        st.subheader(plt_title)
        fig = px.choropleth(
            world,
            geojson=world.geometry,
            locations=world.index,
            color_discrete_sequence=["#FFFFFF"],
        )
        fig.update_traces(marker_line_color="#B8B8B8", marker_line_width=0.6, hoverinfo="skip")
        fig2 = px.choropleth(
            world_modeled,
            geojson=world_modeled.geometry,
            locations=world_modeled.index,
            color=map_color,
            color_continuous_scale="Blues",
            range_color=(0, 100),
            labels={map_color: map_lbl},
            hover_name="NAME",
        )
        fig2.update_traces(marker_line_color="#4D4D4D", marker_line_width=0.8)
        for trace in fig2.data:
            fig.add_trace(trace)
        fig.update_geos(
            fitbounds="locations",
            visible=False,
            showcountries=True,
            countrycolor="#B8B8B8",
            countrywidth=0.6,
            showcoastlines=True,
            coastlinecolor="#B8B8B8",
        )
        fig.update_layout(
            height=800,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                orientation="h",
                title=colorbar_title,
                x=0.5,
                xanchor="center",
                y=0.05,
                yanchor="top",
                len=0.6,
                thickness=15,
            ),
        )
        st.plotly_chart(fig, width="stretch", config={"scrollZoom": False})

    choropleth_map(
        "Probability of extreme heat next quarter (conditional on ENSO)",
        "Probability of extreme heat",
        "Heat probability (%)",
        "Probability of extreme heat",
    )
    choropleth_map(
        "Probability of extreme moisture next quarter (conditional on ENSO)",
        "Probability of extreme moisture",
        "Moisture probability (%)",
        "Probability of extreme moisture",
    )


with tab_scenario:
    st.header("Scenario Impacts")

    forecast_pack = load_forecast_bundle()
    if forecast_pack["path"]:
        st.caption(f"Loaded forecast pickle: `{forecast_pack['path']}`")
    else:
        st.warning(
            "No forecast pickle found. Run `analysis/Dash_Output/gvar_kf_forecast.py` first; "
            "the dashboard will then draw directly from `Dash_Input/gvar_forecast_results.pkl`."
        )

    scenario_countries = st.multiselect(
        "Countries",
        options=country_options,
        default=[country],
        format_func=iso3_to_label,
        key="scenario_countries",
        help=HELP_TEXT["scenario_countries"],
    )

    if not forecast_pack["bundle"]:
        st.subheader("Core forecast")
        enso_img = SCENARIO_OUTPUT_ROOT / "enso_series" / "enso_history_forecast.png"
        if enso_img.exists():
            st.image(str(enso_img), caption="ENSO history and forecast", use_container_width=True)
        else:
            st.info(f"Missing ENSO chart: `{enso_img}`")
    else:
        core_df = build_forecast_plot_df(
            forecast_pack["bundle"],
            panel,
            scenario_countries,
            response_var,
        )
        q_summary, c_summary = summarize_forecast_ranges(core_df)

        st.subheader("Core forecast")
        enso_img = SCENARIO_OUTPUT_ROOT / "enso_series" / "enso_history_forecast.png"
        if enso_img.exists():
            st.image(str(enso_img), caption="ENSO history and forecast", use_container_width=True)
        else:
            st.info(f"Missing ENSO chart: `{enso_img}`")

        fig_core = plot_core_forecast(core_df, response_var)
        if fig_core is None:
            st.info("Forecast pickle does not contain data for the selected countries/response.")
        else:
            st.plotly_chart(fig_core, width="stretch")

        if not q_summary.empty:
            st.subheader("Forecast impact ranges vs no ENSO baseline")
            scenario_start = core_df.loc[
                core_df["period_type"].eq("Scenario forecast"), "quarter"
            ].min()
            scenario_start_label = (
                pd.Period(scenario_start, freq="Q")
                if pd.notna(scenario_start)
                else "the scenario period"
            )
            st.caption(
                f"ENSO and commodity values already observed in the panel are used before {scenario_start_label}. "
                "Rows marked `Gap fill / nowcast` fill missing near-term country outcomes before the "
                "scenario period; cumulative impacts and maps are based on `Scenario forecast` rows."
            )
            for c in scenario_countries:
                c_quarters = q_summary[q_summary["country"] == c].copy()
                c_cum = c_summary[c_summary["country"] == c].copy()
                if c_quarters.empty:
                    continue
                st.markdown(f"**{c}**")
                if not c_cum.empty:
                    r = c_cum.iloc[0]
                    st.metric(
                        f"{response_var} cumulative change vs no ENSO impact",
                        f"{r['cumulative_mean']:+.2f} p.p.",
                        delta=f"{r['cumulative_min']:+.2f} to {r['cumulative_max']:+.2f} p.p.",
                        help="Cumulative forecast difference between the ENSO scenario and a no-ENSO-impact baseline, shown in percentage points.",
                    )
                show_tbl = c_quarters.copy()
                show_tbl["quarter"] = show_tbl["quarter"].dt.to_period("Q").astype(str)
                show_tbl = show_tbl.rename(
                    columns={
                        "quarter": "Quarter",
                        "period_type": "Period type",
                        "value_min": f"{response_var} min",
                        "value_mean": f"{response_var} mean",
                        "value_max": f"{response_var} max",
                        "impact_min": "Impact min vs no ENSO",
                        "impact_mean": "Impact mean vs no ENSO",
                        "impact_max": "Impact max vs no ENSO",
                    }
                )
                st.dataframe(
                    show_tbl[
                        [
                            "Quarter",
                            "Period type",
                            f"{response_var} min",
                            f"{response_var} mean",
                            f"{response_var} max",
                            "Impact min vs no ENSO",
                            "Impact mean vs no ENSO",
                            "Impact max vs no ENSO",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                )

            st.subheader("Cumulative impact maps")
            map_cols = st.columns(2)
            for i, metric in enumerate([v for v in MACRO_IMPACT_VARS if v in panel.columns]):
                metric_df = build_forecast_plot_df(
                    forecast_pack["bundle"],
                    panel,
                    scenario_countries,
                    metric,
                )
                _, metric_summary = summarize_forecast_ranges(metric_df)
                fig_map = plot_metric_impact_map(metric_summary, metric)
                with map_cols[i % 2]:
                    if fig_map is not None:
                        st.plotly_chart(fig_map, width="stretch")

        st.subheader("QR experiment: adaptive vs baseline")
        qr_country = st.selectbox(
            "QR country",
            options=scenario_countries or country_options,
            format_func=iso3_to_label,
            key="qr_country",
            help=HELP_TEXT["qr_country"],
        )
        qr_cols = st.columns(2)
        qr_paths = [
            (
                "Adaptive KF track",
                SCENARIO_OUTPUT_ROOT
                / "qr_experiment"
                / "kf_track"
                / "adaptive_vs_baseline"
                / f"kf_track_qr_{qr_country}.png",
            ),
            (
                "Adaptive climate coefficients",
                SCENARIO_OUTPUT_ROOT
                / "qr_experiment"
                / "climate_coeff"
                / "adaptive_vs_baseline"
                / f"climate_coeff_qr_{qr_country}.png",
            ),
        ]
        for col, (caption, path) in zip(qr_cols, qr_paths):
            with col:
                if path.exists():
                    st.image(str(path), caption=caption, use_container_width=True)
                else:
                    st.info(f"Missing QR chart: `{path}`")
    if not forecast_pack["bundle"]:
        st.subheader("QR experiment: adaptive vs baseline")
        qr_country = st.selectbox(
            "QR country",
            options=scenario_countries or country_options,
            format_func=iso3_to_label,
            key="qr_country_no_forecast",
            help=HELP_TEXT["qr_country"],
        )
        qr_cols = st.columns(2)
        qr_paths = [
            (
                "Adaptive KF track",
                SCENARIO_OUTPUT_ROOT
                / "qr_experiment"
                / "kf_track"
                / "adaptive_vs_baseline"
                / f"kf_track_qr_{qr_country}.png",
            ),
            (
                "Adaptive climate coefficients",
                SCENARIO_OUTPUT_ROOT
                / "qr_experiment"
                / "climate_coeff"
                / "adaptive_vs_baseline"
                / f"climate_coeff_qr_{qr_country}.png",
            ),
        ]
        for col, (caption, path) in zip(qr_cols, qr_paths):
            with col:
                if path.exists():
                    st.image(str(path), caption=caption, use_container_width=True)
                else:
                    st.info(f"Missing QR chart: `{path}`")


with tab_event_study:
    st.header("ENSO Peak Event Study")
    st.caption(
        "Raw data mode shows y(t+k) minus the value in the selected ENSO peak quarter, "
        "with the peak aligned at t=0. Model contribution mode is a separate estimated "
        "ENSO component and should not be read as the raw event-study difference."
    )

    event_cols = st.columns([1, 1, 3])
    with event_cols[0]:
        event_country = st.selectbox(
            "Country",
            country_options,
            index=country_options.index(country),
            format_func=iso3_to_label,
            key="event_country",
        )
    with event_cols[1]:
        event_mode = st.radio(
            "Series",
            ["Raw data", "ENSO model contribution"],
            horizontal=True,
            key="event_mode",
        )

    event_forecast_pack = load_forecast_bundle()
    event_pipeline_pack = load_pipeline_break_scores()
    _, peak_df = build_enso_peak_event_study(
        panel,
        event_forecast_pack["bundle"],
        event_pipeline_pack,
        event_country,
        event_mode,
    )

    peak_table = peak_df.copy()
    peak_table["quarter"] = peak_table["quarter"].dt.to_period("Q").map(format_quarter_label)
    peak_table = peak_table.rename(columns={"quarter": "ENSO peak quarter", "ENSO": "ENSO value"})
    st.dataframe(
        peak_table[["ENSO peak quarter", "ENSO value"]],
        hide_index=True,
        width="stretch",
    )
    peak_options = peak_table["ENSO peak quarter"].tolist()
    selected_peaks = st.multiselect(
        "Events to plot",
        options=peak_options,
        default=peak_options,
        key="event_peaks",
    )
    fig_enso_peaks = plot_enso_peaks(panel, peak_df, selected_peaks)
    if fig_enso_peaks is not None:
        st.plotly_chart(fig_enso_peaks, width="stretch")

    event_df, _ = build_enso_peak_event_study(
        panel,
        event_forecast_pack["bundle"],
        event_pipeline_pack,
        event_country,
        event_mode,
        selected_peak_labels=selected_peaks,
    )

    if not selected_peaks:
        st.info("Select at least one ENSO peak to plot.")
    elif event_df.empty:
        st.info("No event-study data available for the selected country and series.")
    else:
        fig_event = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[v.replace("_", " ") for v in MACRO_IMPACT_VARS],
            horizontal_spacing=0.10,
            vertical_spacing=0.16,
        )
        colors = px.colors.qualitative.Plotly
        events = event_df[["event_label", "peak_quarter"]].drop_duplicates().sort_values("peak_quarter")
        color_map = {event.event_label: colors[i % len(colors)] for i, event in enumerate(events.itertuples())}

        for i, var in enumerate(MACRO_IMPACT_VARS):
            r = i // 2 + 1
            c = i % 2 + 1
            vdf = event_df[event_df["variable"] == var]
            for event_label, edf in vdf.groupby("event_label"):
                fig_event.add_trace(
                    go.Scatter(
                        x=edf["relative_quarter"],
                        y=edf["value"],
                        mode="lines+markers",
                        name=event_label,
                        legendgroup=event_label,
                        showlegend=(i == 0),
                        line=dict(color=color_map[event_label]),
                        marker=dict(color=color_map[event_label]),
                        hovertemplate=(
                            "ENSO peak: %{fullData.name}"
                            "<br>Relative quarter: %{x}"
                            "<br>Difference: %{y:.2f}"
                            "<extra></extra>"
                        ),
                    ),
                    row=r,
                    col=c,
                )
            fig_event.add_hline(y=0, line_dash="dot", line_color="#999", row=r, col=c)
            fig_event.add_vline(x=0, line_dash="dash", line_color="#444", row=r, col=c)

        fig_event.update_layout(
            title=f"{event_country}: top-5 ENSO peak windows ({event_mode})",
            height=720,
            margin=dict(l=30, r=20, t=80, b=80),
            legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0),
        )
        fig_event.update_xaxes(title_text="Quarters from ENSO peak")
        y_title = (
            "Difference from value during ENSO peak"
            if event_mode == "Raw data"
            else "Estimated ENSO model contribution"
        )
        fig_event.update_yaxes(title_text=y_title)
        st.plotly_chart(fig_event, width="stretch")


with tab_structural_break:
    st.header("Structural Break")
    st.markdown(
        "A set of analyses generated from Kalman filter / EM outputs, "
        "with optional LLM (Gemini) structural-break overlays."
    )

    wb_top4_df = load_wb_top4()
    gemini_df = load_gemini_outputs()
    pipeline_pack = load_pipeline_break_scores()
    break_score_df = pipeline_pack["break_score_df"]
    composite_break_df = pipeline_pack["composite_break_df"]
    score_year_df = pipeline_pack["score_year_df"]
    llm_from_pickle_df = pipeline_pack["llm_df"]
    offline_per_country = pipeline_pack["offline_per_country"]
    offline_refit_per_country = pipeline_pack["offline_refit_per_country"]
    llm_overlay_df = llm_from_pickle_df if not llm_from_pickle_df.empty else gemini_df
    if not llm_overlay_df.empty:
        llm_overlay_df = llm_overlay_df.copy()
        if "country" in llm_overlay_df.columns:
            llm_overlay_df["country"] = llm_overlay_df["country"].astype(str).str.strip()
            llm_overlay_df["iso3"] = llm_overlay_df["country"].map(country_to_iso3)
        if "break_year" in llm_overlay_df.columns:
            llm_overlay_df["year"] = pd.to_numeric(llm_overlay_df["break_year"], errors="coerce")
        if "break_supported" not in llm_overlay_df.columns and "raw_output" in llm_overlay_df.columns:
            llm_overlay_df["break_supported"] = pd.to_numeric(
                llm_overlay_df["raw_output"].map(lambda x: _extract_field(x, "break_supported")),
                errors="coerce",
            )
        if "confidence" not in llm_overlay_df.columns and "raw_output" in llm_overlay_df.columns:
            llm_overlay_df["confidence"] = pd.to_numeric(
                llm_overlay_df["raw_output"].map(lambda x: _extract_field(x, "confidence")),
                errors="coerce",
            )
        if "break_type" not in llm_overlay_df.columns and "raw_output" in llm_overlay_df.columns:
            llm_overlay_df["break_type"] = llm_overlay_df["raw_output"].map(
                lambda x: _extract_field(x, "break_type")
            )
        if "summary" not in llm_overlay_df.columns and "raw_output" in llm_overlay_df.columns:
            llm_overlay_df["summary"] = llm_overlay_df["raw_output"].map(lambda x: _extract_field(x, "summary"))
        if "llm_joint_score" not in llm_overlay_df.columns:
            llm_overlay_df["llm_joint_score"] = (
                pd.to_numeric(llm_overlay_df.get("break_supported"), errors="coerce").fillna(0)
                * pd.to_numeric(llm_overlay_df.get("confidence"), errors="coerce").fillna(0)
            )

        # Sentinel cleanup for UI / overlays: -99 -> 0; status "error" -> None
        if "break_supported" in llm_overlay_df.columns:
            _bs = pd.to_numeric(llm_overlay_df["break_supported"], errors="coerce")
            llm_overlay_df["break_supported"] = np.where(_bs == -99, 0.0, _bs)
        if "status" in llm_overlay_df.columns:
            _st = llm_overlay_df["status"].astype(str).str.strip().str.lower()
            llm_overlay_df["status"] = llm_overlay_df["status"].mask(_st == "error", None)
        if "confidence" in llm_overlay_df.columns:
            llm_overlay_df["llm_joint_score"] = (
                pd.to_numeric(llm_overlay_df["break_supported"], errors="coerce").fillna(0)
                * pd.to_numeric(llm_overlay_df["confidence"], errors="coerce").fillna(0)
            )

    if pipeline_pack["path"]:
        st.caption(f"Loaded pipeline bundle: `{pipeline_pack['path']}`")
    else:
        st.info(
            "No pipeline pickle found. Structural-break score panel will use Gemini-derived yearly scores. "
            "If you generate `gvar_pipeline_results.pkl`, richer quarterly scores will be shown automatically."
        )

    available_iso = set()
    if "country" in break_score_df:
        available_iso |= set(break_score_df["country"].dropna().astype(str))
    if "country" in composite_break_df:
        available_iso |= set(composite_break_df["country"].dropna().astype(str))
    if "iso3" in gemini_df:
        available_iso |= set(gemini_df["iso3"].dropna().astype(str))
    if "iso3" in llm_overlay_df:
        available_iso |= set(llm_overlay_df["iso3"].dropna().astype(str))
    available_iso |= set(offline_per_country.keys())
    if "country" in wb_top4_df:
        available_iso |= set(wb_top4_df["country"].map(country_to_iso3).dropna().astype(str))

    available_iso = [x for x in DASHBOARD_COUNTRIES if x in available_iso]
    default_sel = [country] if country in available_iso else available_iso[:3]

    sb_countries = st.multiselect(
        "Select countries (multi-select)",
        options=available_iso,
        default=default_sel,
        format_func=iso3_to_label,
        key="sb_countries",
        help=HELP_TEXT["sb_countries"],
    )
    use_llm_overlay = st.checkbox(
        "Overlay Gemini identified break years",
        value=True,
        key="sb_use_llm_overlay",
        help=HELP_TEXT["llm_overlay"],
    )

    st.subheader("1) Structural break scores")

    def _country_break_scores(iso3):
        if not composite_break_df.empty and "country" in composite_break_df:
            cdf = composite_break_df[composite_break_df["country"] == iso3].copy()
            if not cdf.empty:
                if "quarter" in cdf:
                    cdf["quarter"] = pd.to_datetime(cdf["quarter"], errors="coerce")
                return cdf, "quarterly"
        if not break_score_df.empty and "country" in break_score_df:
            bdf = break_score_df[break_score_df["country"] == iso3].copy()
            if not bdf.empty:
                if "quarter" in bdf:
                    bdf["quarter"] = pd.to_datetime(bdf["quarter"], errors="coerce")
                return bdf, "quarterly"
        gdf = gemini_df[gemini_df["iso3"] == iso3].copy()
        if gdf.empty:
            return pd.DataFrame(), "none"
        gdf = gdf.sort_values("year")
        gdf["llm_break_supported_score"] = gdf["break_supported"]
        gdf["llm_confidence_score"] = gdf["confidence"]
        gdf["llm_joint_score"] = gdf["llm_joint_score"]
        return gdf, "yearly"

    for iso3 in sb_countries:
        df_sc, freq_mode = _country_break_scores(iso3)
        st.markdown(f"**{iso3_to_label(iso3)}**")
        if df_sc.empty:
            st.warning("No structural-break score data available.")
            continue

        if freq_mode == "quarterly":
            score_cols = [
                c
                for c in [
                    "innovation_score",
                    "coefficient_change",
                    "filter_smoother_gap",
                    "score",
                    "composite_score",
                ]
                if c in df_sc.columns
            ]
            x_col = "quarter" if "quarter" in df_sc.columns else None
        else:
            score_cols = [
                c
                for c in [
                    "llm_break_supported_score",
                    "llm_confidence_score",
                    "llm_joint_score",
                ]
                if c in df_sc.columns
            ]
            x_col = "year"

        if not score_cols or x_col is None:
            st.warning("Score columns unavailable for plotting.")
            continue

        score_pick = st.multiselect(
            f"Score series ({iso3})",
            options=score_cols,
            default=score_cols[:3],
            key=f"sb_score_pick_{iso3}",
            help=HELP_TEXT["score_series"],
        )
        if not score_pick:
            st.info("Select at least one score series.")
            continue

        plot_df = df_sc[[x_col] + score_pick].copy()
        if x_col == "quarter":
            plot_df = plot_df.dropna(subset=[x_col]).sort_values(x_col)
        else:
            plot_df = plot_df.dropna(subset=[x_col]).sort_values(x_col)

        fig_sb = px.line(
            plot_df.melt(id_vars=[x_col], value_vars=score_pick, var_name="series", value_name="value"),
            x=x_col,
            y="value",
            color="series",
            markers=True,
            title=f"{iso3}: structural break score series",
        )

        if use_llm_overlay and not llm_overlay_df.empty:
            ov = llm_overlay_df[
                (llm_overlay_df["iso3"] == iso3)
                & (pd.to_numeric(llm_overlay_df["break_supported"], errors="coerce") == 1)
                & (llm_overlay_df["year"].notna())
            ]
            for yr in sorted(set(ov["year"].astype(int))):
                if x_col == "quarter":
                    fig_sb.add_vrect(
                        x0=pd.Timestamp(year=int(yr), month=1, day=1),
                        x1=pd.Timestamp(year=int(yr), month=12, day=31),
                        fillcolor="gold",
                        opacity=0.15,
                        line_width=0,
                    )
                else:
                    fig_sb.add_vline(x=int(yr), line_dash="dot", line_color="goldenrod")

        st.plotly_chart(fig_sb, width="stretch")

    st.subheader("1.1) Core EM coefficient trajectories from pickle")
    if not offline_per_country:
        st.info("No `offline_plot_data.base.per_country` in pipeline bundle.")
    else:
        for iso3 in sb_countries:
            d = offline_per_country.get(iso3)
            if not isinstance(d, dict):
                continue
            st.markdown(f"**{iso3_to_label(iso3)}**")
            coeff_quarters = pd.to_datetime(d.get("coeff_quarters", []), errors="coerce")
            diag_quarters = pd.to_datetime(d.get("diag_quarters", []), errors="coerce")
            diag_coeff_series = d.get("diag_coeff_series", [])
            enso_coeff_series = d.get("enso_coeff_series", [])

            if len(coeff_quarters) > 0 and diag_coeff_series:
                own_df = pd.DataFrame({"quarter": coeff_quarters})
                for s in diag_coeff_series:
                    vals = np.asarray(s.get("values", []), dtype=float)
                    if len(vals) == len(coeff_quarters):
                        own_df[s.get("label", "diag")] = vals
                if own_df.shape[1] > 1:
                    fig_own = px.line(
                        own_df.melt(id_vars=["quarter"], var_name="series", value_name="value"),
                        x="quarter",
                        y="value",
                        color="series",
                        title=f"{iso3}: Diagonal own-variable lag coefficients",
                    )
                    st.plotly_chart(fig_own, width="stretch")

            if len(coeff_quarters) > 0 and enso_coeff_series:
                enso_df = pd.DataFrame({"quarter": coeff_quarters})
                for s in enso_coeff_series:
                    vals = np.asarray(s.get("values", []), dtype=float)
                    if len(vals) == len(coeff_quarters):
                        enso_df[s.get("label", "enso")] = vals
                if enso_df.shape[1] > 1:
                    fig_enso = px.line(
                        enso_df.melt(id_vars=["quarter"], var_name="series", value_name="value"),
                        x="quarter",
                        y="value",
                        color="series",
                        title=f"{iso3}: ENSO coefficients",
                    )
                    st.plotly_chart(fig_enso, width="stretch")


    st.subheader("2) World Bank document information")
    if wb_top4_df.empty:
        st.warning("`structural_break/wb_top4.csv` not found or empty.")
    else:
        wb_use = wb_top4_df.copy()
        wb_use["iso3"] = wb_use["country"].map(country_to_iso3)
        wb_use = wb_use[wb_use["iso3"].isin(sb_countries)] if sb_countries else wb_use.iloc[0:0]
        if wb_use.empty:
            st.info("No WB Top4 records for selected countries.")
        else:
            for iso3 in sb_countries:
                cdf = wb_use[wb_use["iso3"] == iso3].copy()
                if cdf.empty:
                    continue
                st.markdown(f"**{iso3_to_label(iso3)}**")
                years = sorted([int(y) for y in cdf["year"].dropna().unique()])
                if not years:
                    st.caption("No valid years in WB records.")
                    continue
                y_pick = st.selectbox(
                    f"Year ({iso3})",
                    options=years,
                    key=f"wb_year_{iso3}",
                    help=HELP_TEXT["wb_year"],
                )
                ydf = (
                    cdf[cdf["year"] == y_pick]
                    .sort_values("positive_score", ascending=False)
                    .head(4)
                )
                doc_tabs = st.tabs([f"Doc {i}" for i in range(1, len(ydf) + 1)])
                for i, (_, row) in enumerate(ydf.iterrows()):
                    with doc_tabs[i]:
                        positive_hits_val = row.get("positive_hits", None)
                        if pd.isna(positive_hits_val):
                            positive_hits_val = "None"
                        st.markdown(f"**Country**: {row.get('country', '')}")
                        st.markdown(f"**Year**: {int(row.get('year')) if pd.notna(row.get('year')) else ''}")
                        st.markdown(f"**Title**: {row.get('display_title', '')}")
                        st.markdown(f"**Positive score**: {row.get('positive_score', '')}")
                        st.markdown(f"**Positive hits**: {positive_hits_val}")
                        st.markdown("**Abstract**")
                        st.write(str(row.get("abstract_text", "")))

    st.subheader("3) Gemini outputs")
    if llm_overlay_df.empty:
        st.warning("No Gemini / LLM output table in pickle or under `structural_break/gemini output/`.")
    else:
        for iso3 in sb_countries:
            gdf = llm_overlay_df[llm_overlay_df["iso3"] == iso3].copy().sort_values("year")
            if gdf.empty:
                continue
            if "break_type" in gdf.columns:
                gdf["break_type"] = gdf["break_type"].replace(
                    to_replace=[-99, "-99", "-99.0"],
                    value="--",
                )
            st.markdown(f"**{iso3_to_label(iso3)}**")
            show_cols = [
                c
                for c in [
                    "country",
                    "year",
                    "status",
                    "break_supported",
                    "confidence",
                    "break_type",
                    "summary",
                ]
                if c in gdf.columns
            ]
            st.dataframe(
                gdf[show_cols],
                hide_index=True,
                width="stretch",
            )

    st.subheader("3.1) Raw impact vs model surprise overlap")
    st.caption(
        "Raw impact uses percentile-ranked macro changes. Model surprise uses the existing "
        "pickle break diagnostics, preferring composite score and falling back to innovation score. "
        "Both are forward-averaged over at most two years."
    )
    st.markdown(
        """
        <span style="display:inline-block;width:10px;height:10px;background:#FFD700;opacity:0.65;border:1px solid #bbb;margin-right:6px;"></span>
        Information break year&nbsp;&nbsp;&nbsp;
        <span style="display:inline-block;width:10px;height:10px;background:#d95f02;opacity:0.55;border:1px solid #bbb;margin-right:6px;"></span>
        Top-5 raw impact year&nbsp;&nbsp;&nbsp;
        <span style="display:inline-block;width:10px;height:10px;background:#1f77b4;opacity:0.55;border:1px solid #bbb;margin-right:6px;"></span>
        Top-5 model surprise year
        """,
        unsafe_allow_html=True,
    )
    impact_horizon_q = st.slider(
        "Forward averaging window (quarters)",
        min_value=1,
        max_value=8,
        value=8,
        step=1,
        key="sb_impact_horizon_q",
        help=HELP_TEXT["impact_window"],
    )

    for iso3 in sb_countries:
        df_sc, _ = _country_break_scores(iso3)
        raw_yearly = build_raw_macro_impact_yearly(
            panel=panel,
            iso3=iso3,
            horizon_q=impact_horizon_q,
            macro_vars=MACRO_IMPACT_VARS,
        )
        surprise_yearly = build_model_surprise_yearly(
            df_sc=df_sc,
            horizon_q=impact_horizon_q,
        )
        info_years = build_break_evidence_years(llm_overlay_df, iso3)
        climate_years = build_climate_related_break_years(llm_overlay_df, iso3)

        st.markdown(f"**{iso3_to_label(iso3)}**")
        if raw_yearly.empty and surprise_yearly.empty:
            st.info("No raw-impact or model-surprise yearly data available.")
            continue

        fig_overlap, raw_top, surprise_top = plot_impact_overlap(
            raw_yearly=raw_yearly,
            surprise_yearly=surprise_yearly,
            info_years=info_years,
            climate_years=climate_years,
            iso3=iso3,
        )
        st.plotly_chart(fig_overlap, width="stretch")

        years_to_show = sorted(set(raw_top) | set(surprise_top) | set(info_years))
        if years_to_show:
            summary_df = pd.DataFrame({"year": years_to_show})
            summary_df["top5_raw_impact"] = summary_df["year"].isin(raw_top)
            summary_df["top5_model_surprise"] = summary_df["year"].isin(surprise_top)
            summary_df["information_break"] = summary_df["year"].isin(info_years)
            if not raw_yearly.empty:
                summary_df["raw_impact_score"] = summary_df["year"].map(
                    raw_yearly.set_index("year")["raw_impact_score"]
                )
            if not surprise_yearly.empty:
                summary_df["model_surprise_score"] = summary_df["year"].map(
                    surprise_yearly.set_index("year")["model_surprise_score"]
                )
            st.dataframe(summary_df, hide_index=True, width="stretch")

    # st.subheader("3.2) ENSO coefficients after drop-year refit")
    # if not offline_refit_per_country:
    #     st.info(
    #         "No drop-year refit results found in the pipeline pickle. "
    #         "This comparison appears when `offline_plot_data.refit.per_country` is available."
    #     )
    # else:
    #     for iso3 in sb_countries:
    #         df_sc, _ = _country_break_scores(iso3)
    #         raw_yearly = build_raw_macro_impact_yearly(
    #             panel=panel,
    #             iso3=iso3,
    #             horizon_q=impact_horizon_q,
    #             macro_vars=MACRO_IMPACT_VARS,
    #         )
    #         surprise_yearly = build_model_surprise_yearly(
    #             df_sc=df_sc,
    #             horizon_q=impact_horizon_q,
    #         )
    #         info_years = build_break_evidence_years(llm_overlay_df, iso3)
    #         raw_top = (
    #             raw_yearly.nlargest(5, "raw_impact_score")["year"].astype(int).tolist()
    #             if not raw_yearly.empty
    #             else []
    #         )
    #         surprise_top = (
    #             surprise_yearly.nlargest(5, "model_surprise_score")["year"].astype(int).tolist()
    #             if not surprise_yearly.empty
    #             else []
    #         )
    #         drop_years = sorted(set(info_years) & (set(raw_top) | set(surprise_top)))

    #         base_enso = build_enso_coeff_df_from_offline(offline_per_country.get(iso3))
    #         refit_enso = build_enso_coeff_df_from_offline(offline_refit_per_country.get(iso3))
    #         if base_enso.empty and refit_enso.empty:
    #             continue

    #         st.markdown(f"**{iso3_to_label(iso3)}**")
    #         st.caption(
    #             "Drop-year rule: information break year AND top-5 in raw impact or model surprise. "
    #             f"Selected years: {', '.join(map(str, drop_years)) if drop_years else 'None'}."
    #         )
    #         if not base_enso.empty:
    #             fig_base_enso = px.line(
    #                 base_enso.melt(id_vars=["quarter"], var_name="series", value_name="value"),
    #                 x="quarter",
    #                 y="value",
    #                 color="series",
    #                 title=f"{iso3}: Original ENSO coefficients",
    #             )
    #             st.plotly_chart(fig_base_enso, width="stretch")
    #         if not refit_enso.empty:
    #             fig_refit_enso = px.line(
    #                 refit_enso.melt(id_vars=["quarter"], var_name="series", value_name="value"),
    #                 x="quarter",
    #                 y="value",
    #                 color="series",
    #                 title=f"{iso3}: Drop-year refit ENSO coefficients",
    #             )
    #             st.plotly_chart(fig_refit_enso, width="stretch")

    st.subheader("4) Map by year (pre-generated HTML)")
    st.caption(
        "Color guide: Blue = Structural break; Green = Potential climate-related structural break."
    )
    map_files = sorted(PREGENERATED_MAP_DIR.glob("map_*.html"))
    year_to_file = {}
    for f in map_files:
        m = re.match(r"map_(\d{4})\.html$", f.name)
        if m:
            year_to_file[int(m.group(1))] = f
    map_year_options = sorted(year_to_file.keys())

    if not map_year_options:
        st.warning(f"No pre-generated map files found in `{PREGENERATED_MAP_DIR}`.")
    else:
        st.caption("Using pre-generated map HTML files (no country filtering).")
        map_year = st.selectbox(
            "Map year",
            options=map_year_options,
            key="sb_map_year",
            help=HELP_TEXT["map_year"],
        )
        map_path = year_to_file.get(map_year)
        if map_path is None or not map_path.exists():
            st.warning(f"Map file not found for year {map_year}.")
        else:
            try:
                html_text = map_path.read_text(encoding="utf-8")
                components.html(html_text, height=720, scrolling=True)
            except Exception as e:
                st.error(f"Failed to load map HTML: {e}")
