import sys
from pathlib import Path
import html
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
FORECAST_HISTORY_START = pd.Timestamp("2024-07-01")
FORECAST_COEFF_METHOD_OPTIONS = {
    "last": "Last quarter",
    "avg4": "4-quarter average",
    "avg8": "8-quarter average",
}

HELP_TEXT = {
    "country": "Primary country used as the default selection across dashboard tabs.",
    "response": "Macroeconomic response variable used for scenario charts and impact summaries.",
    "enso_forecast": "Forecast ENSO index value for the next quarter. Positive values indicate El Nino-like conditions; negative values indicate La Nina-like conditions.",
    "stress_threshold": "Percentile cutoff used to define an extreme physical stress event. Higher values focus on rarer, more severe heat or moisture outcomes.",
    "baseline_probability": "Unconditional probability implied by the selected stress threshold before applying ENSO information.",
    "enso_probability": "Estimated probability after conditioning on the selected ENSO forecast. The delta compares this value with the stored baseline probability.",
    "risk_summary": "Country-level probabilities generated from the selected ENSO forecast and stress threshold.",
    "scenario_countries": "Countries included in the scenario forecast charts, summary tables, and cumulative impact maps.",
    "sb_countries": "Countries shown in the structural-break score, document-evidence, Gemini output, and overlap panels.",
    "llm_overlay": "Adds highlighted years where the Gemini/LLM evidence flags a supported structural break.",
    "score_series": "Score diagnostics to plot. Innovation captures forecast surprise; coefficient change captures parameter movement; composite combines available signals.",
    "wb_year": "World Bank document year used to display the top supporting document records.",
    "impact_window": "Number of future quarters used when computing observed-impact and model-surprise scores. Larger values emphasize longer-lasting changes and reduce sensitivity to short-term fluctuations.",
    "map_year": "Pre-generated structural-break map year to display.",
}

GUIDE_SECTIONS = [
    (
        "Climate Early-Warning Chain",
        "This tab links projected ENSO conditions to near-term physical climate risks. Historical "
        "relationships between ENSO and local heat or moisture extremes are used to estimate the "
        "probability that each country will experience climate stress during the next quarter.",
        [
            "Forecast ENSO conditions for the next quarter",
            "Probability of extreme heat stress",
            "Probability of extreme moisture stress",
            "Country-level risk comparisons",
            "Heat and moisture risk maps",
        ],
        "These probabilities provide an early-warning indicator of potential climate stress. They "
        "do not directly estimate economic impacts. Instead, they help identify countries and "
        "scenarios that may warrant further investigation in the Scenario Impacts tab.",
        "Which countries face elevated climate risk under the current ENSO outlook?",
    ),
    (
        "Scenario Impacts",
        "This tab estimates the potential macroeconomic consequences of future ENSO conditions. "
        "Forecasts are generated using the climate-macroeconomic model and are compared against "
        "a counterfactual scenario in which future ENSO effects are absent.",
        [
            "Historical and projected ENSO conditions",
            "Country forecasts under alternative ENSO scenarios",
            "GDP growth, inflation, exchange-rate, and export projections",
            "Cumulative impacts relative to a no-ENSO baseline",
            "Geographic maps of projected impacts",
            "Adaptive-policy experiments",
        ],
        "Impact estimates represent differences between the selected ENSO scenario and a no-ENSO "
        "reference case. Positive or negative values indicate the estimated contribution of ENSO "
        "to future economic outcomes.",
        "How much could future ENSO conditions affect economic performance in each country?",
    ),
    (
        "ENSO Peak Event Study",
        "This tab examines historical macroeconomic responses around major ENSO events. Multiple "
        "ENSO peaks are aligned in time so that users can compare economic trajectories before "
        "and after past climate shocks.",
        [
            "Historical ENSO peaks",
            "Event-aligned GDP, inflation, exchange-rate, and export responses",
            "Comparisons across multiple ENSO episodes",
            "Optional model-estimated ENSO contributions",
        ],
        "The event study provides historical context rather than forecasts. It helps users understand "
        "how countries responded during previous ENSO episodes and whether current projections are "
        "consistent with historical experience.",
        "What happened during past major ENSO events?",
    ),
    (
        "Structural Break Analysis",
        "This tab evaluates whether climate-economy relationships have changed over time. Structural "
        "breaks may arise from policy reforms, economic transitions, technological change, trade "
        "shifts, financial crises, or other major events.",
        [
            "Time-varying Kalman filter coefficients",
            "Structural-break scores",
            "Estimated ENSO sensitivities through time",
            "Supporting World Bank documents",
            "Gemini-generated summaries of potential break drivers",
        ],
        "Changes in model coefficients may indicate that historical climate responses are no longer "
        "stable. Structural-break information can help users identify periods when climate-economic "
        "relationships strengthened, weakened, or changed direction.",
        "Can historical climate-economy relationships be assumed to remain valid today?",
    ),
]


def render_tab_description(section_index):
    _, purpose, shows, interpretation, question = GUIDE_SECTIONS[section_index]
    with st.expander("Description", expanded=False):
        st.markdown(f"**Purpose**  \n{purpose}")
        st.markdown("**What this tab shows**")
        for item in shows:
            st.markdown(f"- {item}")
        st.markdown(f"**Interpretation**  \n{interpretation}")
        st.markdown(f"**Key question**  \n{question}")


def inject_global_control_styles():
    st.markdown(
        """
        <style>
        [data-testid="stLayoutWrapper"]:has(.st-key-analysis_scope_panel) {
            position: sticky;
            top: 3.5rem;
            z-index: 999;
        }
        .st-key-analysis_scope_panel {
            background-color: #eef5fc;
            border-left: 5px solid #2c6fb7;
            border-radius: 8px;
            padding: 18px 22px 16px 22px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }
        .st-key-analysis_scope_panel h3 {
            margin: 0 0 10px 0;
            color: #173f73;
            font-size: 1.35rem;
            font-weight: 800;
        }
        .st-key-analysis_scope_panel p {
            margin: 0 0 14px 0;
            color: #34506f;
            font-size: 1.08rem;
            line-height: 1.45;
        }
        .st-key-analysis_scope_panel [data-testid="stSelectbox"] label p {
            color: #27466d;
            font-size: 1.05rem;
            font-weight: 650;
            margin-bottom: 6px;
        }
        .st-key-analysis_scope_panel [data-baseweb="select"] > div {
            background-color: rgba(26, 82, 140, 0.92) !important;
            border-color: rgba(26, 82, 140, 0.92) !important;
            min-height: 54px;
        }
        .st-key-analysis_scope_panel [data-baseweb="select"] div,
        .st-key-analysis_scope_panel [data-baseweb="select"] span,
        .st-key-analysis_scope_panel [data-baseweb="select"] input {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-size: 1.12rem !important;
            font-weight: 700;
        }
        .st-key-analysis_scope_panel [data-baseweb="select"] svg {
            fill: #ffffff !important;
            color: #ffffff !important;
        }
        .current-selection-callout {
            background-color: #f7fbff;
            border-left: 4px solid #2c6fb7;
            border-radius: 6px;
            padding: 9px 12px;
            margin: 4px 0 16px 0;
            color: #1f2d3d;
            font-size: 0.95rem;
        }
        .current-selection-callout strong {
            color: #173f73;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_current_selection(country, response_var):
    st.markdown(
        f"""
        <div class="current-selection-callout">
            <strong>Current Selection:</strong> {html.escape(iso3_to_label(country))} | {html.escape(response_var)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_help_button(container, text, key):
    safe_text = html.escape(str(text), quote=True)
    container.markdown(
        f"""
        <span title="{safe_text}" style="
            display:inline-flex;
            align-items:center;
            justify-content:center;
            width:18px;
            height:18px;
            border:1.5px solid #8A8F98;
            border-radius:50%;
            color:#6B7280;
            font-size:12px;
            font-weight:700;
            line-height:18px;
            cursor:help;
            margin-top:2px;
        ">?</span>
        """,
        unsafe_allow_html=True,
    )


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
    return ISO_TO_NAME.get(iso3, iso3)


def styled_header(text, bg_color="rgba(26, 82, 140, 0.92)", font_size="1.2rem", padding="12px 12px"):
    """
    bg_color defaults to a Streamlit-compatible blue at 92% opacity.
    font_size defaults to 1.2rem (standard subheader size).
    """
    safe_text = html.escape(str(text))
    st.markdown(
        f"""
        <div style="background-color: {bg_color}; padding: {padding}; border-radius: 0px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0; font-size: {font_size}; font-family: sans-serif; font-weight: 600;">
                {safe_text}
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def st_title(text, bg_color="rgba(26, 82, 140, 0.92)", font_size="2.4rem", padding="12px 12px"):
    styled_header(text, bg_color, font_size, padding)


def st_header(text, bg_color="rgba(26, 82, 140, 0.92)", font_size="1.8rem", padding="10px 12px"):
    styled_header(text, bg_color, font_size, padding)


def st_subheader(text, bg_color="rgba(26, 82, 140, 0.92)", font_size="1.5rem", padding="8px 12px"):
    styled_header(text, bg_color, font_size, padding)


def default_option_index(options, preferred, fallback=0):
    return options.index(preferred) if preferred in options else fallback


def response_table_label(response_var):
    return {"GDP_YoY": "GDP Growth"}.get(response_var, response_var)


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


def forecast_pickle_state():
    return tuple(
        (str(p), p.stat().st_mtime_ns, p.stat().st_size)
        for p in FORECAST_PICKLE_CANDIDATES
        if p.exists()
    )


@st.cache_data
def load_forecast_bundle(pickle_state):
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


CLIMATE_TOGGLE_OPTIONS = {
    "ENSO": "ENSO",
    "IOD": "IOD",
    "HeatDry": "Crop-weighted heat/dryness index",
    "OIL_YoY": "Oil price YoY",
}
_CLIMATE_TOGGLE_ORDER = {v: i for i, v in enumerate(CLIMATE_TOGGLE_OPTIONS)}

# Match CLIMATE_VARIANT_COMBOS in gvar_kf_forecast.py.
CLIMATE_VARIANT_CHOICES = {
    "ENSO": {"label": "ENSO only", "vars": ["ENSO"]},
    "ENSO+OIL_YoY": {
        "label": f"ENSO + {CLIMATE_TOGGLE_OPTIONS['OIL_YoY']}",
        "vars": ["ENSO", "OIL_YoY"],
    },
    "ENSO+HeatDry": {
        "label": f"ENSO + {CLIMATE_TOGGLE_OPTIONS['HeatDry']}",
        "vars": ["ENSO", "HeatDry"],
    },
    "ENSO+HeatDry+OIL_YoY": {
        "label": f"ENSO + {CLIMATE_TOGGLE_OPTIONS['HeatDry']} + {CLIMATE_TOGGLE_OPTIONS['OIL_YoY']}",
        "vars": ["ENSO", "HeatDry", "OIL_YoY"],
    },
    "HeatDry+OIL_YoY": {
        "label": f"{CLIMATE_TOGGLE_OPTIONS['OIL_YoY']} + {CLIMATE_TOGGLE_OPTIONS['HeatDry']}",
        "vars": ["OIL_YoY", "HeatDry"],
    },
    "IOD": {"label": "IOD only", "vars": ["IOD"]},
    "IOD+OIL_YoY": {
        "label": f"IOD + {CLIMATE_TOGGLE_OPTIONS['OIL_YoY']}",
        "vars": ["IOD", "OIL_YoY"],
    },
    "IOD+HeatDry": {
        "label": f"IOD + {CLIMATE_TOGGLE_OPTIONS['HeatDry']}",
        "vars": ["IOD", "HeatDry"],
    },
    "IOD+HeatDry+OIL_YoY": {
        "label": f"IOD + {CLIMATE_TOGGLE_OPTIONS['HeatDry']} + {CLIMATE_TOGGLE_OPTIONS['OIL_YoY']}",
        "vars": ["IOD", "HeatDry", "OIL_YoY"],
    },
}


def _climate_variant_key(climate_vars_selected):
    return "+".join(sorted(climate_vars_selected, key=lambda v: _CLIMATE_TOGGLE_ORDER.get(v, 99)))


def _select_climate_variant_scenarios(bundle, climate_vars_selected):
    """Pick the precomputed scenarios dict for the selected ENSO/heat/moisture
    combination (see analysis/Dash_Output/gvar_kf_forecast.py:
    run_forecast_all_climate_variants). Falls back to the default ENSO-only
    scenarios if the pickle predates the climate-variant toggle or the exact
    combination wasn't precomputed."""
    key = _climate_variant_key(climate_vars_selected)
    variants = bundle.get("climate_variants") if isinstance(bundle, dict) else None
    if variants and key in variants:
        return variants[key].get("scenarios", {})
    return _forecast_scenarios(bundle)


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


def _hex_to_rgba(hex_color, alpha):
    color = str(hex_color).lstrip("#")
    if len(color) != 6:
        return f"rgba(31, 119, 180, {alpha})"
    r, g, b = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _forecast_country_frame(
    scenarios,
    panel_df,
    country,
    response_var,
    history_start=FORECAST_HISTORY_START,
    include_observed_overlap=False,
    coeff_method="last",
):
    frames = []
    hist = panel_df[
        (panel_df["country"].astype(str) == country)
        & panel_df[response_var].notna()
    ].copy()
    if not hist.empty:
        hist["quarter"] = pd.to_datetime(hist["quarter"], errors="coerce")
        hist = hist.dropna(subset=["quarter"]).sort_values("quarter")
        if history_start is not None:
            hist = hist.loc[lambda x: x["quarter"] >= history_start]
        frames.append(
            pd.DataFrame(
                {
                    "country": country,
                    "scenario": "actual",
                    "quarter": hist["quarter"].to_numpy(),
                    "period_type": "Actual history",
                    "value": pd.to_numeric(hist[response_var], errors="coerce").to_numpy(),
                    "no_enso_value": np.nan,
                    "impact_vs_no_enso": np.nan,
                    "no_iod_value": np.nan,
                    "impact_vs_no_iod": np.nan,
                    "no_heat_value": np.nan,
                    "impact_vs_no_heat": np.nan,
                    "heat0_var": None,
                    "actual_value": np.nan,
                    "coefficient_method": np.nan,
                    "coefficient_method_available": np.nan,
                }
            )
        )

    # Lookup of real observed values by quarter, used only to annotate forecast/nowcast
    # rows for comparison (see include_observed_overlap below); does not affect what
    # counts as "Actual history" above.
    actual_lookup = {}
    if response_var in panel_df.columns:
        actual_rows = panel_df[panel_df["country"].astype(str) == country][
            ["quarter", response_var]
        ].copy()
        actual_rows["quarter"] = pd.to_datetime(actual_rows["quarter"], errors="coerce")
        actual_rows = actual_rows.dropna(subset=["quarter"])
        actual_lookup = dict(
            zip(
                actual_rows["quarter"].dt.to_period("Q").dt.to_timestamp(),
                pd.to_numeric(actual_rows[response_var], errors="coerce"),
            )
        )

    for scenario_name, scenario_bundle in scenarios.items():
        d = scenario_bundle.get("per_country", {}).get(country)
        if not d or response_var not in d.get("ENDO_use", []):
            continue
        method_pack = d.get("forecast_methods", {}).get(coeff_method)
        method_available = coeff_method == "last" or isinstance(method_pack, dict)
        if coeff_method != "last":
            if not method_available:
                continue
            d = {**d, **method_pack}
        idx = list(d["ENDO_use"]).index(response_var)
        q = pd.to_datetime(d["fc_quarters"])
        if not hist.empty:
            last_actual_q = pd.Timestamp(hist["quarter"].max()).to_period("Q").to_timestamp()
            cutoff = last_actual_q
            if include_observed_overlap and d.get("hist_last_quarter") is not None:
                # Model-internal cutoff (all domestic vars jointly complete) is never
                # later than last_actual_q, so this only ever pulls the cutoff earlier,
                # exposing forecast/nowcast quarters that already have real data for
                # response_var specifically.
                model_hist_q = pd.Timestamp(d["hist_last_quarter"]).to_period("Q").to_timestamp()
                cutoff = min(cutoff, model_hist_q)
            keep = q.to_period("Q").to_timestamp() > cutoff
        else:
            keep = np.ones(len(q), dtype=bool)
        if not np.any(keep):
            continue
        period_type = d.get("period_type")
        if not period_type or len(period_type) != len(q):
            period_type = np.array(["Scenario forecast"] * len(q), dtype=object)
        else:
            period_type = np.asarray(period_type, dtype=object)
        q = q[keep]
        period_type = period_type[keep]
        y = _to_raw_y(d, d["y_hat"], panel_df, country)[:, idx][keep]
        y_lower = None
        y_upper = None
        if d.get("y_lower") is not None and d.get("y_upper") is not None:
            y_lower = _to_raw_y(d, d["y_lower"], panel_df, country)[:, idx][keep]
            y_upper = _to_raw_y(d, d["y_upper"], panel_df, country)[:, idx][keep]
        base = None
        if d.get("y_hat_enso0") is not None:
            base = _to_raw_y(d, d["y_hat_enso0"], panel_df, country)[:, idx][keep]
        iod_base = None
        if d.get("y_hat_iod0") is not None:
            iod_base = _to_raw_y(d, d["y_hat_iod0"], panel_df, country)[:, idx][keep]
        heat_base = None
        if d.get("y_hat_heat0") is not None:
            heat_base = _to_raw_y(d, d["y_hat_heat0"], panel_df, country)[:, idx][keep]
        heat0_var = d.get("heat0_var")
        actual_value = np.array(
            [actual_lookup.get(pd.Timestamp(x).to_period("Q").to_timestamp(), np.nan) for x in q],
            dtype=float,
        )
        frames.append(
            pd.DataFrame(
                {
                    "country": country,
                    "scenario": scenario_name,
                    "quarter": q,
                    "period_type": period_type,
                    "value": y,
                    "lower": y_lower if y_lower is not None else np.nan,
                    "upper": y_upper if y_upper is not None else np.nan,
                    "no_enso_value": base if base is not None else np.nan,
                    "impact_vs_no_enso": y - base if base is not None else np.nan,
                    "no_iod_value": iod_base if iod_base is not None else np.nan,
                    "impact_vs_no_iod": y - iod_base if iod_base is not None else np.nan,
                    "no_heat_value": heat_base if heat_base is not None else np.nan,
                    "impact_vs_no_heat": y - heat_base if heat_base is not None else np.nan,
                    "heat0_var": heat0_var,
                    "actual_value": actual_value,
                    "coefficient_method": FORECAST_COEFF_METHOD_OPTIONS.get(coeff_method, "Last quarter"),
                    "coefficient_method_available": bool(method_available),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_forecast_plot_df(
    forecast_bundle,
    panel_df,
    countries,
    response_var,
    history_start=FORECAST_HISTORY_START,
    include_observed_overlap=False,
    coeff_method="last",
):
    scenarios = _forecast_scenarios(forecast_bundle)
    frames = [
        _forecast_country_frame(
            scenarios,
            panel_df,
            c,
            response_var,
            history_start=history_start,
            include_observed_overlap=include_observed_overlap,
            coeff_method=coeff_method,
        )
        for c in countries
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_forecast_ranges(plot_df):
    if plot_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    plot_df = plot_df[~plot_df["period_type"].eq("Actual history")].copy()
    if plot_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    q_summary = (
        plot_df.groupby(["country", "quarter", "period_type"], as_index=False)
        .agg(
            impact_min=("impact_vs_no_enso", "min"),
            impact_mean=("impact_vs_no_enso", "mean"),
            impact_max=("impact_vs_no_enso", "max"),
            impact_iod_min=("impact_vs_no_iod", "min"),
            impact_iod_mean=("impact_vs_no_iod", "mean"),
            impact_iod_max=("impact_vs_no_iod", "max"),
            impact_heat_min=("impact_vs_no_heat", "min"),
            impact_heat_mean=("impact_vs_no_heat", "mean"),
            impact_heat_max=("impact_vs_no_heat", "max"),
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
        scenario_df.groupby(
            ["country", "scenario"],
            as_index=False,
        )[["impact_vs_no_enso", "impact_vs_no_iod", "impact_vs_no_heat"]]
        .sum(min_count=1)
        .groupby("country", as_index=False)
        .agg(
            cumulative_min=("impact_vs_no_enso", "min"),
            cumulative_mean=("impact_vs_no_enso", "mean"),
            cumulative_max=("impact_vs_no_enso", "max"),
            cumulative_iod_min=("impact_vs_no_iod", "min"),
            cumulative_iod_mean=("impact_vs_no_iod", "mean"),
            cumulative_iod_max=("impact_vs_no_iod", "max"),
            cumulative_heat_min=("impact_vs_no_heat", "min"),
            cumulative_heat_mean=("impact_vs_no_heat", "mean"),
            cumulative_heat_max=("impact_vs_no_heat", "max"),
        )
    )
    return q_summary, c_summary


def plot_core_forecast(plot_df, response_var):
    if plot_df.empty:
        return None
    fig = go.Figure()
    country_order = list(dict.fromkeys(plot_df["country"].astype(str)))
    palette = px.colors.qualitative.Plotly
    color_map = {c: palette[i % len(palette)] for i, c in enumerate(country_order)}

    for c, cdf in plot_df.groupby("country", sort=False):
        color = color_map.get(str(c), "#1f77b4")
        hist = cdf[cdf["period_type"].eq("Actual history")].sort_values("quarter")
        fc = cdf[~cdf["period_type"].eq("Actual history")]

        if not hist.empty:
            fig.add_trace(
                go.Scatter(
                    x=hist["quarter"],
                    y=hist["value"],
                    mode="lines+markers",
                    name=f"{iso3_to_label(c)} actual",
                    line=dict(color=color, width=2),
                    marker=dict(color=color),
                    hovertemplate=f"{c} actual<br>%{{x|%Y-Q%q}}<br>{response_var}: %{{y:.2f}}<extra></extra>",
                )
            )

        piv = fc.pivot_table(index="quarter", columns="scenario", values="value", aggfunc="mean").sort_index()
        if not piv.empty:
            lower = piv.min(axis=1)
            upper = piv.max(axis=1)
            mean = piv["mean"] if "mean" in piv else piv.mean(axis=1)
            lower_piv = fc.pivot_table(index="quarter", columns="scenario", values="lower", aggfunc="mean").sort_index()
            upper_piv = fc.pivot_table(index="quarter", columns="scenario", values="upper", aggfunc="mean").sort_index()
            if not lower_piv.empty and lower_piv.notna().any().any():
                lower = lower_piv.min(axis=1).reindex(mean.index)
            if not upper_piv.empty and upper_piv.notna().any().any():
                upper = upper_piv.max(axis=1).reindex(mean.index)
            if not hist.empty:
                last_hist = hist.iloc[-1]
                first_fc_q = mean.index[0]
                if pd.notna(last_hist["quarter"]) and first_fc_q > last_hist["quarter"]:
                    fig.add_trace(
                        go.Scatter(
                            x=[last_hist["quarter"], first_fc_q],
                            y=[last_hist["value"], mean.iloc[0]],
                            mode="lines",
                            line=dict(color=color, width=1.5),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
            fig.add_trace(
                go.Scatter(
                    x=list(piv.index) + list(piv.index[::-1]),
                    y=list(upper) + list(lower[::-1]),
                    fill="toself",
                    fillcolor=_hex_to_rgba(color, 0.12),
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
                    name=f"{iso3_to_label(c)} forecast",
                    line=dict(color=color, width=2),
                    marker=dict(color=color),
                    hovertemplate=f"{iso3_to_label(c)} forecast<br>%{{x|%Y-Q%q}}<br>{response_var}: %{{y:.2f}}<extra></extra>",
                )
            )
    fig.update_layout(
        title=f"{response_var}: mean forecast with one SD band for ENSO and modeling uncertainties",
        xaxis_title="Quarter",
        yaxis_title=response_var,
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=80, b=110),
        height=560,
    )
    return fig


def plot_selected_country_forecast(plot_df, response_var, country_label, counterfactual="enso", heat_label=None):
    """Draw one forecast path against the selected climate-driver counterfactual."""
    if plot_df.empty:
        return None
    if counterfactual == "heat":
        cf_col = "no_heat_value"
        driver_label = heat_label or "climate driver"
        cf_name = f"No-{driver_label} counterfactual"
        cf_hover_note = f"Future {driver_label} held at its baseline"
        main_line_name = "Forecasted scenario"
        title = f"{country_label}: {response_var} under forecasted scenario vs no-{driver_label} counterfactual"
    elif counterfactual == "iod":
        cf_col = "no_iod_value"
        cf_name = "No-IOD counterfactual (IOD index = 0)"
        cf_hover_note = "Future IOD index set to 0"
        main_line_name = "Forecasted IOD scenario"
        title = f"{country_label}: {response_var} under forecasted IOD vs no-IOD counterfactual"
    else:
        cf_col = "no_enso_value"
        cf_name = "No-ENSO counterfactual (ENSO index = 0)"
        cf_hover_note = "Future ENSO index set to 0"
        main_line_name = "Forecasted ENSO scenario"
        title = f"{country_label}: {response_var} under forecasted ENSO vs no-ENSO counterfactual"

    fig = go.Figure()
    color = px.colors.qualitative.Plotly[0]
    hist = plot_df[plot_df["period_type"].eq("Actual history")].sort_values("quarter")
    fc = plot_df[~plot_df["period_type"].eq("Actual history")]

    if not hist.empty:
        fig.add_trace(
            go.Scatter(
                x=hist["quarter"],
                y=hist["value"],
                mode="lines",
                name="Actual history",
                line=dict(color=color, width=2),
                hovertemplate=f"Actual<br>%{{x|%Y-Q%q}}<br>{response_var}: %{{y:.2f}}<extra></extra>",
            )
        )

    piv = fc.pivot_table(index="quarter", columns="scenario", values="value", aggfunc="mean").sort_index()
    if not piv.empty:
        mean = piv["mean"] if "mean" in piv else piv.mean(axis=1)
        lower = piv.min(axis=1)
        upper = piv.max(axis=1)
        lower_piv = fc.pivot_table(index="quarter", columns="scenario", values="lower", aggfunc="mean").sort_index()
        upper_piv = fc.pivot_table(index="quarter", columns="scenario", values="upper", aggfunc="mean").sort_index()
        if not lower_piv.empty and lower_piv.notna().any().any():
            lower = lower_piv.min(axis=1).reindex(mean.index)
        if not upper_piv.empty and upper_piv.notna().any().any():
            upper = upper_piv.max(axis=1).reindex(mean.index)

        if not hist.empty and mean.index[0] > hist["quarter"].iloc[-1]:
            fig.add_trace(
                go.Scatter(
                    x=[hist["quarter"].iloc[-1], mean.index[0]],
                    y=[hist["value"].iloc[-1], mean.iloc[0]],
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=list(mean.index) + list(mean.index[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill="toself",
                fillcolor=_hex_to_rgba(color, 0.12),
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=mean.index,
                y=mean,
                mode="lines+markers",
                name=main_line_name,
                line=dict(color=color, width=2),
                marker=dict(color=color),
                hovertemplate=f"Forecast<br>%{{x|%Y-Q%q}}<br>{response_var}: %{{y:.2f}}<extra></extra>",
            )
        )

        no_cf = (
            fc.pivot_table(index="quarter", columns="scenario", values=cf_col, aggfunc="mean")
            .sort_index()
            .mean(axis=1)
            .reindex(mean.index)
        )
        if no_cf.notna().any():
            if not hist.empty and no_cf.index[0] > hist["quarter"].iloc[-1]:
                fig.add_trace(
                    go.Scatter(
                        x=[hist["quarter"].iloc[-1], no_cf.index[0]],
                        y=[hist["value"].iloc[-1], no_cf.iloc[0]],
                        mode="lines",
                        line=dict(color="#111111", width=1.5, dash="dash"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=no_cf.index,
                    y=no_cf,
                    mode="lines+markers",
                    name=cf_name,
                    line=dict(color="#111111", width=2, dash="dash"),
                    marker=dict(color="#111111"),
                    hovertemplate=(
                        f"{cf_name}"
                        f"<br>{cf_hover_note}"
                        f"<br>%{{x|%Y-Q%q}}<br>{response_var}: %{{y:.2f}}<extra></extra>"
                    ),
                )
            )

        if "actual_value" in fc.columns:
            actual = (
                fc.pivot_table(index="quarter", columns="scenario", values="actual_value", aggfunc="mean")
                .sort_index()
                .mean(axis=1)
                .dropna()
            )
            if not actual.empty:
                fig.add_trace(
                    go.Scatter(
                        x=actual.index,
                        y=actual,
                        mode="markers",
                        name="Actual observed (comparison)",
                        marker=dict(color="#2ca02c", size=11, symbol="diamond", line=dict(color="white", width=1)),
                        hovertemplate=(
                            "Actual observed"
                            f"<br>%{{x|%Y-Q%q}}<br>{response_var}: %{{y:.2f}}<extra></extra>"
                        ),
                    )
                )

    fig.update_layout(
        title=title,
        xaxis_title="Quarter",
        yaxis_title=response_var,
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=80, b=110),
        height=560,
    )
    return fig


def plot_enso_forecast_online(forecast_bundle, panel_df, plot_start=pd.Timestamp("2014-01-01")):
    scenarios = _forecast_scenarios(forecast_bundle)
    if not scenarios:
        return None

    hist = panel_df[["quarter", "ENSO"]].copy()
    hist["quarter"] = pd.to_datetime(hist["quarter"], errors="coerce").dt.to_period("Q").dt.to_timestamp()
    hist["ENSO"] = pd.to_numeric(hist["ENSO"], errors="coerce")
    hist = (
        hist.dropna(subset=["quarter", "ENSO"])
        .groupby("quarter", as_index=False)["ENSO"]
        .first()
        .sort_values("quarter")
    )
    hist = hist[hist["quarter"] >= plot_start].copy()

    scen_frames = []
    for scenario_name, scenario_bundle in scenarios.items():
        exo = scenario_bundle.get("exo_forecast")
        if not isinstance(exo, pd.DataFrame) or "target_quarter" not in exo or "ENSO" not in exo:
            continue
        tmp = exo[["target_quarter", "ENSO", "ENSO_source"]].copy()
        tmp["quarter"] = pd.to_datetime(tmp["target_quarter"], errors="coerce").dt.to_period("Q").dt.to_timestamp()
        tmp["ENSO"] = pd.to_numeric(tmp["ENSO"], errors="coerce")
        tmp["scenario"] = scenario_name
        scen_frames.append(tmp.dropna(subset=["quarter", "ENSO"]))
    if not scen_frames:
        return None

    exo_df = pd.concat(scen_frames, ignore_index=True)
    fc = exo_df[exo_df["ENSO_source"].astype(str).eq("forecast")].copy()
    if fc.empty:
        fc = exo_df.copy()
    piv = fc.pivot_table(index="quarter", columns="scenario", values="ENSO", aggfunc="mean").sort_index()
    if piv.empty:
        return None

    mean = piv["mean"] if "mean" in piv else piv.mean(axis=1)
    lower = piv.min(axis=1)
    upper = piv.max(axis=1)

    fig = go.Figure()
    if not hist.empty:
        fig.add_trace(
            go.Scatter(
                x=hist["quarter"],
                y=hist["ENSO"],
                mode="lines",
                name="Historical ENSO",
                line=dict(color="#1f77b4", width=2),
                hovertemplate="Historical ENSO<br>%{x|%Y-Q%q}<br>%{y:.2f}<extra></extra>",
            )
        )
        if mean.index[0] > hist["quarter"].iloc[-1]:
            fig.add_trace(
                go.Scatter(
                    x=[hist["quarter"].iloc[-1], mean.index[0]],
                    y=[hist["ENSO"].iloc[-1], mean.iloc[0]],
                    mode="lines",
                    line=dict(color="#888888", width=1.5, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.add_trace(
        go.Scatter(
            x=list(mean.index) + list(mean.index[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself",
            fillcolor="rgba(255, 127, 14, 0.22)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Low-high scenario range",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=mean.index,
            y=mean,
            mode="lines+markers",
            name="Mean forecast",
            line=dict(color="#ff7f0e", width=2.5),
            marker=dict(color="#ff7f0e"),
            hovertemplate="Mean forecast<br>%{x|%Y-Q%q}<br>ENSO: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_vline(x=mean.index[0], line_dash="dash", line_color="#888888", opacity=0.7)
    fig.update_layout(
        title="ENSO history and forecast (low-high scenario range)",
        xaxis_title="Quarter",
        yaxis_title="ENSO",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=70, b=100),
        height=420,
    )
    return fig


def plot_metric_impact_map(
    summary_df,
    response_var,
    countries=None,
    *,
    mean_col="cumulative_mean",
    min_col="cumulative_min",
    max_col="cumulative_max",
    driver_label="ENSO",
):
    if summary_df.empty:
        return None
    required_cols = [min_col, mean_col, max_col]
    if any(col not in summary_df.columns for col in required_cols):
        return None
    if not pd.to_numeric(summary_df[mean_col], errors="coerce").notna().any():
        return None
    countries = list(countries or summary_df["country"].dropna().astype(str).unique())
    if not countries:
        return None
    summary_df = summary_df.copy()
    summary_df["country_name"] = summary_df["country"].map(iso3_to_label)
    summary_df[min_col] = pd.to_numeric(summary_df[min_col], errors="coerce")
    summary_df[mean_col] = pd.to_numeric(summary_df[mean_col], errors="coerce")
    summary_df[max_col] = pd.to_numeric(summary_df[max_col], errors="coerce")
    world = gpd.read_file(_ROOT / "data" / "natural_earth" / "ne_110m_admin_0_countries.shp")
    df = world[world["ISO_A3"].isin(countries)].merge(
        summary_df,
        left_on="ISO_A3",
        right_on="country",
        how="left",
    )
    vmax = float(np.nanmax(np.abs(df[mean_col]))) if df[mean_col].notna().any() else 1.0
    vmax = max(vmax, 1e-6)
    fig = px.choropleth(
        df,
        geojson=df.geometry,
        locations=df.index,
        color=mean_col,
        color_continuous_scale="RdBu_r",
        range_color=(-vmax, vmax),
        hover_name="NAME",
        hover_data={
            "country": False,
            "country_name": True,
            min_col: ":.2f",
            mean_col: ":.2f",
            max_col: ":.2f",
        },
        labels={
            "country_name": "Country",
            min_col: "Min cumulative difference",
            mean_col: "Mean cumulative difference",
            max_col: "Max cumulative difference",
        },
        title=f"{response_var}: Cumulative impact relative to a no-{driver_label} baseline",
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
        coloraxis_colorbar=dict(title="Mean cumulative difference"),
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
                name="Observed impact",
                line=dict(color="#d95f02"),
                customdata=np.asarray(raw_custom, dtype=object),
                hovertemplate=(
                    "Country: " + iso3
                    + "<br>Year: %{x}"
                    + "<br>Observed-impact score: %{y:.3f}"
                    + "<br>Model-surprise score: %{customdata[4]:.3f}"
                    + "<br>Documented break year: %{customdata[0]}"
                    + "<br>Climate-related break year: %{customdata[1]}"
                    + "<br>Top-5 observed-impact year: %{customdata[2]}"
                    + "<br>Top-5 model-surprise year: %{customdata[3]}"
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
                    + "<br>Model-surprise score: %{y:.3f}"
                    + "<br>Observed-impact score: %{customdata[4]:.3f}"
                    + "<br>Documented break year: %{customdata[0]}"
                    + "<br>Climate-related break year: %{customdata[1]}"
                    + "<br>Top-5 observed-impact year: %{customdata[2]}"
                    + "<br>Top-5 model-surprise year: %{customdata[3]}"
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
        title=f"{iso3_to_label(iso3)}: Observed impact, model surprise, and candidate structural-break years",
        xaxis_title="Year",
        yaxis_title="Percentile score, forward-window averaged",
        yaxis=dict(range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=70, b=40),
    )
    return fig, raw_top, surprise_top


def render_break_shading_legend():
    st.caption(
        "Shaded regions indicate years identified as potential structural breaks based on "
        "documentary evidence, unusually large economic changes, or unusually large model surprises."
    )
    legend_items = [
        (
            "#FFD700",
            "Documented break year",
            "supported by documentary evidence (policy reforms, crises, droughts, trade changes, etc.)",
        ),
        (
            "#d95f02",
            "Top-5 observed-impact year",
            "unusually large economic changes in the data",
        ),
        (
            "#1f77b4",
            "Top-5 model-surprise year",
            "unusually difficult to explain changes using the model alone",
        ),
    ]
    cols = st.columns([1.15, 1.25, 1.25])
    for col, (color, label, help_text) in zip(cols, legend_items):
        with col:
            st.markdown(
                f"""
                <span style="display:inline-block;width:10px;height:10px;background:{color};opacity:0.65;border:1px solid #bbb;margin-right:6px;"></span>
                {label}
                """,
                unsafe_allow_html=True,
            )


def render_map_color_legend():
    legend_items = [
        ("#1f77b4", "Structural break"),
        ("#2ca02c", "Potential climate-related structural break"),
    ]
    cols = st.columns([0.35, 0.65])
    for col, (color, label) in zip(cols, legend_items):
        with col:
            st.markdown(
                f"""
                <span style="display:inline-block;width:11px;height:11px;border-radius:50%;background:{color};border:1px solid #777;margin-right:6px;"></span>
                {label}
                """,
                unsafe_allow_html=True,
            )


def prepare_structural_break_map_html(html_text: str) -> str:
    """Tidy the pre-generated Plotly map before embedding it in Streamlit."""
    html_text = re.sub(
        r'"title":\{"text":"Structural Break Map - \d{4}"\}',
        '"title":{"text":""}',
        html_text,
    )
    html_text = html_text.replace(
        '"legend":{"title":{"text":"point_color"},"tracegroupgap":0,"itemsizing":"constant"}',
        '"showlegend":false,"legend":{"title":{"text":""},"tracegroupgap":0,"itemsizing":"constant"}',
    )
    html_text = html_text.replace('"point_color"', '"Structural break score"')
    html_text = html_text.replace('"point_"', '"Structural break score"')
    html_text += """
    <script>
    setTimeout(function() {
      var gd = document.querySelector(".plotly-graph-div");
      if (!gd || !window.Plotly) {
        return;
      }
      window.Plotly.relayout(gd, {
        "title.text": "",
        "showlegend": false
      });
    }, 500);
    </script>
    """
    return html_text


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
    reference_relative_quarter=-2,
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

    if value_mode == "Estimated ENSO Effects":
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
        origin_period = peak_period + reference_relative_quarter
        if peak_period not in df.index or origin_period not in df.index:
            continue
        base = df.loc[origin_period, vars_use]
        if isinstance(base, pd.DataFrame):
            base = base.iloc[0]
        for rel_q in range(reference_relative_quarter, 13):
            q = peak_period + rel_q
            if q not in df.index:
                continue
            row = df.loc[q, vars_use]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for v in vars_use:
                y = pd.to_numeric(row[v], errors="coerce")
                y0 = pd.to_numeric(base[v], errors="coerce")
                if pd.notna(y) and pd.notna(y0):
                    if value_mode == "Estimated ENSO Effects":
                        value = 0.0 if q <= origin_period else y
                    else:
                        value = y - y0
                    rows.append(
                        {
                            "variable": v,
                            "event_label": format_quarter_label(peak.quarter.to_period("Q")),
                            "peak_quarter": peak.quarter,
                            "origin_quarter": origin_period.to_timestamp(),
                            "enso_value": peak.ENSO,
                            "relative_quarter": rel_q,
                            "value": value,
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
        title="ENSO index with peaks labeled",
        xaxis_title="Quarter",
        yaxis_title="ENSO",
        height=380,
        margin=dict(l=30, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def historical_quarterly_enso(panel_df):
    if panel_df.empty or "quarter" not in panel_df.columns or "ENSO" not in panel_df.columns:
        return pd.DataFrame(columns=["quarter", "ENSO"])
    enso = panel_df[["quarter", "ENSO"]].copy()
    enso["quarter"] = pd.to_datetime(enso["quarter"], errors="coerce").dt.to_period("Q").dt.to_timestamp()
    enso["ENSO"] = pd.to_numeric(enso["ENSO"], errors="coerce")
    return (
        enso.dropna(subset=["quarter", "ENSO"])
        .groupby("quarter", as_index=False)["ENSO"]
        .mean()
        .sort_values("quarter")
    )


def _enso_background_rgba(value, max_abs):
    if not np.isfinite(value) or not np.isfinite(max_abs) or max_abs <= 0:
        return "rgba(255,255,255,0)"
    strength = min(1.0, abs(float(value)) / float(max_abs))
    if strength < 0.05:
        return "rgba(255,255,255,0)"
    alpha = 0.04 + 0.18 * strength
    if value >= 0:
        return f"rgba(214, 39, 40, {alpha:.3f})"
    return f"rgba(31, 119, 180, {alpha:.3f})"


def add_enso_intensity_background(fig, panel_df, plot_df, x_col):
    enso = historical_quarterly_enso(panel_df)
    if enso.empty or plot_df.empty:
        return False

    max_abs = float(np.nanmax(np.abs(enso["ENSO"].to_numpy(dtype=float))))
    if not np.isfinite(max_abs) or max_abs <= 0:
        return False

    if x_col == "quarter":
        xmin = pd.to_datetime(plot_df[x_col], errors="coerce").min()
        xmax = pd.to_datetime(plot_df[x_col], errors="coerce").max()
        shade_df = enso[enso["quarter"].between(xmin, xmax)].copy()
        if shade_df.empty:
            return False
        for row in shade_df.itertuples(index=False):
            q0 = pd.Timestamp(row.quarter)
            fig.add_vrect(
                x0=q0,
                x1=q0 + pd.offsets.QuarterEnd(startingMonth=3),
                fillcolor=_enso_background_rgba(row.ENSO, max_abs),
                line_width=0,
                layer="below",
            )
    else:
        annual = enso.assign(year=enso["quarter"].dt.year).groupby("year", as_index=False)["ENSO"].mean()
        years = pd.to_numeric(plot_df[x_col], errors="coerce")
        shade_df = annual[annual["year"].between(years.min(), years.max())].copy()
        if shade_df.empty:
            return False
        for row in shade_df.itertuples(index=False):
            fig.add_vrect(
                x0=int(row.year) - 0.5,
                x1=int(row.year) + 0.5,
                fillcolor=_enso_background_rgba(row.ENSO, max_abs),
                line_width=0,
                layer="below",
            )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                color=[0],
                cmin=-max_abs,
                cmax=max_abs,
                colorscale=[
                    [0.0, "#1f77b4"],
                    [0.5, "#ffffff"],
                    [1.0, "#d62728"],
                ],
                showscale=True,
                colorbar=dict(
                    title="ENSO intensity",
                    len=0.36,
                    thickness=10,
                    x=1.02,
                    y=0.5,
                ),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    return True

# ----- STREAMLIT SETUP
st.set_page_config(
    page_title="Climate–Macro GVAR Explorer",
    layout="wide"
)
inject_global_control_styles()
st_title("Climate-Macroeconomic Risk Explorer")

from apps.modules.el_nino_event import render_el_nino_event_module
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
prob_df = load_probabilities(_panel_path)
# Note: panel/prob_df are intentionally NOT restricted to DASHBOARD_COUNTRIES
# anymore, so tabs reading `panel` directly (e.g. historical charts) work for
# any country. Tabs that depend on precomputed pickles (Scenario Impacts,
# Structural Break) still only have data for whichever countries were
# included when those pickles were last generated -- they already show an
# info/warning message when a selected country isn't found there.

_MORE_COUNTRY_SEP = "───── More Countries ─────"
_all_panel_countries = sorted(set(panel["country"].dropna().astype(str).unique()))
_more_countries = [c for c in _all_panel_countries if c not in set(DASHBOARD_COUNTRIES)]
country_options = list(DASHBOARD_COUNTRIES)
if _more_countries:
    country_options = country_options + [_MORE_COUNTRY_SEP] + _more_countries
if not country_options:
    st.error("No configured dashboard countries are available in the panel.")
    st.stop()

with st.container(key="analysis_scope_panel"):
    st.markdown(
        """
        <h3>🌎 Analysis Scope</h3>
        <p>Country and Response selections apply to all dashboard modules.</p>
        """,
        unsafe_allow_html=True,
    )
    control_cols = st.columns([1.2, 1.2, 2.6])
    with control_cols[0]:
        country = st.selectbox(
            "Country",
            country_options,
            index=default_option_index(country_options, "CHL"),
            format_func=lambda c: "— More Countries —" if c == _MORE_COUNTRY_SEP else iso3_to_label(c),
            key="country_select",
            help=HELP_TEXT["country"],
        )
        if country == _MORE_COUNTRY_SEP:
            country = "CHL"
    with control_cols[1]:
        response_var = st.selectbox(
            "Response",
            [v for v in MACRO_IMPACT_VARS if v in panel.columns],
            key="response_select",
            help=HELP_TEXT["response"],
        )

# ----- STREAMLIT TABS



(
    tab_el_nino_event,
    tab_climate_risk,
    tab_scenario,
    tab_event_study,
    tab_structural_break,
    tab_guide,
    tab_feedback,
) = st.tabs(
    [
        "2026–2027 El Niño Event",
        "Climate Early-Warning Chain",
        "Scenario Impacts",
        "ENSO Peak Event Study",
        "Structural Break Analysis",
        "Dashboard Guide",
        "Feedback",
    ]
)

st.markdown("""
<style>
    /* Style for the text inside ALL tabs */
    .stTabs [data-baseweb="tab"] p {
        font-size: 150%;
    }

    /* 1. Light background for the entire tab row */
    div[data-baseweb="tab-list"] {
        background-color: #EBF2F8;
        border-radius: 8px 8px 0 0;
        padding: 4px 4px 0 4px;
    }

    /* 2. Styling the Feedback Tab specifically (2rd tab) */
    div[data-baseweb="tab-list"] button:nth-of-type(7) {
        background-color: #E8F4F0 !important;
        margin-left: 10px; /* Optional: adds a small gap to separate it */
        border-radius: 6px 6px 0 0;
    }

    /* 3. Text color for the Feedback Tab */
    div[data-baseweb="tab-list"] button:nth-of-type(7) p {
        color: #2E7D6B !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

with tab_el_nino_event:
    st_header("2026–2027 El Niño Event")
    render_current_selection(country, response_var)
    render_el_nino_event_module(
        repo_root=_ROOT,
        dashboard_countries=DASHBOARD_COUNTRIES,
        selected_country=country,
        country_label_func=iso3_to_label,
    )

with tab_climate_risk:
    st_header("Climate Early-Warning Chain")
    render_current_selection(country, response_var)
    render_tab_description(0)

    st.markdown(
        """
        This panel links ENSO conditions today to the probability of localized heat
        and moisture (drought) stress next quarter. Macroeconomic impacts are evaluated
        separately in the Scenario Impacts tab.
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

    st_subheader(f"Results for {country}")
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
            "Baseline probability of moisture stress next quarter",
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
        "and provide context for scenario selection."
    )

    st_subheader("Country comparison of next-quarter climate stress risk")
    st.caption(
        "Probabilities are conditioned on the selected ENSO forecast and represent the likelihood "
        "of exceeding the specified climate-stress threshold during the next quarter."
    )
    st.caption(
        "Interpretation note: Both extreme heat and moisture stress can be influenced by ENSO "
        "conditions. However, moisture stress often occurs even in the absence of ENSO events, "
        "so ENSO-conditioned changes in moisture-stress probabilities are generally smaller and "
        "should be interpreted more cautiously than changes in heat-stress probabilities."
    )
    risk_table = prob_rows_df.sort_values("Heat probability (%)", ascending=False).copy()
    risk_table["Country"] = risk_table["Country"].map(iso3_to_label)
    st.dataframe(
        risk_table,
        column_config={
            "Country": st.column_config.TextColumn("Country"),
            "Heat probability (%)": st.column_config.NumberColumn(
                "Extreme Heat Risk",
                format="%.1f%%",
                help=HELP_TEXT["risk_summary"],
            ),
            "Moisture probability (%)": st.column_config.NumberColumn(
                "Moisture Stress (Drought) Risk",
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
        st_subheader(plt_title)
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
        "Probability of extreme heat next quarter (conditioned on user-selected ENSO forecast)",
        "Probability of extreme heat",
        "Heat probability (%)",
        "Probability of extreme heat",
    )
    choropleth_map(
        "Probability of extreme moisture next quarter (conditioned on user-selected ENSO forecast)",
        "Probability of extreme moisture",
        "Moisture probability (%)",
        "Probability of extreme moisture",
    )


with tab_scenario:
    st_header("Scenario Impacts")
    render_current_selection(country, response_var)
    render_tab_description(1)

    forecast_pack = load_forecast_bundle(forecast_pickle_state())
    if not forecast_pack["path"]:
        st.warning(
            "No forecast pickle found. Run `analysis/Dash_Output/gvar_kf_forecast.py` first; "
            "the dashboard will then draw directly from `Dash_Input/gvar_forecast_results.pkl`."
        )

    if not forecast_pack["bundle"]:
        st_subheader("Core forecast")
        st.info("No forecast bundle is available for the online ENSO forecast chart.")
    else:
        climate_variant_choice = st.radio(
            "Climate drivers used for this forecast",
            options=list(CLIMATE_VARIANT_CHOICES),
            format_func=lambda k: CLIMATE_VARIANT_CHOICES[k]["label"],
            index=0,
            key="scenario_climate_variant",
            horizontal=True,
            help=(
                "Choose one of the pre-estimated external-driver specifications. Oil, "
                "HeatDry, and IOD use merged panel/source values; ENSO specifications "
                "also include the mean/min/max ENSO scenario paths."
            ),
        )
        climate_vars_selected = CLIMATE_VARIANT_CHOICES[climate_variant_choice]["vars"]
        heat_var_active = "HeatDry" if "HeatDry" in climate_vars_selected else None
        enso_active = "ENSO" in climate_vars_selected
        iod_active = "IOD" in climate_vars_selected
        climate_variant_bundle = {
            "scenarios": _select_climate_variant_scenarios(forecast_pack["bundle"], climate_vars_selected),
            "climate_variants": forecast_pack["bundle"].get("climate_variants"),
        }
        if forecast_pack["bundle"].get("climate_variants") is None:
            st.caption(
                "This forecast pickle predates the climate-driver toggle, so it's showing "
                "the default ENSO-only forecast regardless of the selection above. "
                "Regenerate `Dash_Input/gvar_forecast_results.pkl` to enable it."
            )
        elif set(climate_vars_selected) != {"ENSO"}:
            st.caption(
                f"Model refit using: {', '.join(CLIMATE_TOGGLE_OPTIONS[v] for v in climate_vars_selected)}. "
                "This is a separately-estimated model, not a filter on the default ENSO forecast."
            )

        show_actual_overlap = st.checkbox(
            "Compare forecast vs. actual for already-observed quarters",
            value=False,
            help=(
                "Some near-term quarters are treated as forecast/nowcast because another "
                "model variable was still incomplete when the model was run, even though "
                "new actual data for the selected response variable has since become "
                "available. Enabling this shows those quarters' model forecast alongside "
                "the real observed value, for comparison."
            ),
            key="scenario_show_actual_overlap",
        )
        coeff_method = "last"
        if show_actual_overlap:
            coeff_method = st.radio(
                "Forecast coefficient method",
                options=list(FORECAST_COEFF_METHOD_OPTIONS),
                format_func=lambda x: FORECAST_COEFF_METHOD_OPTIONS[x],
                horizontal=True,
                key="scenario_coeff_method",
                help=(
                    "Selects among complete forecast paths saved in the forecast pickle. "
                    "The dashboard does not recompute coefficients, rerun forecasts, or "
                    "read structural-break diagnostic coefficient series."
                ),
            )
        selected_df = build_forecast_plot_df(
            climate_variant_bundle,
            panel,
            [country],
            response_var,
            history_start=None,
            include_observed_overlap=show_actual_overlap,
            coeff_method=coeff_method,
        )
        if (
            show_actual_overlap
            and coeff_method != "last"
            and not selected_df.empty
            and not selected_df.loc[
                ~selected_df["period_type"].eq("Actual history"),
                "coefficient_method_available",
            ].fillna(False).any()
        ):
            st.warning(
                f"`{FORECAST_COEFF_METHOD_OPTIONS[coeff_method]}` is not available "
                "for this country in the saved forecast bundle. Regenerate the offline "
                "forecast pickle before using this method."
            )
        q_summary, c_summary = summarize_forecast_ranges(selected_df)

        st_subheader(f"Forecast path for {iso3_to_label(country)}")
        enso_fig = plot_enso_forecast_online(forecast_pack["bundle"], panel)
        if enso_fig is not None:
            st.plotly_chart(enso_fig, width="stretch")
            st.caption(
                "Source: NOAA CPC CFSv2 seasonal forecast, "
                "https://www.cpc.ncep.noaa.gov/products/CFSv2/CFSv2seasonal.shtml"
            )
        else:
            st.info("Forecast pickle does not contain ENSO forecast data.")

        if enso_active:
            fig_selected = plot_selected_country_forecast(
                selected_df, response_var, iso3_to_label(country), counterfactual="enso"
            )
            if fig_selected is None:
                st.info("Forecast pickle does not contain data for the selected country/response.")
            else:
                st.plotly_chart(fig_selected, width="stretch")
                st.caption(
                    "No-ENSO counterfactual: the same forecasting model is run with future ENSO index "
                    "values set to 0, so the gap from the forecasted ENSO scenario indicates the "
                    "model-implied macroeconomic impact of ENSO."
                )
                if show_actual_overlap:
                    st.caption(
                        "Green diamonds: the real observed value for quarters now shown as "
                        "forecast/nowcast, so you can compare the model's estimate against what "
                        "actually happened once that data became available."
                    )
        else:
            st.info("ENSO is not part of this model variant, so no ENSO counterfactual is shown.")

        if iod_active:
            fig_iod = plot_selected_country_forecast(
                selected_df, response_var, iso3_to_label(country), counterfactual="iod"
            )
            if fig_iod is None:
                st.info("Forecast pickle does not contain an IOD counterfactual for this selection.")
            else:
                st.plotly_chart(fig_iod, width="stretch")
                st.caption(
                    "No-IOD counterfactual: the same forecasting model is run with future IOD index "
                    "values set to 0, so the gap from the forecasted scenario indicates the "
                    "model-implied macroeconomic impact of IOD."
                )

        if heat_var_active:
            heat_label = CLIMATE_TOGGLE_OPTIONS[heat_var_active]
            fig_heat = plot_selected_country_forecast(
                selected_df, response_var, iso3_to_label(country),
                counterfactual="heat", heat_label=heat_label,
            )
            if fig_heat is None:
                st.info(f"Forecast pickle does not contain a {heat_label} counterfactual for this selection.")
            else:
                st.plotly_chart(fig_heat, width="stretch")
                st.caption(
                    f"No-{heat_label} counterfactual: the same forecasting model is run with the "
                    f"future {heat_label} values set to 0, so the gap from the forecasted scenario "
                    "indicates the model-implied macroeconomic impact of "
                    f"{heat_label}."
                )

        if not q_summary.empty:
            _impact_title = (
                "Projected GDP growth and ENSO impacts relative to a no-ENSO baseline"
                if enso_active
                else "Projected GDP growth and climate-driver impacts"
            )
            st_subheader(_impact_title)
            st.caption(
                "Historical observations are used through 2026Q1. Where recent economic data are "
                "unavailable, near-term values are estimated before the forecast period begins. "
                "Reported cumulative impacts and maps reflect the projected effects of the selected "
                "scenario from 2026Q2 onward."
            )
            for c in [country]:
                c_quarters = q_summary[q_summary["country"] == c].copy()
                c_cum = c_summary[c_summary["country"] == c].copy()
                if c_quarters.empty:
                    continue
                st.markdown(f"**{iso3_to_label(c)}**")
                if not c_cum.empty:
                    r = c_cum.iloc[0]
                    n_metrics = int(enso_active) + int(iod_active) + int(bool(heat_var_active))
                    metric_cols = st.columns(n_metrics) if n_metrics > 1 else [st.container()]
                    metric_i = 0
                    if enso_active:
                        with metric_cols[metric_i]:
                            st.metric(
                                f"{response_var}: Cumulative impact relative to a no-ENSO baseline",
                                f"{r['cumulative_mean']:+.2f} p.p.",
                                delta=f"{r['cumulative_min']:+.2f} to {r['cumulative_max']:+.2f} p.p.",
                                help="Cumulative forecast difference between the ENSO scenario and a no-ENSO-impact baseline, shown in percentage points.",
                            )
                        metric_i += 1
                    if iod_active:
                        with metric_cols[metric_i]:
                            st.metric(
                                f"{response_var}: Cumulative impact relative to a no-IOD baseline",
                                f"{r['cumulative_iod_mean']:+.2f} p.p.",
                                delta=f"{r['cumulative_iod_min']:+.2f} to {r['cumulative_iod_max']:+.2f} p.p.",
                                help="Cumulative forecast difference between the IOD scenario and a no-IOD-impact baseline, shown in percentage points.",
                            )
                        metric_i += 1
                    if heat_var_active:
                        heat_label = CLIMATE_TOGGLE_OPTIONS[heat_var_active]
                        with metric_cols[metric_i]:
                            st.metric(
                                f"{response_var}: Cumulative impact relative to a no-{heat_label} baseline",
                                f"{r['cumulative_heat_mean']:+.2f} p.p.",
                                delta=f"{r['cumulative_heat_min']:+.2f} to {r['cumulative_heat_max']:+.2f} p.p.",
                                help=f"Cumulative forecast difference between the {heat_label} scenario and a no-{heat_label}-impact baseline, shown in percentage points.",
                            )
                show_tbl = c_quarters.copy()
                show_tbl["quarter"] = show_tbl["quarter"].dt.to_period("Q").astype(str)
                if "period_type" in show_tbl.columns:
                    show_tbl["period_type"] = show_tbl["period_type"].replace({"Gap fill / nowcast": "Estimated"})
                response_label = response_table_label(response_var)
                rename_map = {"quarter": "Quarter", "period_type": "Period type"}
                show_cols = ["Quarter", "Period type"]
                if enso_active:
                    rename_map.update(
                        {
                            "value_min": f"{response_label} (Low ENSO)",
                            "value_mean": f"{response_label} (Mean ENSO)",
                            "value_max": f"{response_label} (High ENSO)",
                            "impact_min": f"{response_var} (min vs no ENSO)",
                            "impact_mean": f"{response_var} (mean vs no ENSO)",
                            "impact_max": f"{response_var} (max vs no ENSO)",
                        }
                    )
                    show_cols += [
                        f"{response_label} (Low ENSO)",
                        f"{response_label} (Mean ENSO)",
                        f"{response_label} (High ENSO)",
                        f"{response_var} (min vs no ENSO)",
                        f"{response_var} (mean vs no ENSO)",
                        f"{response_var} (max vs no ENSO)",
                    ]
                else:
                    rename_map["value_mean"] = response_label
                    show_cols.append(response_label)
                if iod_active:
                    rename_map.update(
                        {
                            "impact_iod_min": f"{response_var} (min vs no IOD)",
                            "impact_iod_mean": f"{response_var} (mean vs no IOD)",
                            "impact_iod_max": f"{response_var} (max vs no IOD)",
                        }
                    )
                    show_cols += [
                        f"{response_var} (min vs no IOD)",
                        f"{response_var} (mean vs no IOD)",
                        f"{response_var} (max vs no IOD)",
                    ]
                if heat_var_active:
                    heat_label = CLIMATE_TOGGLE_OPTIONS[heat_var_active]
                    rename_map.update(
                        {
                            "impact_heat_min": f"{response_var} (min vs no {heat_label})",
                            "impact_heat_mean": f"{response_var} (mean vs no {heat_label})",
                            "impact_heat_max": f"{response_var} (max vs no {heat_label})",
                        }
                    )
                    show_cols += [
                        f"{response_var} (min vs no {heat_label})",
                        f"{response_var} (mean vs no {heat_label})",
                        f"{response_var} (max vs no {heat_label})",
                    ]
                show_tbl = show_tbl.rename(columns=rename_map)
                st.dataframe(
                    show_tbl[show_cols],
                    hide_index=True,
                    width="stretch",
                )

        previous_scenario_country = st.session_state.get("_scenario_country_select")
        if previous_scenario_country != country:
            st.session_state["scenario_countries"] = [country]
            st.session_state["_scenario_country_select"] = country
        else:
            current_scenario_countries = st.session_state.get("scenario_countries")
            if not current_scenario_countries:
                st.session_state["scenario_countries"] = [country]
            else:
                st.session_state["scenario_countries"] = [
                    c for c in current_scenario_countries if c in country_options
                ] or [country]

        st_subheader("Multi-country comparison and cumulative impact maps")
        scenario_countries = st.multiselect(
            "Countries",
            options=country_options,
            format_func=iso3_to_label,
            key="scenario_countries",
            help=HELP_TEXT["scenario_countries"],
        )
        comparison_df = build_forecast_plot_df(
            climate_variant_bundle,
            panel,
            scenario_countries,
            response_var,
        )
        fig_core = plot_core_forecast(comparison_df, response_var)
        if fig_core is None:
            st.info("Forecast pickle does not contain data for the selected comparison countries/response.")
        else:
            st.plotly_chart(fig_core, width="stretch")

        st.markdown("**Cumulative impact maps**")
        map_specs = []
        if enso_active:
            map_specs.append(
                {
                    "label": "ENSO",
                    "mean_col": "cumulative_mean",
                    "min_col": "cumulative_min",
                    "max_col": "cumulative_max",
                }
            )
        if iod_active:
            map_specs.append(
                {
                    "label": "IOD",
                    "mean_col": "cumulative_iod_mean",
                    "min_col": "cumulative_iod_min",
                    "max_col": "cumulative_iod_max",
                }
            )
        if heat_var_active:
            heat_label = CLIMATE_TOGGLE_OPTIONS[heat_var_active]
            map_specs.append(
                {
                    "label": heat_label,
                    "mean_col": "cumulative_heat_mean",
                    "min_col": "cumulative_heat_min",
                    "max_col": "cumulative_heat_max",
                }
            )
        if not map_specs:
            selected_variant_label = CLIMATE_VARIANT_CHOICES[climate_variant_choice]["label"]
            st.info(f"No climate counterfactual maps are available for the {selected_variant_label} variant.")
        else:
            st.caption(
                "Map range: 2026Q2-2027Q1 cumulative climate-driver impact. Maps show all "
                "12 dashboard countries and do not depend on the multi-country selection above."
            )
            map_countries = country_options
            metrics_for_maps = [v for v in MACRO_IMPACT_VARS if v in panel.columns]
            for spec in map_specs:
                st.markdown(f"**No-{spec['label']} counterfactual maps**")
                map_cols = st.columns(2)
                for i, metric in enumerate(metrics_for_maps):
                    metric_df = build_forecast_plot_df(
                        climate_variant_bundle,
                        panel,
                        map_countries,
                        metric,
                        history_start=None,
                    )
                    _, metric_summary = summarize_forecast_ranges(metric_df)
                    fig_map = plot_metric_impact_map(
                        metric_summary,
                        metric,
                        countries=map_countries,
                        mean_col=spec["mean_col"],
                        min_col=spec["min_col"],
                        max_col=spec["max_col"],
                        driver_label=spec["label"],
                    )
                    with map_cols[i % 2]:
                        if fig_map is not None:
                            st.plotly_chart(fig_map, width="stretch")
                        else:
                            st.info(f"No cumulative no-{spec['label']} map data for {metric}.")


with tab_event_study:
    st_header("ENSO Peak Event Study")
    render_current_selection(country, response_var)
    render_tab_description(2)
    st.caption(
        "The quarter of each ENSO peak is aligned at t=0. Observed Responses show how each "
        "indicator changes relative to the selected reference quarter. Estimated ENSO Effects "
        "show the model-estimated impact of ENSO from the selected reference quarter onward."
    )

    event_cols = st.columns([1, 1.2, 3.8])
    with event_cols[0]:
        event_country = st.selectbox(
            "Country",
            country_options,
            index=default_option_index(country_options, "MEX"),
            format_func=iso3_to_label,
            key="event_country",
        )
    with event_cols[1]:
        reference_relative_quarter = st.selectbox(
            "Reference quarter",
            options=[-2, -1, 0],
            index=2,
            format_func=lambda q: f"t={q}",
            key="event_reference_quarter",
            help="Quarter used as the zero-change reference point. The event-study plots start from this quarter.",
        )
    st.session_state.pop("event_mode", None)

    event_forecast_pack = load_forecast_bundle(forecast_pickle_state())
    event_pipeline_pack = load_pipeline_break_scores()
    _, peak_df = build_enso_peak_event_study(
        panel,
        event_forecast_pack["bundle"],
        event_pipeline_pack,
        event_country,
        "Observed Responses",
        reference_relative_quarter=reference_relative_quarter,
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

    if not selected_peaks:
        st.info("Select at least one ENSO peak to plot.")
    else:
        def render_event_mode_plot(event_mode):
            event_df, _ = build_enso_peak_event_study(
                panel,
                event_forecast_pack["bundle"],
                event_pipeline_pack,
                event_country,
                event_mode,
                selected_peak_labels=selected_peaks,
                reference_relative_quarter=reference_relative_quarter,
            )
            if event_df.empty:
                st.info(f"No event-study data available for {event_mode}.")
                return

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

            if event_mode == "Observed Responses":
                event_title = (
                    f"{iso3_to_label(event_country)}: Observed macroeconomic responses around selected ENSO peaks"
                )
                y_title = "Change from reference"
                st.caption(
                    "Observed responses are indexed to zero at the selected reference quarter "
                    f"(t={reference_relative_quarter}); the ENSO peak occurs at t=0."
                )
            else:
                event_title = (
                    f"{iso3_to_label(event_country)}: Estimated ENSO effects around selected ENSO peaks"
                )
                y_title = "Estimated ENSO effect"
                st.caption(
                    "Estimated effects represent the model-implied ENSO contribution from the selected "
                    f"reference quarter (t={reference_relative_quarter}) onward."
                )

            fig_event.update_layout(
                title=event_title,
                height=720,
                margin=dict(l=55, r=20, t=70, b=80),
                legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0),
            )
            fig_event.update_xaxes(title_text="Quarters from ENSO peak")
            fig_event.update_yaxes(title_text=y_title)
            st.plotly_chart(fig_event, width="stretch")

        render_event_mode_plot("Observed Responses")
        render_event_mode_plot("Estimated ENSO Effects")

    st.divider()
    st_header("Climate Vulnerability Analysis")
    st.caption(
        "Identifies climate-sensitive countries before examining their ENSO responses. "
        "This section uses ENSO only (no heat/moisture -- those are analyzed separately). "
        "The screening below is structural (no model); the validation reuses the existing "
        "production TVP-GVAR EM/Kalman-filter coefficients (ENSO + OIL_YoY) for "
        "candidate countries only. Not dependent on the country selected above."
    )
    _VULN_DIR = _ROOT / "analysis" / "vulnerability"
    _screen_path = _VULN_DIR / "climate_vulnerability_screening.csv"
    _valid_path = _VULN_DIR / "climate_vulnerability_validation.csv"

    if not _screen_path.is_file() or not _valid_path.is_file():
        st.warning(
            "Vulnerability analysis outputs not found. Run "
            "`python analysis/vulnerability/build_climate_vulnerability.py` first."
        )
    else:
        screen_df = pd.read_csv(_screen_path)
        valid_df = pd.read_csv(_valid_path)
        screen_df["Country"] = screen_df["ISO3"].map(iso3_to_label)
        valid_df["Country"] = valid_df["ISO3"].map(iso3_to_label)

        st_subheader("Structural Vulnerability Screening")
        st.caption(
            "Candidates = (Agriculture %GDP above median OR ND-GAIN Sensitivity above median) "
            "AND at least 40 quarters of macroeconomic observations. Screening only -- not a ranking."
        )
        med_agr = screen_df["agr_gdp_pct_median"].iloc[0]
        med_sens = screen_df["ndgain_sensitivity_median"].iloc[0]
        fig_screen = go.Figure()
        non_cand = screen_df[~screen_df["is_candidate"]]
        cand = screen_df[screen_df["is_candidate"]]
        fig_screen.add_trace(
            go.Scatter(
                x=non_cand["agr_gdp_pct"], y=non_cand["ndgain_sensitivity"],
                mode="markers", name="Other countries",
                marker=dict(color="#B8B8B8", size=7),
                text=non_cand["Country"], hovertemplate="%{text}<br>Agr %%GDP: %{x:.2f}<br>ND-GAIN: %{y:.3f}<extra></extra>",
            )
        )
        fig_screen.add_trace(
            go.Scatter(
                x=cand["agr_gdp_pct"], y=cand["ndgain_sensitivity"],
                mode="markers+text", name="Candidate countries",
                marker=dict(color="#d62728", size=10),
                text=cand["Country"], textposition="top center",
                hovertemplate="%{text}<br>Agr %%GDP: %{x:.2f}<br>ND-GAIN: %{y:.3f}<extra></extra>",
            )
        )
        fig_screen.add_vline(x=med_agr, line_dash="dash", line_color="gray")
        fig_screen.add_hline(y=med_sens, line_dash="dash", line_color="gray")
        fig_screen.update_layout(
            xaxis_title="Agriculture, forestry & fishing (% of GDP), 2014-2024 avg",
            yaxis_title="ND-GAIN Sensitivity Index, 2014-2024 avg",
            height=560,
        )
        st.plotly_chart(fig_screen, width="stretch")
        st.markdown(f"**Candidate countries ({len(cand)}):** {', '.join(sorted(cand['Country']))}")

        st_subheader("Dashboard-Based Validation")
        st.caption(
            "Realized ENSO Influence = mean |beta_ENSO(t) x ENSO_z(t)| (standardized/z-scored "
            "units, matching the model's own normalization). Coefficient Reliability = fraction "
            "of quarters where |beta_ENSO/SE| exceeds 1.96, from the Kalman filter's own P_filt "
            "uncertainty. Both are averaged across GDP/CPI/FX/EX and computed only for candidate "
            "countries, reusing the existing EM/Kalman-filter fit (no new methodology)."
        )
        if valid_df.empty:
            st.info("No candidate countries produced a valid EM fit.")
        else:
            fig_valid = go.Figure(
                go.Scatter(
                    x=valid_df["realized_enso_influence"], y=valid_df["coefficient_reliability"],
                    mode="markers+text", marker=dict(color="#1f77b4", size=10),
                    text=valid_df["Country"], textposition="top center",
                    hovertemplate="%{text}<br>Influence: %{x:.4f}<br>Reliability: %{y:.3f}<extra></extra>",
                )
            )
            fig_valid.add_vline(x=float(valid_df["realized_enso_influence"].median()), line_dash="dash", line_color="gray")
            fig_valid.add_hline(y=float(valid_df["coefficient_reliability"].median()), line_dash="dash", line_color="gray")
            fig_valid.update_layout(
                xaxis_title="Realized ENSO Influence (standardized units)",
                yaxis_title="Coefficient Reliability (fraction of quarters, |beta/SE| > 1.96)",
                height=560,
            )
            st.plotly_chart(fig_valid, width="stretch")
            st.dataframe(
                valid_df[["Country", "realized_enso_influence", "coefficient_reliability", "n_valid_quarters"]],
                width="stretch",
            )


with tab_structural_break:
    st_header("Structural Break")
    render_current_selection(country, response_var)
    render_tab_description(3)
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

    if not pipeline_pack["path"]:
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
        "Overlay Gemini identified break years as dotted lines",
        value=False,
        key="sb_use_llm_overlay",
        help=HELP_TEXT["llm_overlay"],
    )

    st_subheader("1) Structural break scores")

    def _model_break_scores_from_offline(iso3):
        d = offline_per_country.get(iso3)
        if not isinstance(d, dict):
            return pd.DataFrame()
        quarters = pd.to_datetime(d.get("diag_quarters", []), errors="coerce")
        if len(quarters) == 0:
            return pd.DataFrame()
        out = pd.DataFrame({"quarter": quarters})
        for col in ["innovation_score", "coefficient_change", "filter_smoother_gap"]:
            vals = np.asarray(d.get(col, []), dtype=float)
            if len(vals) == len(out):
                out[col] = vals
        score_cols_here = [
            c
            for c in ["innovation_score", "coefficient_change", "filter_smoother_gap"]
            if c in out.columns
        ]
        if not score_cols_here:
            return pd.DataFrame()
        out["year"] = out["quarter"].dt.year
        return out.dropna(subset=["quarter"]).sort_values("quarter")

    def _model_break_years(df_sc):
        score_cols_here = [
            c
            for c in ["innovation_score", "coefficient_change", "filter_smoother_gap"]
            if c in df_sc.columns
        ]
        if not score_cols_here:
            return []
        tmp = df_sc[["year"] + score_cols_here].copy()
        for col in score_cols_here:
            x = pd.to_numeric(tmp[col], errors="coerce")
            sd = x.std(skipna=True)
            tmp[f"z_{col}"] = 0.0 if not np.isfinite(sd) or sd < 1e-12 else (x - x.mean(skipna=True)) / sd
        z_cols = [f"z_{c}" for c in score_cols_here]
        tmp["model_composite_score"] = tmp[z_cols].mean(axis=1)
        yearly = (
            tmp.dropna(subset=["year", "model_composite_score"])
            .groupby("year", as_index=False)["model_composite_score"]
            .max()
            .sort_values("year")
            .reset_index(drop=True)
        )
        if yearly.empty:
            return []
        prev_score = yearly["model_composite_score"].shift(1)
        next_score = yearly["model_composite_score"].shift(-1)
        peaks = yearly[
            (yearly["model_composite_score"] > prev_score)
            & (yearly["model_composite_score"] > next_score)
        ].copy()
        if peaks.empty:
            peaks = yearly.copy()
        return sorted(peaks.nlargest(4, "model_composite_score")["year"].astype(int).tolist())

    def _country_break_scores(iso3):
        offline_scores = _model_break_scores_from_offline(iso3)
        if not offline_scores.empty:
            return offline_scores, "quarterly"
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
        return pd.DataFrame(), "none"

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
            f"Score series ({iso3_to_label(iso3)})",
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
            title=f"{iso3_to_label(iso3)}: structural break score series (model diagnostics)",
        )

        enso_background_added = add_enso_intensity_background(fig_sb, panel, plot_df, x_col)
        if enso_background_added:
            st.caption(
                "Background shading shows historical ENSO intensity from the panel series: "
                "red is positive, blue is negative, and near-zero values are transparent. "
                "Quarterly charts use quarterly ENSO; annual charts use calendar-year mean ENSO."
            )

        if use_llm_overlay and not llm_overlay_df.empty:
            ov = llm_overlay_df[
                (llm_overlay_df["iso3"] == iso3)
                & (pd.to_numeric(llm_overlay_df["break_supported"], errors="coerce") == 1)
                & (llm_overlay_df["year"].notna())
            ]
            for yr in sorted(set(ov["year"].astype(int))):
                if x_col == "quarter":
                    fig_sb.add_vline(
                        x=pd.Timestamp(year=int(yr), month=7, day=1),
                        line_dash="dot",
                        line_color="goldenrod",
                        opacity=0.8,
                    )
                else:
                    fig_sb.add_vline(x=int(yr), line_dash="dot", line_color="goldenrod")

        fig_sb.update_layout(margin=dict(l=20, r=90, t=60, b=45))
        st.plotly_chart(fig_sb, width="stretch")

    st_subheader("2) World Bank document information")
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
                    f"Year ({iso3_to_label(iso3)})",
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

    st_subheader("3) Documentary evidence and AI-assisted interpretation")
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
                gdf["break_type"] = (
                    gdf["break_type"]
                    .astype(str)
                    .str.replace("_", " ", regex=False)
                    .str.title()
                    .replace({"Nan": "--", "None": "--"})
                )
            if "break_supported" in gdf.columns:
                supported_num = pd.to_numeric(gdf["break_supported"], errors="coerce")
                gdf["break_supported"] = np.where(
                    supported_num == 1,
                    "Yes",
                    np.where(supported_num == 0, "No", gdf["break_supported"]),
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
            gdf_display = gdf[show_cols].rename(
                columns={
                    "country": "Country",
                    "year": "Year",
                    "status": "Processing Status",
                    "break_supported": "Break Supported",
                    "confidence": "Confidence (1-5)",
                    "break_type": "Break Category",
                    "summary": "Evidence Summary",
                }
            )
            if "Country" in gdf_display.columns:
                gdf_display["Country"] = gdf_display["Country"].map(
                    lambda x: iso3_to_label(country_to_iso3(x) or str(x))
                )
            st.dataframe(
                gdf_display,
                hide_index=True,
                width="stretch",
            )

    st_subheader("3.1) Observed impact vs model surprise overlap")
    st.caption(
        "Observed impact uses percentile-ranked macro changes. Model surprise uses the existing "
        "pickle break diagnostics, preferring composite score and falling back to innovation score. "
        "Both are forward-averaged over at most two years."
    )
    render_break_shading_legend()
    impact_horizon_q = st.slider(
        "Impact persistence window (quarters)",
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
            st.info("No observed-impact or model-surprise yearly data available.")
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
            summary_df["top5_observed_impact"] = summary_df["year"].isin(raw_top)
            summary_df["top5_model_surprise"] = summary_df["year"].isin(surprise_top)
            summary_df["documented_break"] = summary_df["year"].isin(info_years)
            if not raw_yearly.empty:
                summary_df["observed_impact_score"] = summary_df["year"].map(
                    raw_yearly.set_index("year")["raw_impact_score"]
                )
            if not surprise_yearly.empty:
                summary_df["model_surprise_score"] = summary_df["year"].map(
                    surprise_yearly.set_index("year")["model_surprise_score"]
                )
            summary_df = summary_df.rename(
                columns={
                    "year": "Year",
                    "top5_observed_impact": "Top-5 Impact Year",
                    "top5_model_surprise": "Top-5 Surprise Year",
                    "documented_break": "Documented Break Year",
                    "observed_impact_score": "Observed Impact Score",
                    "model_surprise_score": "Model Surprise Score",
                }
            )
            st.dataframe(summary_df, hide_index=True, width="stretch")

    # st_subheader("3.2) ENSO coefficients after drop-year refit")
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

    st_subheader("4) Global structural-break map by year")
    st.caption(
        "This map shows candidate structural-break locations for the selected year. Dot size reflects "
        "the strength of the break signal. Colors distinguish general structural breaks from potential "
        "climate-related structural breaks."
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
            render_map_color_legend()
            try:
                html_text = prepare_structural_break_map_html(map_path.read_text(encoding="utf-8"))
                components.html(html_text, height=720, scrolling=True)
            except Exception as e:
                st.error(f"Failed to load map HTML: {e}")

with tab_guide:
    st_header("Dashboard Guide")
    render_current_selection(country, response_var)
    st_subheader("Climate-Macroeconomic Risk Explorer")
    st.markdown(
        """
        This dashboard explores how El Niño-Southern Oscillation (ENSO) conditions may influence
        macroeconomic outcomes across twelve climate-sensitive countries. It combines historical
        data, climate-risk indicators, dynamic forecasting models, event studies, and structural-break
        analysis to help users understand both short-term risks and longer-term changes in
        climate-economy relationships.
        """
    )

    workflow_steps = [
        (
            "Climate Early-Warning Chain",
            "Assess whether current ENSO conditions increase the likelihood of extreme heat or moisture "
            "stress in the coming quarter.",
        ),
        (
            "Scenario Impacts",
            "Estimate how alternative ENSO scenarios could affect GDP growth, inflation, exchange rates, "
            "and exports relative to a no-ENSO baseline.",
        ),
        (
            "ENSO Peak Event Study",
            "Compare current forecasts with historical ENSO episodes and examine how macroeconomic "
            "indicators behaved around major El Niño events.",
        ),
        (
            "Structural Break Analysis",
            "Investigate whether relationships between climate and economic outcomes have changed over "
            "time because of policy reforms, economic transitions, or external shocks.",
        ),
    ]

    st_subheader("Recommended workflow")
    for i, (step_title, step_text) in enumerate(workflow_steps, start=1):
        st.markdown(f"**{i}. {step_title}**  \n{step_text}")

    guide_sections = [
        (
            "Tab 1: Climate Early-Warning Chain",
            "This tab links projected ENSO conditions to near-term physical climate risks. Historical "
            "relationships between ENSO and local heat or moisture extremes are used to estimate the "
            "probability that each country will experience climate stress during the next quarter.",
            [
                "Forecast ENSO conditions for the next quarter",
                "Probability of extreme heat stress",
                "Probability of extreme moisture stress",
                "Country-level risk comparisons",
                "Heat and moisture risk maps",
            ],
            "These probabilities provide an early-warning indicator of potential climate stress. They "
            "do not directly estimate economic impacts. Instead, they help identify countries and "
            "scenarios that may warrant further investigation in the Scenario Impacts tab.",
            "Which countries face elevated climate risk under the current ENSO outlook?",
        ),
        (
            "Tab 2: Scenario Impacts",
            "This tab estimates the potential macroeconomic consequences of future ENSO conditions. "
            "Forecasts are generated using the climate-macroeconomic model and are compared against "
            "a counterfactual scenario in which future ENSO effects are absent.",
            [
                "Historical and projected ENSO conditions",
                "Country forecasts under alternative ENSO scenarios",
                "GDP growth, inflation, exchange-rate, and export projections",
                "Cumulative impacts relative to a no-ENSO baseline",
                "Geographic maps of projected impacts",
                "Adaptive-policy experiments",
            ],
            "Impact estimates represent differences between the selected ENSO scenario and a no-ENSO "
            "reference case. Positive or negative values indicate the estimated contribution of ENSO "
            "to future economic outcomes.",
            "How much could future ENSO conditions affect economic performance in each country?",
        ),
        (
            "Tab 3: ENSO Peak Event Study",
            "This tab examines historical macroeconomic responses around major ENSO events. Multiple "
            "ENSO peaks are aligned in time so that users can compare economic trajectories before "
            "and after past climate shocks.",
            [
                "Historical ENSO peaks",
                "Event-aligned GDP, inflation, exchange-rate, and export responses",
                "Comparisons across multiple ENSO episodes",
                "Optional model-estimated ENSO contributions",
            ],
            "The event study provides historical context rather than forecasts. It helps users understand "
            "how countries responded during previous ENSO episodes and whether current projections are "
            "consistent with historical experience.",
            "What happened during past major ENSO events?",
        ),
        (
            "Tab 4: Structural Break Analysis",
            "This tab evaluates whether climate-economy relationships have changed over time. Structural "
            "breaks may arise from policy reforms, economic transitions, technological change, trade "
            "shifts, financial crises, or other major events.",
            [
                "Time-varying Kalman filter coefficients",
                "Structural-break scores",
                "Estimated ENSO sensitivities through time",
                "Supporting World Bank documents",
                "Gemini-generated summaries of potential break drivers",
            ],
            "Changes in model coefficients may indicate that historical climate responses are no longer "
            "stable. Structural-break information can help users identify periods when climate-economic "
            "relationships strengthened, weakened, or changed direction.",
            "Can historical climate-economy relationships be assumed to remain valid today?",
        ),
    ]

    st_subheader("Tab details")
    for i, (title, purpose, shows, interpretation, question) in enumerate(guide_sections):
        with st.expander(title, expanded=(i == 0)):
            st.markdown(f"**Purpose**  \n{purpose}")
            st.markdown("**What this tab shows**")
            for item in shows:
                st.markdown(f"- {item}")
            st.markdown(f"**Interpretation**  \n{interpretation}")
            st.markdown(f"**Key question**  \n{question}")

with tab_feedback:
    st_header("Feedback")
    render_current_selection(country, response_var)
    st.markdown(
        "Use this form to report bugs, confusing results, interpretation issues, "
        "or suggestions for improving the dashboard."
    )
    components.iframe(
        "https://forms.microsoft.com/Pages/ResponsePage.aspx?id=OPSkn-axO0eAP4b4rt8N7AeTQAt0SklBhYoUFkbp7hdUMzZDUUhITEYwNDE5M0lNMTZSMUhHUjBYRi4u",
        height=775,
        scrolling=True,
    )
