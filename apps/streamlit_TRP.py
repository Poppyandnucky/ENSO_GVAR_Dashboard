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

from models.config import KalmanConfig, ISO3_TO_IMF_NAME_FULL
from analysis.prediction import build_prediction_df
from analysis.scenario_irf import scenario_irf_exog
from analysis.estimate_var import fit_country_varx_TVP, fit_country_varx_VAR
# tvp_var_with_exog
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
import numpy as np

import plotly.express as px

import warnings
warnings.simplefilter("error", Warning)
# warnings.simplefilter("default", Warning)

STRUCTURAL_BREAK_DIR = _ROOT / "structural_break"
GEMINI_OUTPUT_DIR = STRUCTURAL_BREAK_DIR / "gemini output"
WB_TOP4_PATH = STRUCTURAL_BREAK_DIR / "wb_top4.csv"
PREGENERATED_MAP_DIR = STRUCTURAL_BREAK_DIR / "map1998-2024"
PIPELINE_PICKLE_CANDIDATES = [
    STRUCTURAL_BREAK_DIR / "gvar_pipeline_results.pkl",
    STRUCTURAL_BREAK_DIR / "Dash_Input" / "gvar_pipeline_results.pkl",
    _ROOT / "Dash_Input" / "gvar_pipeline_results.pkl",
]

# ----- NOTES
# irf computed using scenario_irf_exog( res, shock_var, H=12,
#                                       shock_percentile=84, shock_duration_q=1,
#                                       t0=-1, KalmanConfig = None )
# p, R, lam -> fit_models() ->
#              estimate_GVAR: fit_country_varx() -> tvp_var_with_exog(), TVPVARResult()

# ----- Kalman Filter switches
show_Kalman_controls = True

# ----- Early warning using climate data
def prepare_country_stressor_data( panel, country, stressor_var, stressor_pct=90, tail="lower" ):
    """
    Prepare country-level data for ENSO → stressor probability modeling.

    Parameters
    ----------
    panel : DataFrame
        Full panel with country, quarter, ENSO, and stressor variables
    country : str
        ISO3 country code
    stressor_var : str
        Column name (e.g. 'PRITHVI_HEAT_STD' or 'PRITHVI_MOISTURE_STD')
    stressor_pct : int
        Percentile threshold defining an extreme event

    Returns
    -------
    DataFrame
        Columns include ENSO and stressor_event_next (binary)
    """

    df = panel[panel["country"] == country].copy()
    df = df.sort_values("quarter")
    df.index.freq = "QS-OCT"

    # --- Define extreme stressor event
    if tail == "lower":
        thr = df[stressor_var].quantile(1. - stressor_pct / 100)
        df["stressor_event"] = (df[stressor_var] <= thr).astype(int)
    else:
        thr = df[stressor_var].quantile(stressor_pct / 100)
        df["stressor_event"] = (df[stressor_var] >= thr).astype(int)

    # --- ENSO_t → stressor_{t+1}
    df["stressor_event_next"] = df["stressor_event"].shift(-1)

    # --- Keep only rows usable for probability modeling
    df = df.dropna(subset=["ENSO", "stressor_event_next"])

    return df

def suggested_scenario_from_probability(p):
        """
        Map probability to suggested shock parameters.
        """
        if p >= 0.75:
            return dict(percentile=95, duration="1 year (4 quarters)")
        elif p >= 0.5:
            return dict(percentile=90, duration="1 year (4 quarters)")
        elif p >= 0.3:
            return dict(percentile=84, duration="1 quarter")
        else:
            return dict(percentile=70, duration="1 quarter")

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
            per_country = base_pack.get("per_country", {}) if isinstance(base_pack, dict) else {}
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
        "config": {},
    }

# ----- STREAMLIT SETUP
st.title("Climate–Macroeconomic Risk Explorer")
st.set_page_config(
    page_title="Climate–Macro GVAR Explorer",
    layout="wide"
)

# ----- STREAMLIT CACHED LOADERS
# @st.cache_data
# def load_panel():
#     return pd.read_csv(
#         "data/gvar_panel_streamlit.csv",
#         parse_dates=["quarter"]
#     )
# @st.cache_data
# def load_probabilities():
#     return pd.read_csv("data/prithvi_stressor_probabilities.csv")

from data_prep.data_loader import load_gvar_panel, load_stressor_probabilities
from analysis.var_spec import VARSpec

# ----- INITIALIZE VARSpec
STREAMLIT_SPEC = VARSpec(
    endog_vars=["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"],
    exog_vars=["ENSO", "PRITHVI_HEAT_EXTENT", "PRITHVI_MOISTURE_EXTENT"],  # adjust if needed
    include_const=True,      # explicit intercept
    demean=True,             # critical for baseline behavior
    standardize=True,        # critical for TVP stability
    p=2
)

# ----- STREAMLIT CACHED LOADERS (UI-level caching only)
@st.cache_data
def load_panel():
    return load_gvar_panel()
@st.cache_data
def load_probabilities():
    return load_stressor_probabilities()
@st.cache_resource
def fit_models_NEW(panel, R=1.0, lam=0.98, use_TVP=False):
    results = {}
    spec = STREAMLIT_SPEC

    modeled_countries = sorted(results.keys())
    for c in modeled_countries:
        if use_TVP:
            res = fit_country_varx_TVP(
                panel,
                c,
                spec=spec,
                R=R,
                lam=lam,
                prior_var=True
            )
        else:
            res = fit_country_varx_VAR(
                panel,
                c,
                spec=spec
            )

        if res is not None:
            results[c] = res

    return results
@st.cache_resource
def fit_models(panel, p=STREAMLIT_SPEC.p, R=1.0, lam=0.98, use_TVP=False):
    results = {}
    spec = STREAMLIT_SPEC

    for c in sorted(panel["country"].unique()):
        if use_TVP:
            res = fit_country_varx_TVP(panel, c, spec, R=R, lam=lam, prior_var=True)
            # res = fit_country_varx_TVP(panel, c, p=p, R=R, lam=lam, prior_var=True)
        else:
            res = fit_country_varx_VAR(panel, c, spec)
            # res = fit_country_varx_VAR(panel, c, p=p)

        if res is not None:
            results[c] = res
            res.var_spec = spec

    return results
@st.cache_resource
def fit_enso_stressor_model(df):
    """
    Fit ENSO → downstream stressor probability model.

    Expects df to contain ENSO and stressor_event_next (binary).
    Returns None if the sample is degenerate or the Hessian is singular (unidentified covariance).
    """
    y = df["stressor_event_next"].astype(float)
    if len(y) < 10:
        return None
    if y.nunique(dropna=False) < 2:
        return None
    if df["ENSO"].nunique() < 2:
        return None

    X = sm.add_constant(df["ENSO"])
    # Global simplefilter("error", Warning) would turn statsmodels diagnostics into crashes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerfectSeparationWarning)
        warnings.simplefilter("ignore", RuntimeWarning)  # exp overflow in cdf during fit diagnostics
        try:
            model = sm.Logit(y, X).fit(disp=0)
        except (np.linalg.LinAlgError, ValueError):
            return None
    return model

panel = load_panel()
results = fit_models(panel)
prob_df = load_probabilities()
prob_df.index.freq = "QS-OCT"

# Define shock candidates dynamically
SHOCK_VARS = [
    c for c in [
        "ENSO",
        "PRITHVI_HEAT_EXTENT",
        "PRITHVI_MOISTURE_EXTENT",
        "COMMODITY_YoY",
        "US_GDP_YoY",
        "CHN_GDP_YoY",
    ]
    if c in panel.columns
]
print(f"panel.columns: {panel.columns}")

# ----- SIDEBAR CONTROLS
st.sidebar.header("Controls")

country = st.sidebar.selectbox(
    "Country",
    sorted(results.keys()),
    key="country_select"
)
p_row = prob_df[prob_df["country"] == country]

shock_var = st.sidebar.selectbox(
    "Shock",
    SHOCK_VARS,
    format_func=lambda x: {
        "PRITHVI_MOISTURE_EXTENT": "Drought extent (% of land)",
        "PRITHVI_HEAT_EXTENT": "High temperature extent (% of land)",
    }.get(x, x),
    key="shock_select"
)

response_var = st.sidebar.selectbox(
    "Response",
    results[country].var_spec.endog_vars,
    key="response_select"
)

H = st.sidebar.slider(
    "Horizon (quarters)",
    4, 120, 12,
    key="horizon_slider"
)

st.sidebar.markdown("### Shock design")

shock_duration = st.sidebar.selectbox(
    "Shock duration",
    ["1 year (4 quarters)", "1 quarter", ],
    key="shock_duration"
)
shock_duration_q = 1 if shock_duration == "1 quarter" else 4

shock_percentile = st.sidebar.slider(
    "Shock magnitude (percentile)",
    50, 99, 84,
    key="shock_percentile"
)

plot_type = st.sidebar.selectbox(
    "Plot type",
    ["Cumulative response", "Level response", ],
    key="plot_type"
)

st.sidebar.markdown("### Climate early-warning chain")

enso_forecast = st.sidebar.slider("ENSO forecast for next quarter", min_value=-2.5, max_value=2.5,
                          value=1.5, step=0.1, key="enso_forecast")
heat_moist_pct = st.sidebar.slider("Heat and moisture stress threshold (percentile)", min_value=80,
                           max_value=99, value=90, step=1, key="heat_threshold")
# shock_duration_heat_moist = st.sidebar.selectbox("Heat and moisture stress duration",
#                                          ["1 year (4 quarters)", "1 quarter", ], key="country_shock_duration")
# shock_duration_q_heat_moist = 1 if shock_duration_heat_moist == "1 quarter" else 4
shock_duration_q_heat_moist = shock_duration_q

# ----- KALMAN FILTER PARAMETERS
# ----- STREAMSLIT PARAMETERS
if show_Kalman_controls:
    st.subheader("Advanced Dynamics (tvp-VAR)")

    KalmanConfig["use_TVP"] = st.checkbox("Use time-varying VAR", True)
    KalmanConfig["forgetting_lambda"] = st.slider("Memory (λ)", 0.98, 1.000, 0.99)
    KalmanConfig["measurement_noise_R"] = st.slider("Stability (R)", 0.01, 10.0, 1.0)

    # KalmanConfig["shock_scaling"] = st.selectbox(
    #     "Shock scaling",
    #     ["std", "percentile", "unit"]
    # )
    #
    # KalmanConfig["irf_regime"] = st.selectbox(
    #     "IRF regime",
    #     ["typical", "stress", "last"]
    # )

# ----- Forgetting & memory control
lam = KalmanConfig["forgetting_lambda"]
R = KalmanConfig["measurement_noise_R"]
# ----- Fixed VAR vs Kalman VAR [NOT USED]
# if not KalmanConfig["use_TVP"]:
#     # use statsmodels VAR
# else:
#     # use tvp-VAR with forgetting
if KalmanConfig["use_TVP"]:
    results = fit_models(panel, R=R, lam=lam, use_TVP=True)

# ----- Regime selection (this explains VAR vs tvp differences)
# MAY BE BETTER TO REGULATE R RATHER THAN t0
# if KalmanConfig["irf_regime"] == "last":
#     t0 = -1
# elif KalmanConfig["irf_regime"] == "typical":
#     t0 = np.nanargmin(np.abs(shock_series))
# elif KalmanConfig["irf_regime"] == "stress":
#     t0 = np.nanargmax(shock_series)
# elif KalmanConfig["irf_regime"] == "custom":
#     t0 = KalmanConfig["custom_t0"]

# ----- Heat variable transformation (huge stabilizer)
# if KalmanConfig["heat_transform"] == "intensity":
#     heat_used = np.maximum(0, heat - threshold)
# elif KalmanConfig["heat_transform"] == "zscore":
#     heat_used = (heat - heat.mean()) / heat.std()
# else:
#     heat_used = heat_event

# ----- Eigenvalue damping
# if KalmanConfig["cap_eigenvalues"]:
#     eigs = np.linalg.eigvals(A1)
#     if np.max(np.abs(eigs)) > KalmanConfig["max_eigenvalue"]:
#         A1 *= KalmanConfig["max_eigenvalue"] / np.max(np.abs(eigs))

# ----- STREAMLIT TABS
tab_climate_risk, tab_scenario, tab_structural_break = st.tabs(
    ["Climate Risk", "Scenario Impacts", "Structural Break"]
)

with tab_climate_risk:
    # Probability maps: extreme-flag + probability estimation
    st.header("Climate Early-Warning Chain")

    st.markdown(
        """
        This panel links **ENSO conditions today** to the **probability of localized heat stress next quarter**, 
        and combines that risk with the **conditional macroeconomic impact** estimated by the GVAR.
        """
    )

    # ----- DIAGNOSTICS
    show_diag = st.checkbox("Show historical data used to estimate probabilities", value=False)
    if show_diag:
        df_diag = prepare_country_stressor_data(panel, country,
            stressor_var=(
                "PRITHVI_HEAT_EXTENT" if shock_var == "PRITHVI_HEAT_EXTENT"
                else "PRITHVI_MOISTURE_EXTENT"),
            # stressor_var=("PRITHVI_HEAT_STD" if shock_var == "PRITHVI_HEAT_STD"
            #               else "PRITHVI_MOISTURE_STD" ),
            stressor_pct=90,
            # tail=("upper" if shock_var == "PRITHVI_HEAT_STD" else "lower")
            tail=("upper")  # always upper for EXTENT variables
            )

        st.subheader("Country data")
        st.dataframe(df_diag)

        cols = df_diag.columns
        x_axis = st.selectbox("Select X-Axis", cols.to_list(), index=cols.get_loc("ENSO"))
        y_axis = st.selectbox("Select Y-Axis", cols.to_list(), index=cols.get_loc("PRITHVI_MOISTURE_EXTENT"))
        st.scatter_chart(df_diag, x=x_axis, y=y_axis)

    # --- Probability model ---
    # Generate conditional probabilities for all countries
    prob_rows = []  # heat_std, moisture_std, moisture_extent
    rows = []
    # print(f"panel: {panel[['country','PRITHVI_HEAT_STD','PRITHVI_MOISTURE_STD']]}")

    for c in sorted(panel["country"].unique()):
        if c not in results:
            continue
        df_heat = prepare_country_stressor_data(
            panel, c, "PRITHVI_HEAT_EXTENT", stressor_pct=heat_moist_pct, tail="upper" )
        #   panel, c, "PRITHVI_HEAT_STD", stressor_pct = heat_moist_pct, tail = "upper" )
        df_moist = prepare_country_stressor_data(
            panel, c, "PRITHVI_MOISTURE_EXTENT", stressor_pct=heat_moist_pct, tail="upper" )
        #   panel, c, "PRITHVI_MOISTURE_STD", stressor_pct = heat_moist_pct, tail = "lower" )

        if len(df_heat) >= 10:
            model_heat = fit_enso_stressor_model(df_heat)
            if model_heat is not None:
                p_heat = float(
                    model_heat.predict(
                        pd.DataFrame({"const": [1.0], "ENSO": [enso_forecast]})
                    )[0]
                )
            else:
                p_heat = np.nan
        else:
            p_heat = np.nan

        if len(df_moist) >= 10:
            model_moist = fit_enso_stressor_model(df_moist)
            if model_moist is not None:
                p_moist = float(
                    model_moist.predict(
                        pd.DataFrame({"const": [1.0], "ENSO": [enso_forecast]})
                    )[0]
                )
            else:
                p_moist = np.nan
        else:
            p_moist = np.nan

        prob_rows.append({
            "country": c, "P_HEAT_NEXT_Q_pct": p_heat * 100, "P_MOISTURE_NEXT_Q_pct": p_moist * 100 } )

        res_c = results[c]
        resp_idx = res_c.var_spec.endog_vars.index(response_var)
        # resp_idx = list(res_c.names).index(response_var)
        # diff_c_heat = scenario_irf_exog( res_c, shock_var="PRITHVI_HEAT_STD", H=H,
        diff_c_heat = scenario_irf_exog( res_c, shock_var="PRITHVI_HEAT_EXTENT", H=H,
            shock_percentile=heat_moist_pct, shock_duration_q=shock_duration_q_heat_moist )
        # diff_c_moist = scenario_irf_exog( res_c, shock_var="PRITHVI_MOISTURE_STD", H=H,
        diff_c_moist=scenario_irf_exog(res_c, shock_var="PRITHVI_MOISTURE_EXTENT", H=H,
            shock_percentile=heat_moist_pct, shock_duration_q=shock_duration_q_heat_moist )
        # expected_loss_heat = p_heat * diff_c_heat[:, resp_idx].cumsum()[-1]
        # expected_loss_moist = p_moist * diff_c_moist[:, resp_idx].cumsum()[-1]

        def peak_directional_impact( diff, resp_idx, direction_horizon=(1, 2) ):
            resp = diff[:, resp_idx]
            h0, h1 = direction_horizon
            direction_signal = resp[h0:h1 + 1].mean()
            direction = np.sign(direction_signal)
            if direction < 0:
                impact_value = resp.min()
            elif direction > 0:
                impact_value = resp.max()
            else:
                impact_value = 0.0
            return impact_value

        expected_loss_heat = p_heat * peak_directional_impact(diff_c_heat, resp_idx)
        expected_loss_moist = p_moist * peak_directional_impact(diff_c_moist, resp_idx)

        rows.append({ "Country": c,
            "Heat probability (%)": 100 * p_heat, "Expected cumulative heat impact": expected_loss_heat,
            "Moisture probability (%)": 100 * p_moist, "Expected cumulative moisture impact": expected_loss_moist,          })

    prob_rows_df = pd.DataFrame(prob_rows)

    p_heat = prob_rows_df.loc[prob_rows_df['country'] == country, 'P_HEAT_NEXT_Q_pct'].item() / 100.
    p_moist = prob_rows_df.loc[prob_rows_df['country'] == country, 'P_MOISTURE_NEXT_Q_pct'].item() / 100.

    if not np.isnan(p_moist):
        print(f"p_moist: {p_moist:.2f} %")

    st.subheader(f"Results for {country}")

    if not p_row.empty:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "Baseline probability of extreme heat next quarter",
                f"{1.0 - heat_moist_pct / 100:.0%}"
                # f"{p_row['P_HEAT_NEXT_Q'].iloc[0]:.0%}"
            )
        with c2:
            delta_heat = p_heat - p_row["P_HEAT_NEXT_Q"].iloc[0]
            metric_heat = "—" if np.isnan(p_heat) else f"{p_heat:.0%}"
            kw_heat = (
                {"delta": f"{delta_heat:+.0%}"}
                if not np.isnan(p_heat) and not np.isnan(delta_heat)
                else {}
            )
            st.metric(
                "ENSO-conditioned probability of extreme heat next quarter",
                metric_heat,
                **kw_heat,
            )
        with c3:
            st.metric(
                "Probability of moisture stress next quarter",
                f"{1.0 - heat_moist_pct / 100:.0%}"
                # f"{p_row['P_MOISTURE_NEXT_Q'].iloc[0]:.0%}"
            )
        with c4:
            delta_moist = p_moist - p_row["P_MOISTURE_NEXT_Q"].iloc[0]
            metric_moist = "—" if np.isnan(p_moist) else f"{p_moist:.0%}"
            kw_moist = (
                {"delta": f"{delta_moist:+.0%}"}
                if not np.isnan(p_moist) and not np.isnan(delta_moist)
                else {}
            )
            st.metric(
                "ENSO-conditioned probability of extreme moisture next quarter",
                metric_moist,
                **kw_moist,
            )

        st.caption(
            "Probabilities are estimated from historical ENSO → physical stress "
            "relationships and provide context for scenario selection. "
            "Macroeconomic impacts are evaluated separately via scenario analysis."
        )
    else:
        st.info("No probability estimates available for this country.")

    # --- Conditional macro impact from GVAR [NEED TO INCLUDE MOIST] ---
    res = results[country]
    # diff = scenario_irf_exog( res, shock_var="PRITHVI_HEAT_STD", H=H, shock_percentile=heat_moist_pct,
    diff_heat = scenario_irf_exog( res, shock_var="PRITHVI_HEAT_EXTENT", H=H, shock_percentile=heat_moist_pct,
        shock_duration_q=shock_duration_q_heat_moist )
    diff_moist = scenario_irf_exog( res, shock_var="PRITHVI_MOISTURE_EXTENT", H=H, shock_percentile=heat_moist_pct,
        shock_duration_q=shock_duration_q_heat_moist )

    st.write("DEBUG: raw diff first 5 quarters", diff_heat[:5, resp_idx])
    st.write("DEBUG: mean abs diff", np.mean(np.abs(diff_heat[:, resp_idx])))

    resp_idx = res.var_spec.endog_vars.index(response_var)

    # --- Plot: conditional vs expected ---
    def plot_shock_effect(event,p_event,diff_event):
        conditional = diff_event[:, resp_idx]
        expected = p_event * conditional
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(conditional, label=f"Conditional impact (if {event} occurs)",
            linestyle="--" )
        ax.plot(expected, label="Expected impact (probability-weighted given ENSO)",
            linewidth=2 )
        # ax.plot(conditional.cumsum(), label=f"Conditional impact (if {event} occurs)",
        #     linestyle="--" )
        # ax.plot(expected.cumsum(), label="Expected impact (probability-weighted given ENSO)",
        #     linewidth=2 )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Quarters after onset")
        ax.set_ylabel(f"Cumulative Δ {response_var}")
        ax.set_title(f"{country}: Expected downside from {event} risk")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.caption( f"Expected impact = Probability of {event} event × Conditional macro response. "
                    "This supports early-warning and portfolio risk assessment." )

    col1_climate_risk, col2_climate_risk = st.columns(2)

    with col1_climate_risk:
        plot_shock_effect("heat",p_heat,diff_heat)
    with col2_climate_risk:
        plot_shock_effect("drought",p_moist,diff_moist)

    # ----- Comparison across countries
    st.subheader("Early-warning comparison across countries")

    col1a_climate_risk, col2a_climate_risk = st.columns(2)

    with col1a_climate_risk:
        # compare_countries = st.multiselect("Compare countries", sorted(results.keys()),
        #     default=["IND", "BRA", "ZAF"], key="ew_compare")

        world = gpd.read_file("data/natural_earth/ne_110m_admin_0_countries.shp")
        world_modeled = world[world["ISO_A3"].isin(prob_df["country"])]
        # Merge probabilities with countries
        world_modeled = world_modeled.merge(prob_rows_df, left_on="ISO_A3", right_on="country", how="left")

        risk_table = pd.DataFrame(rows).sort_values("Expected cumulative heat impact")

        # st.dataframe(risk_table, width='stretch')
        # ----- TABLE WITH COUNTRIES IN COLUMN 1
        st.write("#### Country Risk Summary")

        # 1. (Optional) Filter your table if needed
        # filtered_table = risk_table[risk_table['Country'].isin(compare_countries)]

        # 2. Display the table with professional formatting
        st.dataframe(
            risk_table,
            column_config={
                "Country": st.column_config.TextColumn("Country"),
                "Heat probability (%)": st.column_config.NumberColumn(
                    "Heat Probability",
                    format="%.1f%%"  # Adds the % sign and 1 decimal place
                ),
                "Expected cumulative heat impact": st.column_config.NumberColumn(
                    "Heat Impact",
                    format="%.2f"  # 2 decimal places
                ),
                "Moisture probability (%)": st.column_config.NumberColumn(
                    "Moisture Probability",
                    format="%.1f%%"
                ),
                "Expected cumulative moisture impact": st.column_config.NumberColumn(
                    "Moisture Impact",
                    format="%.2f"
                ),
            },
            hide_index=True,
            width='stretch'
        )

    with col2a_climate_risk:
        # ----- Risk quadrant plot
        fig, ax = plt.subplots()
        ax.scatter( risk_table["Heat probability (%)"], risk_table["Expected cumulative heat impact"],
                    color="blue", label="Heat")
        ax.scatter( risk_table["Moisture probability (%)"], risk_table["Expected cumulative moisture impact"],
                    color="red", label="Moisture" )
        for _, r in risk_table.iterrows():
            ax.text( r["Heat probability (%)"], r["Expected cumulative heat impact"], r["Country"] )
            ax.text( r["Moisture probability (%)"], r["Expected cumulative moisture impact"], r["Country"],
                     color="red" )

        ax.legend()
        ax.set_xlabel("Probability of heat or moisture event (%)")
        ax.set_ylabel("Expected cumulative macro impact")
        ax.set_title("Climate early-warning risk map")
        st.pyplot(fig)

    # ----- CLOROPLETH MAP
    def choropleth_map(plt_title, colorbar_title, map_color, map_lbl):
        st.subheader(plt_title)

        fig = px.choropleth(
            world,
            geojson=world.geometry,
            locations=world.index,
            color_discrete_sequence=["#FFFFFF"],
        )
        fig.update_traces(
            marker_line_color="gray",
            marker_line_width=0.5,
            hoverinfo="skip"
        )
        # Overlay: modeled countries with probabilities
        fig2 = px.choropleth(
            world_modeled,
            geojson=world_modeled.geometry,
            locations=world_modeled.index,
            color=map_color,  # or P_MOISTURE_NEXT_Q
            color_continuous_scale="Reds",
            range_color=(0, 1),
            labels={map_color: map_lbl},
        )
        # Combine layers
        for trace in fig2.data:
            fig.add_trace(trace)
        fig.update_geos(
            fitbounds="locations",
            visible=False
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
            )
        )
        st.plotly_chart(fig, width='stretch', config={'scrollZoom': False})

    # Base map: all country outlines
    plt_title = "Probability of extreme heat next quarter (conditional on ENSO)"
    colorbar_title = "Probability of extreme heat"
    map_color = "P_HEAT_NEXT_Q_pct"
    map_lbl = "Probability of extreme heat"

    choropleth_map(plt_title, colorbar_title, map_color, map_lbl)

    plt_title = "Probability of extreme moisture next quarter (conditional on ENSO)"
    colorbar_title = "Probability of extreme moisture"
    map_color = "P_MOISTURE_NEXT_Q_pct"
    map_lbl = "Probability of extreme moisture"

    choropleth_map(plt_title, colorbar_title, map_color, map_lbl)

with tab_scenario:
    st.header("Macroeconomic impact under selected scenario")

    col1_scenario, col2_scenario = st.columns(2)

    with col1_scenario:
        st.subheader(f"Results for country {country}")
        diff = scenario_irf_exog(
            results[country],
            shock_var,
            H=H,
            shock_percentile=shock_percentile,
            shock_duration_q=shock_duration_q,
        )

        resp_idx = results[country].var_spec.endog_vars.index(response_var)

        if plot_type == "Cumulative response":
            y_plot = diff[:, resp_idx].cumsum()
            ylabel = f"Cumulative Δ {response_var}"
        else:
            y_plot = diff[:, resp_idx]
            ylabel = f"Δ {response_var}"

        fig = plt.figure(figsize=(6,4))
        plt.plot(y_plot, marker="o")
        plt.axhline(0, color="black", lw=0.8)
        plt.xlabel("Quarters after shock")
        plt.ylabel(ylabel)
        plt.title(
            f"{country}: {response_var} response to {shock_var}\n"
            f"{shock_percentile}th percentile, {shock_duration}"
        )
        plt.tight_layout()

        st.pyplot(fig)

        # ----- Streamlit panel: Predicted vs Observed
        st.subheader("Observed vs projected")

        pred_df = build_prediction_df(
            panel,
            results,
            country=country,
            response_var=response_var,
            shock_var=shock_var,
            H=H, shock_percentile=shock_percentile, shock_duration_q=shock_duration_q,
        )
        pred_df.index.freq = "QS-OCT"

        fig3 = plt.figure(figsize=(7,4))
        pred_df["observed"].plot(label="Observed", marker="o")
        pred_df["baseline"].plot(label="Baseline forecast", linestyle="--")

        if "scenario" in pred_df:
            pred_df["scenario"].plot(label=f"{shock_var} scenario", linestyle=":")

        plt.axvline(pred_df.index[-H], color="gray", linestyle=":")
        plt.legend()
        plt.title(f"{country}: {response_var} observed vs projected")
        plt.ylabel(response_var)
        plt.tight_layout()

        st.pyplot(fig3)

        st.subheader("Macroeconomic regime over time")

        regime_df = get_country_regime_ts(panel, country)

        import matplotlib.ticker as mticker

        fig_reg, ax = plt.subplots(figsize=(7, 4))

        ax.plot(
            regime_df["year"],
            regime_df["CPI_YoY_annual"],
            label="Inflation (CPI, %)",
            marker="o"
        )

        ax.plot(
            regime_df["year"],
            regime_df["FX_YoY_annual"],
            label="FX depreciation (% YoY)",
            marker="s"
        )

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Year")
        ax.set_ylabel("Percent")
        ax.set_title(f"{country}: inflation and FX regime")
        # 🔹 Force integer (yearly) ticks
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend()
        ax.grid(alpha=0.3)
        ax.fill_between(
            regime_df["year"],
            regime_df["CPI_YoY_annual"],
            where=regime_df["CPI_YoY_annual"] > 5,
            alpha=0.2,
            label="High inflation regime"
        )
        st.pyplot(fig_reg)

    with col2_scenario:
        st.subheader("Relative vulnerability across countries")

        compare_list = st.multiselect(
            "Compare countries",
            sorted(results.keys()),
            default=["IND", "BRA", "ZAF"],
            key="compare"
        )

        fig2 = plt.figure(figsize=(7,5))

        for c in compare_list:
            if c not in results:
                continue
            diff = scenario_irf_exog(
                results[c],
                shock_var,
                H=H,
                shock_percentile=shock_percentile,
                shock_duration_q=shock_duration_q,
            )
            idx = results[c].var_spec.endog_vars.index(response_var)
            plt.plot(diff[:, idx].cumsum(), label=c)

        plt.axhline(0, color="black", lw=0.8)
        plt.xlabel("Quarters after shock")
        plt.ylabel(f"Cumulative Δ {response_var}")
        plt.title(f"Cumulative impact of {shock_var}")
        plt.legend()
        plt.tight_layout()

        st.pyplot(fig2)

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

    available_iso = sorted(x for x in available_iso if x)
    default_sel = [country] if country in available_iso else available_iso[:3]

    sb_countries = st.multiselect(
        "Select countries (multi-select)",
        options=available_iso,
        default=default_sel,
        format_func=iso3_to_label,
        key="sb_countries",
    )
    use_llm_overlay = st.checkbox(
        "Overlay Gemini identified break years",
        value=True,
        key="sb_use_llm_overlay",
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
    if gemini_df.empty:
        st.warning("No Gemini output CSV found under `structural_break/gemini output/`.")
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

    st.subheader("4) Map by year (pre-generated HTML)")
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
        map_year = st.selectbox("Map year", options=map_year_options, key="sb_map_year")
        map_path = year_to_file.get(map_year)
        if map_path is None or not map_path.exists():
            st.warning(f"Map file not found for year {map_year}.")
        else:
            try:
                html_text = map_path.read_text(encoding="utf-8")
                components.html(html_text, height=720, scrolling=True)
            except Exception as e:
                st.error(f"Failed to load map HTML: {e}")
