"""
Code Map (approximate, current version):
- Climate data prep and merge helpers, then VARX diagnostics.
- Core KF/EM math (VARX init, KF, RTS smoother, diagnostics, EM M-step,
  `run_kf_em`) now lives in `trp.kalman_core` (imported below) rather than
  being defined locally -- see that module for the Kalman/EM implementation.
- Panel data config, per-country prep/EM caching, coefficient / diagnostics /
  Q-R trace plotting, LLM mapping + break-score panel builder, optional LLM
  overlay/refit outputs.

Debug tip:
- If output seems wrong, start from `trp.kalman_core.run_kf_em`,
  `build_em_break_score_panel`, `build_drop_quarters_dict_from_llm_points`,
  and final output flag blocks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import pickle
import os

try:
    from llm_break_visualization import (
        plot_structural_break_score_with_llm_overlay,
        plot_break_supported_ratio,
        plot_break_type_distribution,
        build_time_slice_maps,
    )
    _HAS_LLM_VIS = True
except Exception as _e:
    _HAS_LLM_VIS = False
    print(f"[WARN] llm_break_visualization import failed: {_e}")

# ---------- helpers ----------
def intersect_stable(a_list, b_list):
    b_index = {v: i for i, v in enumerate(b_list)}
    ia, ib, c_vals = [], [], []
    for i, v in enumerate(a_list):
        if v in b_index:
            ia.append(i)
            ib.append(b_index[v])
            c_vals.append(v)
    return c_vals, np.array(ia, dtype=int), np.array(ib, dtype=int)

# ---------- VARX benchmark ----------
def varx_rolling_predict(Y: np.ndarray,
                         Z: np.ndarray,
                         lags: int = 2,
                         window: int | None = None,
                         ridge: float = 1e-6):
    n, mY = Y.shape
    mX = Z.shape[1]
    m = lags * mY + mX
    Y_pred = np.full((n, mY), np.nan)
    e_raw  = np.full((n, mY), np.nan)

    for t in range(lags + 1, n - 1):
        if window is None:
            s0 = lags+1
        else:
            s0 = max(lags+1, t - 1 - window + 1)
        T = (t - 1) - s0 + 1
        if T <= 0:
            continue

        X_train = np.zeros((T, m))
        Y_train = np.zeros((T, mY))
        for k, s in enumerate(range(s0, t)):
            regY = np.concatenate([Y[s - i, :] for i in range(1, lags + 1)], axis=0)
            X_train[k, :] = np.concatenate([regY, Z[s-1, :]], axis=0)
            Y_train[k, :] = Y[s, :]

        A = X_train.T @ X_train + ridge * np.eye(m)
        B = X_train.T @ Y_train
        coef = np.linalg.solve(A, B)

        regY_next = np.concatenate([Y[t + 1 - i, :] for i in range(1, lags + 1)], axis=0)
        x_next = np.concatenate([regY_next, Z[t, :]], axis=0)

        yhat = x_next @ coef
        Y_pred[t + 1, :] = yhat
        e_raw[t + 1, :] = Y[t + 1, :] - yhat

    rmse = np.sqrt(np.nanmean(e_raw**2, axis=1))
    return Y_pred, e_raw, rmse

import sys
_KC_ROOT = Path(__file__).resolve().parent.parent
if str(_KC_ROOT) not in sys.path:
    sys.path.insert(0, str(_KC_ROOT))
from trp.kalman_core import (
    NumericalInstabilityError,
    init_from_varx_rolling,
    kalman_multilag_filter,
    kf_e_step_store,
    rts_smoother,
    compute_kf_smoother_diagnostics,
    em_m_step_update,
    run_kf_em,
)

# ============================================================
# New data: long panel + fixed exogenous variables (COMMODITY, ENSO, OIL)
# This section includes data loading + multi-country four-column heatmap panel_y_yhat_heatmap
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Dashboard run configuration ----------
PATH = str(Path(__file__).resolve().parent.parent / "analysis" / "gvar_panel_streamlit (8 + EGY + PER).csv")
COUNTRIES = [
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
countries_to_run = list(COUNTRIES)
MAX_EM_ITER = 50

ISO3_TO_COUNTRY = {
    "AFG": "Afghanistan", "ALB": "Albania", "DZA": "Algeria", "AND": "Andorra",
    "AGO": "Angola", "ATG": "Antigua and Barbuda", "ARG": "Argentina", "ARM": "Armenia",
    "AUS": "Australia", "AUT": "Austria", "AZE": "Azerbaijan", "BHS": "Bahamas",
    "BHR": "Bahrain", "BGD": "Bangladesh", "BRB": "Barbados", "BLR": "Belarus",
    "BEL": "Belgium", "BLZ": "Belize", "BEN": "Benin", "BTN": "Bhutan",
    "BOL": "Bolivia", "BIH": "Bosnia and Herzegovina", "BWA": "Botswana", "BRA": "Brazil",
    "BRN": "Brunei", "BGR": "Bulgaria", "BFA": "Burkina Faso", "BDI": "Burundi",
    "CPV": "Cabo Verde", "KHM": "Cambodia", "CMR": "Cameroon", "CAN": "Canada",
    "CAF": "Central African Republic", "TCD": "Chad", "CHL": "Chile", "CHN": "China",
    "COL": "Colombia", "COM": "Comoros", "COG": "Republic of the Congo", "COD": "Democratic Republic of the Congo",
    "CRI": "Costa Rica", "CIV": "Cote d'Ivoire", "HRV": "Croatia", "CUB": "Cuba",
    "CYP": "Cyprus", "CZE": "Czech Republic", "DNK": "Denmark", "DJI": "Djibouti",
    "DMA": "Dominica", "DOM": "Dominican Republic", "ECU": "Ecuador", "EGY": "Egypt",
    "SLV": "El Salvador", "GNQ": "Equatorial Guinea", "ERI": "Eritrea", "EST": "Estonia",
    "SWZ": "Eswatini", "ETH": "Ethiopia", "FJI": "Fiji", "FIN": "Finland",
    "FRA": "France", "GAB": "Gabon", "GMB": "Gambia", "GEO": "Georgia",
    "DEU": "Germany", "GHA": "Ghana", "GRC": "Greece", "GRD": "Grenada",
    "GTM": "Guatemala", "GIN": "Guinea", "GNB": "Guinea-Bissau", "GUY": "Guyana",
    "HTI": "Haiti", "HND": "Honduras", "HUN": "Hungary", "ISL": "Iceland",
    "IND": "India", "IDN": "Indonesia", "IRN": "Iran", "IRQ": "Iraq",
    "IRL": "Ireland", "ISR": "Israel", "ITA": "Italy", "JAM": "Jamaica",
    "JPN": "Japan", "JOR": "Jordan", "KAZ": "Kazakhstan", "KEN": "Kenya",
    "KIR": "Kiribati", "KWT": "Kuwait", "KGZ": "Kyrgyzstan", "LAO": "Laos",
    "LVA": "Latvia", "LBN": "Lebanon", "LSO": "Lesotho", "LBR": "Liberia",
    "LBY": "Libya", "LIE": "Liechtenstein", "LTU": "Lithuania", "LUX": "Luxembourg",
    "MDG": "Madagascar", "MWI": "Malawi", "MYS": "Malaysia", "MDV": "Maldives",
    "MLI": "Mali", "MLT": "Malta", "MHL": "Marshall Islands", "MRT": "Mauritania",
    "MUS": "Mauritius", "MEX": "Mexico", "FSM": "Micronesia", "MDA": "Moldova",
    "MCO": "Monaco", "MNG": "Mongolia", "MNE": "Montenegro", "MAR": "Morocco",
    "MOZ": "Mozambique", "MMR": "Myanmar", "NAM": "Namibia", "NRU": "Nauru",
    "NPL": "Nepal", "NLD": "Netherlands", "NZL": "New Zealand", "NIC": "Nicaragua",
    "NER": "Niger", "NGA": "Nigeria", "PRK": "North Korea", "MKD": "North Macedonia",
    "NOR": "Norway", "OMN": "Oman", "PAK": "Pakistan", "PLW": "Palau",
    "PAN": "Panama", "PNG": "Papua New Guinea", "PRY": "Paraguay", "PER": "Peru",
    "PHL": "Philippines", "POL": "Poland", "PRT": "Portugal", "QAT": "Qatar",
    "ROU": "Romania", "RUS": "Russia", "RWA": "Rwanda", "KNA": "Saint Kitts and Nevis",
    "LCA": "Saint Lucia", "VCT": "Saint Vincent and the Grenadines", "WSM": "Samoa",
    "SMR": "San Marino", "STP": "Sao Tome and Principe", "SAU": "Saudi Arabia",
    "SEN": "Senegal", "SRB": "Serbia", "SYC": "Seychelles", "SLE": "Sierra Leone",
    "SGP": "Singapore", "SVK": "Slovakia", "SVN": "Slovenia", "SLB": "Solomon Islands",
    "SOM": "Somalia", "ZAF": "South Africa", "KOR": "South Korea", "SSD": "South Sudan",
    "ESP": "Spain", "LKA": "Sri Lanka", "SDN": "Sudan", "SUR": "Suriname",
    "SWE": "Sweden", "CHE": "Switzerland", "SYR": "Syria", "TJK": "Tajikistan",
    "TZA": "Tanzania", "THA": "Thailand", "TLS": "Timor-Leste", "TGO": "Togo",
    "TON": "Tonga", "TTO": "Trinidad and Tobago", "TUN": "Tunisia", "TUR": "Turkey",
    "TKM": "Turkmenistan", "TUV": "Tuvalu", "UGA": "Uganda", "UKR": "Ukraine",
    "ARE": "United Arab Emirates", "GBR": "United Kingdom", "USA": "United States",
    "URY": "Uruguay", "UZB": "Uzbekistan", "VUT": "Vanuatu", "VEN": "Venezuela",
    "VNM": "Vietnam", "YEM": "Yemen", "ZMB": "Zambia", "ZWE": "Zimbabwe"
}


def country_display_name(iso3: str, iso3_to_country: dict | None = None) -> str:
    """ISO3 -> full country name (for titles, heatmap y-axis, etc.)."""
    if not iso3_to_country:
        return str(iso3)
    return str(iso3_to_country.get(iso3, iso3))


def style_quarter_axis_every_four_quarters(ax) -> None:
    """Set x-axis major ticks/grid every 4 quarters (12 months), label as YYYY-MM."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, which="major", axis="x", alpha=0.35)


def _fig_save_or_show(fig, pdf: PdfPages | None) -> None:
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show(block=False)


# Shared caches: reduce repeated CSV loads and repeated EM runs.
_PANEL_DF_CACHE: dict[tuple, pd.DataFrame] = {}
_EM_RESULT_CACHE: dict[tuple, dict] = {}
_COUNTRY_PREP_CACHE: dict[tuple, dict | None] = {}
NUMERICAL_DIAGNOSTICS: list[dict] = []


def _spec_label(exo_vars: list[str] | tuple[str, ...]) -> str:
    return " + ".join(exo_vars) if exo_vars else "(none)"


def _record_numerical_diagnostic(
    *,
    country: str,
    exo_vars: list[str] | tuple[str, ...],
    status: str,
    failure: NumericalInstabilityError | None = None,
) -> None:
    NUMERICAL_DIAGNOSTICS.append(
        {
            "country": country,
            "external_variables": _spec_label(exo_vars),
            "status": status,
            "operation": getattr(failure, "operation", None),
            "matrix": getattr(failure, "matrix", None),
            "reason": getattr(failure, "reason", None),
            "em_iteration": getattr(failure, "em_iteration", None),
            "time_index": getattr(failure, "time_index", None),
            "quarter": getattr(failure, "quarter", None),
            "first_failure": "-" if failure is None else str(failure),
        }
    )


def print_numerical_diagnostics() -> None:
    if not NUMERICAL_DIAGNOSTICS:
        return
    df_diag = pd.DataFrame(NUMERICAL_DIAGNOSTICS).drop_duplicates(
        subset=["country", "external_variables", "status", "first_failure"],
        keep="first",
    )
    print("\n== Kalman/EM numerical diagnostics ==")
    cols = ["country", "external_variables", "status", "first_failure"]
    print(df_diag[cols].sort_values(["external_variables", "country"]).to_string(index=False))


def _load_panel_df_cached(PATH: str, COL_COUNTRY: str, COL_TIME: str) -> pd.DataFrame:
    key = (str(PATH), COL_COUNTRY, COL_TIME)
    if key in _PANEL_DF_CACHE:
        return _PANEL_DF_CACHE[key]
    df = pd.read_csv(PATH)
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df = df.dropna(subset=[COL_TIME]).sort_values([COL_COUNTRY, COL_TIME]).copy()
    _PANEL_DF_CACHE[key] = df
    return df


def _prepare_country_panel_cached(
    PATH: str,
    country: str,
    COL_COUNTRY: str,
    COL_TIME: str,
    ENDO: list[str],
    EXO: list[str],
    min_T: int,
):
    pkey = (str(PATH), country, COL_COUNTRY, COL_TIME, tuple(ENDO), tuple(EXO), int(min_T))
    if pkey in _COUNTRY_PREP_CACHE:
        return _COUNTRY_PREP_CACHE[pkey]

    df = _load_panel_df_cached(PATH, COL_COUNTRY, COL_TIME)
    ENDO_use = [c for c in ENDO if c in df.columns and df[c].notna().any()]
    missing_exo = [c for c in EXO if c not in df.columns or not df[c].notna().any()]
    if missing_exo:
        _COUNTRY_PREP_CACHE[pkey] = None
        return None
    EXO_use = list(EXO)
    if not ENDO_use or not EXO_use:
        _COUNTRY_PREP_CACHE[pkey] = None
        return None

    g = df[df[COL_COUNTRY] == country]
    if g.empty:
        _COUNTRY_PREP_CACHE[pkey] = None
        return None
    g = g.sort_values(COL_TIME)
    mask = (
        np.isfinite(g[ENDO_use].to_numpy(float)).all(axis=1)
        & np.isfinite(g[EXO_use].to_numpy(float)).all(axis=1)
    )
    g = g.loc[mask].copy().reset_index(drop=True)
    if len(g) < min_T:
        _COUNTRY_PREP_CACHE[pkey] = None
        return None

    Yd = g[ENDO_use].to_numpy(float)
    mu_y = np.nanmean(Yd, axis=0)
    sd_y = np.nanstd(Yd, axis=0) + 1e-8
    Yd = (Yd - mu_y) / sd_y

    Xd = g[EXO_use].to_numpy(float)
    mu_x = np.nanmean(Xd, axis=0)
    sd_x = np.nanstd(Xd, axis=0) + 1e-8
    Xd = (Xd - mu_x) / sd_x

    out = {
        "g": g,
        "Yd": Yd,
        "Xd": Xd,
        "quarters": pd.to_datetime(g[COL_TIME].values),
        "ENDO_use": ENDO_use,
        "EXO_use": EXO_use,
    }
    _COUNTRY_PREP_CACHE[pkey] = out
    return out


def _run_kf_em_cached(
    PATH: str,
    country: str,
    COL_COUNTRY: str,
    COL_TIME: str,
    ENDO: list[str],
    EXO: list[str],
    lags: int,
    min_T: int,
    max_em_iter: int,
    exo_mode: str = "all",  # "all" | "none"
    update_P0: bool = True,
):
    """Production EM entry point.

    update_P0=True (default): P0 is refreshed each EM iteration from the
    first valid smoothed covariance, in addition to theta0 (see
    trp.kalman_core.run_kf_em). Adopted as the default after comparing
    against the fixed-P0 path with analysis/validation/kf_em_diagnostics.py
    across all 12 dashboard countries: no divergence, NaN/Inf, or
    singularity issues, and no material change in the final one-step
    residual objective -- only a higher achieved log-likelihood and a much
    tighter (data-informed) P0. Pass update_P0=False to fall back to the
    original fixed-P0 behavior.
    """
    prep = _prepare_country_panel_cached(
        PATH=PATH,
        country=country,
        COL_COUNTRY=COL_COUNTRY,
        COL_TIME=COL_TIME,
        ENDO=ENDO,
        EXO=EXO,
        min_T=min_T,
    )
    if prep is None:
        return None, None

    Yd = prep["Yd"]
    Xd = prep["Xd"]
    Z = Xd if exo_mode == "all" else np.zeros((len(Yd), 0))
    ekey = (
        str(PATH),
        country,
        tuple(prep["ENDO_use"]),
        tuple(prep["EXO_use"]) if exo_mode == "all" else tuple(),
        int(lags),
        int(max_em_iter),
        bool(update_P0),
    )
    if ekey not in _EM_RESULT_CACHE:
        try:
            _EM_RESULT_CACHE[ekey] = run_kf_em(
                Y=Yd,
                Z=Z,
                lags=lags,
                window=40,
                max_em_iter=max_em_iter,
                tol=1e-4,
                em_damping=0.0,
                verbose=False,
                update_P0=update_P0,
            )
            _record_numerical_diagnostic(country=country, exo_vars=prep["EXO_use"], status="OK")
        except NumericalInstabilityError as exc:
            quarters = prep.get("quarters")
            if exc.quarter is None and exc.time_index is not None and quarters is not None and exc.time_index < len(quarters):
                exc.quarter = pd.Timestamp(quarters[exc.time_index]).to_period("Q")
                exc.args = (str(exc),)
            _record_numerical_diagnostic(country=country, exo_vars=prep["EXO_use"], status="FAILED", failure=exc)
            return prep, None
    return prep, _EM_RESULT_CACHE[ekey]


# ---------- Column names (edit here if your file uses different names) ----------
COL_COUNTRY = "country"
COL_TIME = "quarter"
ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
EXO = ["ENSO", "OIL_YoY"]

lags = 1

# ---------- Read data ----------
df = _load_panel_df_cached(PATH, COL_COUNTRY, COL_TIME)

endo_use = [c for c in ENDO if df[c].notna().any()]
if len(endo_use) < len(ENDO):
    dropped = set(ENDO) - set(endo_use)
    print(f"[INFO] drop all-NaN endogenous columns: {dropped}")
ENDO = endo_use

if len(COUNTRIES) == 0:
    countries_to_run = sorted(df[COL_COUNTRY].dropna().unique())
    print(f"[INFO] COUNTRIES empty → run all: n={len(countries_to_run)}")
else:
    countries_to_run = list(COUNTRIES)

def plot_coeff_trajectories_countries_em(
    PATH,
    COUNTRIES,
    COL_COUNTRY="country",
    COL_TIME="quarter",
    ENDO=None,
    EXO=None,
    lags=1,
    min_T=None,
    max_em_iter=MAX_EM_ITER,
    iso3_to_country: dict | None = None,
    pdf: PdfPages | None = None,
):
    """If pdf=PdfPages(...), write each figure to that file; otherwise call plt.show()."""

    if ENDO is None:
        ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
    if EXO is None:
        EXO = ["ENSO", "OIL_YoY"]
    if min_T is None:
        min_T = lags + 5

    df = _load_panel_df_cached(PATH, COL_COUNTRY, COL_TIME)
    ENDO_use = [c for c in ENDO if c in df.columns and df[c].notna().any()]
    EXO_use = [c for c in EXO if c in df.columns and df[c].notna().any()]


    for country in COUNTRIES:
        cname = country_display_name(country, iso3_to_country)
        prep, res = _run_kf_em_cached(
            PATH=PATH,
            country=country,
            COL_COUNTRY=COL_COUNTRY,
            COL_TIME=COL_TIME,
            ENDO=ENDO_use,
            EXO=EXO_use,
            lags=lags,
            min_T=min_T,
            max_em_iter=max_em_iter,
            exo_mode="all",
        )
        if prep is None or res is None:
            print(f"[SKIP] {cname}: missing/short panel or EM failed")
            continue
        g = prep["g"]

        theta_est = res["theta_est"]  # (T, p)
        Tn, p = theta_est.shape
        mY = len(ENDO_use)
        mX = len(EXO_use)
        m = lags * mY + mX

        valid = ~np.isnan(theta_est).any(axis=1)
        if valid.sum() <= 2:
            print(f"[SKIP] {cname}: no valid theta trajectory")
            continue

        quarters = pd.to_datetime(g[COL_TIME].values)

        # ----- Plot 1: diagonal own-lag coefficients only (GDP<-L*.GDP, CPI<-L*.CPI, ...) -----
        diag_k, diag_lbl = [], []
        for eq in range(mY):
            for li in range(1, lags + 1):
                off = (li - 1) * mY + eq
                k = eq * m + off
                if k < p:
                    diag_k.append(k)
                    diag_lbl.append(f"{ENDO_use[eq]}<-L{li}.{ENDO_use[eq]}")

        plt.figure(figsize=(12, 5))
        for k, lbl in zip(diag_k, diag_lbl):
            plt.plot(
                quarters[valid],
                theta_est[valid, k],
                linewidth=1.3,
                marker="o",
                markersize=2,
                alpha=0.9,
                label=lbl,
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
        _fig_save_or_show(plt.gcf(), pdf)

        # ----- Plot 2: ENSO-related coefficients only (one line per endogenous equation) -----
        if "ENSO" not in EXO_use:
            print(f"[INFO] {cname}: no ENSO column in EXO_use, skip ENSO-only plot")
        else:
            idx_enso = EXO_use.index("ENSO")
            enso_k, enso_lbl = [], []
            for eq in range(mY):
                off = lags * mY + idx_enso
                k = eq * m + off
                if k < p:
                    enso_k.append(k)
                    enso_lbl.append(f"{ENDO_use[eq]}<-ENSO")

            plt.figure(figsize=(12, 5))
            for k, lbl in zip(enso_k, enso_lbl):
                plt.plot(
                    quarters[valid],
                    theta_est[valid, k],
                    linewidth=1.3,
                    linestyle="-",
                    marker="o",
                    markersize=2,
                    alpha=0.9,
                    label=lbl,
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
            _fig_save_or_show(plt.gcf(), pdf)


def plot_em_diagnostics_countries(
    PATH,
    COUNTRIES,
    COL_COUNTRY="country",
    COL_TIME="quarter",
    ENDO=None,
    EXO=None,
    lags=1,
    min_T=None,
    max_em_iter=MAX_EM_ITER,
    iso3_to_country: dict | None = None,
    pdf: PdfPages | None = None,
):
    """
    For each country, plot three diagnostics after training:
    - innovation score: v_t' S_t^{-1} v_t
    - coefficient change: ||beta_smooth,t - beta_smooth,t-1||
    - filter–smoother gap: ||beta_smooth,t - beta_filt,t||
    Data come from run_kf_em outputs:
    innovation_score / coefficient_change / filter_smoother_gap.
    If pdf=PdfPages(...), write each figure to PDF instead of showing.
    """
    if ENDO is None:
        ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
    if EXO is None:
        EXO = ["ENSO", "OIL_YoY"]
    if min_T is None:
        min_T = lags + 5

    df = _load_panel_df_cached(PATH, COL_COUNTRY, COL_TIME)
    ENDO_use = [c for c in ENDO if c in df.columns and df[c].notna().any()]
    EXO_use = [c for c in EXO if c in df.columns and df[c].notna().any()]

    for country in COUNTRIES:
        cname = country_display_name(country, iso3_to_country)
        prep, res = _run_kf_em_cached(
            PATH=PATH,
            country=country,
            COL_COUNTRY=COL_COUNTRY,
            COL_TIME=COL_TIME,
            ENDO=ENDO_use,
            EXO=EXO_use,
            lags=lags,
            min_T=min_T,
            max_em_iter=max_em_iter,
            exo_mode="all",
        )
        if prep is None or res is None:
            print(f"[SKIP] {cname}: missing/short panel or EM failed")
            continue
        g = prep["g"]

        innov = res.get("innovation_score")
        dcoef = res.get("coefficient_change")
        fsgap = res.get("filter_smoother_gap")
        if innov is None or len(innov) != len(g):
            print(f"[SKIP] {cname}: missing diagnostics")
            continue

        quarters = pd.to_datetime(g[COL_TIME].values)
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(quarters, innov, color="C0", linewidth=1.2)
        axes[0].set_ylabel("Innovation score (v' S^{-1} v)")
        axes[0].grid(True, which="major", axis="y", alpha=0.25)
        axes[0].set_title(f"{cname} — EM diagnostics")

        axes[1].plot(quarters, dcoef, color="C1", linewidth=1.2)
        axes[1].set_ylabel("Coefficient change ||b_s,t - b_s,t-1||")
        axes[1].grid(True, which="major", axis="y", alpha=0.25)

        axes[2].plot(quarters, fsgap, color="C2", linewidth=1.2)
        axes[2].set_ylabel("Filter–smoother gap ||b_s,t - b_f,t||")
        axes[2].set_xlabel("Quarter")
        axes[2].grid(True, which="major", axis="y", alpha=0.25)

        style_quarter_axis_every_four_quarters(axes[2])
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        _fig_save_or_show(fig, pdf)


def plot_em_qr_traces_countries(
    PATH,
    COUNTRIES,
    COL_COUNTRY="country",
    COL_TIME="quarter",
    ENDO=None,
    EXO=None,
    lags=1,
    min_T=None,
    max_em_iter=MAX_EM_ITER,
    iso3_to_country: dict | None = None,
    pdf: PdfPages | None = None,
):
    """
    Plot EM iteration traces by country:
    - trace(Q)
    - trace(R)
    If pdf=PdfPages(...), write each figure to PDF instead of showing.
    """
    if ENDO is None:
        ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
    if EXO is None:
        EXO = ["ENSO", "OIL_YoY"]
    if min_T is None:
        min_T = lags + 5

    df = _load_panel_df_cached(PATH, COL_COUNTRY, COL_TIME)
    ENDO_use = [c for c in ENDO if c in df.columns and df[c].notna().any()]
    EXO_use = [c for c in EXO if c in df.columns and df[c].notna().any()]

    for country in COUNTRIES:
        cname = country_display_name(country, iso3_to_country)
        prep, res = _run_kf_em_cached(
            PATH=PATH,
            country=country,
            COL_COUNTRY=COL_COUNTRY,
            COL_TIME=COL_TIME,
            ENDO=ENDO_use,
            EXO=EXO_use,
            lags=lags,
            min_T=min_T,
            max_em_iter=max_em_iter,
            exo_mode="all",
        )
        if prep is None or res is None:
            print(f"[SKIP] {cname}: missing/short panel or EM failed")
            continue

        hist = res.get("em_history", {})
        q_tr = np.asarray(hist.get("trace_Q", []), dtype=float)
        r_tr = np.asarray(hist.get("trace_R", []), dtype=float)
        it = np.arange(1, len(q_tr) + 1)

        if len(it) == 0:
            print(f"[SKIP] {cname}: empty EM history")
            continue

        plt.figure(figsize=(9, 4.5))
        plt.plot(it, q_tr, "-o", linewidth=1.4, markersize=3, label="trace(Q)")
        plt.plot(it, r_tr, "-s", linewidth=1.4, markersize=3, label="trace(R)")
        plt.title(f"{cname} - EM iteration traces")
        plt.xlabel("EM iteration")
        plt.ylabel("trace value")
        plt.grid(alpha=0.3)
        plt.legend(frameon=False)
        plt.tight_layout()
        _fig_save_or_show(plt.gcf(), pdf)


def build_em_offline_plot_pack(
    PATH,
    COUNTRIES,
    COL_COUNTRY="country",
    COL_TIME="quarter",
    ENDO=None,
    EXO=None,
    lags=1,
    min_T=None,
    max_em_iter=MAX_EM_ITER,
    iso3_to_country: dict | None = None,
):
    """
    Build a self-contained plotting pack so figures can be rendered offline
    from pickle without rerunning run_kf_em.
    """
    if ENDO is None:
        ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
    if EXO is None:
        EXO = ["ENSO", "OIL_YoY"]
    if min_T is None:
        min_T = lags + 5

    df = _load_panel_df_cached(PATH, COL_COUNTRY, COL_TIME)
    ENDO_use = [c for c in ENDO if c in df.columns and df[c].notna().any()]
    EXO_use = [c for c in EXO if c in df.columns and df[c].notna().any()]
    mY = len(ENDO_use)
    mX = len(EXO_use)
    m = lags * mY + mX if mY > 0 else 0

    pack = {
        "meta": {
            "path": PATH,
            "countries": list(COUNTRIES),
            "ENDO_use": ENDO_use,
            "EXO_use": EXO_use,
            "lags": lags,
            "max_em_iter": max_em_iter,
        },
        "per_country": {},
    }
    if mY == 0 or mX == 0:
        return pack

    for country in COUNTRIES:
        cname = country_display_name(country, iso3_to_country)
        prep, res = _run_kf_em_cached(
            PATH=PATH,
            country=country,
            COL_COUNTRY=COL_COUNTRY,
            COL_TIME=COL_TIME,
            ENDO=ENDO_use,
            EXO=EXO_use,
            lags=lags,
            min_T=min_T,
            max_em_iter=max_em_iter,
            exo_mode="all",
        )
        if prep is None or res is None:
            continue
        g = prep["g"]

        theta_est = np.asarray(res.get("theta_est"), dtype=float)
        if theta_est.ndim != 2 or theta_est.shape[0] != len(g):
            continue

        valid = ~np.isnan(theta_est).any(axis=1)
        q_all = pd.to_datetime(g[COL_TIME].values)
        q_valid = q_all[valid]
        if len(q_valid) <= 2:
            continue

        p = theta_est.shape[1]
        diag_series = []
        for eq in range(mY):
            for li in range(1, lags + 1):
                off = (li - 1) * mY + eq
                k = eq * m + off
                if k < p:
                    diag_series.append(
                        {
                            "label": f"{ENDO_use[eq]}<-L{li}.{ENDO_use[eq]}",
                            "values": theta_est[valid, k].astype(float).tolist(),
                        }
                    )

        enso_series = []
        if "ENSO" in EXO_use:
            idx_enso = EXO_use.index("ENSO")
            for eq in range(mY):
                off = lags * mY + idx_enso
                k = eq * m + off
                if k < p:
                    enso_series.append(
                        {
                            "label": f"{ENDO_use[eq]}<-ENSO",
                            "values": theta_est[valid, k].astype(float).tolist(),
                        }
                    )

        hist = res.get("em_history", {})
        q_tr = np.asarray(hist.get("trace_Q", []), dtype=float)
        r_tr = np.asarray(hist.get("trace_R", []), dtype=float)
        innov = np.asarray(res.get("innovation_score"), dtype=float)
        dcoef = np.asarray(res.get("coefficient_change"), dtype=float)
        fsgap = np.asarray(res.get("filter_smoother_gap"), dtype=float)
        if len(innov) != len(q_all) or len(dcoef) != len(q_all) or len(fsgap) != len(q_all):
            continue

        pack["per_country"][country] = {
            "country_name": cname,
            "coeff_quarters": [pd.Timestamp(x).isoformat() for x in q_valid],
            "diag_quarters": [pd.Timestamp(x).isoformat() for x in q_all],
            "diag_coeff_series": diag_series,
            "enso_coeff_series": enso_series,
            "innovation_score": innov.astype(float).tolist(),
            "coefficient_change": dcoef.astype(float).tolist(),
            "filter_smoother_gap": fsgap.astype(float).tolist(),
            "trace_Q": q_tr.astype(float).tolist(),
            "trace_R": r_tr.astype(float).tolist(),
            "breakscore": innov.astype(float).tolist(),
        }

    return pack


def plot_offline_core_from_pack(
    offline_pack: dict,
    COUNTRIES: list[str],
    pdf: PdfPages | None = None,
):
    per_country = offline_pack.get("per_country", {})
    for country in COUNTRIES:
        d = per_country.get(country)
        if not d:
            continue
        cname = d.get("country_name", country)
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
            _fig_save_or_show(plt.gcf(), pdf)

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
            _fig_save_or_show(plt.gcf(), pdf)

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
            _fig_save_or_show(fig, pdf)

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
            _fig_save_or_show(plt.gcf(), pdf)


def plot_offline_breakscore_from_pack(
    offline_pack: dict,
    COUNTRIES: list[str],
    drop_quarters_dict: dict[str, list[pd.Timestamp]] | None = None,
    pdf: PdfPages | None = None,
    save_dir: str | None = None,
):
    per_country = offline_pack.get("per_country", {})
    for country in COUNTRIES:
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
        if save_dir:
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"breakscore_{country}.jpg", dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            _fig_save_or_show(fig, pdf)


if os.environ.get("GVAR_IMPORT_ONLY") != "1":
    # Country trajectory plots: full countries_to_run list.
    # Titles/y-axis labels use full names from ISO3_TO_COUNTRY.
    # If non-None, all three plot groups are written into one PDF.
    DASH_INPUT_DIR = Path("Dash_Input")
    DASH_INPUT_DIR.mkdir(parents=True, exist_ok=True)


    def _dash_out(name: str) -> str:
        return str(DASH_INPUT_DIR / name)


    PLOTS_PDF_PATH: str | None = _dash_out("GVAR_LLM_EM_plots.pdf")
    RESULTS_PICKLE_PATH = _dash_out("gvar_pipeline_results.pkl")
    COUNTRIES_FOR_COEF_PLOT = list(countries_to_run)
    COUNTRIES_FOR_DIAG_PLOT = list(countries_to_run)
    COUNTRIES_FOR_QR_PLOT = list(countries_to_run)
    _plot_kw = dict(
        PATH=PATH,
        COL_COUNTRY=COL_COUNTRY,
        COL_TIME=COL_TIME,
        ENDO=ENDO,
        EXO=EXO,
        lags=lags,
        max_em_iter=MAX_EM_ITER,
        iso3_to_country=ISO3_TO_COUNTRY,
    )


    def _map_llm_country_to_iso3(llm_df: pd.DataFrame, iso3_to_country: dict) -> pd.DataFrame:
        """Map country names in LLM table (full name / ISO3 mixed) into ISO3 codes."""
        out = llm_df.copy()
        rev = {str(v).strip().lower(): k for k, v in iso3_to_country.items()}

        def _one(v):
            s = str(v).strip()
            su = s.upper()
            if su in iso3_to_country:
                return su
            return rev.get(s.lower(), s)

        out["country"] = out["country"].apply(_one)
        return out


    def build_em_break_score_panel(
        PATH,
        COUNTRIES,
        COL_COUNTRY="country",
        COL_TIME="quarter",
        ENDO=None,
        EXO=None,
        lags=1,
        min_T=None,
        max_em_iter=MAX_EM_ITER,
    ) -> pd.DataFrame:
        """
        Build a quarterly break-score panel from run_kf_em innovation_score.
        Output columns: country, quarter, year, score
        """
        if ENDO is None:
            ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
        if EXO is None:
            EXO = ["ENSO", "OIL_YoY"]
        if min_T is None:
            min_T = lags + 5

        df = _load_panel_df_cached(PATH, COL_COUNTRY, COL_TIME)

        ENDO_use = [c for c in ENDO if c in df.columns and df[c].notna().any()]
        EXO_use = [c for c in EXO if c in df.columns and df[c].notna().any()]
        rows = []

        for country in COUNTRIES:
            prep, res = _run_kf_em_cached(
                PATH=PATH,
                country=country,
                COL_COUNTRY=COL_COUNTRY,
                COL_TIME=COL_TIME,
                ENDO=ENDO_use,
                EXO=EXO_use,
                lags=lags,
                min_T=min_T,
                max_em_iter=max_em_iter,
                exo_mode="all",
            )
            if prep is None or res is None:
                continue
            g = prep["g"]

            innov = np.asarray(res.get("innovation_score"), dtype=float)
            if innov.shape[0] != len(g):
                continue

            q = pd.to_datetime(g[COL_TIME].values)
            for qt, sc in zip(q, innov):
                rows.append(
                    {
                        "country": country,
                        "quarter": qt,
                        "year": int(pd.Timestamp(qt).year),
                        "score": float(sc) if np.isfinite(sc) else np.nan,
                    }
                )

        return pd.DataFrame(rows)


    def build_em_composite_break_score_panel(
        PATH,
        COUNTRIES,
        COL_COUNTRY="country",
        COL_TIME="quarter",
        ENDO=None,
        EXO=None,
        lags=1,
        min_T=None,
        max_em_iter=MAX_EM_ITER,
    ):
        """
        Build quarter-level panel with standardized diagnostics and composite score:
        composite_score = (z_innovation + z_coef_change + z_filter_smoother_gap) / 3
        """
        if ENDO is None:
            ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
        if EXO is None:
            EXO = ["ENSO", "OIL_YoY"]
        if min_T is None:
            min_T = lags + 5

        df = _load_panel_df_cached(PATH, COL_COUNTRY, COL_TIME)

        ENDO_use = [c for c in ENDO if c in df.columns and df[c].notna().any()]
        EXO_use = [c for c in EXO if c in df.columns and df[c].notna().any()]
        if not ENDO_use or not EXO_use:
            return pd.DataFrame(
                columns=[
                    "country",
                    "quarter",
                    "year",
                    "innovation_score",
                    "coefficient_change",
                    "filter_smoother_gap",
                    "z_innovation_score",
                    "z_coefficient_change",
                    "z_filter_smoother_gap",
                    "composite_score",
                ]
            )

        def _safe_z(a: np.ndarray) -> np.ndarray:
            x = np.asarray(a, dtype=float)
            mu = np.nanmean(x)
            sd = np.nanstd(x)
            if not np.isfinite(sd) or sd < 1e-12:
                return np.zeros_like(x, dtype=float)
            return (x - mu) / sd

        rows: list[dict] = []
        for country in COUNTRIES:
            prep, res = _run_kf_em_cached(
                PATH=PATH,
                country=country,
                COL_COUNTRY=COL_COUNTRY,
                COL_TIME=COL_TIME,
                ENDO=ENDO_use,
                EXO=EXO_use,
                lags=lags,
                min_T=min_T,
                max_em_iter=max_em_iter,
                exo_mode="all",
            )
            if prep is None or res is None:
                continue
            g = prep["g"]

            innov = np.asarray(res.get("innovation_score"), dtype=float)
            dcoef = np.asarray(res.get("coefficient_change"), dtype=float)
            fsgap = np.asarray(res.get("filter_smoother_gap"), dtype=float)
            if len(innov) != len(g) or len(dcoef) != len(g) or len(fsgap) != len(g):
                continue

            z_innov = _safe_z(innov)
            z_dcoef = _safe_z(dcoef)
            z_fsgap = _safe_z(fsgap)
            comp = (z_innov + z_dcoef + z_fsgap) / 3.0

            for qt, a, b, c, za, zb, zc, sc in zip(
                g[COL_TIME].values, innov, dcoef, fsgap, z_innov, z_dcoef, z_fsgap, comp
            ):
                rows.append(
                    {
                        "country": country,
                        "quarter": pd.Timestamp(qt),
                        "year": int(pd.Timestamp(qt).year),
                        "innovation_score": float(a) if np.isfinite(a) else np.nan,
                        "coefficient_change": float(b) if np.isfinite(b) else np.nan,
                        "filter_smoother_gap": float(c) if np.isfinite(c) else np.nan,
                        "z_innovation_score": float(za) if np.isfinite(za) else np.nan,
                        "z_coefficient_change": float(zb) if np.isfinite(zb) else np.nan,
                        "z_filter_smoother_gap": float(zc) if np.isfinite(zc) else np.nan,
                        "composite_score": float(sc) if np.isfinite(sc) else np.nan,
                    }
                )

        return pd.DataFrame(rows)


    def build_llm_input_candidates_from_composite(
        composite_df: pd.DataFrame,
        country_col: str = "country",
        year_col: str = "year",
        score_col: str = "composite_score",
        top_k_peaks: int = 4,
    ) -> list[dict]:
        """
        Build LLM input list:
        [
            {"country": "CHL", "years": [...]},
            ...
        ]
        """
        if composite_df is None or composite_df.empty:
            return []

        df = composite_df.dropna(subset=[country_col, year_col, score_col]).copy()
        if df.empty:
            return []

        out: list[dict] = []
        for country, g in df.groupby(country_col):
            ys = (
                g.groupby(year_col, as_index=False)[score_col]
                .max()
                .sort_values(year_col)
                .reset_index(drop=True)
            )
            if ys.empty:
                out.append({"country": country, "years": []})
                continue

            s_prev = ys[score_col].shift(1)
            s_next = ys[score_col].shift(-1)
            peak_mask = (ys[score_col] > s_prev) & (ys[score_col] > s_next)
            peaks = ys.loc[peak_mask].copy()
            if peaks.empty:
                out.append({"country": country, "years": []})
                continue

            top_peaks = peaks.nlargest(top_k_peaks, score_col)
            cand_years: set[int] = set()
            for y in top_peaks[year_col].astype(int).tolist():
                cand_years.update([y - 1, y, y + 1])

            out.append({"country": country, "years": sorted(cand_years)})

        return out


    # ============================================================
    # LLM integration: break-score overlay + summary charts + time-slice maps
    # ============================================================
    ENABLE_LLM_INTEGRATION = True
    LLM_RESULTS_CSV = "merged_clean_dataset.csv"  # replace with your prepared CSV
    SHOW_LLM_BREAK_OVERLAY = True
    LLM_OVERLAY_FIG_PATH = _dash_out("gvar_breakscore_llm_overlay.png")
    LLM_STATS_FIG_PATH = _dash_out("gvar_llm_stats.png")
    LLM_MAP_OUTPUT_DIR = _dash_out("gvar_llm_time_slice_maps")
    INPUT_LLM_PICKLE_PATH = _dash_out("input_LLM.pkl")

    pickle_bundle: dict = {
        "config": {
            "PATH": PATH,
            "COUNTRIES_FOR_COEF_PLOT": COUNTRIES_FOR_COEF_PLOT,
            "COUNTRIES_FOR_DIAG_PLOT": COUNTRIES_FOR_DIAG_PLOT,
            "COUNTRIES_FOR_QR_PLOT": COUNTRIES_FOR_QR_PLOT,
            "PLOTS_PDF_PATH": PLOTS_PDF_PATH,
            "plot_kw": _plot_kw,
            "COL_COUNTRY": COL_COUNTRY,
            "COL_TIME": COL_TIME,
            "ENDO": ENDO,
            "EXO": EXO,
            "lags": lags,
            "max_em_iter": MAX_EM_ITER,
            "iso3_to_country": ISO3_TO_COUNTRY,
            "SHOW_LLM_BREAK_OVERLAY": SHOW_LLM_BREAK_OVERLAY,
            "LLM_OVERLAY_FIG_PATH": LLM_OVERLAY_FIG_PATH,
            "LLM_STATS_FIG_PATH": LLM_STATS_FIG_PATH,
            "LLM_MAP_OUTPUT_DIR": LLM_MAP_OUTPUT_DIR,
        }
    }

    # Build offline plotting data (base path) so pickle can render without rerun.
    pickle_bundle["offline_plot_data"] = {
        "base": build_em_offline_plot_pack(
            PATH=PATH,
            COUNTRIES=COUNTRIES_FOR_COEF_PLOT,
            COL_COUNTRY=COL_COUNTRY,
            COL_TIME=COL_TIME,
            ENDO=ENDO,
            EXO=EXO,
            lags=lags,
            max_em_iter=MAX_EM_ITER,
            iso3_to_country=ISO3_TO_COUNTRY,
        )
    }

    if ENABLE_LLM_INTEGRATION:
        if not _HAS_LLM_VIS:
            print("[WARN] skip LLM integration: llm_break_visualization not available")
        elif not Path(LLM_RESULTS_CSV).exists():
            print(f"[WARN] skip LLM integration: file not found -> {LLM_RESULTS_CSV}")
        else:
            llm_raw_df = pd.read_csv(LLM_RESULTS_CSV)
            llm_df = _map_llm_country_to_iso3(llm_raw_df, ISO3_TO_COUNTRY)
            llm_df = llm_df[llm_df["country"].isin(list(countries_to_run))].copy()
            manual_rows = []
            for _country in ["IND", "IDN"]:
                for _year in [2020, 2021, 2022]:
                    manual_rows.append(
                        {
                            "country": _country,
                            "break_year": _year,
                            "n_docs": 0,
                            "status": "manual",
                            "error_message": np.nan,
                            "raw_output": (
                                "break_supported: 1\n"
                                "break_type: climate_shock, enso\n"
                                "duration: 3 year\n"
                                "climate_related: 1\n"
                                "confidence: 5\n"
                                "summary: Manual ENSO break-year overlay for QR experiment."
                            ),
                            "source_file": "manual_enso_breaks",
                            "break_supported": 1,
                            "confidence": 5,
                            "break_type": "climate_shock, enso",
                            "duration": "3 year",
                            "climate_related": 1,
                            "summary": "Manual ENSO break-year overlay for QR experiment.",
                        }
                    )
            if manual_rows:
                for _col in ["duration", "climate_related"]:
                    if _col not in llm_df.columns:
                        llm_df[_col] = np.nan
                _manual_keys = {(_r["country"], _r["break_year"]) for _r in manual_rows}
                llm_df = (
                    pd.concat([llm_df, pd.DataFrame(manual_rows)], ignore_index=True)
                    .assign(
                        _manual_rank=lambda _d: [
                            1 if (str(_r.country).upper(), int(_r.break_year)) in _manual_keys else 0
                            for _r in _d[["country", "break_year"]].itertuples(index=False)
                        ]
                    )
                    .sort_values(["country", "break_year", "_manual_rank"])
                    .drop_duplicates(["country", "break_year"], keep="last")
                    .drop(columns=["_manual_rank"])
                    .reset_index(drop=True)
                )
            print(f"[INFO] LLM matched rows: {len(llm_df)}")

            break_score_df = build_em_break_score_panel(
                PATH=PATH,
                COUNTRIES=list(countries_to_run),
                COL_COUNTRY=COL_COUNTRY,
                COL_TIME=COL_TIME,
                ENDO=ENDO,
                EXO=EXO,
                lags=lags,
                max_em_iter=MAX_EM_ITER,
            )
            break_score_df = break_score_df.dropna(subset=["score"]).copy()
            print(f"[INFO] break_score rows: {len(break_score_df)}")

            composite_break_df = build_em_composite_break_score_panel(
                PATH=PATH,
                COUNTRIES=list(countries_to_run),
                COL_COUNTRY=COL_COUNTRY,
                COL_TIME=COL_TIME,
                ENDO=ENDO,
                EXO=EXO,
                lags=lags,
                max_em_iter=MAX_EM_ITER,
            )
            llm_input_list = build_llm_input_candidates_from_composite(
                composite_df=composite_break_df,
                country_col="country",
                year_col="year",
                score_col="composite_score",
                top_k_peaks=4,
            )
            with open(INPUT_LLM_PICKLE_PATH, "wb") as _f_llm_input:
                pickle.dump(llm_input_list, _f_llm_input)

            # Persist data first; visualization is optional.
            pickle_bundle["llm_integration"] = {
                "llm_df": llm_df,
                "break_score_df": break_score_df,
                "composite_break_df": composite_break_df,
                "input_llm_list": llm_input_list,
                "input_llm_pickle_path": INPUT_LLM_PICKLE_PATH,
            }

            # Task 3 input prep: yearly map source data
            score_year_df = (
                break_score_df.groupby(["country", "year"], as_index=False)["score"]
                .mean()
                .rename(columns={"score": "score"})
            )
            pickle_bundle["llm_integration"]["score_year_df"] = score_year_df



    def plot_breakscore_countries_em(
        PATH,
        COUNTRIES,
        COL_COUNTRY="country",
        COL_TIME="quarter",
        ENDO=None,
        EXO=None,
        lags=1,
        max_em_iter=MAX_EM_ITER,
        drop_quarters_dict: dict[str, list[pd.Timestamp]] | None = None,
        iso3_to_country: dict | None = None,
        pdf: PdfPages | None = None,
        save_dir: str | None = None,
    ):
        """Plot per-country break score trajectories from EM innovation score.
        If drop_quarters_dict is provided, those quarters are removed before plotting.
        """
        bs = build_em_break_score_panel(
            PATH=PATH,
            COUNTRIES=COUNTRIES,
            COL_COUNTRY=COL_COUNTRY,
            COL_TIME=COL_TIME,
            ENDO=ENDO,
            EXO=EXO,
            lags=lags,
            max_em_iter=max_em_iter,
        ).dropna(subset=["score"])
        if bs.empty:
            print("[WARN] break score panel empty; skip breakscore plot")
            return

        for country in COUNTRIES:
            g = bs[bs["country"] == country].copy()
            if drop_quarters_dict and country in drop_quarters_dict:
                drop_q = set(pd.to_datetime(drop_quarters_dict[country]))
                g = g[~pd.to_datetime(g["quarter"]).isin(drop_q)].copy()
            if g.empty:
                continue
            cname = country_display_name(country, iso3_to_country)
            quarters = pd.to_datetime(g["quarter"].values)
            vals = g["score"].to_numpy(float)
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
            if save_dir:
                out_dir = Path(save_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(out_dir / f"breakscore_{country}.jpg", dpi=150, bbox_inches="tight")
                plt.close(fig)
            else:
                _fig_save_or_show(fig, pdf)


    def build_drop_quarters_dict_from_llm_points(
        base_path: str,
        llm_csv: str,
        countries: list[str],
        near_break_years: int = 2,
        top_score_k: int = 5,
    ) -> tuple[dict[str, list[pd.Timestamp]], pd.DataFrame]:
        """Build country->quarters-to-drop dictionary from top-k score + LLM break-year rule."""
        if not Path(llm_csv).exists():
            print(f"[WARN] skip refit filter: file not found -> {llm_csv}")
            return {}, pd.DataFrame(columns=["country", "quarter", "year", "score"])

        llm_raw = pd.read_csv(llm_csv)
        llm = _map_llm_country_to_iso3(llm_raw, ISO3_TO_COUNTRY)
        llm = llm[llm["country"].isin(countries)].copy()
        if "break_supported" not in llm.columns or "break_year" not in llm.columns:
            print("[WARN] skip refit filter: missing break_supported/break_year")
            return {}, pd.DataFrame(columns=["country", "quarter", "year", "score"])

        llm["break_year"] = pd.to_numeric(llm["break_year"], errors="coerce")
        llm = llm.dropna(subset=["break_year"]).copy()
        llm["break_year"] = llm["break_year"].astype(int)
        llm["break_supported_bin"] = (
            llm["break_supported"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"1": 1, "true": 1, "yes": 1, "0": 0, "false": 0, "no": 0})
            .fillna(0)
            .astype(int)
        )

        score_df = build_em_break_score_panel(
            PATH=base_path,
            COUNTRIES=countries,
            COL_COUNTRY=COL_COUNTRY,
            COL_TIME=COL_TIME,
            ENDO=ENDO,
            EXO=EXO,
            lags=lags,
            max_em_iter=MAX_EM_ITER,
        ).dropna(subset=["score"])
        if score_df.empty:
            print("[WARN] skip refit filter: break score panel empty")
            return {}, pd.DataFrame(columns=["country", "quarter", "year", "score"])

        topk = (
            score_df.groupby("country", group_keys=False)
            .apply(lambda g: g.nlargest(top_score_k, "score"))
            .reset_index(drop=True)
        )
        supported_years = (
            llm[llm["break_supported_bin"] == 1]
            .groupby("country")["break_year"]
            .apply(list)
            .to_dict()
        )

        drop_rows = []
        for r in topk.itertuples(index=False):
            years = supported_years.get(r.country, [])
            if any(abs(int(r.year) - int(y)) <= near_break_years for y in years):
                drop_rows.append(
                    {
                        "country": r.country,
                        "quarter": pd.Timestamp(r.quarter),
                        "year": int(r.year),
                        "score": float(r.score),
                    }
                )
        if not drop_rows:
            print("[INFO] no points matched condition; keep original panel")
            return {}, pd.DataFrame(columns=["country", "quarter", "year", "score"])

        drop_df = pd.DataFrame(drop_rows).drop_duplicates(subset=["country", "quarter"]).copy()
        drop_df["quarter"] = pd.to_datetime(drop_df["quarter"])

        # Aggressive expansion: drop the whole year (all 4 quarters) for matched points.
        expanded_rows = []
        for r in drop_df.itertuples(index=False):
            y = int(r.year)
            for qn in (1, 2, 3, 4):
                q = pd.Period(f"{y}Q{qn}", freq="Q").to_timestamp("Q")
                expanded_rows.append(
                    {
                        "country": r.country,
                        "quarter": q,
                        "year": y,
                        "score": float(r.score),
                    }
                )
        drop_df = pd.DataFrame(expanded_rows).drop_duplicates(subset=["country", "quarter"]).copy()

        drop_quarters_dict = (
            drop_df.groupby("country")["quarter"]
            .apply(lambda s: sorted(set(pd.to_datetime(s).tolist())))
            .to_dict()
        )
        return drop_quarters_dict, drop_df


    def build_filtered_panel_from_llm_points(
        base_path: str,
        out_filtered_csv: str,
        drop_quarters_dict: dict[str, list[pd.Timestamp]] | None = None,
        exclude_years: list[int] | None = None,
    ) -> str:
        """Drop rows by precomputed country->quarter dict and return filtered panel path.
        Note: actual filtering is done by (country, year) to avoid timestamp alignment issues.
        """
        if not drop_quarters_dict:
            return base_path

        src = pd.read_csv(base_path)
        src[COL_TIME] = pd.to_datetime(src[COL_TIME], errors="coerce")
        src["_drop_year"] = src[COL_TIME].dt.year
        key_rows = []
        for c, q_list in drop_quarters_dict.items():
            for q in q_list:
                key_rows.append({COL_COUNTRY: c, "_drop_year": int(pd.Timestamp(q).year)})
        key = pd.DataFrame(key_rows).drop_duplicates()
        if key.empty:
            return base_path

        out = src.merge(key.assign(_drop=1), on=[COL_COUNTRY, "_drop_year"], how="left")
        out = out[out["_drop"].isna()].copy()
        if exclude_years:
            ex_years = set(int(y) for y in exclude_years)
            out = out[~out["_drop_year"].isin(ex_years)].copy()
        out = out.drop(columns=["_drop", "_drop_year"])
        out.to_csv(out_filtered_csv, index=False)
        print(f"[SAVE] {out_filtered_csv} (dropped={len(src)-len(out)})")
        return out_filtered_csv


    # Extra pass only: drop points then rerun KF/EM and export coeff + breakscore PDF.
    ENABLE_LLM_REFIT_PDF = False
    PLOTS_REFIT_PDF_PATH: str | None = _dash_out("GVAR_LLM_EM_plots_refit.pdf")
    DROP_POINTS_LIST_CSV = _dash_out("llm_top5_near_break_points_to_drop.csv")
    REFIT_PANEL_CSV = _dash_out("gvar_panel_refit_filtered.csv")
    TOP_SCORE_K = 5
    NEAR_BREAK_YEARS = 2
    REFIT_EXCLUDE_YEARS = [2020,2021,2022,2023]
    BREAKSCORE_OUTPUT_MODE = "pdf"   # "pdf" | "plot" | "jpg"
    BREAKSCORE_JPG_DIR = _dash_out("breakscore_jpg")

    if ENABLE_LLM_REFIT_PDF:
        drop_quarters_dict, drop_df = build_drop_quarters_dict_from_llm_points(
            base_path=PATH,
            llm_csv=LLM_RESULTS_CSV,
            countries=list(countries_to_run),
            near_break_years=NEAR_BREAK_YEARS,
            top_score_k=TOP_SCORE_K,
        )
        if not drop_df.empty:
            drop_df.to_csv(DROP_POINTS_LIST_CSV, index=False)
            print(f"[SAVE] {DROP_POINTS_LIST_CSV} (n={len(drop_df)})")
        refit_path = build_filtered_panel_from_llm_points(
            base_path=PATH,
            out_filtered_csv=REFIT_PANEL_CSV,
            drop_quarters_dict=drop_quarters_dict,
            exclude_years=REFIT_EXCLUDE_YEARS,
        )
        refit_kw = dict(_plot_kw)
        refit_kw["PATH"] = refit_path
        pickle_bundle["refit"] = {
            "drop_quarters_dict": drop_quarters_dict,
            "drop_df": drop_df,
            "refit_path": refit_path,
            "BREAKSCORE_OUTPUT_MODE": BREAKSCORE_OUTPUT_MODE,
        }
        pickle_bundle["offline_plot_data"]["refit"] = build_em_offline_plot_pack(
            PATH=refit_path,
            COUNTRIES=COUNTRIES_FOR_COEF_PLOT,
            COL_COUNTRY=COL_COUNTRY,
            COL_TIME=COL_TIME,
            ENDO=ENDO,
            EXO=EXO,
            lags=lags,
            max_em_iter=MAX_EM_ITER,
            iso3_to_country=ISO3_TO_COUNTRY,
        )

    # Persist pipeline artifacts only; visualization is handled in SDR_visualize_from_pickle.py.
    print_numerical_diagnostics()
    with open(RESULTS_PICKLE_PATH, "wb") as f:
        pickle.dump(pickle_bundle, f)
    print(f"[PICKLE] Saved: {RESULTS_PICKLE_PATH}")
