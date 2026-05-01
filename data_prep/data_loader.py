# analysis/data_loader.py
#
# Shared data-loading and preparation utilities.
# These functions are UI-agnostic and may be used by:
#   - Streamlit apps
#   - Validation scripts
#   - Batch analysis
#
# No Streamlit imports, no caching, no side effects.

import pandas as pd

def load_gvar_panel():
    """
    Load the macro panel used by the Streamlit app and validation code.

    Returns
    -------
    panel : pandas.DataFrame
        Must include columns:
        - country
        - quarter
        - macro variables (GDP_YoY, CPI_YoY, FX_YoY, ENSO, etc.)
    """
    return pd.read_csv(
        "data/gvar_panel_streamlit.csv",
        parse_dates=["quarter"]
    )

def load_stressor_probabilities():
    """
    Load precomputed stressor probabilities.

    Returns
    -------
    prob_df : pandas.DataFrame
    """
    return pd.read_csv("data/prithvi_stressor_probabilities.csv")

def load_country_dataframe(country):
    """
    Load and prepare macro data for a single country.

    Parameters
    ----------
    country : str

    Returns
    -------
    df_country : pandas.DataFrame
        Prepared data for VAR / TVP-VAR estimation
    """
    df = load_gvar_panel()

    # 🔧 MOVE this logic from streamlit_TRP.py
    df_country = df[df["country"] == country].copy()
    df_country = df_country.sort_values("quarter")

    # any cleaning / filtering currently in Streamlit
    # e.g. dropping NaNs, selecting date range, etc.

    return df_country

def get_country_arrays(country, y_vars, x_vars):
    """
    Convenience wrapper returning numpy arrays for modeling.

    Parameters
    ----------
    country : str
    y_vars : list[str]
        Endogenous variable names
    x_vars : list[str]
        Exogenous variable names

    Returns
    -------
    Y : (T, k) ndarray
    X : (T, q) ndarray
    df_country : DataFrame (for reference)
    """
    df_country = load_country_dataframe(country)

    Y = df_country[y_vars].values
    X = df_country[x_vars].values

    return Y, X, df_country