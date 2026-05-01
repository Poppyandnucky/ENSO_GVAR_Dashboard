# analysis/validation/run_break.py

from statsmodels.tsa.api import VAR

from analysis.validation.simulate_data import simulate_var2_with_break
from analysis.estimate_var import tvp_var_with_exog_and_var_prior  # your wrapper
from data_prep.data_loader import load_country_dataframe
from analysis.validation.compare_betas import plot_A1_tracking, summarize_pre_post

def run_break_experiment_on_real_data( Y, X, p=2, lam=0.995, R_scale=200.0,
    shrink=1e-3, eps_scale=1e-8 ):
    # --- use real data in Y and X,
    #     then compute VAR versus TVP + prior VAR
    res_var = VAR(Y, exog=X).fit(p, trend="n")
    A_blocks_var = res_var.coefs

    # TVP-VAR anchored to VAR
    beta_path, Z = tvp_var_with_exog_and_var_prior(Y=Y, Xexo=X, p=p, lam=lam, R=R_scale, shrink=shrink,
                                                   eps_scale=eps_scale)

    return { "df": None, "true": None,   # no known truth
        "A_blocks_var": A_blocks_var, "beta_tvp_path": beta_path, "p": p, "k": Y.shape[1] }

def run_break_experiment( T=2000, break_frac=0.5, p=2, lam=0.995, R_scale=200.0,
    shrink=1e-3, eps_scale=1e-8, seed=123 ):
    # --- simulate data with break,
    #     then compute VAR versus TVP + prior VAR
    df, true_params = simulate_var2_with_break(T=T, break_frac=break_frac, seed=seed)

    Y = df[["y1", "y2"]].values
    X = df[["x1"]].values

    # --- fixed VAR (for comparison)
    res_var = VAR(Y, exog=X).fit(p, trend="n")
    A_blocks_var = res_var.coefs

    # --- TVP-VAR anchored to VAR prior
    beta_path, Z = tvp_var_with_exog_and_var_prior(Y=Y, Xexo=X, p=p, lam=lam, R=R_scale, shrink=shrink,
                                                   eps_scale=eps_scale)

    return { "df": df, "true": true_params, "A_blocks_var": A_blocks_var,
        "beta_tvp_path": beta_path, "p": p, "k": 2 }

def main_real_data():
    # Step 4b: run break experiment on real (Streamlit) data
    country = "BRA"   # or parameterize
    df = load_country_dataframe(country)

    Y = df[["GDP_YoY", "CPI_YoY", "FX_YoY"]].values
    X = df[["ENSO"]].values

    results = run_break_experiment_on_real_data( Y, X, p=2, lam=0.995, R_scale=200.0, shrink=1e-3 )

    plot_A1_tracking(results, entry=(0, 0))
    summarize_pre_post(results)

def main_simulated():
    # Step 4a: run break experiment on simulated data
    results = run_break_experiment( T=3000+10000, break_frac=0.7, p=2, lam=0.995, R_scale=200.0,
        shrink=1e-3, seed=123 )

    plot_A1_tracking(results, entry=(0, 0))
    plot_A1_tracking(results, entry=(0, 1))
    plot_A1_tracking(results, entry=(1, 1))
    summarize_pre_post(results)

if __name__ == "__main__":
    # main_simulated()
    main_real_data()
