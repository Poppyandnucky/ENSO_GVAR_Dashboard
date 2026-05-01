# analysis/validation/baseline/run_no_break.py
#
# Frozen baseline verification for TVP-VAR.
#
# This script verifies that, in the absence of structural breaks,
# a VAR-anchored TVP-VAR converges exactly to the fixed VAR.
#
# Do not modify this file except to fix bugs.

import numpy as np
from statsmodels.tsa.api import VAR

from analysis.estimate_var import tvp_var_with_exog_and_var_prior
from analysis.validation.simulate_data import simulate_var2_fixed
from analysis.validation.compare_betas import compare_convergence, extract_A_blocks_from_beta
# from tvp.kalman_var import tvp_var_with_exog

def run_convergence_experiment(
    T=1000,
    p=2,
    lam=1.0,
    R_scale=1000.0,
    shrink=1e-3,
    eps_scale=1e-8,
    prior_var_n_data=1000
):
    # --- simulate data
    df, true_params = simulate_var2_fixed(T=T)

    Y = df[["y1", "y2"]].values
    X = df[["x1"]].values

    # --- fixed VAR (for comparison only)
    res_var = VAR(Y, exog=X).fit(p, trend="n")
    A_blocks_var = res_var.coefs

    # --- TVP-VAR anchored to VAR
    beta_path, Z = tvp_var_with_exog_and_var_prior(Y=Y, Xexo=X, p=p, lam=lam, R=R_scale, shrink=shrink,
        eps_scale=eps_scale, prior_var_n_data=prior_var_n_data)

    from analysis.results import TVPVARResult  # or wherever it lives

    res_tvp = TVPVARResult( beta_path=beta_path, dom_vars=["y1", "y2"], p=p, endog=Y,
        exog=X, exog_columns=["x1"] )
    res_tvp.sigma_u = res_var.sigma_u.copy()

    print("VAR A:")
    print(res_var.coefs)
    print("TVP A:")
    # print(beta_path[-1][:p * 2].reshape(p, 2, 2))
    print(extract_A_blocks_from_beta(beta_path[-1], p, k=2))

    H = 20
    irf_var = res_var.irf(H)
    irf_tvp = res_tvp.irf(H, t0=-1)

    # response of y1 to shock in y1
    irf_var_y1 = irf_var.irfs[:, 0, 0]
    irf_tvp_y1 = irf_tvp.irfs[:, 0, 0]

    print("max abs diff:", np.max(np.abs(irf_var_y1 - irf_tvp_y1)))

    # one-step-ahead forecast comparison
    y_last = Y[-p:]
    f_var = res_var.forecast(y_last, steps=1, exog_future=X[-1:])
    f_tvp = res_tvp.forecast(y_last, steps=1, exog_future=X[-1:])

    print("max abs forecast:",np.max(np.abs(f_var - f_tvp)))

    return {
        "true": true_params,
        "A_blocks_var": A_blocks_var,
        "beta_tvp_path": beta_path,
    }

def main():
    results = run_convergence_experiment(
        T=3000, # need 30000 to see many oscillations around mean
        lam=0.995 + 0.005, # 1.0 -> oscillates around initial (prior) value
        R_scale=10.0 * 0.001, # smaller -> smaller oscillations
        shrink=1e-0, # smaller value -> longer time to converge to true value
        eps_scale=1e-1, # minor effect
        prior_var_n_data=300 # used to get prior var
    )
    compare_convergence(results)

if __name__ == "__main__":
    main()