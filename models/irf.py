# models/irf.py
#
# Impulse response function (IRF) utilities for VAR(p) models.
#
# Responsibilities:
#   - Companion-form construction
#   - IRF computation given VAR coefficient matrices
#
# This module contains no estimation logic and no model state.
import numpy as np
from analysis.validation.compare_betas import extract_A_blocks_from_beta

def irf_var_consistent(res_var, H):
    A_blocks_var = res_var.coefs.copy()

    # Convert VAR A into Kalman orientation (regressor × equation)
    # so that innovation_irf can apply the SAME transpose logic
    p, k, _ = A_blocks_var.shape
    beta_like = np.vstack([A_blocks_var[lag].T for lag in range(p)])

    # Reconstruct A_blocks via the SAME extractor used for TVP
    from analysis.validation.compare_betas import extract_A_blocks_from_beta
    A_blocks = extract_A_blocks_from_beta(beta_like, p=p, k=k)

    # return innovation_irf(A_blocks, H)
    return innovation_irf(A_blocks, H, Dy=res_var.Dy)

def innovation_irf(A_blocks, H, Dy=None):
    """
    Compute innovation IRFs Ψ_h from VAR(p) coefficients.

    Returns array of shape (H+1, k, k) with
    [h, response, impulse].
    """
    irfs_std = compute_irf_varp(A_blocks, horizon=H)
    # irfs_std = compute_irf_var1(A_blocks, H)

    # IMPORTANT: match statsmodels convention
    #   cols are equations
    for h in range(H + 1):
        irfs_std[h] = irfs_std[h].T

    if Dy is not None:
        Dy_inv = np.linalg.inv(Dy)
        irfs = np.empty_like(irfs_std)
        for h in range(H + 1):
            irfs[h] = Dy_inv @ irfs_std[h] @ Dy
            # irfs[h] = Dy @ irfs_std[h] @ Dy_inv
        return irfs

    return irfs_std

def irf_from_var(res_var, H):
    x
    return innovation_irf(
        A_blocks=res_var.coefs,
        H=H,
        Dy=None   # VAR already in original units
    )

def irf_from_tvp(res_tvp, H, t0=-1):
    beta_t = res_tvp.beta_path[t0]
    k = len(res_tvp.names)
    p = res_tvp.k_ar

    # A_blocks extracted from beta_t after taking transpose
    #   each row is an equation
    A_blocks = extract_A_blocks_from_beta(beta_t, p, k)

    return innovation_irf(
        A_blocks=A_blocks,
        H=H,
        Dy=getattr(res_tvp, "Dy", None)
    )

def compute_irf_varp(A_blocks, horizon=12):
    # A_blocks[i] @ x is used for computing y
    p, k, _ = A_blocks.shape
    F = np.zeros((p*k, p*k))
    # first k rows are the equations, rest are y_i-1 = y_i-1
    F[:k, :] = np.hstack(A_blocks)
    if p > 1:
        F[k:, :-k] = np.eye((p-1)*k)

    irfs = np.zeros((horizon+1, k, k))
    irfs[0] = np.eye(k)

    Fpow = np.eye(p*k)
    J = np.hstack([np.eye(k), np.zeros((k, (p-1)*k))])

    for h in range(1, horizon+1):
        Fpow = F @ Fpow # seems to be slightly better - not clear why
        # Fpow = Fpow @ F
        irfs[h] = J @ Fpow @ J.T

    return irfs

# NOT USED, BUT MAY HAVE BEEN HELPFUL IN DEBUGGING
#   use A @ irfs[h-1] instead of irfs[h-1] @ A, not sure why this is slightly better
def compute_irf1(A, horizon=12):
    # """
    # Compute impulse responses for VAR(1):
    # A : (k, k) coefficient matrix
    # 	•	Takes one fixed VAR(1) coefficient matrix A
    # 	•	Computes powers: I, A, A^2, A^3, \dots
	#     •	Returns the mathematical impulse responses
    # """
    k = A.shape[0]
    irfs = np.zeros((horizon + 1, k, k))
    irfs[0] = np.eye(k)

    for h in range(1, horizon + 1):
        irfs[h] = A @ irfs[h - 1]

    return irfs
