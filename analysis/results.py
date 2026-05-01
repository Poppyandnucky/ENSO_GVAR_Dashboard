# analysis/results.py
#
# Result container classes for VAR and TVP-VAR models.
#
# These classes:
#   - store estimation outputs
#   - provide a common interface (forecast, irf)
#   - are compatible with statsmodels-style usage
#
# No numerical estimation is performed here.

import numpy as np
from matplotlib import pyplot as plt

from models.irf import compute_irf_varp


class TVPVARResult:
    def __init__( self, beta_path, dom_vars, endog, exog, exog_columns, p=2, R=1.0, lam=0.98 ):
        self.beta_path = beta_path
        self.dom_vars = dom_vars
        self.p = p

        # --- statsmodels compatibility
        self.names = list(dom_vars)
        self.k_ar = p
        self.endog = endog
        self.exog = exog
        self._exog_columns = list(exog_columns)

        self.R = R
        self.lam = lam

    def forecast(self, y, steps, exog_future, t0=-1):
        """
        Conditional forecast in STANDARDIZED space.
        exog_future ALREADY includes the constant column.
        """
        beta_t = self.beta_path[t0]

        k = y.shape[1]
        p = self.k_ar

        # --- extract A matrices (correct orientation)
        beta_mat = beta_t
        A = np.zeros((p, k, k))
        for lag in range(p):
            rows = slice(lag * k, (lag + 1) * k)
            A[lag] = beta_mat[rows, :]

        # --- extract B to match exog dimension (NO extra constant)
        q = exog_future.shape[1]  # includes constant
        B = beta_mat[p * k: p * k + q, :]  # shape (q, k)

        y_hist = np.asarray(y)
        forecasts = []

        for h in range(steps):
            y_next = np.zeros(k)

            # lagged endogenous
            for i in range(p):
                # A cols are equations, need transpose
                y_next += A[i].T @ y_hist[-(i + 1)]

            # exogenous (including constant already)
            x = exog_future[h]  # shape (q,)
            # B cols are equations, need transpose
            y_next += (B.T @ x).ravel()

            forecasts.append(y_next)
            y_hist = np.vstack([y_hist, y_next])

        return np.asarray(forecasts)

    def irf(self, H, t0=-1):
        """
        Time-specific IRF evaluated at coefficient state t0
        """
        print("USING innovation IRF")
        beta_t = self.beta_path[t0]

        k = len(self.dom_vars)
        p = self.k_ar

        # unpack VAR(p) coefficients with correct orientation
        beta_mat = beta_t  # (m, k)
        A_blocks = np.zeros((p, k, k))
        for lag in range(p):
            rows = slice(lag * k, (lag + 1) * k)
            A_blocks[lag] = beta_mat[rows, :]

        # expose statsmodels-like attributes
        self.coefs = A_blocks.T

        # VAR(p) IRF in standardized space
        irfs_std = compute_irf_varp(A_blocks.T, horizon=H)

        # FIX: match statsmodels convention (response, impulse)
        for h in range(H + 1):
            irfs_std[h] = irfs_std[h].T

        # un-standardize IRFs
        if hasattr(self, "Dy") and self.Dy is not None:
            Dy = self.Dy
            # Dy_inv = np.linalg.inv(Dy)
            irfs = np.empty_like(irfs_std)
            for h in range(H + 1):
                # irfs[h] = Dy @ irfs_std[h] @ Dy_inv
                irfs[h] = Dy @ irfs_std[h]
        else:
            irfs = irfs_std
            Dy = None

        return TVPIRF(irfs, self.dom_vars, Dy=Dy)

class TVPIRF:
    def __init__(self, irfs, var_names, Dy=None):
        self.irfs = irfs
        self.var_names = var_names
        self.Dy = Dy

    def plot(self, impulse, response, orth=False):
        i = self.var_names.index(response)
        j = self.var_names.index(impulse)

        y = self.irfs[:, i, j]

        plt.figure(figsize=(6, 4))
        plt.plot(y, lw=2)
        plt.axhline(0, color="k", ls="--", lw=1)
        plt.title(f"IRF: {response} ← {impulse}")
        plt.xlabel("Horizon")
        plt.ylabel("Response")
        plt.tight_layout()
        plt.show()
