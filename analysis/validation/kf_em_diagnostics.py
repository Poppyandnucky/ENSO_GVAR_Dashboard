"""Standalone KF/EM diagnostic utility -- run from the terminal for one country.

This tool is intentionally decoupled from the dashboard and from
`analysis/regenerate_dashboard_artifacts.py`: it does not import
`structural_break/GVAR_LLM_pickle.py` or any Streamlit code, and it never
writes to any file the dashboard reads. It reuses only the shared, tested
Kalman/EM math in `trp/kalman_core.py` (`init_from_varx_rolling`,
`kf_e_step_store`, `rts_smoother`, `em_m_step_update`) -- it does not
reimplement any filtering/smoothing/M-step math itself. The only thing it
duplicates is the *orchestration loop* (the same loop shape as
`trp.kalman_core.run_kf_em`), so that per-iteration internals (full Q, R,
theta0, not just their traces) can be captured for diagnostics and so that
literal breakpoints can be placed inside the loop body.

ENDO/EXO/lags below mirror the production config in
`structural_break/GVAR_LLM_pickle.py` (`ENDO`, `EXO`, `lags`). If that
config changes, update it here too for the diagnostics to stay
representative of what the dashboard actually estimates.

Usage (from repo root):
    python analysis/validation/kf_em_diagnostics.py --country KEN --max-iter 30
    python analysis/validation/kf_em_diagnostics.py --country KEN --max-iter 30 --update-p0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trp.kalman_core import (  # noqa: E402
    init_from_varx_rolling,
    kf_e_step_store,
    rts_smoother,
    em_m_step_update,
    _kf_log_likelihood,
)
from trp.inputs import load_gvar_panel  # noqa: E402

# ---------------------------------------------------------------------------
# Production config mirror (see module docstring).
# ---------------------------------------------------------------------------
COL_COUNTRY = "country"
COL_TIME = "quarter"
ENDO = ["GDP_YoY", "CPI_YoY", "FX_YoY", "EX_YoY"]
EXO = ["ENSO", "OIL_YoY"]
LAGS = 1
WINDOW = 40
MIN_T = LAGS + 5


# ---------------------------------------------------------------------------
# Data prep (standalone -- does not import GVAR_LLM_pickle.py)
# ---------------------------------------------------------------------------
def _prepare_country(country: str) -> dict | None:
    df = load_gvar_panel()
    endo_use = [c for c in ENDO if c in df.columns and df[c].notna().any()]
    exo_use = [c for c in EXO if c in df.columns and df[c].notna().any()]
    if not endo_use or not exo_use:
        return None

    g = df[df[COL_COUNTRY] == country].sort_values(COL_TIME)
    if g.empty:
        return None
    mask = (
        np.isfinite(g[endo_use].to_numpy(float)).all(axis=1)
        & np.isfinite(g[exo_use].to_numpy(float)).all(axis=1)
    )
    g = g.loc[mask].copy().reset_index(drop=True)
    if len(g) < MIN_T:
        return None

    Yd = g[endo_use].to_numpy(float)
    mu_y, sd_y = np.nanmean(Yd, axis=0), np.nanstd(Yd, axis=0) + 1e-8
    Yd = (Yd - mu_y) / sd_y

    Xd = g[exo_use].to_numpy(float)
    mu_x, sd_x = np.nanmean(Xd, axis=0), np.nanstd(Xd, axis=0) + 1e-8
    Xd = (Xd - mu_x) / sd_x

    return {
        "g": g,
        "Yd": Yd,
        "Xd": Xd,
        "quarters": pd.to_datetime(g[COL_TIME].values),
        "ENDO_use": endo_use,
        "EXO_use": exo_use,
        "mu_y": mu_y,
        "sd_y": sd_y,
        "mu_x": mu_x,
        "sd_x": sd_x,
    }


def _theta_index_groups(ENDO_use: list[str], EXO_use: list[str], lags: int, m: int) -> dict:
    """Map readable labels -> flat theta index, grouped by equation.

    Layout matches trp.kalman_core: theta has p = mY*m entries, block j
    (equation for ENDO_use[j]) occupies [j*m, (j+1)*m), with regressor order
    [Y_{t-1}..Y_{t-lags} (each length mY), Z_{t-1} (length mX)].
    """
    mY = len(ENDO_use)
    mX = len(EXO_use)
    groups: dict[str, dict[str, int]] = {}
    for j, eq_name in enumerate(ENDO_use):
        block = j * m
        labels: dict[str, int] = {}
        for lag in range(1, lags + 1):
            for k, var_name in enumerate(ENDO_use):
                idx = block + (lag - 1) * mY + k
                tag = "own_lag" if var_name == eq_name else "cross_lag"
                labels[f"{tag}:{var_name}(t-{lag})"] = idx
        for k, var_name in enumerate(EXO_use):
            idx = block + lags * mY + k
            tag = "enso" if var_name.upper() == "ENSO" else "exo"
            labels[f"{tag}:{var_name}(t-1)"] = idx
        groups[eq_name] = labels
    return groups


# ---------------------------------------------------------------------------
# EM loop with full per-iteration history (reuses kalman_core math functions)
# ---------------------------------------------------------------------------
def _run_em_with_full_history(
    Y: np.ndarray,
    Z: np.ndarray,
    lags: int,
    window: int,
    max_iter: int,
    tol: float,
    em_damping: float,
    update_P0: bool,
    verbose: bool,
) -> dict:
    theta0, Q, R, P0, m, p = init_from_varx_rolling(Y=Y, Z=Z, lags=lags, window=window)
    # BREAKPOINT 1: inspect initial theta0, P0, Q, R here (the one-time VAR
    # burn-in output, before any EM iteration has run). Useful for checking
    # that the rolling-VARX initializer produced a sane starting point (no
    # NaNs, no absurd scale) before blaming the EM loop for anything later.
    init_state = {"theta0": theta0.copy(), "Q0": Q.copy(), "R0": R.copy(), "P0": P0.copy()}

    iters = []
    last_obj = np.inf
    best_pack = None
    theta0_prev, Q_prev, R_prev = theta0, Q, R

    for it in range(max_iter):
        theta_filt, P_filt, theta_pred, P_pred, H_list, y_list, valid_mask = kf_e_step_store(
            Y=Y, Z_all=Z, theta0=theta0, Q=Q, R=R, P0=P0, lags=lags
        )
        theta_smooth, P_smooth, J_hist = rts_smoother(
            theta_filt, P_filt, theta_pred, P_pred, valid_mask
        )
        Q_new, R_new = em_m_step_update(
            Y=Y, H_list=H_list, y_list=y_list, valid_mask=valid_mask,
            theta_smooth=theta_smooth, P_smooth=P_smooth, J_hist=J_hist,
            Q_old=Q, R_old=R, em_damping=em_damping,
        )

        valid_idx = np.where(valid_mask)[0]
        theta0_new = theta_smooth[valid_idx[0]].reshape(-1, 1) if len(valid_idx) else theta0
        if update_P0 and len(valid_idx):
            P0_new = P_smooth[valid_idx[0]]
        else:
            P0_new = P0

        obj = 0.0
        cnt = 0
        for t in valid_idx:
            e = y_list[t] - H_list[t] @ theta_smooth[t].reshape(-1, 1)
            obj += float((e.T @ e).item())
            cnt += 1
        obj = obj / max(cnt, 1)
        ll = _kf_log_likelihood(H_list, y_list, valid_mask, theta_pred, P_pred, R_new)

        theta_change_norm = float(np.linalg.norm(theta0_new - theta0_prev))
        if len(valid_idx):
            diag0 = np.diag(P_smooth[valid_idx[0]])
            sd0 = np.sqrt(diag0) if np.all(diag0 > 0) else np.full_like(diag0, np.nan, dtype=float)
            theta_change_std = float(np.linalg.norm((theta0_new - theta0_prev).ravel() / sd0))
        else:
            theta_change_std = np.nan
        q_rel_change = float(np.linalg.norm(Q_new - Q_prev) / max(np.linalg.norm(Q_prev), 1e-12))
        r_rel_change = float(np.linalg.norm(R_new - R_prev) / max(np.linalg.norm(R_prev), 1e-12))

        ev_q = np.linalg.eigvalsh(0.5 * (Q_new + Q_new.T))
        ev_r = np.linalg.eigvalsh(0.5 * (R_new + R_new.T))
        ev_p0 = np.linalg.eigvalsh(0.5 * (P0_new + P0_new.T))

        record = {
            "iter": it + 1,
            "obj": obj,
            "convergence_metric": abs(last_obj - obj),  # what run_kf_em's stopping rule uses
            "log_likelihood": ll,
            "theta0_change_norm": theta_change_norm,
            "theta0_change_std": theta_change_std,
            "Q_rel_change": q_rel_change,
            "R_rel_change": r_rel_change,
            "trace_P0": float(np.trace(P0_new)),
            "trace_Q": float(np.trace(Q_new)),
            "trace_R": float(np.trace(R_new)),
            "eig_min_Q": float(ev_q.min()),
            "eig_min_R": float(ev_r.min()),
            "eig_min_P0": float(ev_p0.min()),
        }
        iters.append(record)
        # BREAKPOINT 2: inspect this iteration's likelihood, convergence
        # metric (|delta obj|), theta/Q/R changes, and the new P0/Q/R here.
        # Useful for stepping through EM iteration-by-iteration when
        # convergence looks suspicious (e.g. oscillating obj, a sudden jump
        # in theta0_change_std, or Q/R moving in an unexpected direction).
        if verbose:
            print(
                f"[EM {it + 1:02d}] obj={obj:.6f} |dobj|={record['convergence_metric']:.3e} "
                f"ll={ll:.2f} dtheta0_std={theta_change_std:.2f} "
                f"dQrel={q_rel_change:.3e} dRrel={r_rel_change:.3e}"
            )

        best_pack = {
            "theta_filt": theta_filt, "P_filt": P_filt, "theta_pred": theta_pred,
            "P_pred": P_pred, "theta_smooth": theta_smooth, "P_smooth": P_smooth,
            "J_hist": J_hist, "valid_mask": valid_mask, "H_list": H_list, "y_list": y_list,
        }

        theta0, Q, R, P0 = theta0_new, Q_new, R_new, P0_new
        theta0_prev, Q_prev, R_prev = theta0_new, Q_new, R_new

        if abs(last_obj - obj) < tol:
            if verbose:
                print(f"[EM] converged at iter={it + 1}")
            break
        last_obj = obj

    # BREAKPOINT 3: inspect final filtered/smoothed state trajectories here
    # (best_pack["theta_filt"], ["theta_smooth"], ["P_filt"], ["P_smooth"]).
    # This is the state right after the last E-step + smoother pass, before
    # any innovation/forecast computation -- the natural place to check
    # whether the filtered and smoothed coefficient paths look sensible
    # (e.g. smoothed path much smoother than filtered, no wild swings).
    return {
        "init_state": init_state,
        "iters": iters,
        "theta0": theta0, "Q": Q, "R": R, "P0": P0,
        "e_step_store": best_pack,
        "m": m, "p": p,
        "converged": len(iters) < max_iter,
    }


# ---------------------------------------------------------------------------
# Innovations (using the model's actual S_t, not an approximation)
# ---------------------------------------------------------------------------
def _compute_innovations(pack: dict, R: np.ndarray, ENDO_use: list[str], eps: float = 1e-8) -> dict:
    valid_idx = np.where(pack["valid_mask"])[0]
    mY = len(ENDO_use)
    n_valid = len(valid_idx)
    innov = np.full((n_valid, mY), np.nan)
    innov_std = np.full((n_valid, mY), np.nan)
    S_diag = np.full((n_valid, mY), np.nan)

    for row, t in enumerate(valid_idx):
        H_t = pack["H_list"][t]
        y_t = pack["y_list"][t]
        theta_pr = pack["theta_pred"][t].reshape(-1, 1)
        P_pr = pack["P_pred"][t]
        v = y_t - H_t @ theta_pr
        S = H_t @ P_pr @ H_t.T + R
        S = 0.5 * (S + S.T) + eps * np.eye(mY)
        # BREAKPOINT 4: inspect innovation `v`, model-implied covariance `S`,
        # and standardized innovation `v / sqrt(diag(S))` here. Useful for
        # spotting a single bad time point (e.g. one huge outlier quarter)
        # versus a systematic problem (e.g. S too small/large across the
        # whole sample), before it gets summarized into aggregate stats.
        innov[row, :] = v.ravel()
        S_diag[row, :] = np.diag(S)
        innov_std[row, :] = v.ravel() / np.sqrt(np.clip(np.diag(S), eps, None))

    quarters = None  # filled in by caller using prep["quarters"][valid_idx]
    return {
        "valid_idx": valid_idx,
        "innovation": innov,
        "innovation_std": innov_std,
        "S_diag": S_diag,
    }


# ---------------------------------------------------------------------------
# One-step forecast illustration (NOT the dashboard's multi-step MC scenario
# forecast -- see analysis/Dash_Output/gvar_kf_forecast.py for that; this is
# a minimal, self-contained illustration of where P/Q/R enter a one-step KF
# prediction, kept here only so the tool can inspect it end to end).
# ---------------------------------------------------------------------------
def _one_step_forecast_illustration(pack: dict, Q: np.ndarray, R: np.ndarray, m: int, mY: int) -> dict:
    valid_idx = np.where(pack["valid_mask"])[0]
    last_t = valid_idx[-1]
    theta_last = pack["theta_filt"][last_t].reshape(-1, 1)
    P_last = pack["P_filt"][last_t]
    H_last = pack["H_list"][last_t]

    # BREAKPOINT 5: inspect the exact theta, P, Q, R, and regressor row H
    # passed into the one-step forecast here -- confirm these are the FINAL
    # EM estimates (not zeros/defaults) before the prediction equations run.
    theta_pred = theta_last  # point-forecast equation: y_hat = H_next @ theta_pred (mean unaffected by P, Q, R)
    P_pred = P_last + Q  # state covariance propagation: P and Q both enter here
    y_hat = H_last @ theta_pred
    S_pred = H_last @ P_pred @ H_last.T + R  # R enters only here, alongside P and Q

    # BREAKPOINT 6: inspect the forecast mean (y_hat) and covariance
    # components (P_pred, S_pred) here -- this is where you can see that P,
    # Q, and R only ever change S_pred (forecast uncertainty), never y_hat
    # (forecast mean), for this random-walk-state model.
    return {
        "theta_pred": theta_pred.ravel(),
        "P_pred": P_pred,
        "y_hat": y_hat.ravel(),
        "S_pred": S_pred,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plot_convergence(iters_df: pd.DataFrame, tol: float, out_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
    final_it = iters_df["iter"].iloc[-1]
    panels = [
        ("log_likelihood", "Log-likelihood", False),
        ("convergence_metric", "Convergence metric |Δobj| (run_kf_em stopping rule)", True),
        ("theta0_change_std", "Standardized θ0 change", False),
        ("Q_rel_change", "Relative change in Q (‖ΔQ‖/‖Q_prev‖)", False),
        ("R_rel_change", "Relative change in R (‖ΔR‖/‖R_prev‖)", False),
        ("trace_P0", "trace(P0)", False),
        ("trace_Q", "trace(Q)", False),
        ("trace_R", "trace(R)", False),
    ]
    for ax, (col, title, show_tol) in zip(axes.ravel(), panels):
        ax.plot(iters_df["iter"], iters_df[col], marker="o", ms=3)
        if show_tol:
            ax.axhline(tol, color="red", linestyle="--", linewidth=1, label=f"tol={tol:g}")
            ax.legend(fontsize=8)
        ax.axvline(final_it, color="gray", linestyle=":", linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("EM iteration")
    fig.suptitle("EM convergence diagnostics (final iteration marked with dotted line)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_covariance_time_traces(P_filt, P_smooth, valid_mask, quarters, out_path: Path) -> None:
    valid_idx = np.where(valid_mask)[0]
    tr_filt = [np.trace(P_filt[t]) for t in valid_idx]
    tr_smooth = [np.trace(P_smooth[t]) for t in valid_idx]
    q = quarters[valid_idx]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(q, tr_filt, label="trace(P_filt[t])", marker="o", ms=3)
    ax.plot(q, tr_smooth, label="trace(P_smooth[t])", marker="o", ms=3)
    ax.set_title("Filtered vs. smoothed state covariance trace across historical time (final EM iteration)")
    ax.set_xlabel("Quarter")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_theta_paths(theta_filt, theta_smooth, valid_mask, quarters, groups: dict, out_dir: Path) -> None:
    valid_idx = np.where(valid_mask)[0]
    q = quarters[valid_idx]
    for eq_name, labels in groups.items():
        own_lag = {k: v for k, v in labels.items() if k.startswith("own_lag")}
        enso = {k: v for k, v in labels.items() if k.startswith("enso")}
        other = {k: v for k, v in labels.items() if k.startswith("cross_lag") or k.startswith("exo")}
        panel_groups = [("Own-lag coefficient", own_lag), ("ENSO coefficient", enso), ("Other (cross-lag / exogenous)", other)]
        panel_groups = [(t, g) for t, g in panel_groups if g]
        fig, axes = plt.subplots(len(panel_groups), 1, figsize=(11, 3.4 * len(panel_groups)), sharex=True)
        if len(panel_groups) == 1:
            axes = [axes]
        for ax, (title, group) in zip(axes, panel_groups):
            for label, idx in group.items():
                ax.plot(q, theta_filt[valid_idx, idx], linestyle="--", alpha=0.6, label=f"filt: {label}")
                ax.plot(q, theta_smooth[valid_idx, idx], linestyle="-", label=f"smooth: {label}")
            ax.set_title(f"{eq_name} equation -- {title}", fontsize=10)
            ax.legend(fontsize=7, ncol=2)
        fig.suptitle(f"Filtered vs. smoothed coefficient paths -- {eq_name} equation")
        fig.tight_layout()
        fig.savefig(out_dir / f"theta_paths_{eq_name}.png", dpi=140)
        plt.close(fig)


def _plot_observed_vs_fitted(Y, theta_filt, theta_smooth, H_list, valid_mask, quarters, ENDO_use, out_dir: Path) -> pd.DataFrame:
    valid_idx = np.where(valid_mask)[0]
    q = quarters[valid_idx]
    mY = len(ENDO_use)
    y_obs = Y[valid_idx, :]
    y_fit_filt = np.full((len(valid_idx), mY), np.nan)
    y_fit_smooth = np.full((len(valid_idx), mY), np.nan)
    for row, t in enumerate(valid_idx):
        H_t = H_list[t]
        y_fit_filt[row, :] = (H_t @ theta_filt[t].reshape(-1, 1)).ravel()
        y_fit_smooth[row, :] = (H_t @ theta_smooth[t].reshape(-1, 1)).ravel()

    rows = []
    for j, var in enumerate(ENDO_use):
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(q, y_obs[:, j], label="Observed (standardized)", color="black", linewidth=1.5)
        ax.plot(q, y_fit_filt[:, j], label="Reconstructed from theta_filt (in-sample, filtered)", linestyle="--")
        ax.plot(q, y_fit_smooth[:, j], label="Reconstructed from theta_smooth (in-sample, smoothed)", linestyle=":")
        ax.set_title(
            f"{var}: observed vs. in-sample KF fit -- NOT an out-of-sample forecast",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"observed_vs_fitted_{var}.png", dpi=140)
        plt.close(fig)
        for row in range(len(valid_idx)):
            rows.append({
                "quarter": q[row], "variable": var,
                "observed": y_obs[row, j],
                "fitted_filt": y_fit_filt[row, j],
                "fitted_smooth": y_fit_smooth[row, j],
                "resid_filt": y_obs[row, j] - y_fit_filt[row, j],
                "resid_smooth": y_obs[row, j] - y_fit_smooth[row, j],
            })
    return pd.DataFrame(rows)


def _plot_innovation_diagnostics(
    innov_res: dict, ENDO_use: list[str], quarters, out_dir: Path, Y: np.ndarray | None = None
) -> pd.DataFrame:
    valid_idx = innov_res["valid_idx"]
    q = quarters[valid_idx]
    innov = innov_res["innovation"]
    innov_std = innov_res["innovation_std"]
    S_diag = innov_res["S_diag"]

    report_rows = []
    for j, var in enumerate(ENDO_use):
        z = innov_std[:, j]
        z_ok = z[np.isfinite(z)]

        fig, axes = plt.subplots(3, 2, figsize=(11, 12))
        axes[0, 0].plot(q, z, marker="o", ms=3)
        axes[0, 0].axhline(1.96, color="red", linestyle="--", linewidth=1)
        axes[0, 0].axhline(-1.96, color="red", linestyle="--", linewidth=1)
        axes[0, 0].set_title(f"{var}: standardized innovation time series")

        axes[0, 1].hist(z_ok, bins=20, density=True, alpha=0.7)
        xs = np.linspace(-4, 4, 200)
        axes[0, 1].plot(xs, stats.norm.pdf(xs), color="red", label="N(0,1)")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].set_title(f"{var}: histogram vs N(0,1)")

        stats.probplot(z_ok, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f"{var}: QQ plot vs normal")

        acf_vals = acf(z_ok, nlags=min(20, len(z_ok) - 1), fft=True)
        axes[1, 1].bar(range(len(acf_vals)), acf_vals, width=0.3)
        ci = 1.96 / np.sqrt(len(z_ok))
        axes[1, 1].axhline(ci, color="red", linestyle="--", linewidth=1)
        axes[1, 1].axhline(-ci, color="red", linestyle="--", linewidth=1)
        axes[1, 1].set_title(f"{var}: ACF of standardized innovations")

        # ---- extra panel: standardized (z-scored) innovation, 8Q rolling mean, mean, cumsum ----
        z_innov = innov_std[:, j]
        mean_innov = float(np.nanmean(z_innov))
        roll_mean_innov = pd.Series(z_innov).rolling(8, min_periods=1).mean().to_numpy()
        cumsum_innov = np.nancumsum(np.nan_to_num(z_innov, nan=0.0))

        ax = axes[2, 0]
        ax.plot(q, z_innov, alpha=0.35, color="tab:blue", label="innovation (z-scored)")
        ax.plot(q, roll_mean_innov, color="tab:blue", linewidth=2, label="8Q rolling mean")
        ax.axhline(mean_innov, color="tab:green", linestyle=":", label=f"mean={mean_innov:.3f}")
        ax_r = ax.twinx()
        ax_r.plot(q, cumsum_innov, color="tab:purple", linestyle="--", label="cumsum")
        ax_r.set_ylabel("cumulative sum", color="tab:purple")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_r.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
        ax.set_title(f"{var}: innovation (z-scored) rolling mean & cumulative sum")

        # ---- extra panel: z-scored economic series, mean, cumsum ----
        if Y is not None:
            z_series = Y[valid_idx, j]
        else:
            z_series = np.full(len(valid_idx), np.nan)
        mean_series = float(np.nanmean(z_series))
        cumsum_series = np.nancumsum(np.nan_to_num(z_series, nan=0.0))

        ax2 = axes[2, 1]
        ax2.plot(q, z_series, color="tab:blue", label=f"{var} (z-scored)")
        ax2.axhline(mean_series, color="tab:green", linestyle=":", label=f"mean={mean_series:.3f}")
        ax2_r = ax2.twinx()
        ax2_r.plot(q, cumsum_series, color="tab:purple", linestyle="--", label="cumsum")
        ax2_r.set_ylabel("cumulative sum", color="tab:purple")
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
        ax2.set_title(f"{var}: z-scored economic series mean & cumulative sum")

        fig.suptitle(f"Innovation diagnostics -- {var}")
        fig.tight_layout()
        fig.savefig(out_dir / f"innovation_diagnostics_{var}.png", dpi=140)
        plt.close(fig)

        pct_1_96 = float(np.mean(np.abs(z_ok) > 1.96) * 100)
        pct_2_58 = float(np.mean(np.abs(z_ok) > 2.58) * 100)
        emp_var = float(np.var(innov[:, j][np.isfinite(innov[:, j])], ddof=1))
        model_var_mean = float(np.nanmean(S_diag[:, j]))
        sw_stat, sw_p = stats.shapiro(z_ok) if len(z_ok) >= 3 else (np.nan, np.nan)
        lag1_acf = float(acf_vals[1]) if len(acf_vals) > 1 else np.nan

        report_rows.append({
            "variable": var,
            "n": len(z_ok),
            "mean_std_innov": float(np.mean(z_ok)),
            "var_std_innov": float(np.var(z_ok, ddof=1)),
            "empirical_innovation_variance": emp_var,
            "model_implied_S_mean": model_var_mean,
            "pct_outside_1_96": pct_1_96,
            "pct_outside_2_58": pct_2_58,
            "lag1_acf": lag1_acf,
            "shapiro_stat": float(sw_stat) if np.isfinite(sw_stat) else np.nan,
            "shapiro_p": float(sw_p) if np.isfinite(sw_p) else np.nan,
        })

    return pd.DataFrame(report_rows)


def _print_innovation_narrative(report_df: pd.DataFrame) -> None:
    print("\n--- innovation diagnostics narrative (per variable) ---")
    for _, r in report_df.iterrows():
        notes = []
        if abs(r["mean_std_innov"]) > 0.2:
            notes.append(f"mean bias present (mean={r['mean_std_innov']:.2f}, not ~0)")
        if r["var_std_innov"] < 0.5 or r["var_std_innov"] > 2.0:
            notes.append(f"variance mismatch vs. nominal 1.0 (var={r['var_std_innov']:.2f})")
        if abs(r["lag1_acf"]) > 2 / np.sqrt(r["n"]):
            notes.append(f"lag-1 autocorrelation exceeds the 95% band (acf1={r['lag1_acf']:.2f})")
        if np.isfinite(r["shapiro_p"]) and r["shapiro_p"] < 0.05:
            notes.append(f"Shapiro-Wilk rejects normality (p={r['shapiro_p']:.3f})")
        if r["pct_outside_1_96"] > 7.5 or r["pct_outside_1_96"] < 2.5:
            notes.append(f"tail rate at ±1.96 ({r['pct_outside_1_96']:.1f}%) deviates from the nominal 5%")
        verdict = "; ".join(notes) if notes else "no obvious departure from white noise found (does not by itself prove correctness)"
        print(f"  [{r['variable']}] {verdict}")


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def run_kf_em_diagnostics(
    country: str,
    max_iter: int = 30,
    update_P0: bool = False,
    output_dir: str | Path | None = None,
    lags: int = LAGS,
    window: int = WINDOW,
    tol: float = 1e-4,
    em_damping: float = 0.0,
    verbose: bool = True,
) -> dict:
    prep = _prepare_country(country)
    if prep is None:
        raise ValueError(f"No usable data for country={country!r} (insufficient rows or all-NaN columns).")

    out_dir = Path(output_dir) if output_dir else (
        _ROOT / "analysis" / "Dash_Output" / "kf_em_diagnostics" / f"{country}_updateP0-{update_P0}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    Y, Z = prep["Yd"], prep["Xd"]
    ENDO_use, EXO_use = prep["ENDO_use"], prep["EXO_use"]

    run = _run_em_with_full_history(
        Y=Y, Z=Z, lags=lags, window=window, max_iter=max_iter, tol=tol,
        em_damping=em_damping, update_P0=update_P0, verbose=verbose,
    )
    pack = run["e_step_store"]
    m, p = run["m"], run["p"]

    # ---- 1. EM convergence diagnostics ----
    iters_df = pd.DataFrame(run["iters"])
    iters_df.to_csv(out_dir / "em_convergence.csv", index=False)
    _plot_convergence(iters_df, tol=tol, out_path=out_dir / "em_convergence.png")

    # ---- 2. Covariance diagnostics ----
    _plot_covariance_time_traces(
        pack["P_filt"], pack["P_smooth"], pack["valid_mask"], prep["quarters"],
        out_path=out_dir / "covariance_time_traces.png",
    )

    # ---- 3. Filtered vs smoothed coefficient paths ----
    groups = _theta_index_groups(ENDO_use, EXO_use, lags, m)
    _plot_theta_paths(pack["theta_filt"], pack["theta_smooth"], pack["valid_mask"], prep["quarters"], groups, out_dir)
    valid_idx = np.where(pack["valid_mask"])[0]
    theta_cols = {}
    for eq_name, labels in groups.items():
        for label, idx in labels.items():
            theta_cols[f"{eq_name}|{label}|filt"] = pack["theta_filt"][valid_idx, idx]
            theta_cols[f"{eq_name}|{label}|smooth"] = pack["theta_smooth"][valid_idx, idx]
    theta_df = pd.DataFrame({"quarter": prep["quarters"][valid_idx], **theta_cols})
    theta_df.to_csv(out_dir / "theta_filt_smooth_paths.csv", index=False)

    # ---- 4. Observed vs KF estimates (in-sample fit, NOT forecast) ----
    fit_df = _plot_observed_vs_fitted(
        Y, pack["theta_filt"], pack["theta_smooth"], pack["H_list"], pack["valid_mask"],
        prep["quarters"], ENDO_use, out_dir,
    )
    fit_df.to_csv(out_dir / "observed_vs_fitted_residuals.csv", index=False)

    # ---- 5. Innovation diagnostics ----
    innov_res = _compute_innovations(pack, run["R"], ENDO_use)
    innov_report = _plot_innovation_diagnostics(innov_res, ENDO_use, prep["quarters"], out_dir, Y=Y)
    innov_report.to_csv(out_dir / "innovation_diagnostics_summary.csv", index=False)
    innov_rows = []
    for row, t in enumerate(innov_res["valid_idx"]):
        for j, var in enumerate(ENDO_use):
            innov_rows.append({
                "quarter": prep["quarters"][t], "variable": var,
                "innovation": innov_res["innovation"][row, j],
                "innovation_std": innov_res["innovation_std"][row, j],
                "model_S_diag": innov_res["S_diag"][row, j],
            })
    pd.DataFrame(innov_rows).to_csv(out_dir / "innovation_timeseries.csv", index=False)
    if verbose:
        _print_innovation_narrative(innov_report)

    # ---- one-step forecast illustration (P/Q/R tracing, item 9) ----
    fc_illustration = _one_step_forecast_illustration(pack, run["Q"], run["R"], m, len(ENDO_use))

    summary = {
        "country": country,
        "update_P0": update_P0,
        "n_iterations": len(run["iters"]),
        "converged": run["converged"],
        "final_log_likelihood": run["iters"][-1]["log_likelihood"],
        "final_obj": run["iters"][-1]["obj"],
        "final_trace_P0": run["iters"][-1]["trace_P0"],
        "final_trace_Q": run["iters"][-1]["trace_Q"],
        "final_trace_R": run["iters"][-1]["trace_R"],
        "output_dir": str(out_dir),
        "one_step_forecast_illustration": fc_illustration,
    }
    if verbose:
        print(f"\n[SAVE] diagnostics written to {out_dir}")
        print(
            f"[{country}] update_P0={update_P0}: n_iter={summary['n_iterations']} "
            f"converged={summary['converged']} final_ll={summary['final_log_likelihood']:.2f}"
        )
    return summary


def _cli():
    parser = argparse.ArgumentParser(description="Standalone KF/EM diagnostics for one country.")
    parser.add_argument("--country", required=True, help="ISO3 country code, e.g. KEN")
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--update-p0", action="store_true", help="Refresh P0 each EM iteration (experimental).")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_kf_em_diagnostics(
        country=args.country,
        max_iter=args.max_iter,
        update_P0=args.update_p0,
        output_dir=args.output_dir,
        tol=args.tol,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    _cli()
