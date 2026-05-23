"""
Train/test backtest: train≤2013Q4, predict 2014Q1–2018Q4.

Two predictors compared against actual (1-step ahead, in z-space) — 3 lines total:
- KF frozen   : θ fixed at the final smoothed state after training EM; no measurement
                update on test.
- VARX frozen : fit one OLS VARX on all training quarters; coefficients frozen
                for the whole test window. 1-step-ahead uses actual lagged values.

Lightweight; reuses GVAR_LLM_pickle / gvar_kf_forecast without modification.

Note: panel is standardized on the full sample (via _prepare_country_panel_cached).
This is a small leak only in mean/std scaling and affects both methods equally,
so relative comparisons stay fair.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("GVAR_IMPORT_ONLY", "1")
import GVAR_LLM_pickle as gp  # noqa: E402

TRAIN_END = pd.Timestamp("2013-12-31")
TEST_START = pd.Timestamp("2014-01-01")
TEST_END = pd.Timestamp("2018-12-31")
OUTPUT_DIR = Path("Dash_Output/traintest_2014_2018")
RIDGE = 1e-6
EPS = 1e-8
MIN_EM_ITER = 10

# Default backtest countries (ISO3).
DEFAULT_COUNTRIES: list[str] = [
    "BRA",  # Brazil
    "MEX",  # Mexico
    "CHL",  # Chile
    "PHL",  # Philippines
    "IND",  # India
    "IDN",  # Indonesia
    "PER",  # Peru
    "THA",  # Thailand
    "COL",  # Colombia
    "KEN",  # Kenya
    "EGY",  # Egypt
    "ZAF",  # South Africa
]


def _build_xt(Y: np.ndarray, Z: np.ndarray, t: int, lags: int) -> np.ndarray:
    pieces = [Y[t - i, :] for i in range(1, lags + 1)]
    if Z is not None and Z.size > 0:
        pieces.append(Z[t - 1, :])
    return np.concatenate(pieces)


def _build_Ht(X_t: np.ndarray, mY: int, p: int, m: int) -> np.ndarray:
    H = np.zeros((mY, p))
    for j in range(mY):
        H[j, j * m : (j + 1) * m] = X_t
    return H


def _varx_fit_train(
    Y: np.ndarray, Z: np.ndarray, t_train_end: int, lags: int, mY: int, m: int
) -> np.ndarray | None:
    """One-shot OLS VARX fit on training rows [lags+1 .. t_train_end].

    Returns coef (m, mY) so that y_t ≈ x_t @ coef. None if too few rows.
    """
    s0 = lags + 1
    T = t_train_end + 1 - s0
    if T <= max(m, 5):
        return None
    X_tr = np.zeros((T, m))
    Y_tr = np.zeros((T, mY))
    for k, s in enumerate(range(s0, t_train_end + 1)):
        X_tr[k] = _build_xt(Y, Z, s, lags)
        Y_tr[k] = Y[s, :]
    return np.linalg.solve(X_tr.T @ X_tr + RIDGE * np.eye(m), X_tr.T @ Y_tr)


def _final_smoothed_state_after_em(
    Y_train: np.ndarray, Z_train: np.ndarray, res: dict, lags: int
) -> np.ndarray:
    """Run one final smoother with EM-final Q/R and return the latest training state."""
    theta_filt, P_filt, theta_pred, P_pred, *_rest, valid_mask = gp.kf_e_step_store(
        Y=Y_train,
        Z_all=Z_train,
        theta0=res["theta0"],
        Q=res["Q"],
        R=res["R"],
        P0=res["P0"],
        lags=lags,
        eps=EPS,
    )
    theta_smooth, _, _ = gp.rts_smoother(
        theta_filt, P_filt, theta_pred, P_pred, valid_mask, eps=EPS
    )
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        raise ValueError("No valid KF states in training sample")
    return theta_smooth[int(valid_idx[-1])].reshape(-1, 1)


def run_traintest_one_country(
    country: str,
    *,
    train_end: pd.Timestamp = TRAIN_END,
    test_start: pd.Timestamp = TEST_START,
    test_end: pd.Timestamp = TEST_END,
    max_em_iter: int = 10,
) -> dict | None:
    prep = gp._prepare_country_panel_cached(
        gp.PATH,
        country,
        gp.COL_COUNTRY,
        gp.COL_TIME,
        list(gp.ENDO),
        list(gp.EXO),
        gp.lags + 5,
    )
    if prep is None:
        return None

    Y = np.asarray(prep["Yd"], dtype=float)
    Z = np.asarray(prep["Xd"], dtype=float)
    quarters = pd.to_datetime(prep["quarters"])
    ENDO_use = list(prep["ENDO_use"])
    mY = len(ENDO_use)
    lags = int(gp.lags)
    mX = Z.shape[1] if Z is not None and Z.size > 0 else 0
    m = lags * mY + mX
    p = m * mY

    train_mask = quarters <= train_end
    test_mask = (quarters >= test_start) & (quarters <= test_end)
    if train_mask.sum() < lags + 10 or test_mask.sum() < 1:
        return None

    t_train_end = int(np.where(train_mask)[0][-1])
    test_idx = np.where(test_mask)[0]

    Y_train = Y[: t_train_end + 1]
    Z_train = Z[: t_train_end + 1]
    res = gp.run_kf_em(
        Y_train,
        Z_train,
        lags=lags,
        max_em_iter=max(int(max_em_iter), MIN_EM_ITER),
        tol=-1.0,
        verbose=False,
    )
    theta_frozen = _final_smoothed_state_after_em(Y_train, Z_train, res, lags)

    varx_coef = _varx_fit_train(Y, Z, t_train_end, lags, mY, m)

    test_q = quarters[test_idx]
    y_true = Y[test_idx]
    y_kf_frozen = np.full((len(test_idx), mY), np.nan)
    y_varx = np.full((len(test_idx), mY), np.nan)

    for k, t in enumerate(test_idx):
        X_t = _build_xt(Y, Z, t, lags)
        H = _build_Ht(X_t, mY, p, m)
        y_kf_frozen[k] = (H @ theta_frozen).ravel()
        if varx_coef is not None:
            y_varx[k] = X_t @ varx_coef

    def _rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        diff = a - b
        return np.sqrt(np.nanmean(diff**2, axis=0))

    return {
        "country": country,
        "ENDO_use": ENDO_use,
        "train_end_quarter": quarters[t_train_end],
        "test_quarters": test_q,
        "y_true": y_true,
        "y_kf_frozen": y_kf_frozen,
        "y_varx": y_varx,
        "rmse_kf_frozen": _rmse(y_true, y_kf_frozen),
        "rmse_varx": _rmse(y_true, y_varx),
    }


def plot_traintest(out: dict, *, output_dir: Path = OUTPUT_DIR) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    country = out["country"]
    endo = out["ENDO_use"]
    mY = len(endo)

    fig, axes = plt.subplots(mY, 1, figsize=(11, 3.0 * mY), sharex=True, squeeze=False)
    axes = axes.flatten()
    tq = pd.to_datetime(out["test_quarters"])

    for j in range(mY):
        ax = axes[j]
        ax.plot(tq, out["y_true"][:, j], color="black", linewidth=1.8, label="actual")
        ax.plot(
            tq,
            out["y_kf_frozen"][:, j],
            color="C1",
            linestyle="--",
            marker="o",
            markersize=3,
            label=f"KF frozen 2013Q4 (RMSE={out['rmse_kf_frozen'][j]:.3f})",
        )
        ax.plot(
            tq,
            out["y_varx"][:, j],
            color="C2",
            linewidth=1.4,
            label=f"VARX frozen 2013Q4 (RMSE={out['rmse_varx'][j]:.3f})",
        )
        ax.set_ylabel(endo[j])
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)

    cname = gp.country_display_name(country, gp.ISO3_TO_COUNTRY)
    fig.suptitle(
        f"{cname} — train ≤ {pd.Timestamp(out['train_end_quarter']).date()} / "
        f"test 2014Q1–2018Q4 (z-space, 1-step ahead)",
        y=1.01,
        fontsize=11,
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / f"traintest_{country}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="2014–2018 train/test backtest: KF vs VARX")
    parser.add_argument("--countries", type=str, default="", help="Comma-separated ISO3 list")
    parser.add_argument("--max-em-iter", type=int, default=10)
    args = parser.parse_args()
    countries = [c.strip() for c in args.countries.split(",") if c.strip()] or list(DEFAULT_COUNTRIES)

    rows = []
    for c in countries:
        out = run_traintest_one_country(c, max_em_iter=args.max_em_iter)
        if out is None:
            print(f"[SKIP] {c}: insufficient data")
            continue
        plot_traintest(out)
        for j, name in enumerate(out["ENDO_use"]):
            rows.append(
                {
                    "country": c,
                    "endo": name,
                    "kf_frozen": float(out["rmse_kf_frozen"][j]),
                    "varx_frozen": float(out["rmse_varx"][j]),
                }
            )
        print(
            f"[OK] {c}: kf_frozen={out['rmse_kf_frozen'].mean():.3f}, "
            f"varx_frozen={out['rmse_varx'].mean():.3f}"
        )

    if rows:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        out_csv = OUTPUT_DIR / "rmse_summary.csv"
        df.to_csv(out_csv, index=False)
        print(f"[SAVE] {out_csv}")
        print("\nMean RMSE across all countries × equations:")
        print(df[["kf_frozen", "varx_frozen"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
