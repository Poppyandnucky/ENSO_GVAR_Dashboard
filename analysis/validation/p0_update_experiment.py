"""Compare fixed-P0 (current) vs. refreshed-P0 (experimental) EM behavior.

For each of the 12 dashboard countries, runs `trp.kalman_core.run_kf_em` twice
on the same prepared data:
  - update_P0=False (current production behavior: theta0 refreshes each EM
    iteration from the first valid smoothed state; P0 stays fixed at the
    one-time VAR-derived value)
  - update_P0=True (experimental: P0 also refreshes each iteration from the
    first valid smoothed covariance, symmetrized with an eigenvalue floor)

Records per-iteration diagnostics (trace/eigenvalue range of P0, Q, R; theta0
change norm and its standardized version relative to smoothed uncertainty;
log-likelihood; objective) for both methods, flags objective numerical
problems, and writes a comparison CSV. Does not change any production
default -- this is a read-only experiment.

Run from repo root:
    python analysis/validation/p0_update_experiment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SB_DIR = _ROOT / "structural_break"
if str(_SB_DIR) not in sys.path:
    sys.path.insert(0, str(_SB_DIR))

from trp.kalman_core import run_kf_em  # noqa: E402

import GVAR_LLM_pickle as gp  # noqa: E402

OUT_CSV = _ROOT / "analysis" / "Dash_Output" / "p0_update_experiment.csv"
COUNTRIES = list(gp.COUNTRIES)


def flag_issues(country: str, method: str, hist: dict) -> list[str]:
    flags = []
    n = len(hist["obj"])
    if n == 0:
        flags.append("no iterations recorded")
        return flags

    # Non-finite anywhere.
    for key in ("trace_Q", "trace_R", "trace_P0", "eig_min_Q", "eig_min_R", "eig_min_P0",
                "theta0_change_norm", "theta0_change_std", "log_likelihood", "obj"):
        arr = np.asarray(hist[key], dtype=float)
        if not np.isfinite(arr).all():
            flags.append(f"non-finite values in {key}")

    # Indefinite / singular covariance at any iteration.
    if any(v < -1e-6 for v in hist["eig_min_Q"]):
        flags.append("Q went indefinite (negative eigenvalue) at some iteration")
    if any(v < -1e-6 for v in hist["eig_min_R"]):
        flags.append("R went indefinite (negative eigenvalue) at some iteration")
    if any(v < -1e-6 for v in hist["eig_min_P0"]):
        flags.append("P0 went indefinite (negative eigenvalue) at some iteration")

    # Convergence: did |delta obj| ever drop below the run_kf_em tol (1e-4) before max_em_iter?
    if n >= gp.MAX_EM_ITER:
        deltas = np.abs(np.diff(hist["obj"]))
        if len(deltas) == 0 or deltas[-1] >= 1e-4:
            flags.append(f"did not converge within max_em_iter={gp.MAX_EM_ITER} (ran full budget)")

    # Trace trajectories: exploding or collapsing (order-of-magnitude check across iterations).
    for key in ("trace_Q", "trace_R", "trace_P0"):
        arr = np.asarray(hist[key], dtype=float)
        if arr.min() > 0 and (arr.max() / arr.min()) > 1e4:
            flags.append(f"{key} spans >1e4x across iterations (min={arr.min():.3e}, max={arr.max():.3e})")

    # R monotonically increasing without stabilizing (never decreases after iter 3).
    tr_r = np.asarray(hist["trace_R"], dtype=float)
    if n >= 5:
        diffs = np.diff(tr_r[2:])
        if np.all(diffs > 0):
            flags.append("trace(R) increased every iteration after warmup, never stabilized")

    # Unusually large standardized theta change (loose descriptive threshold, not a claim of "wrong").
    std_changes = np.asarray(hist["theta0_change_std"], dtype=float)
    std_changes = std_changes[np.isfinite(std_changes)]
    if len(std_changes) and std_changes.max() > 10:
        flags.append(
            f"standardized theta0 change reached {std_changes.max():.1f} "
            f"(large relative to smoothed uncertainty) at some iteration"
        )

    return flags


def run_one(country: str, update_P0: bool):
    prep = gp._prepare_country_panel_cached(
        PATH=gp.PATH, country=country, COL_COUNTRY=gp.COL_COUNTRY, COL_TIME=gp.COL_TIME,
        ENDO=gp.ENDO, EXO=gp.EXO, min_T=gp.lags + 5,
    )
    if prep is None:
        return None, None
    res = run_kf_em(
        Y=prep["Yd"], Z=prep["Xd"], lags=gp.lags, window=40, max_em_iter=gp.MAX_EM_ITER,
        tol=1e-4, em_damping=0.0, verbose=False, update_P0=update_P0,
    )
    return prep, res


def main():
    rows = []
    summary_lines = []
    for country in COUNTRIES:
        for method, update_P0 in (("fixed_P0", False), ("refreshed_P0", True)):
            prep, res = run_one(country, update_P0)
            if prep is None or res is None:
                summary_lines.append(f"[{country}] {method}: SKIPPED (insufficient data)")
                continue
            hist = res["em_history"]
            n_iter = len(hist["obj"])
            for it in range(n_iter):
                rows.append({
                    "country": country,
                    "method": method,
                    "iter": it + 1,
                    "obj": hist["obj"][it],
                    "log_likelihood": hist["log_likelihood"][it],
                    "trace_Q": hist["trace_Q"][it],
                    "eig_min_Q": hist["eig_min_Q"][it],
                    "eig_max_Q": hist["eig_max_Q"][it],
                    "trace_R": hist["trace_R"][it],
                    "eig_min_R": hist["eig_min_R"][it],
                    "eig_max_R": hist["eig_max_R"][it],
                    "trace_P0": hist["trace_P0"][it],
                    "eig_min_P0": hist["eig_min_P0"][it],
                    "eig_max_P0": hist["eig_max_P0"][it],
                    "theta0_change_norm": hist["theta0_change_norm"][it],
                    "theta0_change_std": hist["theta0_change_std"][it],
                })
            flags = flag_issues(country, method, hist)
            flag_str = "; ".join(flags) if flags else "no issues flagged"
            summary_lines.append(
                f"[{country}] {method}: n_iter={n_iter} final_ll={hist['log_likelihood'][-1]:.2f} "
                f"final_obj={hist['obj'][-1]:.5f} -- {flag_str}"
            )

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[SAVE] {OUT_CSV}  ({len(df)} rows)")
    print()
    for line in summary_lines:
        print(line)

    print("\n=== final-iteration comparison: fixed_P0 vs refreshed_P0 ===")
    final = df.sort_values("iter").groupby(["country", "method"]).tail(1)
    piv_ll = final.pivot(index="country", columns="method", values="log_likelihood")
    piv_obj = final.pivot(index="country", columns="method", values="obj")
    piv_trP0 = final.pivot(index="country", columns="method", values="trace_P0")
    for c in COUNTRIES:
        if c not in piv_ll.index:
            continue
        print(
            f"[{c}] final_ll: fixed={piv_ll.loc[c,'fixed_P0']:.2f} vs refreshed={piv_ll.loc[c,'refreshed_P0']:.2f}  |  "
            f"final_obj: fixed={piv_obj.loc[c,'fixed_P0']:.5f} vs refreshed={piv_obj.loc[c,'refreshed_P0']:.5f}  |  "
            f"trace(P0): fixed={piv_trP0.loc[c,'fixed_P0']:.3e} vs refreshed={piv_trP0.loc[c,'refreshed_P0']:.3e}"
        )


if __name__ == "__main__":
    main()
