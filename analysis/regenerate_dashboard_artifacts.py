"""Regenerate precomputed artifacts consumed by the Streamlit dashboard.

This is the dashboard-local orchestration layer for the heavier KF/EM pipeline.
It runs the same generation steps that used to live in the separate
`KF_main_code` working directory, but keeps all outputs inside this repository.

Run from anywhere:
    python analysis/regenerate_dashboard_artifacts.py

Outputs refreshed:
  - Dash_Input/gvar_pipeline_results.pkl
  - structural_break/gvar_pipeline_results.pkl
  - Dash_Input/GVAR_LLM_EM_plots.pdf and LLM summary files
  - structural_break/map1998-2024/*.html
  - analysis/Dash_Input/gvar_forecast_results.pkl
  - Dash_Input/gvar_forecast_results.pkl
  - analysis/Dash_Output/forecast_enso_{mean,min,max}/...
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)


def run_step(name: str, args: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(f"\n== {name} ==")
    print("cwd:", cwd)
    print("cmd:", " ".join(args))
    subprocess.run(args, cwd=cwd, env=env, check=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Expected artifact not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[copy] {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"Expected artifact directory not found: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    print(f"[copy] {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT),
            str(ROOT / "structural_break"),
            env.get("PYTHONPATH", ""),
        ]
    ).rstrip(os.pathsep)
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate TRP dashboard artifacts.")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip EM/pipeline pickle generation.")
    parser.add_argument("--skip-visuals", action="store_true", help="Skip structural-break PDFs/maps.")
    parser.add_argument("--skip-forecast", action="store_true", help="Skip forecast pickle and scenario PNGs.")
    args = parser.parse_args()

    env = build_env()

    if not args.skip_pipeline:
        env_pipeline = env.copy()
        env_pipeline["GVAR_IMPORT_ONLY"] = "0"
        run_step(
            "Generate pipeline pickle",
            [str(PYTHON), "structural_break/GVAR_LLM_pickle.py"],
            cwd=ROOT,
            env=env_pipeline,
        )
        copy_file(
            ROOT / "Dash_Input" / "gvar_pipeline_results.pkl",
            ROOT / "structural_break" / "gvar_pipeline_results.pkl",
        )

    if not args.skip_visuals:
        run_step(
            "Generate structural-break visuals and maps",
            [
                str(PYTHON),
                "structural_break/Dashboard_visualize_from_pickle.py",
                "--pickle",
                "Dash_Input/gvar_pipeline_results.pkl",
            ],
            cwd=ROOT,
            env=env,
        )
        generated_maps = ROOT / "Dash_Input" / "gvar_llm_time_slice_maps"
        if generated_maps.is_dir():
            copy_tree(generated_maps, ROOT / "structural_break" / "map1998-2024")

    if not args.skip_forecast:
        run_step(
            "Generate ENSO and commodity history/forecast plots",
            [str(PYTHON), "analysis/plot_exogenous_series.py"],
            cwd=ROOT,
            env=env,
        )
        run_step(
            "Generate forecast pickle and scenario charts",
            [str(PYTHON), "Dash_Output/gvar_kf_forecast.py"],
            cwd=ROOT / "analysis",
            env=env,
        )
        copy_file(
            ROOT / "analysis" / "Dash_Input" / "gvar_forecast_results.pkl",
            ROOT / "Dash_Input" / "gvar_forecast_results.pkl",
        )

    print("\nDone. Dashboard artifacts regenerated.")


if __name__ == "__main__":
    main()
