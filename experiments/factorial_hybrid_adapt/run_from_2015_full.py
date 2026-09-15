"""Restore / run pipeline from 2015-06-03 (feature availability date)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def run(step: str, argv: list[str]) -> None:
    print(f"\n========== {step} ==========", flush=True)
    cmd = [PY, *argv]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--from-step", type=int, default=1)
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()
    s = args.from_step
    if s <= 1:
        run("1 features 2015-06-03", ["-m", "bvc_recommender.scripts.run_step2", "--use-processed", "--skip-supabase-write"])
    if s <= 2:
        run("2 ml_dataset", ["-m", "bvc_recommender.scripts.run_step3", "--use-processed", "--use-features", "--skip-supabase-write"])
    if s <= 3:
        run("3 régimes", ["-m", "bvc_recommender.scripts.run_step4", "--use-processed", "--start-year", "2015", "--end-year", "2026", "--skip-supabase"])
    if s <= 4:
        run("4 plis WF", ["-m", "experiments.factorial_hybrid_adapt.run_wf_etape1_folds"])
        run("4b régimes WF", ["-m", "experiments.factorial_hybrid_adapt.run_wf_etape2_regimes"])
    if (not args.skip_train) and s <= 5:
        run("5 train C1-C4", ["-m", "experiments.factorial_hybrid_adapt.run_wf_etape3_train"])
    if (not args.skip_train) and s <= 6:
        run("6 recos", ["experiments/factorial_hybrid_adapt/run_stage1_recommendations.py", "--folds", "14"])
    if (not args.skip_train) and s <= 7:
        run(
            "7 NSGA Pareto",
            [
                "experiments/factorial_hybrid_adapt/run_stage2_nsga.py",
                "--liquidity-mode",
                "pareto",
                "--cells",
                "1,2,3,4",
                "--eval-start",
                "2015-06",
                "--eval-end",
                "2025-06",
                "--w-max",
                "0.10",
                "--output-dir",
                "experiments/factorial_hybrid_adapt/outputs/portfolios_stage2_final_pareto",
            ],
        )
    if (not args.skip_train) and s <= 8:
        run("8 assemble", ["experiments/factorial_hybrid_adapt/assemble_final_experiment.py"])
        run("8b mart plateforme", ["experiments/factorial_hybrid_adapt/build_platform_mart.py"])
    print("OK pipeline depuis 2015-06-03", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
