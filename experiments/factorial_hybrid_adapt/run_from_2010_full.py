"""
Relance A–Z depuis 2010 (mêmes méthodes C1–C4, train 36m, NSGA 3 obj, Knee).

Étapes :
  1. Features techniques + fondamentales (INDICATORS_MIN_DATE=2010-01-01)
  2. Contexte marché + ml_dataset
  3. Régimes (seuils documentés 2015–2020, appliqués dès 2010)
  4. Plis walk-forward
  5. Réentraînement C1–C4 sur chaque pli
  6. Recommandations + inputs Stage 2
  7. NSGA-III Pareto C1–C4
  8. Backtest + mart plateforme

Usage :
  py -3 experiments/factorial_hybrid_adapt/run_from_2010_full.py
  py -3 experiments/factorial_hybrid_adapt/run_from_2010_full.py --from-step 5
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def run(step: str, args: list[str]) -> None:
    print(f"\n========== {step} ==========", flush=True)
    cmd = [PY, *args]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--from-step", type=int, default=1, help="Reprendre à l'étape N (1–8)")
    args = p.parse_args()
    s = args.from_step

    if s <= 1:
        run(
            "1 features (step2)",
            [
                "-m",
                "bvc_recommender.scripts.run_step2",
                "--use-processed",
                "--skip-supabase-write",
            ],
        )
    if s <= 2:
        run(
            "2 ml_dataset (step3)",
            [
                "-m",
                "bvc_recommender.scripts.run_step3",
                "--use-processed",
                "--use-features",
                "--skip-supabase-write",
            ],
        )
    if s <= 3:
        run(
            "3 régimes (step4)",
            [
                "-m",
                "bvc_recommender.scripts.run_step4",
                "--use-processed",
                "--start-year",
                "2010",
                "--end-year",
                "2026",
                "--skip-supabase",
            ],
        )
    if s <= 4:
        run("4 plis WF", ["-m", "experiments.factorial_hybrid_adapt.run_wf_etape1_folds"])
        run("4b couverture régimes", ["-m", "experiments.factorial_hybrid_adapt.run_wf_etape2_regimes"])
    if s <= 5:
        run("5 train C1–C4 WF", ["-m", "experiments.factorial_hybrid_adapt.run_wf_etape3_train"])
    if s <= 6:
        run(
            "6 recommandations",
            ["experiments/factorial_hybrid_adapt/run_stage1_recommendations.py", "--folds", "14"],
        )
    if s <= 7:
        run(
            "7 NSGA Pareto",
            [
                "experiments/factorial_hybrid_adapt/run_stage2_nsga.py",
                "--liquidity-mode",
                "pareto",
                "--cells",
                "1,2,3,4",
                "--eval-start",
                "2010-01",
                "--eval-end",
                "2025-06",
                "--w-max",
                "0.10",
                "--output-dir",
                "experiments/factorial_hybrid_adapt/outputs/portfolios_stage2_final_pareto",
            ],
        )
    if s <= 8:
        run("8 assemble + mart", ["experiments/factorial_hybrid_adapt/assemble_final_experiment.py"])
        run("8b mart plateforme", ["experiments/factorial_hybrid_adapt/build_platform_mart.py"])
    print("\nTerminé : pipeline 2010 → 2025.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
