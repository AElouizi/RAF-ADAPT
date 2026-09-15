"""
Pipeline complet C1–C5 : Bloc B → Bloc D → mart plateforme → rapport PDF.

Usage :
    py experiments/factorial_hybrid_adapt/run_c1c5_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    py = sys.executable
    run(
        [
            py,
            "-m",
            "experiments.factorial_hybrid_adapt.run_bloc_b_eval",
            "--run",
            "--folds",
            "all",
            "--no-resume",
        ]
    )
    run(
        [
            py,
            "-m",
            "experiments.factorial_hybrid_adapt.run_bloc_d_eval",
            "--run",
            "--no-resume",
        ]
    )
    run([py, str(BASE / "build_c1c5_platform_mart.py")])
    run([py, str(BASE / "build_best_algo_platform_mart.py")])
    run([py, str(BASE / "build_rapport_scientifique.py")])
    print("\n=== Pipeline C1–C5 terminé ===")
    print(f"Mart : {BASE / 'outputs' / 'platform_mart' / 'c1c5'}")
    print(f"PDF  : {BASE / 'outputs' / 'reports' / 'rapport_projet_RAF-ADAPT.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
