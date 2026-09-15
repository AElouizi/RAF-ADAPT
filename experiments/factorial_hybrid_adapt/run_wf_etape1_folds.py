"""
CLI Étape 1 WF — génère et affiche les plis rolling (sans entraînement).

Usage :
    python -m experiments.factorial_hybrid_adapt.run_wf_etape1_folds
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import DATASET_DIR, load_env_file  # noqa: E402
from experiments.factorial_hybrid_adapt.paths import (  # noqa: E402
    REPORTS_DIR,
    ensure_output_dirs,
)
from experiments.factorial_hybrid_adapt.walk_forward_folds import (  # noqa: E402
    available_months_from_dates,
    folds_to_frame,
    generate_nonoverlapping_blocks,
    generate_rolling_folds,
    summarize_fold_grid,
)

load_env_file(ROOT / ".env")


def main() -> int:
    ensure_output_dirs()
    path = DATASET_DIR / "ml_dataset.parquet"
    dates = pd.read_parquet(path, columns=["date_cours"])["date_cours"]
    months = available_months_from_dates(dates)

    overlapping = generate_rolling_folds(months)
    blocks = generate_nonoverlapping_blocks(months)
    summary = summarize_fold_grid(overlapping, blocks, months=months)

    ov_df = folds_to_frame(overlapping)
    bl_df = folds_to_frame(blocks)

    ov_path = REPORTS_DIR / "wf_folds_overlapping.csv"
    bl_path = REPORTS_DIR / "wf_folds_nonoverlapping.csv"
    sum_path = REPORTS_DIR / "wf_etape1_folds_summary.json"
    ov_df.to_csv(ov_path, index=False)
    bl_df.to_csv(bl_path, index=False)
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Walk-forward Etape 1 — Generateur de plis ===")
    print(
        f"Donnees : {summary['data_first_month']} -> {summary['data_last_month']} "
        f"({summary['n_months_available']} mois)"
    )
    print(
        f"Design : train={summary['train_months']}m | test={summary['test_months']}m | "
        f"pas_chevauchant={summary['step_months_overlapping']}m | "
        f"pas_blocs={summary['step_months_nonoverlapping']}m | embargo=aucun"
    )
    print()
    print(
        f"Plis mensuels chevauchants : {summary['n_folds_overlapping']} "
        f"(estimation ~{summary['expected_overlapping_approx']}, "
        f"delta={summary['delta_overlapping_vs_expected']:+d})"
    )
    print(
        f"Blocs non-chevauchants 6m  : {summary['n_blocks_nonoverlapping']} "
        f"(estimation ~{summary['expected_blocks_approx']}, "
        f"delta={summary['delta_blocks_vs_expected']:+d})"
    )

    if summary["alert_overlapping"] or summary["alert_blocks"]:
        print("\nALERTE : ecart notable vs estimation (~78 / ~13) — verifier le bornage.")
    else:
        print("\nOK : effectifs proches de l'estimation (~78 / ~13).")

    def _show(label: str, folds_df: pd.DataFrame) -> None:
        print(f"\n--- {label} : 3 premiers ---")
        cols = [
            "fold_id",
            "train_start_month",
            "train_end_month",
            "test_start_month",
            "test_end_month",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
        ]
        print(folds_df.head(3)[cols].to_string(index=False))
        print(f"--- {label} : 3 derniers ---")
        print(folds_df.tail(3)[cols].to_string(index=False))

    _show("Plis chevauchants", ov_df)
    _show("Blocs non-chevauchants", bl_df)

    print(f"\nArtefacts :\n  - {ov_path}\n  - {bl_path}\n  - {sum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
