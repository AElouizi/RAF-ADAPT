"""
CLI Étape 3 WF — réentraînement 4 cellules × plis chevauchants.

Usage :
    python -m experiments.factorial_hybrid_adapt.run_wf_etape3_train
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import DATASET_DIR, load_env_file  # noqa: E402
from experiments.factorial_hybrid_adapt.data_utils import (  # noqa: E402
    attach_regime_to_panel,
    load_ml_dataset,
)
from experiments.factorial_hybrid_adapt.paths import (  # noqa: E402
    OUTPUT_DIR,
    REPORTS_DIR,
    ensure_output_dirs,
)
from experiments.factorial_hybrid_adapt.walk_forward_folds import (  # noqa: E402
    available_months_from_dates,
    generate_rolling_folds,
)
from experiments.factorial_hybrid_adapt.walk_forward_stage1 import (  # noqa: E402
    DEFAULT_RF_PARAMS,
    consolidate_predictions,
    run_all_overlapping_folds,
)

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    ensure_output_dirs()
    parts_dir = OUTPUT_DIR / "wf_fold_parts_from_2015"
    parts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Chargement panel + régime…")
    df = load_ml_dataset()
    df = attach_regime_to_panel(df)
    months = available_months_from_dates(df["date_cours"])
    folds = generate_rolling_folds(months)
    logger.info("Plis à traiter : %s | RF params=%s", len(folds), DEFAULT_RF_PARAMS)

    results = run_all_overlapping_folds(
        df, folds, parts_dir=parts_dir, rf_params=DEFAULT_RF_PARAMS, resume=True
    )

    metrics = results["metrics"]
    metrics_path = REPORTS_DIR / "wf_etape3_fold_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    preds_path = REPORTS_DIR / "wf_stage1_predictions.parquet"
    logger.info("Consolidation des prédictions…")
    preds = consolidate_predictions(parts_dir, preds_path)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_folds_expected": results["n_folds_expected"],
        "n_folds_done": results["n_folds_done"],
        "n_prediction_rows": int(len(preds)),
        "rf_params": results["rf_params"],
        "note": (
            "Hyperparams RF figés (pas de re-tuning par pli). "
            "Métriques descriptives par pli sauvegardées ; "
            "agrégation inter-plis = étape 4."
        ),
        "preview_mean_rank_ic_by_cell": metrics.groupby("cell_id")["rank_ic"]
        .mean()
        .round(4)
        .to_dict(),
    }
    sum_path = REPORTS_DIR / "wf_etape3_summary.json"
    # json keys str
    summary["preview_mean_rank_ic_by_cell"] = {
        str(k): float(v) for k, v in summary["preview_mean_rank_ic_by_cell"].items()
    }
    sum_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Walk-forward Etape 3 — Reentrainement termine ===")
    print(f"Plis faits : {results['n_folds_done']} / {results['n_folds_expected']}")
    print(f"Lignes predictions : {len(preds)}")
    print(f"RF params : {results['rf_params']}")
    print("\nApercu Rank-IC moyen par cellule (brut, avant etape 4) :")
    for cell_id, g in metrics.groupby("cell_id"):
        print(
            f"  C{cell_id}: mean={g['rank_ic'].mean():.4f} | "
            f"median={g['rank_ic'].median():.4f} | std={g['rank_ic'].std():.4f}"
        )
    print(f"\nArtefacts :\n  - {metrics_path}\n  - {preds_path}\n  - {sum_path}\n  - {parts_dir}")
    print(f"Termine : {datetime.now(timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
