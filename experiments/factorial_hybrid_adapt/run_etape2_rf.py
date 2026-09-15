"""
CLI Étape 2 — entraînement Random Forest (expérience factorial_hybrid_adapt).

Usage (depuis la racine du projet) :
    python -m experiments.factorial_hybrid_adapt.run_etape2_rf
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import (  # noqa: E402
    DATASET_DIR,
    SPLIT_TEST_START,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
    load_env_file,
)
from experiments.factorial_hybrid_adapt.paths import (  # noqa: E402
    MODELS_DIR,
    REPORTS_DIR,
    ensure_output_dirs,
)
from experiments.factorial_hybrid_adapt.random_forest_scorer import (  # noqa: E402
    save_rf_artifacts,
    train_and_evaluate_random_forest,
)

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_ml_dataset() -> pd.DataFrame:
    path = DATASET_DIR / "ml_dataset.parquet"
    if not path.is_file():
        path = DATASET_DIR / "ml_dataset.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "ml_dataset manquant. Lancer run_step3 --use-processed --use-features."
        )
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def main() -> int:
    ensure_output_dirs()
    logger.info(
        "Étape 2 RF | splits train≤%s | val≤%s | test≥%s",
        SPLIT_TRAIN_END,
        SPLIT_VAL_END,
        SPLIT_TEST_START,
    )
    df = load_ml_dataset()
    logger.info("Dataset chargé : %s lignes | colonnes=%s", len(df), len(df.columns))

    results = train_and_evaluate_random_forest(df, include_regime=False)
    paths = save_rf_artifacts(results, models_dir=MODELS_DIR, reports_dir=REPORTS_DIR)

    print("\n=== Étape 2 — Random Forest (sans régime) ===")
    print(f"Features          : {results['n_features']}")
    print(f"Best params       : {results['best_params']}")
    print(f"Tuning métrique   : {results['tuning_metric']}")
    for split_name, split_data in results["splits"].items():
        m = split_data["metrics"]
        print(
            f"[{split_name}] rows={split_data['rows']} | "
            f"Rank-IC(Spearman)={m.get('rank_ic_mean'):.4f} | "
            f"IC(Pearson)={m.get('ic_mean'):.4f} | "
            f"Hit={100 * (m.get('hit_ratio') or 0):.1f}% | "
            f"Sharpe L/S={m.get('sharpe_long_short'):.2f}"
        )
    print(f"\nArtefacts :")
    for k, p in paths.items():
        print(f"  - {k}: {p}")
    print(f"Terminé : {datetime.now(timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
