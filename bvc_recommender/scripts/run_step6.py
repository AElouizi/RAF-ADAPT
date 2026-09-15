"""
Sous-module B — Scoring TFT (pytorch-forecasting) vs baselines.

Usage (depuis la racine du projet) :
    python -m bvc_recommender.scripts.run_step6
    python -m bvc_recommender.scripts.run_step6 --max-epochs 8 --max-train-rows 40000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import (  # noqa: E402
    DATASET_DIR,
    FEATURES_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SPLIT_TEST_START,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
    load_env_file,
)
from bvc_recommender.models.regime_detector import REGIME_COLUMNS  # noqa: E402
from bvc_recommender.models.stock_scorer import (  # noqa: E402
    DEFAULT_MAX_TRAIN_ROWS,
    compare_with_baselines,
    train_and_evaluate_tft,
)

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_step6_report(tft_results: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "BVC Recommender — Sous-module B (TFT)",
        "random_state": RANDOM_STATE,
        "model": "TemporalFusionTransformer",
        "target": "alpha_ajuste_risque",
        "temporal_validation": {
            "train": f"≤ {SPLIT_TRAIN_END}",
            "validation": f"{SPLIT_TRAIN_END} → {SPLIT_VAL_END}",
            "test": f"≥ {SPLIT_TEST_START}",
        },
        "regime_injection": {
            "variables": ["is_bull", "is_neutral", "is_bear"],
            "role": "time_varying_known_reals",
            "source": "features_indices (règle à seuils, sous-module A)",
        },
        "quantiles": {
            "q10": "pessimiste",
            "q50": "central",
            "q90": "optimiste",
            "values": tft_results.get("quantiles"),
        },
        "data_version": tft_results.get("data_version"),
        "train_rows": tft_results.get("train_rows"),
        "training_history": tft_results.get("epoch_history"),
        "tft_splits": tft_results.get("splits"),
        "comparison_vs_baselines": comparison,
    }


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Sous-module B — TFT vs Baselines",
        "",
        f"- **Modèle** : Temporal Fusion Transformer (pytorch-forecasting)",
        f"- **Cible** : alpha ajusté au risque",
        f"- **Régime injecté** : is_bull, is_neutral, is_bear (seuils one-hot, known future)",
        f"- **Quantiles** : Q10 / Q50 / Q90",
        f"- **Validation** : train ≤ {SPLIT_TRAIN_END} | val ≤ {SPLIT_VAL_END} | test ≥ {SPLIT_TEST_START}",
        "",
        "## TFT — métriques",
        "",
        "| Split | Lignes | IC | Hit Ratio | Sharpe L/S |",
        "|-------|--------|-----|-----------|------------|",
    ]
    for split_name, data in report.get("tft_splits", {}).items():
        m = data.get("metrics", {})
        lines.append(
            f"| {split_name} | {data.get('rows', 0)} | {m.get('ic_mean', 0):.4f} | "
            f"{m.get('hit_ratio', 0):.1%} | {m.get('sharpe_long_short', 0):.2f} |"
        )

    comp = report.get("comparison_vs_baselines", {})
    lines.extend(["", "## Comparaison test (IC)", ""])
    tft_ic = comp.get("tft", {}).get("ic_mean")
    lines.append(f"- **TFT** : {tft_ic:.4f}" if tft_ic is not None else "- **TFT** : —")
    for name, row in comp.get("baselines", {}).items():
        lines.append(f"- **{name}** : {row.get('ic_mean', 0):.4f}")
    if comp.get("winner_ic"):
        lines.append(f"\n**Meilleur IC test** : {comp['winner_ic']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="BVC Recommender — Sous-module B (TFT)")
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_MAX_TRAIN_ROWS)
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    logger.info(
        "Sous-module B — TFT (epochs=%s, train_rows≤%s) | splits train≤%s / val≤%s / test≥%s",
        args.max_epochs,
        args.max_train_rows,
        SPLIT_TRAIN_END,
        SPLIT_VAL_END,
        SPLIT_TEST_START,
    )

    ml_path = DATASET_DIR / "ml_dataset.parquet"
    features_path = FEATURES_DIR / "features_indices.parquet"
    if not ml_path.is_file() or not features_path.is_file():
        raise FileNotFoundError(
            "ml_dataset ou features_indices manquant. Lancer étape 3 puis sous-module A (seuils)."
        )

    dataset = pd.read_parquet(ml_path)
    regime = pd.read_parquet(features_path)
    missing = [c for c in REGIME_COLUMNS if c not in regime.columns]
    if missing:
        raise ValueError(
            f"Colonnes régime absentes de features_indices : {missing}. "
            "Relancer python -m bvc_recommender.scripts.run_step4"
        )
    logger.info(
        "Dataset %s lignes | features_indices %s j | colonnes régime=%s",
        len(dataset),
        len(regime),
        REGIME_COLUMNS,
    )

    tft_results = train_and_evaluate_tft(
        dataset,
        regime,
        known_reals=REGIME_COLUMNS,
        max_epochs=args.max_epochs,
        max_train_rows=args.max_train_rows,
    )

    preds = tft_results.pop("predictions", pd.DataFrame())
    if not preds.empty:
        pred_path = DATASET_DIR / "tft_predictions.parquet"
        try:
            preds.to_parquet(pred_path, index=False)
        except Exception:
            pred_path = DATASET_DIR / "tft_predictions.csv"
            preds.to_csv(pred_path, index=False)
        logger.info(
            "Prédictions TFT : %s (%s lignes, quantiles q10/q50/q90)",
            pred_path,
            len(preds),
        )

    comparison = compare_with_baselines(tft_results, REPORTS_DIR / "step5_validation_report.json")
    report = build_step6_report(tft_results, comparison)

    json_path = REPORTS_DIR / "step6_validation_report.json"
    md_path = REPORTS_DIR / "tft_vs_baselines.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_markdown_report(report, md_path)

    logger.info("Rapport sous-module B : %s", json_path)
    if comparison.get("winner_ic"):
        logger.info("Meilleur IC test : %s", comparison["winner_ic"])

    logger.info("Sous-module B terminé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
