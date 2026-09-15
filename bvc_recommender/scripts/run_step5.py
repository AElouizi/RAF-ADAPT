"""
Pipeline Étape 5 — modèles baseline et métriques IC / Hit Ratio / Sharpe.

Usage (depuis la racine du projet) :
    python -m bvc_recommender.scripts.run_step5
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
    RANDOM_STATE,
    REPORTS_DIR,
    SPLIT_TEST_START,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
    load_env_file,
)
from bvc_recommender.features.dataset_builder import TARGET_COLUMN  # noqa: E402
from bvc_recommender.models.baseline_models import (  # noqa: E402
    BASELINE_MODEL_NAMES,
    train_and_evaluate_baselines,
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


def build_step5_report(results: dict[str, Any], dataset: pd.DataFrame) -> dict[str, Any]:
    split_counts = dataset["split"].value_counts().to_dict() if "split" in dataset.columns else {}

    leaderboard: list[dict[str, Any]] = []
    for model_name, model_data in results.get("models", {}).items():
        for split_name, split_data in model_data.get("splits", {}).items():
            m = split_data.get("metrics", {})
            leaderboard.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "ic_mean": m.get("ic_mean"),
                    "rank_ic_mean": m.get("rank_ic_mean"),
                    "hit_ratio": m.get("hit_ratio"),
                    "sharpe_long_short": m.get("sharpe_long_short"),
                }
            )

    test_rows = [r for r in leaderboard if r["split"] == "test"]
    best_ic = max(test_rows, key=lambda r: r.get("ic_mean") or -999) if test_rows else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "BVC Recommender — Étape 5",
        "random_state": RANDOM_STATE,
        "target": TARGET_COLUMN,
        "splits_config": {
            "train_end": SPLIT_TRAIN_END,
            "val_end": SPLIT_VAL_END,
            "test_start": SPLIT_TEST_START,
            "counts": split_counts,
        },
        "n_features": results.get("n_features"),
        "clip_bounds": results.get("clip_bounds"),
        "models": results.get("models"),
        "leaderboard": leaderboard,
        "best_test_ic": best_ic,
    }


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Baselines — Étape 5",
        "",
        f"- **Cible** : `{report['target']}`",
        f"- **Features** : {report.get('n_features')} colonnes z-scorées",
        f"- **Train** : ≤ {report['splits_config']['train_end']}",
        f"- **Val** : → {report['splits_config']['val_end']}",
        f"- **Test** : ≥ {report['splits_config']['test_start']}",
        "",
        "## Leaderboard (IC / Hit Ratio / Sharpe)",
        "",
        "| Modèle | Split | IC moyen | Rank IC | Hit Ratio | Sharpe L/S |",
        "|--------|-------|----------|---------|-----------|------------|",
    ]
    def _fmt(v: Any, fmt: str) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return format(v, fmt)

    for row in report.get("leaderboard", []):
        lines.append(
            f"| {row['model']} | {row['split']} | "
            f"{_fmt(row.get('ic_mean'), '.4f')} | {_fmt(row.get('rank_ic_mean'), '.4f')} | "
            f"{_fmt(row.get('hit_ratio'), '.1%')} | {_fmt(row.get('sharpe_long_short'), '.2f')} |"
        )

    if report.get("best_test_ic"):
        best = report["best_test_ic"]
        lines.extend(
            [
                "",
                "## Meilleur modèle (test — IC)",
                "",
                f"- **{best['model']}** : IC = {best['ic_mean']:.4f}, "
                f"Hit = {best['hit_ratio']:.1%}, Sharpe = {best['sharpe_long_short']:.2f}",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="BVC Recommender — Étape 5")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(BASELINE_MODEL_NAMES),
        choices=list(BASELINE_MODEL_NAMES),
        help="Modèles baseline à entraîner",
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    logger.info("Étape 5 — baselines : %s", args.models)

    dataset = load_ml_dataset()
    results = train_and_evaluate_baselines(dataset, models=tuple(args.models))

    preds = results.pop("predictions", pd.DataFrame())
    if not preds.empty:
        pred_path = DATASET_DIR / "baseline_predictions.parquet"
        try:
            preds.to_parquet(pred_path, index=False)
        except Exception:
            pred_path = DATASET_DIR / "baseline_predictions.csv"
            preds.to_csv(pred_path, index=False)
        logger.info("Prédictions sauvegardées : %s", pred_path)

    report = build_step5_report(results, dataset)
    json_path = REPORTS_DIR / "step5_validation_report.json"
    md_path = REPORTS_DIR / "baseline_models_report.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_markdown_report(report, md_path)

    logger.info("Rapport étape 5 : %s", json_path)
    if report.get("best_test_ic"):
        best = report["best_test_ic"]
        logger.info(
            "Meilleur test : %s | IC=%.4f | Hit=%.1f%% | Sharpe=%.2f",
            best["model"],
            best["ic_mean"],
            best["hit_ratio"] * 100,
            best["sharpe_long_short"],
        )

    logger.info("Étape 5 terminée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
