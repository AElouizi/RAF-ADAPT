"""
Exécute les benchmarks RAF-ADAPT sur la période de test commune.

Usage (depuis la racine du projet) :
    py -m bvc_recommender.benchmarking.run_all_benchmarks
    py -m bvc_recommender.benchmarking.run_all_benchmarks --levels 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.benchmarking.config import (  # noqa: E402
    RANDOM_SEED,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRANSACTION_COST,
    VAL_END,
)
from bvc_recommender.benchmarking.data_access import (  # noqa: E402
    index_returns,
    load_cours,
    load_indices,
    load_technical,
    load_tft_scores,
    price_panel,
)
from bvc_recommender.benchmarking.models import get_models  # noqa: E402
from bvc_recommender.config import DATASET_DIR, REPORTS_DIR, load_env_file  # noqa: E402

load_env_file(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = DATASET_DIR / "benchmarking"
REPORT_DIR = REPORTS_DIR / "benchmarking"


def _parse_levels(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="RAF-ADAPT — run_all_benchmarks")
    parser.add_argument(
        "--levels",
        default="1",
        help="Niveaux à exécuter (ex. 1 ou 1,2,3). Défaut : 1",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Fusionne avec les métriques / rendements déjà sauvegardés",
    )
    args = parser.parse_args()
    levels = _parse_levels(args.levels)

    np.random.seed(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Benchmarking | train≤%s | val≤%s | test %s→%s | coût=%.1f%% | levels=%s",
        TRAIN_END,
        VAL_END,
        TEST_START,
        TEST_END,
        TRANSACTION_COST * 100,
        sorted(levels) if levels else "all",
    )

    cours = load_cours()
    indices = load_indices()
    technical = load_technical()
    scores = load_tft_scores()
    prices = price_panel(cours)
    masi = index_returns(indices, "MASI")

    if prices.empty or masi.empty:
        logger.error("Données marché insuffisantes.")
        return 2

    data_end = min(prices.index.max(), masi.index.max())
    logger.info("Données dispo jusqu'au %s | scores TFT=%s lignes", data_end.date(), len(scores))

    models = get_models(levels=levels)
    if not models:
        logger.error("Aucun modèle à exécuter.")
        return 2

    monthly_cols: dict[str, pd.Series] = {}
    daily_cols: dict[str, pd.Series] = {}
    metrics_rows: list[dict] = []

    if args.append:
        prev_m = OUT_DIR / "benchmark_metrics.csv"
        prev_mo = OUT_DIR / "benchmark_monthly_returns.csv"
        prev_d = OUT_DIR / "benchmark_daily_returns.csv"
        if prev_m.is_file():
            old = pd.read_csv(prev_m)
            metrics_rows.extend(old.to_dict(orient="records"))
        if prev_mo.is_file():
            old_mo = pd.read_csv(prev_mo, index_col=0, parse_dates=True)
            for c in old_mo.columns:
                if c != "masi":
                    monthly_cols[c] = old_mo[c]
        if prev_d.is_file():
            old_d = pd.read_csv(prev_d, index_col=0, parse_dates=True)
            for c in old_d.columns:
                daily_cols[c] = old_d[c]

    for model in models:
        # Évite les doublons si --append
        metrics_rows = [r for r in metrics_rows if r.get("model_id") != model.model_id]
        monthly_cols.pop(model.model_id, None)
        daily_cols.pop(model.model_id, None)

        logger.info("▶ %s (niveau %s) — %s", model.model_id, model.level, model.replaces)
        try:
            result = model.run(
                cours=cours,
                indices=indices,
                prices=prices,
                masi_returns=masi,
                scores=scores,
                technical=technical,
            )
        except Exception as exc:
            logger.exception("Échec %s : %s", model.model_id, exc)
            metrics_rows.append(
                {
                    "model_id": model.model_id,
                    "model_name": model.model_name,
                    "level": model.level,
                    "replaces": model.replaces,
                    "error": str(exc),
                }
            )
            continue

        result.with_monthly()
        daily_cols[model.model_id] = result.daily_returns
        if result.monthly_returns is not None:
            monthly_cols[model.model_id] = result.monthly_returns
        row = {
            "model_id": model.model_id,
            "model_name": model.model_name,
            "level": model.level,
            "replaces": model.replaces,
            **{k: v for k, v in result.metrics.items() if k != "name"},
        }
        metrics_rows.append(row)
        logger.info(
            "  alpha=%.1f%% | Sharpe=%s | MaxDD=%.1f%% | Hit=%s | Omega=%s",
            (result.metrics.get("alpha_annualized") or 0) * 100,
            result.metrics.get("sharpe"),
            (result.metrics.get("max_drawdown") or 0) * 100,
            result.metrics.get("hit_ratio"),
            result.metrics.get("omega"),
        )

    masi_test = masi.loc[(masi.index >= TEST_START) & (masi.index <= data_end)]
    monthly_cols["masi"] = (1 + masi_test).resample("ME").prod() - 1

    monthly_df = pd.DataFrame(monthly_cols).sort_index()
    daily_df = pd.DataFrame(daily_cols).sort_index()
    metrics_df = pd.DataFrame(metrics_rows)
    # Ordre stable par model_id
    if "model_id" in metrics_df.columns:
        metrics_df = metrics_df.sort_values("model_id").reset_index(drop=True)

    monthly_path = OUT_DIR / "benchmark_monthly_returns.csv"
    daily_path = OUT_DIR / "benchmark_daily_returns.csv"
    metrics_path = OUT_DIR / "benchmark_metrics.csv"
    monthly_df.to_csv(monthly_path, index_label="date")
    daily_df.to_csv(daily_path, index_label="date")
    metrics_df.to_csv(metrics_path, index=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "RAF-ADAPT benchmarking",
        "train_end": TRAIN_END,
        "val_end": VAL_END,
        "test_start": TEST_START,
        "test_end": str(data_end.date()),
        "transaction_cost": TRANSACTION_COST,
        "random_state": RANDOM_SEED,
        "levels_run": sorted(levels) if levels else "all",
        "n_models": len(metrics_rows),
        "outputs": {
            "monthly_returns": str(monthly_path),
            "daily_returns": str(daily_path),
            "metrics": str(metrics_path),
        },
        "metrics": metrics_rows,
    }
    json_path = REPORT_DIR / "benchmark_run_report.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    logger.info("CSV mensuels : %s", monthly_path)
    logger.info("CSV métriques : %s", metrics_path)
    logger.info("Rapport : %s", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
