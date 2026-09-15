"""
Pipeline Étape 1 — chargement, nettoyage, fusion, rapport qualité.

Usage (depuis la racine du projet) :
    python -m bvc_recommender.scripts.run_step1
    python -m bvc_recommender.scripts.run_step1 --max-financial-rows 50000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    load_env_file,
)
from bvc_recommender.data.cleaner import clean_all_tables  # noqa: E402
from bvc_recommender.data.loader import load_all_tables  # noqa: E402
from bvc_recommender.data.merger import merge_step1_datasets  # noqa: E402
from bvc_recommender.data.masi_coverage import check_masi_coverage  # noqa: E402
from bvc_recommender.data.quality_report import (  # noqa: E402
    build_quality_report,
    write_quality_report,
)

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _save_parquet(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="BVC Recommender — Étape 1")
    parser.add_argument(
        "--max-financial-rows",
        type=int,
        default=None,
        help="Limite lignes donnees_financieres (défaut : toutes)",
    )
    parser.add_argument(
        "--max-cours-rows",
        type=int,
        default=None,
        help="Limite lignes market_data_cours_historique",
    )
    parser.add_argument(
        "--max-indices-rows",
        type=int,
        default=None,
        help="Limite lignes market_data_indices_historique",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore le cache raw et recharge depuis Supabase",
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    logger.info("random_state=%s", RANDOM_STATE)

    raw_cache_dir = DATA_PROCESSED_DIR / "raw_cache"
    logger.info("Chargement Supabase (cache=%s)...", raw_cache_dir)
    raw = load_all_tables(
        max_financial_rows=args.max_financial_rows,
        max_cours_rows=args.max_cours_rows,
        max_indices_rows=args.max_indices_rows,
        cache_dir=raw_cache_dir,
        refresh=args.refresh,
    )

    masi_raw = check_masi_coverage(raw.get("market_data_indices_historique", pd.DataFrame()))
    logger.info(
        "MASI chargé : %s lignes, %s → %s, couverture 2010=%s",
        masi_raw["row_count"],
        masi_raw["date_min"],
        masi_raw["date_max"],
        masi_raw["covers_from_2010"],
    )

    logger.info("Nettoyage...")
    cleaned = clean_all_tables(raw)

    logger.info("Fusion Étape 1...")
    merged = merge_step1_datasets(cleaned)

    processed_dir = DATA_PROCESSED_DIR
    for name, df in cleaned.items():
        _save_parquet(df, processed_dir / f"{name}.parquet")
        logger.info("Sauvegardé %s (%s lignes)", name, len(df))

    wide = merged.get("fondamentaux_wide")
    if wide is not None and not wide.empty:
        _save_parquet(wide, processed_dir / "fondamentaux_wide.parquet")

    report = build_quality_report(raw, cleaned, merged.get("summary"))
    from bvc_recommender.config import get_settings

    settings = get_settings()
    report.metrics["supabase_url"] = settings.supabase_url
    report.metrics["tables_chargees"] = sum(len(df) for df in raw.values())
    if report.metrics["tables_chargees"] == 0:
        report.warnings.append(
            "Toutes les tables sont vides via l'API anon — vérifier le projet Supabase "
            "dans .env (URL/clé) et les policies RLS."
        )
    md_path, json_path = write_quality_report(report, REPORTS_DIR)

    logger.info("Rapport qualité : %s", md_path)
    logger.info("Rapport JSON : %s", json_path)

    if report.blockers:
        for b in report.blockers:
            logger.warning("BLOCKER: %s", b)
        return 2

    logger.info("Étape 1 terminée avec succès.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
