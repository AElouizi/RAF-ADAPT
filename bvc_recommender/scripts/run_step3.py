"""
Pipeline Étape 3 — contexte marché + dataset ML.

Usage (depuis la racine du projet) :
    python -m bvc_recommender.scripts.run_step3 --use-processed --use-features
    python -m bvc_recommender.scripts.run_step3 --use-processed --use-features --tickers ATW IAM BCP
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
    DATA_PROCESSED_DIR,
    DATASET_DIR,
    FEATURES_DIR,
    FORWARD_HORIZON_DAYS,
    RANDOM_STATE,
    REPORTS_DIR,
    SPLIT_TEST_START,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
    load_env_file,
)
from bvc_recommender.data.loader import (
    fetch_table,
    get_supabase_client,
    load_histo_const_ind,
    load_indicateurs_financiers,
)
from bvc_recommender.features.dataset_builder import (  # noqa: E402
    TARGET_COLUMN,
    build_ml_dataset,
    save_ml_dataset,
)
from bvc_recommender.features.fundamental import (  # noqa: E402
    FUNDAMENTAL_FEATURE_COLUMNS,
    build_fundamental_features,
    listing_dates_from_histo,
)
from bvc_recommender.features.market_context import (  # noqa: E402
    MARKET_CONTEXT_COLUMNS,
    build_market_context,
)
from bvc_recommender.features.technical import (  # noqa: E402
    TECHNICAL_FEATURE_COLUMNS,
    build_technical_features,
)
from bvc_recommender.features.writer import (  # noqa: E402
    publish_market_context,
    save_features_local,
)

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_TICKERS = ["ATW", "IAM", "BCP"]


def _load_parquet(name: str, directory: Path) -> pd.DataFrame:
    for ext in (".parquet", ".csv"):
        path = directory / f"{name}{ext}"
        if path.is_file():
            return pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
    return pd.DataFrame()


def load_reference_tables(client) -> dict[str, pd.DataFrame]:
    refs: dict[str, pd.DataFrame] = {}
    for name, table in [
        ("periodes", "periodes"),
        ("entreprises", "entreprises"),
        ("secteurs", "secteurs"),
    ]:
        try:
            refs[name] = fetch_table(client, table)
        except Exception as exc:
            logger.warning("Référence %s non chargée : %s", table, exc)
            refs[name] = pd.DataFrame()
    return refs


def load_step3_inputs(
    client,
    *,
    tickers: list[str] | None,
    use_processed: bool,
    use_features: bool,
    refresh_sources: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Charge ou recalcule les entrées étape 3."""
    refs = load_reference_tables(client)

    if use_features and (FEATURES_DIR / "features_techniques.parquet").is_file():
        logger.info("Chargement features depuis %s ...", FEATURES_DIR)
        fundamental = _load_parquet("features_fondamentales", FEATURES_DIR)
        technical = _load_parquet("features_techniques", FEATURES_DIR)
        market_context = _load_parquet("features_indices", FEATURES_DIR)
        cours = _load_parquet("market_data_cours_historique", DATA_PROCESSED_DIR)
        indices = _load_parquet("market_data_indices_historique", DATA_PROCESSED_DIR)
        if tickers:
            if not fundamental.empty:
                fundamental = fundamental[fundamental["ticker"].isin(tickers)]
            if not technical.empty:
                technical = technical[technical["ticker"].isin(tickers)]
            if not cours.empty:
                cours = cours[cours["ticker"].isin(tickers)]
        return fundamental, technical, market_context, cours, indices, refs

    if not use_processed or not (DATA_PROCESSED_DIR / "market_data_cours_historique.parquet").is_file():
        raise FileNotFoundError(
            "Données processed/ ou features/ manquantes. "
            "Lancer run_step1 --refresh puis run_step2 --use-processed."
        )

    logger.info("Chargement sources fraîches et recalcul features ...")
    cours = _load_parquet("market_data_cours_historique", DATA_PROCESSED_DIR)
    indices = _load_parquet("market_data_indices_historique", DATA_PROCESSED_DIR)

    fin_cache = DATA_PROCESSED_DIR / "donnees_financieres_full.parquet"
    if fin_cache.is_file() and not refresh_sources:
        fin = _load_parquet("donnees_financieres_full", DATA_PROCESSED_DIR)
        logger.info("Cache donnees_financieres : %s lignes", len(fin))
    else:
        fin = fetch_table(client, "donnees_financieres", filters={"statut_validation": "VALIDE"})
        if not fin.empty:
            fin["valeur"] = pd.to_numeric(fin["valeur"], errors="coerce")
            DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            fin.to_parquet(fin_cache, index=False)

    histo = load_histo_const_ind(client, cache_dir=DATA_PROCESSED_DIR, refresh=refresh_sources)
    indicateurs = load_indicateurs_financiers(
        client, cache_dir=DATA_PROCESSED_DIR, refresh=refresh_sources
    )

    if tickers and not cours.empty:
        cours = cours[cours["ticker"].isin(tickers)]

    listing_dates = listing_dates_from_histo(histo, refs.get("entreprises", pd.DataFrame()))

    fundamental = build_fundamental_features(
        cours=cours,
        donnees_financieres=fin,
        periodes=refs.get("periodes"),
        entreprises=refs.get("entreprises"),
        secteurs=refs.get("secteurs"),
        histo_const_ind=histo,
        indicateurs_financiers=indicateurs,
        tickers=tickers,
    )
    technical = build_technical_features(
        cours,
        indices=indices,
        entreprises=refs.get("entreprises"),
        tickers=tickers,
        listing_dates=listing_dates if not listing_dates.empty else None,
    )
    market_context = build_market_context(cours, indices)
    return fundamental, technical, market_context, cours, indices, refs


def build_step3_report(
    market_context: pd.DataFrame,
    dataset: pd.DataFrame,
    sample_tickers: list[str],
    publish_status: dict[str, Any],
    dataset_path: Path,
) -> dict[str, Any]:
    def _coverage(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
        if df.empty:
            return {}
        return {
            c: round(float(df[c].notna().mean() * 100), 1)
            for c in cols
            if c in df.columns
        }

    universe = {
        "technical_tickers": sorted(dataset["ticker"].dropna().unique().tolist()) if not dataset.empty else [],
        "fundamental_tickers": sorted(
            dataset.loc[dataset[[c for c in FUNDAMENTAL_FEATURE_COLUMNS if c in dataset.columns]].notna().any(axis=1), "ticker"]
            .dropna()
            .unique()
            .tolist()
        )
        if not dataset.empty
        else [],
    }

    split_counts = {}
    if not dataset.empty and "split" in dataset.columns:
        split_counts = dataset["split"].value_counts().to_dict()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "BVC Recommender — Étape 3",
        "random_state": RANDOM_STATE,
        "scope": "all",
        "forward_horizon_days": FORWARD_HORIZON_DAYS,
        "splits": {
            "train_end": SPLIT_TRAIN_END,
            "val_end": SPLIT_VAL_END,
            "test_start": SPLIT_TEST_START,
            "counts": split_counts,
        },
        "universe": universe,
        "sample_tickers": sample_tickers,
        "market_context": {
            "rows": len(market_context),
            "date_min": str(market_context["date_cours"].min().date()) if not market_context.empty else None,
            "date_max": str(market_context["date_cours"].max().date()) if not market_context.empty else None,
            "coverage_pct": _coverage(market_context, MARKET_CONTEXT_COLUMNS),
        },
        "ml_dataset": {
            "rows": len(dataset),
            "tickers": int(dataset["ticker"].nunique()) if not dataset.empty else 0,
            "path": str(dataset_path),
            "target_coverage_pct": round(float(dataset[TARGET_COLUMN].notna().mean() * 100), 1)
            if not dataset.empty and TARGET_COLUMN in dataset.columns
            else 0.0,
            "target_stats": {
                "mean": float(dataset[TARGET_COLUMN].mean()) if not dataset.empty else None,
                "std": float(dataset[TARGET_COLUMN].std()) if not dataset.empty else None,
            },
        },
        "supabase_publish": publish_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BVC Recommender — Étape 3")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Sous-ensemble de tickers (défaut : tout l'univers)",
    )
    parser.add_argument(
        "--use-processed",
        action="store_true",
        help="Utiliser bvc_recommender/data/processed/",
    )
    parser.add_argument(
        "--use-features",
        action="store_true",
        help="Réutiliser features_fondamentales/techniques si disponibles",
    )
    parser.add_argument(
        "--refresh-sources",
        action="store_true",
        help="Recharger donnees_financieres et histo_const_ind depuis Supabase (ignore le cache)",
    )
    parser.add_argument(
        "--skip-supabase-write",
        action="store_true",
        help="Ne pas tenter l'écriture Supabase",
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    tickers = args.tickers
    logger.info("random_state=%s | tickers=%s", RANDOM_STATE, tickers or "ALL")

    client = get_supabase_client()
    fundamental, technical, market_context, cours, indices, _refs = load_step3_inputs(
        client,
        tickers=tickers,
        use_processed=args.use_processed,
        use_features=args.use_features,
        refresh_sources=args.refresh_sources,
    )

    if not args.use_features and not fundamental.empty:
        save_features_local(fundamental, "features_fondamentales", FEATURES_DIR)
        save_features_local(technical, "features_techniques", FEATURES_DIR)

    if market_context.empty:
        logger.info("Calcul contexte marché ...")
        market_context = build_market_context(cours, indices)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    save_features_local(market_context, "features_indices", FEATURES_DIR)

    publish_status: dict[str, Any] = {}
    if args.skip_supabase_write:
        publish_status = {"skipped": True}
    else:
        publish_status = {"market_context": publish_market_context(client, market_context)}

    logger.info("Assemblage dataset ML (horizon=%s jours) ...", FORWARD_HORIZON_DAYS)
    dataset = build_ml_dataset(
        technical,
        fundamental,
        market_context,
        cours,
        indices,
        tickers=tickers,
    )

    if dataset.empty:
        logger.error("Dataset ML vide — vérifier les features et les cours.")
        return 2

    dataset_path = save_ml_dataset(dataset, "ml_dataset")

    sample_tickers = tickers or [
        t for t in SAMPLE_TICKERS if t in set(dataset.get("ticker", []))
    ]
    report = build_step3_report(
        market_context, dataset, list(sample_tickers), publish_status, dataset_path
    )
    report_path = REPORTS_DIR / "step3_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Rapport étape 3 : %s", report_path)

    logger.info(
        "Univers dataset : %s tickers | train=%s val=%s test=%s",
        dataset["ticker"].nunique(),
        int((dataset["split"] == "train").sum()),
        int((dataset["split"] == "val").sum()),
        int((dataset["split"] == "test").sum()),
    )
    for t in sample_tickers:
        sub = dataset[dataset["ticker"] == t]
        logger.info("[%s] %s lignes | cible moyenne=%.4f", t, len(sub), sub[TARGET_COLUMN].mean())

    logger.info("Étape 3 terminée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
