"""
Pipeline Étape 2 — calcul des features fondamentales et techniques.

Usage (depuis la racine du projet) :
    python -m bvc_recommender.scripts.run_step2 --use-processed
    python -m bvc_recommender.scripts.run_step2 --use-processed --tickers ATW IAM BCP
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
    FEATURES_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    load_env_file,
)
from bvc_recommender.data.cleaner import clean_all_tables  # noqa: E402
from bvc_recommender.data.loader import (
    fetch_table,
    get_supabase_client,
    load_fondamentaux_rs_bis,
    load_histo_const_ind,
    load_indicateurs_financiers,
)
from bvc_recommender.data.masi_coverage import check_masi_coverage  # noqa: E402
from bvc_recommender.features.fundamental import (  # noqa: E402
    FUNDAMENTAL_FEATURE_COLUMNS,
    build_fundamental_features,
    build_fundamental_features_from_bis,
    listing_dates_from_histo,
)
from bvc_recommender.features.technical import (  # noqa: E402
    TECHNICAL_FEATURE_COLUMNS,
    build_technical_features,
)
from bvc_recommender.features.writer import publish_features, save_features_local  # noqa: E402

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_TICKERS = ["ATW", "IAM", "BCP"]
MASI_EXPECTED_MIN_DATE = "2010-01-01"


def _load_parquet(name: str, directory: Path) -> pd.DataFrame:
    for ext in (".parquet", ".csv"):
        path = directory / f"{name}{ext}"
        if path.is_file():
            return pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
    return pd.DataFrame()


def verify_masi_history(indices: pd.DataFrame) -> dict[str, Any]:
    """Vérifie la couverture historique MASI."""
    result = check_masi_coverage(indices)
    result["expected_min_date"] = MASI_EXPECTED_MIN_DATE
    if result.get("status") == "OK":
        result["notes"] = result.get("notes") or [
            f"Historique MASI couvre depuis {result.get('date_min')}"
        ]
    elif result.get("status") == "WARNING":
        result["notes"] = result.get("notes") or [
            f"Historique MASI commence le {result.get('date_min')} — attendu ~{MASI_EXPECTED_MIN_DATE}. "
            "Beta vs MASI et benchmarks seront limités à cette fenêtre."
        ]
    return result


def load_step2_data(
    client,
    *,
    tickers: list[str] | None,
    use_processed: bool,
    refresh_sources: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Charge uniquement les données nécessaires à l'étape 2."""
    refs = load_reference_tables(client)

    if use_processed and (DATA_PROCESSED_DIR / "market_data_cours_historique.parquet").is_file():
        logger.info("Chargement cours/indices depuis processed/ ...")
        cleaned = {
            "market_data_cours_historique": _load_parquet(
                "market_data_cours_historique", DATA_PROCESSED_DIR
            ),
            "market_data_indices_historique": _load_parquet(
                "market_data_indices_historique", DATA_PROCESSED_DIR
            ),
        }
    else:
        logger.info("Chargement Supabase (cours / indices)...")
        if tickers:
            cours_parts = []
            for t in tickers:
                part = fetch_table(
                    client,
                    "market_data_cours_historique",
                    filters={"ticker": t},
                    order_by="date_cours",
                )
                if not part.empty:
                    cours_parts.append(part)
            cours = pd.concat(cours_parts, ignore_index=True) if cours_parts else pd.DataFrame()
        else:
            from bvc_recommender.config import COURS_MIN_DATE

            cours = fetch_table(
                client,
                "market_data_cours_historique",
                gte_filters={"date_cours": COURS_MIN_DATE},
            )
        indices = fetch_table(
            client, "market_data_indices_historique", filters={"code_index": "MASI"}
        )
        raw = {
            "market_data_cours_historique": cours,
            "market_data_indices_historique": indices,
        }
        cleaned = clean_all_tables(raw)

    logger.info("Chargement donnees_financieres (VALIDE, CONSOLIDE + SOCIAL) ...")
    fin_cache = DATA_PROCESSED_DIR / "donnees_financieres_full.parquet"
    if fin_cache.is_file() and use_processed and not refresh_sources:
        fin = _load_parquet("donnees_financieres_full", DATA_PROCESSED_DIR)
        logger.info("Cache donnees_financieres : %s lignes", len(fin))
    else:
        fin = fetch_table(client, "donnees_financieres", filters={"statut_validation": "VALIDE"})
        if not fin.empty:
            fin["valeur"] = pd.to_numeric(fin["valeur"], errors="coerce")
            DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            fin.to_parquet(fin_cache, index=False)
            logger.info("Cache donnees_financieres écrit : %s lignes", len(fin))

    if tickers and refs.get("entreprises") is not None and not refs["entreprises"].empty:
        ent = refs["entreprises"]
        ent_ids = ent.loc[ent["ticker"].isin(tickers), "id"].astype(str).tolist()
        if ent_ids:
            fin = fin[fin["entreprise_id"].astype(str).isin(ent_ids)].copy()
            logger.info("donnees_financieres filtrées tickers=%s : %s lignes", tickers, len(fin))

    cleaned["donnees_financieres"] = fin

    if tickers and not cleaned["market_data_cours_historique"].empty:
        cleaned["market_data_cours_historique"] = cleaned["market_data_cours_historique"][
            cleaned["market_data_cours_historique"]["ticker"].isin(tickers)
        ]

    refs["histo_const_ind"] = load_histo_const_ind(
        client, cache_dir=DATA_PROCESSED_DIR, refresh=refresh_sources
    )
    refs["indicateurs_financiers"] = load_indicateurs_financiers(
        client, cache_dir=DATA_PROCESSED_DIR, refresh=refresh_sources
    )
    refs["fondamentaux_rs_bis"] = load_fondamentaux_rs_bis(
        client, cache_dir=DATA_PROCESSED_DIR, refresh=refresh_sources
    )
    return cleaned, refs


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


def update_quality_report_masi(masi_check: dict[str, Any]) -> tuple[Path, Path]:
    """Met à jour data_quality_report avec la section MASI."""
    json_path = REPORTS_DIR / "data_quality_report.json"
    md_path = REPORTS_DIR / "data_quality_report.md"

    if json_path.is_file():
        report = json.loads(json_path.read_text(encoding="utf-8"))
    else:
        report = {"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

    report["masi_verification"] = masi_check
    report["metrics"] = report.get("metrics", {})
    report["metrics"]["masi_date_min"] = masi_check.get("date_min")
    report["metrics"]["masi_date_max"] = masi_check.get("date_max")
    report["metrics"]["masi_row_count"] = masi_check.get("row_count")
    report["metrics"]["masi_covers_from_2010"] = masi_check.get("covers_from_2010")

    if masi_check.get("status") == "WARNING" and masi_check.get("notes"):
        warnings = report.get("warnings", [])
        note = masi_check["notes"][0]
        if note not in warnings:
            warnings.append(note)
        report["warnings"] = warnings

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    masi_section = [
        "",
        "## Vérification MASI (Étape 2)",
        "",
        f"- **code_index** : {masi_check.get('code_index')}",
        f"- **Lignes MASI** : {masi_check.get('row_count'):,}",
        f"- **Date min** : {masi_check.get('date_min') or '—'}",
        f"- **Date max** : {masi_check.get('date_max') or '—'}",
        f"- **Couverture depuis 2010** : {'Oui' if masi_check.get('covers_from_2010') else 'Non'}",
        f"- **Statut** : {masi_check.get('status')}",
    ]
    if masi_check.get("notes"):
        masi_section.append(f"- **Note** : {masi_check['notes'][0]}")

    if md_path.is_file():
        md = md_path.read_text(encoding="utf-8")
        marker = "## Vérification MASI (Étape 2)"
        if marker in md:
            md = md.split(marker)[0].rstrip()
        md = md + "\n".join(masi_section) + "\n"
    else:
        md = "# Rapport qualité des données\n" + "\n".join(masi_section) + "\n"

    md_path.write_text(md, encoding="utf-8")
    return md_path, json_path


def build_step2_report(
    masi_check: dict[str, Any],
    fundamental: pd.DataFrame,
    technical: pd.DataFrame,
    sample_tickers: list[str],
    publish_status: dict[str, Any],
    *,
    scope: str = "all",
) -> dict[str, Any]:
    """Rapport JSON de validation étape 2."""
    def _coverage(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
        if df.empty:
            return {}
        return {
            c: round(float(df[c].notna().mean() * 100), 1)
            for c in cols
            if c in df.columns
        }

    sample: dict[str, Any] = {}
    for t in sample_tickers:
        fsub = fundamental[fundamental["ticker"] == t] if not fundamental.empty else pd.DataFrame()
        tsub = technical[technical["ticker"] == t] if not technical.empty else pd.DataFrame()
        sample[t] = {
            "fundamental_rows": len(fsub),
            "technical_rows": len(tsub),
            "fundamental_last": (
                fsub.sort_values("date_fin").tail(1).to_dict(orient="records")[0]
                if not fsub.empty
                else None
            ),
            "technical_last": (
                tsub.sort_values("date_cours").tail(1).to_dict(orient="records")[0]
                if not tsub.empty
                else None
            ),
        }

    fundamental_universe = (
        sorted(fundamental["ticker"].dropna().unique().tolist()) if not fundamental.empty else []
    )
    technical_universe = (
        sorted(technical["ticker"].dropna().unique().tolist()) if not technical.empty else []
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "BVC Recommender — Étape 2",
        "random_state": RANDOM_STATE,
        "scope": scope,
        "universe": {
            "fundamental_tickers": fundamental_universe,
            "technical_tickers": technical_universe,
        },
        "sample_tickers": sample_tickers,
        "masi_verification": masi_check,
        "fundamental": {
            "rows": len(fundamental),
            "tickers": int(fundamental["ticker"].nunique()) if not fundamental.empty else 0,
            "coverage_pct": _coverage(fundamental, FUNDAMENTAL_FEATURE_COLUMNS),
        },
        "technical": {
            "rows": len(technical),
            "tickers": int(technical["ticker"].nunique()) if not technical.empty else 0,
            "coverage_pct": _coverage(technical, TECHNICAL_FEATURE_COLUMNS),
        },
        "ticker_samples": sample,
        "supabase_publish": publish_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="BVC Recommender — Étape 2")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Sous-ensemble de tickers (défaut : tout l'univers disponible)",
    )
    parser.add_argument(
        "--use-processed",
        action="store_true",
        help="Recharger depuis bvc_recommender/data/processed/ si disponible",
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
    cleaned, refs = load_step2_data(
        client,
        tickers=tickers,
        use_processed=args.use_processed,
        refresh_sources=args.refresh_sources,
    )

    indices = cleaned.get("market_data_indices_historique", pd.DataFrame())
    masi_check = verify_masi_history(indices)
    logger.info(
        "MASI : %s lignes, %s → %s, couverture 2010=%s",
        masi_check["row_count"],
        masi_check["date_min"],
        masi_check["date_max"],
        masi_check["covers_from_2010"],
    )
    update_quality_report_masi(masi_check)

    cours = cleaned.get("market_data_cours_historique", pd.DataFrame())
    if tickers and not cours.empty:
        cours = cours[cours["ticker"].isin(tickers)]

    listing_dates = pd.DataFrame()
    histo = refs.get("histo_const_ind")
    ent = refs.get("entreprises")
    if histo is not None and not histo.empty and ent is not None and not ent.empty:
        listing_dates = listing_dates_from_histo(histo, ent)

    fondamentaux_bis = refs.get("fondamentaux_rs_bis")
    if fondamentaux_bis is not None and not fondamentaux_bis.empty:
        logger.info(
            "Calcul features fondamentales depuis fondamentaux_rs_bis (%s lignes brutes)...",
            len(fondamentaux_bis),
        )
        fundamental = build_fundamental_features_from_bis(
            fondamentaux_bis,
            entreprises=refs.get("entreprises"),
            secteurs=refs.get("secteurs"),
            histo_const_ind=refs.get("histo_const_ind"),
            tickers=tickers,
        )
    else:
        logger.warning(
            "fondamentaux_rs_bis indisponible — repli sur donnees_financieres + histo_const_ind."
        )
        fundamental = build_fundamental_features(
            cours=cours,
            donnees_financieres=cleaned.get("donnees_financieres"),
            periodes=refs.get("periodes"),
            entreprises=refs.get("entreprises"),
            secteurs=refs.get("secteurs"),
            histo_const_ind=refs.get("histo_const_ind"),
            indicateurs_financiers=refs.get("indicateurs_financiers"),
            tickers=tickers,
        )

    logger.info("Calcul features techniques...")
    technical = build_technical_features(
        cours,
        indices=indices,
        entreprises=refs.get("entreprises"),
        tickers=tickers,
        listing_dates=listing_dates if not listing_dates.empty else None,
    )

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    save_features_local(fundamental, "features_fondamentales", FEATURES_DIR)
    save_features_local(technical, "features_techniques", FEATURES_DIR)

    publish_status: dict[str, Any] = {}
    if args.skip_supabase_write:
        publish_status = {"skipped": True}
    else:
        publish_status = publish_features(
            client,
            fundamental,
            technical,
            full_universe=tickers is None,
            listing_dates=listing_dates if not listing_dates.empty else None,
        )

    sample_tickers = tickers or [
        t for t in SAMPLE_TICKERS if t in set(fundamental.get("ticker", [])) | set(technical.get("ticker", []))
    ]
    report = build_step2_report(
        masi_check,
        fundamental,
        technical,
        list(sample_tickers),
        publish_status,
        scope="subset" if tickers else "all",
    )
    report_path = REPORTS_DIR / "step2_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Rapport étape 2 : %s", report_path)

    logger.info(
        "Univers : fondamental=%s tickers | technique=%s tickers",
        fundamental["ticker"].nunique() if not fundamental.empty else 0,
        technical["ticker"].nunique() if not technical.empty else 0,
    )
    for t in sample_tickers:
        fsub = fundamental[fundamental["ticker"] == t] if not fundamental.empty else pd.DataFrame()
        tsub = technical[technical["ticker"] == t] if not technical.empty else pd.DataFrame()
        logger.info("[%s] fondamental=%s lignes | technique=%s lignes", t, len(fsub), len(tsub))
        if not fsub.empty:
            last = fsub.sort_values("date_fin").tail(1)
            cols = [c for c in FUNDAMENTAL_FEATURE_COLUMNS if c in last.columns][:6]
            logger.info("  fond. dernier semestre : %s", last[cols].to_dict(orient="records"))
        if not tsub.empty:
            last = tsub.sort_values("date_cours").tail(1)
            cols = [c for c in TECHNICAL_FEATURE_COLUMNS if c in last.columns][:6]
            logger.info("  tech. dernier jour : %s", last[cols].to_dict(orient="records"))

    if fundamental.empty or technical.empty:
        logger.error("Features vides — vérifier les sources.")
        return 2

    logger.info("Étape 2 terminée.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
