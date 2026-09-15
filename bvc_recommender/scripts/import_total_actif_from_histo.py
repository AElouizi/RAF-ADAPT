"""
Remplace Total_Actif dans donnees_financieres par histo_total_actif.

Étapes :
  1. Vérifier indicateurs_financiers (code Total_Actif)
  2. DELETE donnees_financieres WHERE indicateur_id = Total_Actif
  3. INSERT depuis histo_total_actif (dédupliqué)

Usage :
    py bvc_recommender/scripts/import_total_actif_from_histo.py --dry-run
    py bvc_recommender/scripts/import_total_actif_from_histo.py
    py bvc_recommender/scripts/import_total_actif_from_histo.py --ticker IAM --dry-run

Préférer SUPABASE_SERVICE_ROLE_KEY (RLS sur histo_total_actif).
Alternative SQL : scripts/sql/migrate_total_actif_from_histo.sql
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import load_env_file  # noqa: E402
from bvc_recommender.data.loader import (  # noqa: E402
    fetch_table,
    get_service_role_client,
    get_supabase_client,
)

load_env_file(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TABLE = "donnees_financieres"
HISTO_TABLE = "histo_total_actif"
INDICATEUR_ID = "86a4a181-e1d3-4189-9c79-9d72e3d8eae9"
INDICATEUR_CODE = "Total_Actif"
TYPE_ETAT_ID = "fc14edf7-6caf-4dbf-b7ab-e9204617f079"
SOURCE_ID = "bdd88eb6-09fa-497e-820e-1b7c64711900"
ITEM_OK = {"total_actif", "total_actif_buk1", "total_actif_buk2", "total_actif_buk3"}


def _client():
    return get_service_role_client() or get_supabase_client()


def _norm_item(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "total_actif"
    return re.sub(r"\s+", "_", str(value).strip().lower())


def _parse_date(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    return pd.to_datetime(str(value).strip()[:10], errors="coerce")


def _parse_montant(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().replace(" ", "").replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _check_indicateur(client) -> None:
    resp = (
        client.table("indicateurs_financiers")
        .select("id,code")
        .eq("id", INDICATEUR_ID)
        .limit(1)
        .execute()
    )
    if not resp.data or resp.data[0].get("code") != INDICATEUR_CODE:
        raise RuntimeError(
            f"Indicateur {INDICATEUR_CODE} ({INDICATEUR_ID}) introuvable dans indicateurs_financiers"
        )
    logger.info("OK — indicateur %s présent", INDICATEUR_CODE)


def _load_histo(client, *, ticker: str | None) -> pd.DataFrame:
    rows: list[dict] = []
    offset = 0
    while True:
        q = client.table(HISTO_TABLE).select("ticker,date,montant,type_compte,item_id").range(offset, offset + 999)
        if ticker:
            q = q.eq("ticker", ticker.upper())
        batch = q.execute().data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    return pd.DataFrame(rows)


def _build_rows(histo: pd.DataFrame, entreprises: pd.DataFrame, periodes: pd.DataFrame) -> pd.DataFrame:
    if histo.empty:
        return pd.DataFrame()

    by_ticker = {
        str(r["ticker"]).upper(): str(r["id"])
        for _, r in entreprises.iterrows()
        if pd.notna(r.get("ticker"))
    }
    per_map = {
        pd.Timestamp(r["date_fin"]).normalize(): str(r["id"])
        for _, r in periodes.iterrows()
    }

    records: list[dict[str, Any]] = []
    for _, row in histo.iterrows():
        if _norm_item(row.get("item_id")) not in ITEM_OK:
            continue
        type_compte = str(row.get("type_compte") or "").strip().upper()
        if type_compte not in {"CONSOLIDE", "SOCIAL"}:
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        eid = by_ticker.get(ticker)
        if not eid:
            continue
        date_fin = _parse_date(row.get("date"))
        if pd.isna(date_fin):
            continue
        periode_id = per_map.get(pd.Timestamp(date_fin).normalize())
        if not periode_id:
            continue
        montant = _parse_montant(row.get("montant"))
        if montant is None or montant == 0:
            continue
        records.append(
            {
                "entreprise_id": eid,
                "ticker": ticker,
                "periode_id": periode_id,
                "date_fin": pd.Timestamp(date_fin).normalize(),
                "type_compte": type_compte,
                "valeur": montant,
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(["entreprise_id", "periode_id", "type_compte", "date_fin"])
    return df.drop_duplicates(["entreprise_id", "periode_id", "type_compte"], keep="last")


def _count_existing(client) -> int:
    resp = (
        client.table(TABLE)
        .select("id", count="exact")
        .eq("indicateur_id", INDICATEUR_ID)
        .limit(1)
        .execute()
    )
    return int(resp.count or 0)


def _delete_existing(client, *, dry_run: bool) -> int:
    n = _count_existing(client)
    logger.info("Lignes Total_Actif à supprimer : %s", n)
    if not dry_run and n > 0:
        client.table(TABLE).delete().eq("indicateur_id", INDICATEUR_ID).execute()
    return n


def _payload(rec: pd.Series) -> dict[str, Any]:
    return {
        "entreprise_id": rec["entreprise_id"],
        "indicateur_id": INDICATEUR_ID,
        "periode_id": rec["periode_id"],
        "source_id": SOURCE_ID,
        "type_etat_id": TYPE_ETAT_ID,
        "valeur": float(rec["valeur"]),
        "unite": "KMAD",
        "type_compte": rec["type_compte"],
        "score_qualite": 0,
        "statut_validation": "VALIDE",
    }


def run(*, dry_run: bool, ticker: str | None) -> dict[str, int]:
    client = _client()
    if get_service_role_client() is None:
        logger.warning(
            "SUPABASE_SERVICE_ROLE_KEY absente — histo_total_actif peut être vide (RLS). "
            "Utilisez migrate_total_actif_from_histo.sql dans SQL Editor."
        )

    _check_indicateur(client)
    histo = _load_histo(client, ticker=ticker)
    logger.info("histo_total_actif : %s lignes brutes", len(histo))
    if histo.empty:
        raise RuntimeError("histo_total_actif vide ou inaccessible (RLS ?)")

    entreprises = fetch_table(client, "entreprises")
    periodes = fetch_table(client, "periodes")
    rows = _build_rows(histo, entreprises, periodes)
    logger.info("Lignes mappées (après dédup) : %s", len(rows))

    stats = {"deleted": 0, "inserted": 0, "unmapped": len(histo) - len(rows)}

    stats["deleted"] = _delete_existing(client, dry_run=dry_run)

    inserted = 0
    batch: list[dict] = []
    for _, rec in rows.iterrows():
        batch.append(_payload(rec))
        if len(batch) >= 200:
            if not dry_run:
                client.table(TABLE).insert(batch).execute()
            inserted += len(batch)
            batch = []
    if batch:
        if not dry_run:
            client.table(TABLE).insert(batch).execute()
        inserted += len(batch)

    stats["inserted"] = inserted
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrer Total_Actif depuis histo_total_actif")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ticker", help="Limiter à un ticker (ex. IAM)")
    args = parser.parse_args()
    stats = run(dry_run=args.dry_run, ticker=args.ticker)
    logger.info(
        "Terminé (%s) — deleted=%s inserted=%s unmapped=%s",
        "dry-run" if args.dry_run else "écrit",
        stats["deleted"],
        stats["inserted"],
        stats["unmapped"],
    )


if __name__ == "__main__":
    main()
