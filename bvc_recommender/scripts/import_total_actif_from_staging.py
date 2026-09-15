"""
Import Total_Actif depuis wafanalytics_v2.comptes_financiers_staging
→ public.donnees_financieres (idempotent, sans doublons).

Usage (racine projet) :
    py bvc_recommender/scripts/import_total_actif_from_staging.py --dry-run
    py bvc_recommender/scripts/import_total_actif_from_staging.py
    py bvc_recommender/scripts/import_total_actif_from_staging.py --ticker IAM --dry-run

Alternative SQL : bvc_recommender/scripts/sql/upsert_total_actif_from_staging.sql
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
from bvc_recommender.data.loader import fetch_table, get_supabase_client  # noqa: E402

load_env_file(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TABLE = "donnees_financieres"
STAGING_SCHEMA = "wafanalytics_v2"
STAGING_TABLE = "comptes_financiers_staging"

INDICATEUR_ID = "86a4a181-e1d3-4189-9c79-9d72e3d8eae9"  # Total_Actif
TYPE_ETAT_ID = "fc14edf7-6caf-4dbf-b7ab-e9204617f079"  # Actifs
SOURCE_ID = "bdd88eb6-09fa-497e-820e-1b7c64711900"

ITEM_ALIASES = {"total_actif"}


def _norm_item(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return re.sub(r"\s+", "_", str(value).strip().lower())


def _parse_date(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return pd.Timestamp(pd.to_datetime(text, format=fmt))
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(text, errors="coerce")


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


def _to_kmad(montant: float, unite: Any) -> float:
    if unite is None or (isinstance(unite, float) and pd.isna(unite)):
        return montant
    u = str(unite).upper()
    if "MAD" in u and "K" not in u:
        return montant / 1000.0
    return montant


def _load_staging(client, *, ticker: str | None) -> pd.DataFrame:
    rows: list[dict] = []
    offset = 0
    while True:
        q = (
            client.schema(STAGING_SCHEMA)
            .table(STAGING_TABLE)
            .select("*")
            .range(offset, offset + 999)
        )
        if ticker:
            q = q.ilike("titre", ticker)
        batch = q.execute().data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    return pd.DataFrame(rows)


def _entreprise_lookup(entreprises: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    by_ticker = {
        str(r["ticker"]).upper(): str(r["id"])
        for _, r in entreprises.iterrows()
        if pd.notna(r.get("ticker"))
    }
    by_code = {
        str(r["code_original"]).upper(): str(r["id"])
        for _, r in entreprises.iterrows()
        if pd.notna(r.get("code_original"))
    }
    return by_ticker, by_code


def _periode_lookup(periodes: pd.DataFrame) -> dict[pd.Timestamp, str]:
    out: dict[pd.Timestamp, str] = {}
    for _, row in periodes.iterrows():
        dt = pd.Timestamp(row["date_fin"])
        out[dt.normalize()] = str(row["id"])
    return out


def build_rows(
    staging: pd.DataFrame,
    entreprises: pd.DataFrame,
    periodes: pd.DataFrame,
    *,
    ticker: str | None,
) -> pd.DataFrame:
    if staging.empty:
        return pd.DataFrame()

    by_ticker, by_code = _entreprise_lookup(entreprises)
    per_map = _periode_lookup(periodes)

    records: list[dict[str, Any]] = []
    for _, row in staging.iterrows():
        if _norm_item(row.get("item_id")) not in ITEM_ALIASES:
            continue
        etat = str(row.get("type_etat_financier") or "").lower()
        if etat and "actif" not in etat:
            continue

        titre = str(row.get("titre") or "").strip().upper()
        if ticker and titre != ticker.upper():
            continue

        eid = by_ticker.get(titre) or by_code.get(titre)
        if not eid:
            continue

        type_compte = str(row.get("type_compte") or "").strip().upper()
        if type_compte not in {"CONSOLIDE", "SOCIAL"}:
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

        valeur = _to_kmad(montant, row.get("unite"))
        records.append(
            {
                "entreprise_id": eid,
                "ticker": titre,
                "periode_id": periode_id,
                "date_fin": pd.Timestamp(date_fin).normalize(),
                "type_compte": type_compte,
                "valeur": valeur,
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(["entreprise_id", "periode_id", "type_compte", "date_fin"])
    return df.drop_duplicates(["entreprise_id", "periode_id", "type_compte"], keep="last")


def _find_existing(client, *, entreprise_id: str, periode_id: str, type_compte: str) -> dict | None:
    resp = (
        client.table(TABLE)
        .select("id,valeur")
        .eq("entreprise_id", entreprise_id)
        .eq("indicateur_id", INDICATEUR_ID)
        .eq("periode_id", periode_id)
        .eq("type_etat_id", TYPE_ETAT_ID)
        .eq("type_compte", type_compte)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


def _payload(entreprise_id: str, periode_id: str, type_compte: str, valeur: float) -> dict[str, Any]:
    return {
        "entreprise_id": entreprise_id,
        "indicateur_id": INDICATEUR_ID,
        "periode_id": periode_id,
        "source_id": SOURCE_ID,
        "type_etat_id": TYPE_ETAT_ID,
        "valeur": valeur,
        "unite": "KMAD",
        "type_compte": type_compte,
        "score_qualite": 0,
        "statut_validation": "VALIDE",
    }


def run(*, dry_run: bool, ticker: str | None) -> dict[str, int]:
    client = get_supabase_client()
    staging = _load_staging(client, ticker=ticker)
    logger.info("Staging : %s lignes brutes", len(staging))

    entreprises = fetch_table(client, "entreprises")
    periodes = fetch_table(client, "periodes")
    rows = build_rows(staging, entreprises, periodes, ticker=ticker)
    logger.info("Lignes Total_Actif mappées : %s", len(rows))

    stats = {"insert": 0, "update": 0, "skip": 0, "unmapped_staging": len(staging) - len(rows)}

    for _, rec in rows.iterrows():
        existing = _find_existing(
            client,
            entreprise_id=rec["entreprise_id"],
            periode_id=rec["periode_id"],
            type_compte=rec["type_compte"],
        )
        if existing is not None and abs(float(existing["valeur"]) - float(rec["valeur"])) < 0.01:
            stats["skip"] += 1
            continue
        if existing is not None:
            logger.info(
                "UPDATE %s %s %s : %s → %s",
                rec["ticker"],
                rec["date_fin"].date(),
                rec["type_compte"],
                existing["valeur"],
                rec["valeur"],
            )
            if not dry_run:
                client.table(TABLE).update({"valeur": rec["valeur"]}).eq("id", existing["id"]).execute()
            stats["update"] += 1
        else:
            logger.info(
                "INSERT %s %s %s : %s",
                rec["ticker"],
                rec["date_fin"].date(),
                rec["type_compte"],
                rec["valeur"],
            )
            if not dry_run:
                client.table(TABLE).insert(
                    _payload(rec["entreprise_id"], rec["periode_id"], rec["type_compte"], rec["valeur"])
                ).execute()
            stats["insert"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Total_Actif depuis staging wafanalytics_v2")
    parser.add_argument("--dry-run", action="store_true", help="Simuler sans écrire")
    parser.add_argument("--ticker", help="Limiter à un ticker (ex. IAM)")
    args = parser.parse_args()
    stats = run(dry_run=args.dry_run, ticker=args.ticker)
    logger.info(
        "Terminé (%s) — insert=%s update=%s skip=%s",
        "dry-run" if args.dry_run else "écrit",
        stats["insert"],
        stats["update"],
        stats["skip"],
    )


if __name__ == "__main__":
    main()
