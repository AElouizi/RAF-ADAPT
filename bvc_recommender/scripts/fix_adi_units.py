"""
Correction d'unité ADI — donnees_financieres (KMAD).

Période : clôtures semestrielles CONSOLIDE du 30/06/2015 au 30/06/2022.
États :
- Actifs, Passifs → valeur × 1 000 (sous-échelle vs 2023)
- CPC : RNPG uniquement → valeur ÷ 1 000 (sur-échelle ; CA/RN/rex déjà en KMAD)

Usage (depuis la racine du projet) :
    py bvc_recommender/scripts/fix_adi_units.py --dry-run
    py bvc_recommender/scripts/fix_adi_units.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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
TICKER = "ADI"
ENTREPRISE_ID = "93bf92ba-3730-496f-8a8a-85700819df47"
TYPE_COMPTE = "CONSOLIDE"
TYPE_PERIODE = "SEMESTRIELLE"
DATE_MIN = "2015-06-30"
DATE_MAX = "2022-06-30"
FACTOR = 1_000
BATCH_SIZE = 100

# type_etat_id
ACTIFS_ID = "fc14edf7-6caf-4dbf-b7ab-e9204617f079"
PASSIFS_ID = "e8a7cb89-643f-46fd-81f1-79f17030bdc9"
CPC_ID = "1202c838-e080-4d87-97c5-314169a582a0"
RNPG_ID = "f0a04583-1206-4d30-b1b3-bc988fbd451c"

MULTIPLY_TYPE_ETAT_IDS = (ACTIFS_ID, PASSIFS_ID)


def _load_target_rows(client, *, type_etat_ids: tuple[str, ...], indicateur_id: str | None = None) -> pd.DataFrame:
    periodes = fetch_table(client, "periodes")
    periodes["date_fin"] = pd.to_datetime(periodes["date_fin"], errors="coerce")
    valid_periodes = periodes.loc[
        (periodes["type_periode"] == TYPE_PERIODE)
        & (periodes["date_fin"] >= pd.Timestamp(DATE_MIN))
        & (periodes["date_fin"] <= pd.Timestamp(DATE_MAX)),
        ["id", "date_fin"],
    ].rename(columns={"id": "periode_id"})

    rows: list[dict] = []
    offset = 0
    while True:
        query = (
            client.table(TABLE)
            .select("id,periode_id,type_etat_id,valeur")
            .eq("entreprise_id", ENTREPRISE_ID)
            .eq("type_compte", TYPE_COMPTE)
            .in_("type_etat_id", list(type_etat_ids))
        )
        if indicateur_id:
            query = query.eq("indicateur_id", indicateur_id)
        resp = query.range(offset, offset + 999).execute()
        batch = resp.data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.merge(valid_periodes, on="periode_id", how="inner")
    df["valeur"] = pd.to_numeric(df["valeur"], errors="coerce")
    return df.dropna(subset=["id", "valeur"]).reset_index(drop=True)


def _apply_factor(client, df: pd.DataFrame, factor: float) -> int:
    if df.empty:
        return 0
    updated = 0
    for start in range(0, len(df), BATCH_SIZE):
        chunk = df.iloc[start : start + BATCH_SIZE]
        for _, row in chunk.iterrows():
            new_val = float(row["valeur"]) * factor
            client.table(TABLE).update({"valeur": new_val}).eq("id", row["id"]).execute()
            updated += 1
        logger.info("  … %s / %s lignes mises à jour", updated, len(df))
    return updated


def run(*, dry_run: bool, skip_cpc_rnpg: bool) -> dict[str, int]:
    client = get_supabase_client()

    multiply_df = _load_target_rows(client, type_etat_ids=MULTIPLY_TYPE_ETAT_IDS)
    rnpg_df = pd.DataFrame()
    if not skip_cpc_rnpg:
        rnpg_df = _load_target_rows(client, type_etat_ids=(CPC_ID,), indicateur_id=RNPG_ID)

    stats = {"multiply_rows": len(multiply_df), "multiply_updated": 0, "rnpg_rows": len(rnpg_df), "rnpg_updated": 0}

    if multiply_df.empty and rnpg_df.empty:
        logger.warning("Aucune ligne ADI à corriger pour la période %s → %s.", DATE_MIN, DATE_MAX)
        return stats

    if not multiply_df.empty:
        logger.info(
            "ADI Actifs/Passifs : %s lignes × %s (%s → %s)",
            len(multiply_df),
            FACTOR,
            DATE_MIN,
            DATE_MAX,
        )
        for te, n in multiply_df.groupby("type_etat_id").size().items():
            label = {ACTIFS_ID: "Actifs", PASSIFS_ID: "Passifs"}.get(te, te)
            logger.info("  %s : %s lignes", label, n)
        if not dry_run:
            stats["multiply_updated"] = _apply_factor(client, multiply_df, FACTOR)

    if not rnpg_df.empty:
        logger.info("ADI CPC RNPG : %s lignes ÷ %s", len(rnpg_df), FACTOR)
        sample = rnpg_df.head(2)
        for _, row in sample.iterrows():
            logger.info("  ex. %s : %s → %s", row["date_fin"].date(), row["valeur"], row["valeur"] / FACTOR)
        if not dry_run:
            stats["rnpg_updated"] = _apply_factor(client, rnpg_df, 1 / FACTOR)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Correction unité KMAD — ADI")
    parser.add_argument("--dry-run", action="store_true", help="Simuler sans écrire")
    parser.add_argument("--skip-cpc-rnpg", action="store_true", help="Ne pas corriger RNPG CPC")
    args = parser.parse_args()
    stats = run(dry_run=args.dry_run, skip_cpc_rnpg=args.skip_cpc_rnpg)
    logger.info(
        "Terminé (%s) — actifs/passifs=%s (maj %s) | RNPG CPC=%s (maj %s)",
        "dry-run" if args.dry_run else "écrit",
        stats["multiply_rows"],
        stats["multiply_updated"],
        stats["rnpg_rows"],
        stats["rnpg_updated"],
    )


if __name__ == "__main__":
    main()
