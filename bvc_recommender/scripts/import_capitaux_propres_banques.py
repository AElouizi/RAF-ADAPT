"""
Import / mise à jour capitaux_propres (Passifs, CONSOLIDE) dans donnees_financieres.

Usage (depuis la racine du projet) :
    py bvc_recommender/scripts/import_capitaux_propres_banques.py --dry-run
    py bvc_recommender/scripts/import_capitaux_propres_banques.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import load_env_file  # noqa: E402
from bvc_recommender.data.loader import fetch_table, get_supabase_client  # noqa: E402

load_env_file(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TABLE = "donnees_financieres"
INDICATEUR_ID = "b9c2dc9f-e1ba-448b-a84b-aefa42e620ea"  # Capitaux_propres / Passifs
TYPE_ETAT_ID = "e8a7cb89-643f-46fd-81f1-79f17030bdc9"  # Passifs
SOURCE_ID = "bdd88eb6-09fa-497e-820e-1b7c64711900"

ENTREPRISE_IDS = {
    "BCP": "ad996576-6fd2-4372-adf7-853773a6e546",
    "BOA": "eb78b221-7006-4de6-9572-f20a7894f565",
    "SBM": "23ac7cc4-5784-43f5-8b8b-2e5acfe82e57",
    "ATW": "245923cb-9950-45df-9cb8-c7a08c849108",
}

# Valeurs en milliers MAD (KMAD) — clôtures annuelles 31/12
CAPITAUX_PROPRES: dict[str, dict[int, int]] = {
    "BCP": {2015: 38_839_564, 2016: 41_370_924, 2017: 43_483_573},
    "BOA": {
        2016: 23_582_687,
        2017: 24_684_424,
        2018: 23_841_511,
        2019: 29_435_162,
        2020: 29_943_306,
        2021: 31_390_520,
    },
    "SBM": {
        2015: 1_587_653,
        2016: 1_660_155,
        2017: 1_741_968,
        2018: 1_673_601,
        2019: 1_686_386,
        2020: 1_537_640,
    },
    "ATW": {2015: 40_401_978, 2016: 47_411_083, 2017: 46_058_720},
}


def _annual_periode_map(periodes) -> dict[int, str]:
    df = periodes.copy()
    df["date_fin"] = df["date_fin"].astype(str)
    annual = df[df["type_periode"] == "ANNUELLE"]
    out: dict[int, str] = {}
    for _, row in annual.iterrows():
        year = int(str(row["date_fin"])[:4])
        out[year] = str(row["id"])
    return out


def _find_existing(
    client,
    *,
    entreprise_id: str,
    periode_id: str,
) -> dict[str, Any] | None:
    resp = (
        client.table(TABLE)
        .select("id,valeur")
        .eq("entreprise_id", entreprise_id)
        .eq("indicateur_id", INDICATEUR_ID)
        .eq("periode_id", periode_id)
        .eq("type_etat_id", TYPE_ETAT_ID)
        .eq("type_compte", "CONSOLIDE")
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


def _base_payload(entreprise_id: str, periode_id: str, valeur: int) -> dict[str, Any]:
    return {
        "entreprise_id": entreprise_id,
        "indicateur_id": INDICATEUR_ID,
        "periode_id": periode_id,
        "source_id": SOURCE_ID,
        "type_etat_id": TYPE_ETAT_ID,
        "valeur": valeur,
        "unite": "KMAD",
        "type_compte": "CONSOLIDE",
        "score_qualite": 0,
        "statut_validation": "VALIDE",
    }


def run(*, dry_run: bool) -> dict[str, int]:
    client = get_supabase_client()
    periodes = fetch_table(client, "periodes")
    per_map = _annual_periode_map(periodes)

    stats = {"insert": 0, "update": 0, "skip": 0}

    for ticker, by_year in CAPITAUX_PROPRES.items():
        eid = ENTREPRISE_IDS[ticker]
        for year, valeur in sorted(by_year.items()):
            periode_id = per_map.get(year)
            if not periode_id:
                raise KeyError(f"Période annuelle {year}-12-31 introuvable")

            existing = _find_existing(client, entreprise_id=eid, periode_id=periode_id)
            if existing is not None and float(existing["valeur"]) == float(valeur):
                logger.info("%s %s : inchangé (%s)", ticker, year, valeur)
                stats["skip"] += 1
                continue

            if existing is not None:
                logger.info(
                    "%s %s : UPDATE %s → %s",
                    ticker,
                    year,
                    existing["valeur"],
                    valeur,
                )
                if not dry_run:
                    client.table(TABLE).update({"valeur": valeur}).eq("id", existing["id"]).execute()
                stats["update"] += 1
            else:
                logger.info("%s %s : INSERT %s", ticker, year, valeur)
                if not dry_run:
                    client.table(TABLE).insert(_base_payload(eid, periode_id, valeur)).execute()
                stats["insert"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Import capitaux_propres banques")
    parser.add_argument("--dry-run", action="store_true", help="Afficher sans écrire")
    args = parser.parse_args()
    stats = run(dry_run=args.dry_run)
    logger.info(
        "Terminé (%s) — insert=%s update=%s skip=%s",
        "dry-run" if args.dry_run else "écrit",
        stats["insert"],
        stats["update"],
        stats["skip"],
    )


if __name__ == "__main__":
    main()
