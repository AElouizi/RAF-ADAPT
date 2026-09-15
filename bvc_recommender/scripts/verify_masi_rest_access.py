"""
Vérifie que l'API REST (clé anon) voit le MASI depuis 2010.

Usage :
    py -m bvc_recommender.scripts.verify_masi_rest_access

Si ÉCHEC : exécuter scripts/sql/fix_rls_masi_quick.sql dans Supabase SQL Editor,
puis relancer ce script et run_step1 --refresh.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import MASI_MIN_DATE, load_env_file  # noqa: E402
from bvc_recommender.data.loader import fetch_table, get_supabase_client  # noqa: E402
from bvc_recommender.data.masi_coverage import check_masi_coverage  # noqa: E402

load_env_file(ROOT / ".env")

# Ligne visible dans le SQL Editor (capture utilisateur) mais absente via anon si RLS actif
SCREENSHOT_UUID = "9c49e61a-e4e5-40a4-b58d-80ba95b869ea"
SCREENSHOT_DATE = "2010-01-04"


def main() -> int:
    print("=== Vérification accès REST MASI (clé anon) ===\n")
    client = get_supabase_client()

    by_uuid = (
        client.table("market_data_indices_historique")
        .select("id,date_index,valeur_index")
        .eq("id", SCREENSHOT_UUID)
        .execute()
        .data
        or []
    )
    by_date = (
        client.table("market_data_indices_historique")
        .select("id,date_index,valeur_index")
        .eq("code_index", "MASI")
        .eq("date_index", SCREENSHOT_DATE)
        .execute()
        .data
        or []
    )
    masi_count = (
        client.table("market_data_indices_historique")
        .select("*", count="exact")
        .eq("code_index", "MASI")
        .limit(1)
        .execute()
        .count
    )

    print(f"Ligne UUID {SCREENSHOT_UUID[:8]}… : {'OK' if by_uuid else 'ABSENTE (RLS probable)'}")
    print(f"MASI {SCREENSHOT_DATE}          : {'OK' if by_date else 'ABSENTE (RLS probable)'}")
    if by_date:
        print(f"  valeur_index = {by_date[0].get('valeur_index')}")
    print(f"Compteur MASI (anon)            : {masi_count:,}")

    df = fetch_table(client, "market_data_indices_historique", filters={"code_index": "MASI"})
    check = check_masi_coverage(df)
    print(f"\nCouverture MASI via pagination  : {check['date_min']} -> {check['date_max']}")
    print(f"Couverture depuis {MASI_MIN_DATE}     : {'OUI' if check['covers_from_2010'] else 'NON'}")

    if check["covers_from_2010"] and by_uuid:
        print("\nSUCCÈS — l'API REST voit l'historique MASI complet.")
        return 0

    print("\nÉCHEC — les données existent en base (SQL Editor) mais pas via l'API anon.")
    print("Action : ouvrir Supabase SQL Editor et exécuter TOUT le fichier :")
    print("  bvc_recommender/scripts/sql/fix_masi_all.sql")
    print("Puis : py -m bvc_recommender.scripts.refresh_indices_only")
    print("       py -m bvc_recommender.scripts.verify_masi_rest_access")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
