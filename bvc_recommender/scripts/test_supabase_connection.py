"""Vérifie la configuration et teste la connexion Supabase (Étape 1)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import (  # noqa: E402
    TABLE_COURS_HISTORIQUE,
    TABLE_DONNEES_FINANCIERS,
    TABLE_FONDAMENTAUX_RS,
    TABLE_INDICES_HISTORIQUE,
    get_settings,
    load_env_file,
)
from bvc_recommender.data.loader import get_supabase_client, probe_connection  # noqa: E402

load_env_file(ROOT / ".env")


def main() -> int:
    settings = get_settings()
    print("=== Test connexion Supabase — BVC Recommender (Étape 1) ===")
    print(f"URL : {settings.supabase_url or '(vide)'}")
    print(f"SSL : {'activé' if settings.verify_ssl else 'DÉSACTIVÉ'}")

    try:
        settings.validate()
    except ValueError as exc:
        print(f"\nConfiguration invalide : {exc}")
        print("\nCopiez .env.example vers .env et renseignez SUPABASE_URL / SUPABASE_KEY.")
        return 1

    try:
        client = get_supabase_client()
    except ModuleNotFoundError:
        print("\nDépendances manquantes. Exécutez :")
        print("  pip install -r bvc_recommender/requirements.txt")
        return 1
    except Exception as exc:
        print(f"\nErreur client Supabase : {exc}")
        return 2

    tables = [
        TABLE_FONDAMENTAUX_RS,
        TABLE_DONNEES_FINANCIERS,
        TABLE_COURS_HISTORIQUE,
        TABLE_INDICES_HISTORIQUE,
    ]
    print("\nTables configurées :")
    for t in tables:
        print(f"  - {t}")

    print("\nProbe (1 ligne par table) :")
    results = probe_connection(client)
    ok = True
    for table, info in results.items():
        if info.get("ok"):
            count = info.get("count", "?")
            ncols = len(info.get("columns", []))
            print(f"  OK  {table} (count~{count}, {ncols} colonnes)")
        else:
            print(f"  ERR {table} : {info.get('error')}")
            ok = False

    if ok:
        print("\nConnexion Supabase : SUCCÈS")
        return 0
    print("\nConnexion Supabase : ÉCHEC partiel ou total")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
