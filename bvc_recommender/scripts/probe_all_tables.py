"""Probe rapide des 4 tables principales — Analyste_IA_26."""

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
from bvc_recommender.data.loader import get_supabase_client  # noqa: E402
from bvc_recommender.data.masi_coverage import check_masi_coverage  # noqa: E402

load_env_file(ROOT / ".env")


def main() -> int:
    settings = get_settings()
    print(f"URL : {settings.supabase_url}")
    client = get_supabase_client()
    tables = [
        TABLE_FONDAMENTAUX_RS,
        TABLE_DONNEES_FINANCIERS,
        TABLE_COURS_HISTORIQUE,
        TABLE_INDICES_HISTORIQUE,
    ]
    ok = True
    for table in tables:
        try:
            resp = (
                client.table(table)
                .select("*", count="exact")
                .limit(1)
                .execute()
            )
            count = resp.count or 0
            status = "OK" if count > 0 else "VIDE (API)"
            if count == 0:
                ok = False
            print(f"  {status:12} {table:35} count={count:,}")
        except Exception as exc:
            ok = False
            print(f"  ERR          {table:35} {exc}")

    idx = (
        client.table(TABLE_INDICES_HISTORIQUE)
        .select("*")
        .eq("code_index", "MASI")
        .order("date_index")
        .limit(5000)
        .execute()
        .data
        or []
    )
    import pandas as pd

    masi = check_masi_coverage(pd.DataFrame(idx) if idx else pd.DataFrame())
    print(
        f"\nMASI : {masi['row_count']} lignes (échantillon), "
        f"{masi['date_min']} -> {masi['date_max']}, 2010={masi['covers_from_2010']}"
    )
    if not ok:
        print("\nSi SQL Editor affiche des lignes : exécuter scripts/sql/fix_rls_analyste_all.sql")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
