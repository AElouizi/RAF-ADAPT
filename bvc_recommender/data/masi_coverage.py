"""Vérification de la couverture historique MASI (benchmark depuis 2010)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from bvc_recommender.config import MASI_MIN_DATE

MASI_CODE = "MASI"


def check_masi_coverage(indices: pd.DataFrame) -> dict[str, Any]:
    """
    Contrôle la présence et l'étendue temporelle du MASI dans les indices chargés.

    Attendu : code_index='MASI', historique depuis MASI_MIN_DATE (2010-01-01).
    """
    result: dict[str, Any] = {
        "code_index": MASI_CODE,
        "expected_min_date": MASI_MIN_DATE,
        "row_count": 0,
        "date_min": None,
        "date_max": None,
        "covers_from_2010": False,
        "status": "BLOCKER",
        "notes": [],
    }
    if indices.empty or "code_index" not in indices.columns:
        result["notes"].append("Table indices vide ou sans colonne code_index")
        return result

    masi = indices.loc[indices["code_index"].astype(str).str.upper() == MASI_CODE].copy()
    if masi.empty:
        result["notes"].append("Aucune ligne code_index='MASI'")
        return result

    date_col = "date_index" if "date_index" in masi.columns else "date"
    masi[date_col] = pd.to_datetime(masi[date_col], errors="coerce")
    masi = masi.dropna(subset=[date_col])

    result["row_count"] = int(len(masi))
    result["date_min"] = str(masi[date_col].min().date())
    result["date_max"] = str(masi[date_col].max().date())
    min_ts = masi[date_col].min()
    # Première séance BVC souvent 2010-01-04 (pas le 01/01)
    result["covers_from_2010"] = int(min_ts.year) <= pd.Timestamp(MASI_MIN_DATE).year

    if result["covers_from_2010"]:
        result["status"] = "OK"
        result["notes"].append(
            f"Historique MASI couvre depuis {result['date_min']} ({result['row_count']:,} lignes)"
        )
    else:
        result["status"] = "WARNING"
        result["notes"].append(
            f"MASI visible depuis {result['date_min']} seulement ({result['row_count']:,} lignes) — "
            f"attendu depuis {MASI_MIN_DATE}. Les lignes 2010+ existent en base (SQL Editor) "
            "mais sont filtrées par RLS pour l'API anon. Exécuter "
            "bvc_recommender/scripts/sql/fix_rls_masi_quick.sql dans Supabase SQL Editor, "
            "puis relancer run_step1 --refresh. Alternative : DATABASE_URL ou "
            "SUPABASE_SERVICE_ROLE_KEY dans .env."
        )
    return result
