"""
Correction outliers cours BVC — SNA / SLF / TMA / RDS.

Regles (demande utilisateur) :
- SNA : supprimer TOUT l'historique
- SLF : supprimer les cours autour de 26 (echelle basse aberrante)
- TMA : supprimer les spikes isoles (fins de trimestre + 2010-01-04)
- RDS : supprimer les cours haute echelle aberrants en 2018 (~80-100
  intercales dans une serie ~23)

Applique :
1) parquet locaux (processed + raw_cache + cache diagnostic)
2) tentative DELETE Supabase REST (anon) — echoue souvent sans service_role
3) ecrit SQL pour execution manuelle dans le SQL Editor Supabase

Usage :
  py -3 diagnostics/fix_cours_outliers_sna_slf_tma_rds.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    TABLE_COURS_HISTORIQUE,
    load_env_file,
)

OUT_DIR = Path(__file__).resolve().parent
SQL_PATH = OUT_DIR / "fix_cours_outliers_sna_slf_tma_rds.sql"
REPORT_PATH = OUT_DIR / "fix_cours_outliers_report.txt"

# SLF : cluster ~26.25 (et marge)
SLF_LOW_MAX = 40.0

# TMA : dates de spikes isoles a supprimer (prix bas aberrants fin de trimestre)
TMA_DELETE_DATES = {
    "2010-01-04",  # 262 alors que la suite est ~1900
    "2022-12-31",  # 180
    "2023-06-30",  # 385
    "2023-09-30",  # 559
    "2023-12-31",  # 600
    "2024-03-31",  # 660
}

# RDS : en 2018, supprimer les cotes haute echelle (>= 50) meleés à ~23
RDS_YEAR = 2018
RDS_HIGH_MIN = 50.0


def _mask_to_delete(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    d["date_cours"] = pd.to_datetime(d["date_cours"], errors="coerce")
    d["prix_cloture"] = pd.to_numeric(d.get("prix_cloture"), errors="coerce")
    ticker = d["ticker"].astype(str).str.strip().str.upper()

    m_sna = ticker == "SNA"
    m_slf = (ticker == "SLF") & (d["prix_cloture"] <= SLF_LOW_MAX)
    date_str = d["date_cours"].dt.strftime("%Y-%m-%d")
    m_tma = (ticker == "TMA") & date_str.isin(TMA_DELETE_DATES)
    m_rds = (
        (ticker == "RDS")
        & (d["date_cours"].dt.year == RDS_YEAR)
        & (d["prix_cloture"] >= RDS_HIGH_MIN)
    )
    return m_sna | m_slf | m_tma | m_rds


def _apply_to_parquet(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "status": "absent"}
    df = pd.read_parquet(path)
    before = len(df)
    mask = _mask_to_delete(df)
    n_del = int(mask.sum())
    detail = {}
    if "ticker" in df.columns:
        sub = df.loc[mask].copy()
        if not sub.empty:
            detail = sub.assign(ticker=sub["ticker"].astype(str).str.upper())[
                "ticker"
            ].value_counts().to_dict()
    cleaned = df.loc[~mask].copy()
    cleaned.to_parquet(path, index=False)
    return {
        "path": str(path),
        "status": "updated",
        "before": before,
        "deleted": n_del,
        "after": len(cleaned),
        "by_ticker": detail,
    }


def _write_sql() -> Path:
    tma_list = ", ".join(f"'{d}'" for d in sorted(TMA_DELETE_DATES))
    sql = f"""-- Correction outliers market_data_cours_historique
-- Genere par diagnostics/fix_cours_outliers_sna_slf_tma_rds.py
-- A executer dans le SQL Editor Supabase (role postgres / service_role)

BEGIN;

-- 1) SNA : supprimer tout l'historique
DELETE FROM public.market_data_cours_historique
WHERE upper(ticker) = 'SNA';

-- 2) SLF : supprimer cours autour de 26 (echelle basse)
DELETE FROM public.market_data_cours_historique
WHERE upper(ticker) = 'SLF'
  AND prix_cloture <= {SLF_LOW_MAX};

-- 3) TMA : supprimer spikes isoles
DELETE FROM public.market_data_cours_historique
WHERE upper(ticker) = 'TMA'
  AND date_cours::date IN ({tma_list});

-- 4) RDS : supprimer haute echelle aberrante en 2018
DELETE FROM public.market_data_cours_historique
WHERE upper(ticker) = 'RDS'
  AND EXTRACT(YEAR FROM date_cours::date) = {RDS_YEAR}
  AND prix_cloture >= {RDS_HIGH_MIN};

COMMIT;
"""
    SQL_PATH.write_text(sql, encoding="utf-8")
    return SQL_PATH


def _delete_in_batches(client, build_query, label: str, page_size: int = 500) -> dict:
    """
    PostgREST ne renvoie / n'applique souvent que ~1000 lignes par requete.
    On boucle tant qu'il reste des lignes matching.
    """
    total = 0
    rounds = 0
    try:
        while rounds < 50:
            # Prefer delete by id list for reliable batching
            sel = build_query(
                client.table(TABLE_COURS_HISTORIQUE).select("id")
            ).limit(page_size)
            resp = sel.execute()
            ids = [row["id"] for row in (resp.data or []) if row.get("id") is not None]
            if not ids:
                break
            client.table(TABLE_COURS_HISTORIQUE).delete().in_("id", ids).execute()
            total += len(ids)
            rounds += 1
            if len(ids) < page_size:
                break
        return {"ok": True, "deleted": total, "rounds": rounds, "label": label}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "deleted": total, "label": label}


def _try_supabase_delete() -> dict:
    load_env_file()
    from bvc_recommender.data.loader import get_service_role_client, get_supabase_client

    sr = get_service_role_client()
    client = sr or get_supabase_client()
    role = "service_role" if sr else "anon"

    results = {
        "role": role,
        "SNA_all": _delete_in_batches(
            client, lambda q: q.eq("ticker", "SNA"), "SNA_all"
        ),
        "SLF_low": _delete_in_batches(
            client,
            lambda q: q.eq("ticker", "SLF").lte("prix_cloture", SLF_LOW_MAX),
            "SLF_low",
        ),
        "TMA_spikes": _delete_in_batches(
            client,
            lambda q: q.eq("ticker", "TMA").in_(
                "date_cours", sorted(TMA_DELETE_DATES)
            ),
            "TMA_spikes",
        ),
        "RDS_2018_high": _delete_in_batches(
            client,
            lambda q: q.eq("ticker", "RDS")
            .gte("date_cours", f"{RDS_YEAR}-01-01")
            .lte("date_cours", f"{RDS_YEAR}-12-31")
            .gte("prix_cloture", RDS_HIGH_MIN),
            "RDS_2018_high",
        ),
    }
    return results


def main() -> int:
    load_env_file()
    lines: list[str] = []
    lines.append("=== Correction outliers SNA / SLF / TMA / RDS ===")

    parquet_targets = [
        DATA_PROCESSED_DIR / "market_data_cours_historique.parquet",
        DATA_PROCESSED_DIR / "raw_cache" / "market_data_cours_historique.parquet",
        OUT_DIR / "_cache_cours_rest.parquet",
    ]

    lines.append("\n--- Parquet locaux ---")
    for path in parquet_targets:
        info = _apply_to_parquet(path)
        lines.append(str(info))
        print(info)

    sql_path = _write_sql()
    lines.append(f"\nSQL ecrit : {sql_path}")
    print(f"SQL ecrit : {sql_path}")

    lines.append("\n--- Tentative DELETE Supabase REST ---")
    print("Tentative DELETE Supabase...")
    sb = _try_supabase_delete()
    lines.append(str(sb))
    print(sb)

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapport : {REPORT_PATH}")

    ok_ops = [k for k, v in sb.items() if isinstance(v, dict) and v.get("ok")]
    fail_ops = [
        k for k, v in sb.items() if isinstance(v, dict) and v.get("ok") is False
    ]
    if fail_ops:
        print(
            "\nATTENTION : DELETE Supabase incomplete. "
            f"Echecs={fail_ops}. Executer le SQL : {sql_path}"
        )
    else:
        print(f"\nDELETE Supabase OK ({ok_ops}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
