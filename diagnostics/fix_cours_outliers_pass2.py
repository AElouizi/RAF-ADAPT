"""Pass 2 — residual TMA (oct 2022 low scale) + RDS high-scale pre-2024."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from bvc_recommender.config import (
    DATA_PROCESSED_DIR,
    TABLE_COURS_HISTORIQUE,
    load_env_file,
)
from bvc_recommender.data.loader import get_supabase_client

OUT_DIR = Path(__file__).resolve().parent

# TMA: bloc basse echelle juste avant le regime ~1350
TMA_LOW_START = "2022-10-01"
TMA_LOW_END = "2022-10-23"
TMA_LOW_MAX = 400.0

# RDS: supprimer haute echelle (~180) avant le regime recent 2024+
RDS_HIGH_MIN = 100.0
RDS_BEFORE = "2024-05-01"


def mask_pass2(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    d["date_cours"] = pd.to_datetime(d["date_cours"], errors="coerce")
    d["prix_cloture"] = pd.to_numeric(d.get("prix_cloture"), errors="coerce")
    t = d["ticker"].astype(str).str.strip().str.upper()
    m_tma = (
        (t == "TMA")
        & (d["date_cours"] >= TMA_LOW_START)
        & (d["date_cours"] <= TMA_LOW_END)
        & (d["prix_cloture"] <= TMA_LOW_MAX)
    )
    m_rds = (
        (t == "RDS")
        & (d["date_cours"] < RDS_BEFORE)
        & (d["prix_cloture"] >= RDS_HIGH_MIN)
    )
    return m_tma | m_rds


def apply_parquet(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "status": "absent"}
    df = pd.read_parquet(path)
    mask = mask_pass2(df)
    n = int(mask.sum())
    by = {}
    if n:
        by = (
            df.loc[mask]
            .assign(ticker=lambda x: x["ticker"].astype(str).str.upper())["ticker"]
            .value_counts()
            .to_dict()
        )
        df.loc[~mask].to_parquet(path, index=False)
    return {"path": str(path), "deleted": n, "by_ticker": by, "after": len(df) - n}


def delete_batches(client, build_select, label: str, page_size: int = 500) -> dict:
    total = 0
    rounds = 0
    try:
        while rounds < 50:
            resp = build_select(
                client.table(TABLE_COURS_HISTORIQUE).select("id")
            ).limit(page_size).execute()
            ids = [r["id"] for r in (resp.data or []) if r.get("id") is not None]
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


def main() -> int:
    load_env_file()
    paths = [
        DATA_PROCESSED_DIR / "market_data_cours_historique.parquet",
        DATA_PROCESSED_DIR / "raw_cache" / "market_data_cours_historique.parquet",
        OUT_DIR / "_cache_cours_rest.parquet",
    ]
    for p in paths:
        print(apply_parquet(p))

    client = get_supabase_client()
    r1 = delete_batches(
        client,
        lambda q: q.eq("ticker", "TMA")
        .gte("date_cours", TMA_LOW_START)
        .lte("date_cours", TMA_LOW_END)
        .lte("prix_cloture", TMA_LOW_MAX),
        "TMA_oct2022_low",
    )
    r2 = delete_batches(
        client,
        lambda q: q.eq("ticker", "RDS")
        .lt("date_cours", RDS_BEFORE)
        .gte("prix_cloture", RDS_HIGH_MIN),
        "RDS_high_pre_2024",
    )
    print(r1)
    print(r2)

    # append SQL note
    sql = OUT_DIR / "fix_cours_outliers_sna_slf_tma_rds.sql"
    extra = f"""
-- Pass 2 (residus)
DELETE FROM public.market_data_cours_historique
WHERE upper(ticker)='TMA'
  AND date_cours::date BETWEEN '{TMA_LOW_START}' AND '{TMA_LOW_END}'
  AND prix_cloture <= {TMA_LOW_MAX};

DELETE FROM public.market_data_cours_historique
WHERE upper(ticker)='RDS'
  AND date_cours::date < '{RDS_BEFORE}'
  AND prix_cloture >= {RDS_HIGH_MIN};
"""
    with sql.open("a", encoding="utf-8") as f:
        f.write(extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
