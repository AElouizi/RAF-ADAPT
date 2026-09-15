"""Diagnostic MASI — comptages Supabase REST et analyse locale."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import DATA_PROCESSED_DIR, load_env_file  # noqa: E402
from bvc_recommender.data.loader import fetch_table, get_supabase_client  # noqa: E402

load_env_file(ROOT / ".env")

TABLE = "market_data_indices_historique"


def count_table(client, table: str, **kwargs) -> int | None:
    q = client.table(table).select("*", count="exact").limit(0)
    for key, value in kwargs.get("filters", {}).items():
        q = q.eq(key, value)
    for key, value in kwargs.get("gte_filters", {}).items():
        q = q.gte(key, value)
    for key, value in kwargs.get("lte_filters", {}).items():
        q = q.lte(key, value)
    for key, value in kwargs.get("ilike_filters", {}).items():
        q = q.ilike(key, value)
    return q.execute().count


def analyze_indices(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"empty": True}

    date_col = "date_index" if "date_index" in df.columns else "date"
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    stats = []
    for code, grp in out.groupby("code_index"):
        stats.append(
            {
                "code_index": code,
                "count": int(len(grp)),
                "date_min": str(grp[date_col].min().date()) if grp[date_col].notna().any() else None,
                "date_max": str(grp[date_col].max().date()) if grp[date_col].notna().any() else None,
            }
        )
    stats.sort(key=lambda x: x["count"], reverse=True)

    pre2020 = out[out[date_col] < "2020-01-01"]
    pre_stats = []
    for code, grp in pre2020.groupby("code_index"):
        pre_stats.append(
            {
                "code_index": code,
                "count": int(len(grp)),
                "date_min": str(grp[date_col].min().date()),
                "date_max": str(grp[date_col].max().date()),
            }
        )

    masi = out[out["code_index"] == "MASI"]
    masi_variants = out[out["code_index"].astype(str).str.upper().str.contains("MASI", na=False)]
    variant_stats = []
    for code, grp in masi_variants.groupby("code_index"):
        variant_stats.append(
            {
                "code_index": code,
                "count": int(len(grp)),
                "date_min": str(grp[date_col].min().date()),
                "date_max": str(grp[date_col].max().date()),
            }
        )

    return {
        "columns": list(out.columns),
        "global_date_min": str(out[date_col].min().date()),
        "global_date_max": str(out[date_col].max().date()),
        "code_index_stats": stats,
        "rows_before_2020": int(len(pre2020)),
        "pre2020_by_code": pre_stats,
        "masi": {
            "count": int(len(masi)),
            "date_min": str(masi[date_col].min().date()) if len(masi) else None,
            "date_max": str(masi[date_col].max().date()) if len(masi) else None,
            "covers_2010": bool(len(masi) and masi[date_col].min() <= pd.Timestamp("2010-01-01")),
        },
        "masi_variants": variant_stats,
    }


def load_local_parquet() -> dict:
    path = DATA_PROCESSED_DIR / "market_data_indices_historique.parquet"
    if not path.is_file():
        csv_path = path.with_suffix(".csv")
        if csv_path.is_file():
            df = pd.read_csv(csv_path)
        else:
            return {"available": False, "path": str(path)}
    else:
        df = pd.read_parquet(path)

    analysis = analyze_indices(df)
    analysis["available"] = True
    analysis["path"] = str(path)
    analysis["row_count"] = len(df)
    return analysis


def main() -> int:
    client = get_supabase_client()
    results: dict = {}

    results["total_count_exact"] = count_table(client, TABLE)

    for name in ["features_fondamentales", "features_techniques"]:
        try:
            resp = client.table(name).select("*", count="exact").limit(1).execute()
            results[f"table_{name}"] = {
                "exists": True,
                "count": resp.count,
                "sample_cols": list(resp.data[0].keys()) if resp.data else [],
            }
        except Exception as exc:
            results[f"table_{name}"] = {"exists": False, "error": str(exc)[:300]}

    print("Fetching all indices (order id)...", flush=True)
    df_id = fetch_table(client, TABLE, order_by="id", page_size=1000)
    results["fetched_rows_order_id"] = len(df_id)
    results["pagination_complete_id"] = len(df_id) == results["total_count_exact"]

    print("Fetching all indices (order date_index)...", flush=True)
    df_date = fetch_table(client, TABLE, order_by="date_index", page_size=1000)
    results["fetched_rows_order_date"] = len(df_date)
    results["pagination_complete_date"] = len(df_date) == results["total_count_exact"]

    results["supabase_analysis"] = analyze_indices(df_id)
    results["count_masi_filter"] = count_table(client, TABLE, filters={"code_index": "MASI"})
    results["count_code_masi_lower"] = count_table(client, TABLE, filters={"code_index": "masi"})
    results["count_pre2020_api"] = count_table(client, TABLE, lte_filters={"date_index": "2019-12-31"})
    results["count_masi_from_2010"] = count_table(
        client,
        TABLE,
        filters={"code_index": "MASI"},
        gte_filters={"date_index": "2010-01-01"},
    )

    for alt in ["masi_historique", "market_data_masi", "indices_masi", "masi_historical"]:
        try:
            resp = client.table(alt).select("*", count="exact").limit(1).execute()
            results[f"alt_table_{alt}"] = {"exists": True, "count": resp.count}
        except Exception:
            results[f"alt_table_{alt}"] = {"exists": False}

    db_url = os.getenv("DATABASE_URL", "")
    results["database_url_configured"] = bool(db_url and not db_url.startswith("#"))
    results["local_parquet"] = load_local_parquet()

    out_path = ROOT / "bvc_recommender" / "reports" / "masi_diagnostic_raw.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
