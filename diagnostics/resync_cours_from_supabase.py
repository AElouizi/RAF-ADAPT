"""Re-telecharge market_data_cours_historique depuis Supabase vers les parquets locaux."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bvc_recommender.config import (
    DATA_PROCESSED_DIR,
    TABLE_COURS_HISTORIQUE,
    load_env_file,
)
from bvc_recommender.data.loader import fetch_table, get_supabase_client

OUT_DIR = Path(__file__).resolve().parent


def main() -> int:
    load_env_file()
    client = get_supabase_client()
    print("Download REST full table...")
    df = fetch_table(
        client,
        TABLE_COURS_HISTORIQUE,
        columns="*",
        order_by="id",
        page_size=1000,
        checkpoint_path=OUT_DIR / "_checkpoints" / "cours_resync.parquet",
    )
    print(f"Downloaded n={len(df)}")
    targets = [
        DATA_PROCESSED_DIR / "market_data_cours_historique.parquet",
        DATA_PROCESSED_DIR / "raw_cache" / "market_data_cours_historique.parquet",
        OUT_DIR / "_cache_cours_rest.parquet",
    ]
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"Wrote {path} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
