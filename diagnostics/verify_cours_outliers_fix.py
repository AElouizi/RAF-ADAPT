"""Verification post-correction outliers SNA/SLF/TMA/RDS."""
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


def main() -> int:
    load_env_file()
    client = get_supabase_client()
    for t in ["SNA", "SLF", "TMA", "RDS"]:
        r = (
            client.table(TABLE_COURS_HISTORIQUE)
            .select("id", count="exact")
            .eq("ticker", t)
            .limit(1)
            .execute()
        )
        print(f"DB {t} count={r.count}")
    rslf = (
        client.table(TABLE_COURS_HISTORIQUE)
        .select("id", count="exact")
        .eq("ticker", "SLF")
        .lte("prix_cloture", 40)
        .limit(1)
        .execute()
    )
    print(f"DB SLF prix<=40 remaining: {rslf.count}")

    df = pd.read_parquet(DATA_PROCESSED_DIR / "market_data_cours_historique.parquet")
    df["date_cours"] = pd.to_datetime(df["date_cours"])
    df["prix_cloture"] = pd.to_numeric(df["prix_cloture"], errors="coerce")
    df = df.dropna(subset=["date_cours", "prix_cloture", "ticker"])
    df = df[(df["date_cours"] >= "2015-01-01") & (df["date_cours"] <= "2025-12-31")]
    df = df.sort_values(["ticker", "date_cours"]).drop_duplicates(
        ["ticker", "date_cours"], keep="last"
    )
    print(f"Local SNA remaining 2015-2025: {(df['ticker']=='SNA').sum()}")

    df["daily_ret"] = df.groupby("ticker")["prix_cloture"].pct_change()
    panel = df.dropna(subset=["daily_ret"])
    disp = panel.groupby("date_cours")["daily_ret"].std()
    print("\n=== DISPERSION APRES CORRECTION (local) ===")
    print(
        f"n={len(disp)} med={disp.median():.6f} mean={disp.mean():.6f} "
        f"std={disp.std():.6f}"
    )
    print(
        f"p90={disp.quantile(0.9):.6f} p95={disp.quantile(0.95):.6f} "
        f"p99={disp.quantile(0.99):.6f} p99.9={disp.quantile(0.999):.6f}"
    )
    print(f"max={disp.max():.6f} date={disp.idxmax().date()}")
    print(f"n>10x med={(disp > 10 * disp.median()).sum()}")

    print("\nTop 10 dates:")
    for dt, v in disp.nlargest(10).items():
        day = panel[panel["date_cours"] == dt].copy()
        med = day["daily_ret"].median()
        day["absdev"] = (day["daily_ret"] - med).abs()
        top3 = day.nlargest(3, "absdev")
        parts = [f"{r.ticker}:{r.daily_ret:+.1%}" for r in top3.itertuples()]
        print(f"  {dt.date()} disp={v:.4f} | " + ", ".join(parts))

    print("\n=== Jumps >30% restants ===")
    for t in ["SLF", "TMA", "RDS"]:
        g = df[df["ticker"] == t].sort_values("date_cours").copy()
        g["ret"] = g["prix_cloture"].pct_change()
        j = g[g["ret"].abs() > 0.3]
        print(f"{t}: {len(j)} jumps")
        if len(j):
            show = j[["date_cours", "prix_cloture", "ret"]].head(20)
            print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
