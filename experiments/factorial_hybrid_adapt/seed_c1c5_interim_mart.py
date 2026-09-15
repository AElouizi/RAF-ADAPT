"""
Mart C1–C5 provisoire depuis le livrable legacy (C2 Ridge+régime).

Usage (pendant le pipeline complet) :
    py experiments/factorial_hybrid_adapt/seed_c1c5_interim_mart.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.factorial_hybrid_adapt.bloc_b_models import MODEL_IDS, STRATEGY_LABELS

BASE = Path(__file__).resolve().parent
LEGACY = BASE / "outputs" / "platform_mart"
OUT = LEGACY / "c1c5"


def _tag(df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    out = df.copy()
    out["model_id"] = model_id
    out["strategy_label"] = STRATEGY_LABELS[model_id]
    return out


def main() -> int:
    titles_src = LEGACY / "fact_titles_month_C4.parquet"
    hold_src = LEGACY / "fact_portfolio_holdings_C4.parquet"
    monthly_src = LEGACY / "fact_portfolio_monthly_C4.parquet"
    if not titles_src.is_file():
        raise FileNotFoundError(f"Legacy mart introuvable : {titles_src}")

    titles_base = pd.read_parquet(titles_src)
    hold_base = pd.read_parquet(hold_src) if hold_src.is_file() else pd.DataFrame()
    monthly_base = pd.read_parquet(monthly_src) if monthly_src.is_file() else pd.DataFrame()

    titles_parts = [_tag(titles_base, mid) for mid in MODEL_IDS]
    hold_parts = [_tag(hold_base, mid) for mid in MODEL_IDS] if not hold_base.empty else []
    monthly_parts = [_tag(monthly_base, mid) for mid in MODEL_IDS] if not monthly_base.empty else []

    titles_all = pd.concat(titles_parts, ignore_index=True)
    hold_all = pd.concat(hold_parts, ignore_index=True) if hold_parts else pd.DataFrame()
    monthly_all = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame()

    OUT.mkdir(parents=True, exist_ok=True)
    titles_all.to_parquet(OUT / "fact_titles_month.parquet", index=False)
    titles_all.to_parquet(OUT / "fact_recommendations.parquet", index=False)
    if not hold_all.empty:
        hold_all.to_parquet(OUT / "fact_portfolio_holdings.parquet", index=False)
    if not monthly_all.empty:
        monthly_all.to_parquet(OUT / "fact_portfolio_monthly.parquet", index=False)

    months = sorted(titles_all["month"].astype(str).unique().tolist())
    meta = {
        "strategies": STRATEGY_LABELS,
        "model_ids": list(MODEL_IDS),
        "months": months,
        "n_months": len(months),
        "window": f"{months[0]} → {months[-1]}" if months else None,
        "interim": True,
        "interim_note": (
            "Données provisoires (legacy C2 Ridge+régime dupliquées sur C1–C5). "
            "Recalcul en cours via run_c1c5_pipeline.py."
        ),
        "read_only": True,
    }
    (OUT / "meta_c1c5.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    masi = LEGACY / "fact_masi_since_2010.csv"
    if masi.is_file():
        shutil.copy2(masi, OUT / "fact_masi_since_2010.csv")

    dim = LEGACY / "dim_ticker.csv"
    if dim.is_file():
        pd.read_csv(dim).to_parquet(OUT / "dim_ticker.parquet", index=False)

    print(f"Mart interim C1-C5 -> {OUT}")
    print(f"  {len(titles_all)} lignes titres | {len(months)} mois | interim=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
