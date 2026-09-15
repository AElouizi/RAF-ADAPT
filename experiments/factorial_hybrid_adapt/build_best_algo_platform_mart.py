"""
Mart plateforme « meilleur algorithme par titre » + portefeuille champion NSGA.

Sélection : pour chaque (mois, ticker), la reco du modèle C1–C5 avec le plus haut alpha.
Allocation : portefeuille NSGA du champion financier (Sharpe moyen 14 blocs).

Prérequis : platform_mart/c1c5/ (run_c1c5_pipeline ou build_c1c5_platform_mart).

Usage :
    py experiments/factorial_hybrid_adapt/build_best_algo_platform_mart.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.factorial_hybrid_adapt.bloc_b_models import MODEL_IDS, STRATEGY_LABELS
from experiments.factorial_hybrid_adapt.stage2_allocation import Stage2Config

BASE = Path(__file__).resolve().parent
MART_C1C5 = BASE / "outputs" / "platform_mart" / "c1c5"
OUT = BASE / "outputs" / "platform_mart" / "best"
REPORTS = BASE / "outputs" / "reports"
CHAMPION_PORTFOLIO_MODEL = "c3_lightgbm"


def _pick_champion_portfolio_model() -> str:
    summary = REPORTS / "bloc_d_summary_14blocks.csv"
    if summary.is_file():
        df = pd.read_csv(summary)
        if not df.empty and "sharpe" in df.columns:
            top = df.sort_values("sharpe", ascending=False).iloc[0]
            mid = str(top["model_id"])
            if mid in MODEL_IDS:
                return mid
    return CHAMPION_PORTFOLIO_MODEL


def build() -> int:
    titles_path = MART_C1C5 / "fact_titles_month.parquet"
    hold_path = MART_C1C5 / "fact_portfolio_holdings.parquet"
    monthly_path = MART_C1C5 / "fact_portfolio_monthly.parquet"
    if not titles_path.is_file():
        raise FileNotFoundError(f"Mart C1–C5 introuvable : {titles_path}")

    champion = _pick_champion_portfolio_model()
    titles_all = pd.read_parquet(titles_path)
    titles_all["alpha"] = pd.to_numeric(titles_all["alpha"], errors="coerce")

    idx = titles_all.groupby(["month", "ticker"], sort=False)["alpha"].idxmax()
    best = titles_all.loc[idx].copy()
    best["selected_model_id"] = best["model_id"]
    best["selected_strategy"] = best["strategy_label"]
    best["selection_rule"] = "max_alpha_per_title_c1c5"

    hold_all = pd.read_parquet(hold_path) if hold_path.is_file() else pd.DataFrame()
    monthly_all = pd.read_parquet(monthly_path) if monthly_path.is_file() else pd.DataFrame()

    hold_champion = (
        hold_all[hold_all["model_id"] == champion].copy() if not hold_all.empty else pd.DataFrame()
    )
    monthly_champion = (
        monthly_all[monthly_all["model_id"] == champion].copy()
        if not monthly_all.empty
        else pd.DataFrame()
    )

    if not hold_champion.empty:
        port_keys = hold_champion[["month", "ticker", "weight", "weight_pct"]].drop_duplicates(
            ["month", "ticker"]
        )
        drop_cols = [c for c in ("weight", "weight_pct", "in_portfolio") if c in best.columns]
        base = best.drop(columns=drop_cols)
        unified = base.merge(port_keys, on=["month", "ticker"], how="left")
        unified["in_portfolio"] = unified["weight"].notna().astype(int)
        unified["weight"] = pd.to_numeric(unified["weight"], errors="coerce").fillna(0.0)
        unified["weight_pct"] = pd.to_numeric(unified["weight_pct"], errors="coerce").fillna(0.0)
    else:
        unified = best.copy()
        unified["in_portfolio"] = 0
        unified["weight"] = 0.0
        unified["weight_pct"] = 0.0

    OUT.mkdir(parents=True, exist_ok=True)
    unified.to_parquet(OUT / "fact_titles_month.parquet", index=False)
    unified.to_parquet(OUT / "fact_recommendations.parquet", index=False)
    if not hold_champion.empty:
        hold_out = hold_champion.copy()
        hold_out["portfolio_model_id"] = champion
        hold_out["portfolio_strategy"] = STRATEGY_LABELS[champion]
        hold_out.to_parquet(OUT / "fact_portfolio_holdings.parquet", index=False)
    if not monthly_champion.empty:
        mm = monthly_champion.copy()
        mm["portfolio_model_id"] = champion
        mm["portfolio_strategy"] = STRATEGY_LABELS[champion]
        mm.to_parquet(OUT / "fact_portfolio_monthly.parquet", index=False)

    months = sorted(unified["month"].astype(str).unique().tolist())
    nsga_cfg = Stage2Config()
    meta = {
        "mode": "best_algo_per_title",
        "selection_rule": "max_alpha_per_title_c1c5",
        "portfolio_champion_model_id": champion,
        "portfolio_champion_label": STRATEGY_LABELS[champion],
        "strategies_pool": STRATEGY_LABELS,
        "model_ids": list(MODEL_IDS),
        "months": months,
        "n_months": len(months),
        "window": f"{months[0]} → {months[-1]}" if months else None,
        "n_recommendation_rows": int(len(unified)),
        "n_holding_rows": int(len(hold_champion)),
        "source_c1c5": str(MART_C1C5),
        "nsga_protocol": {
            "liquidity_mode": nsga_cfg.liquidity_mode,
            "nsga_objectives": [
                "max preference-adjusted predicted_return",
                "min CVaR 95%",
                "max portfolio L=sigmoid(VMQ_20j)",
            ],
            "vmq_eligibility_filter": False,
            "liquidity_role": "nsga_pareto_objective",
            "selection_rule": nsga_cfg.selection_rule,
            "w_max": nsga_cfg.w_max,
        },
        "read_only": True,
    }
    (OUT / "meta_platform.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    # KPI champion
    kpi_src = MART_C1C5 / "kpi_c1c5.csv"
    if kpi_src.is_file():
        kpi = pd.read_csv(kpi_src)
        row = kpi[kpi["model_id"] == champion]
        if not row.empty:
            row.to_csv(OUT / "kpi_champion.csv", index=False)

    masi = MART_C1C5 / "fact_masi_since_2010.csv"
    if masi.is_file():
        import shutil

        shutil.copy2(masi, OUT / "fact_masi_since_2010.csv")

    print(f"Mart best-algo -> {OUT}")
    print(f"  titres : {len(unified)} | holdings champion {champion} : {len(hold_champion)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
