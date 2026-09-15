"""
Construit le mart plateforme C1–C5 depuis Bloc B (scores) + Bloc D (poids NSGA).

Prérequis :
    py -m experiments.factorial_hybrid_adapt.run_bloc_b_eval --run --folds all --no-resume
    py -m experiments.factorial_hybrid_adapt.run_bloc_d_eval --run --no-resume
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bvc_recommender.config import DATA_PROCESSED_DIR
from experiments.factorial_hybrid_adapt.allocation import load_market_panels
from experiments.factorial_hybrid_adapt.bloc_b_models import MODEL_IDS, STRATEGY_LABELS
from experiments.factorial_hybrid_adapt.stage2_allocation import Stage2Config
from experiments.factorial_hybrid_adapt.bloc_d_multi_model import PORTFOLIO, simulate_block_portfolio
from experiments.factorial_hybrid_adapt.data_utils import load_ml_dataset
from experiments.factorial_hybrid_adapt.stage2_bloc_b_loader import (
    block_fold_ids_from_summary,
    build_stage2_inputs_for_model_fold,
)
from experiments.factorial_hybrid_adapt.stage2_metrics import _masi_returns, _price_panel
from experiments.factorial_hybrid_adapt.walk_forward_folds import (
    available_months_from_dates,
    generate_rolling_folds,
)

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs" / "platform_mart" / "c1c5"
BLOC_D_PARTS = BASE / "outputs" / "bloc_d_wf_14blocks"
REPORTS = BASE / "outputs" / "reports"
LEGACY_MART = BASE / "outputs" / "platform_mart"


def _ticker_names() -> pd.DataFrame:
    h = pd.read_parquet(DATA_PROCESSED_DIR / "histo_const_ind.parquet")
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    names = (
        h.sort_values("date")
        .groupby("ticker", as_index=False)
        .tail(1)[["ticker", "libelle"]]
        .rename(columns={"libelle": "name"})
    )
    names["ticker"] = names["ticker"].astype(str)
    return names


def _blocks() -> list:
    block_ids = block_fold_ids_from_summary()
    months = available_months_from_dates(load_ml_dataset()["date_cours"])
    by_id = {f.fold_id: f for f in generate_rolling_folds(months)}
    return [by_id[fid] for fid in block_ids if fid in by_id]


def _stage2_to_titles(stage2: pd.DataFrame, model_id: str, names: pd.DataFrame) -> pd.DataFrame:
    df = stage2.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["ticker"] = df["ticker"].astype(str)
    df["recommendation"] = df["recommendation"].astype(str).str.upper()
    df["alpha"] = pd.to_numeric(df.get("predicted_return", df.get("score")), errors="coerce")
    df["conviction_score"] = pd.to_numeric(df.get("conviction_score", df["alpha"]), errors="coerce")
    df["liquidity_L"] = pd.to_numeric(df.get("L_sigmoid", df.get("liquidity_L")), errors="coerce")
    df["vmq"] = pd.to_numeric(df.get("liquidity_vmq_20j", df.get("vmq")), errors="coerce")
    df["model_id"] = model_id
    df["strategy_label"] = STRATEGY_LABELS[model_id]
    df = df.merge(names, on="ticker", how="left")
    df["name"] = df["name"].fillna(df["ticker"])
    if "regime_market" not in df.columns:
        df["regime_market"] = np.nan
    return df


def _weights_to_holdings(
    weights: pd.DataFrame,
    model_id: str,
    names: pd.DataFrame,
) -> pd.DataFrame:
    w = weights[
        (weights["model_id"] == model_id) & (weights["portfolio"] == PORTFOLIO)
    ].copy()
    if w.empty:
        return pd.DataFrame()
    w["month"] = w["mois"].astype(str)
    w["ticker"] = w["ticker"].astype(str)
    w["weight"] = pd.to_numeric(w["weight"], errors="coerce")
    w["weight_pct"] = w["weight"] * 100
    w["alpha"] = pd.to_numeric(w.get("predicted_return"), errors="coerce")
    w["liquidity_L"] = pd.to_numeric(w.get("L_sigmoid"), errors="coerce")
    w["vmq"] = pd.to_numeric(w.get("liquidity_vmq_20j"), errors="coerce")
    w["recommendation"] = w.get("recommendation", "").astype(str).str.upper()
    w["model_id"] = model_id
    w["strategy_label"] = STRATEGY_LABELS[model_id]
    w = w.merge(names, on="ticker", how="left")
    w["name"] = w["name"].fillna(w["ticker"])
    w["in_portfolio"] = 1
    return w


def build() -> int:
    blocks = _blocks()
    names = _ticker_names()
    cours, indices, technical = load_market_panels()
    prices = _price_panel(cours)
    masi = _masi_returns(indices)

    all_titles: list[pd.DataFrame] = []
    all_hold: list[pd.DataFrame] = []
    all_monthly: list[pd.DataFrame] = []

    for model_id in MODEL_IDS:
        title_parts: list[pd.DataFrame] = []
        hold_parts: list[pd.DataFrame] = []
        monthly_parts: list[pd.DataFrame] = []

        for fold in blocks:
            wpath = BLOC_D_PARTS / f"fold_{fold.fold_id:04d}_{model_id}_weights.parquet"
            if not wpath.is_file():
                raise FileNotFoundError(f"Poids Bloc D manquant : {wpath}")

            stage2 = build_stage2_inputs_for_model_fold(model_id, fold)
            titles = _stage2_to_titles(stage2, model_id, names)
            title_parts.append(titles)

            weights = pd.read_parquet(wpath)
            hold = _weights_to_holdings(weights, model_id, names)
            if not hold.empty:
                hold_parts.append(hold)

            sim = simulate_block_portfolio(
                weights,
                prices,
                technical,
                masi,
                model_id=model_id,
                fold_id=fold.fold_id,
            )
            mdf = sim.get("monthly")
            if mdf is not None and not mdf.empty:
                mm = mdf.copy()
                mm["model_id"] = model_id
                mm["strategy_label"] = STRATEGY_LABELS[model_id]
                mm["month"] = mm["mois"].astype(str)
                mm["net_return"] = mm["ret_month"]
                monthly_parts.append(mm)

        tdf = pd.concat(title_parts, ignore_index=True)
        port_keys = (
            pd.concat(hold_parts, ignore_index=True)[["month", "ticker", "weight", "weight_pct"]]
            if hold_parts
            else pd.DataFrame(columns=["month", "ticker", "weight", "weight_pct"])
        )
        if not port_keys.empty:
            unified = tdf.merge(port_keys, on=["month", "ticker"], how="left")
            unified["in_portfolio"] = unified["weight"].notna().astype(int)
            unified["weight"] = pd.to_numeric(unified["weight"], errors="coerce").fillna(0.0)
            unified["weight_pct"] = pd.to_numeric(unified["weight_pct"], errors="coerce").fillna(0.0)
        else:
            unified = tdf.copy()
            unified["in_portfolio"] = 0
            unified["weight"] = 0.0
            unified["weight_pct"] = 0.0

        all_titles.append(unified)
        if hold_parts:
            all_hold.append(pd.concat(hold_parts, ignore_index=True))
        if monthly_parts:
            mm_all = pd.concat(monthly_parts, ignore_index=True).sort_values("month")
            mm_all["wealth100_net"] = 100 * (1 + mm_all["net_return"].fillna(0)).cumprod()
            all_monthly.append(mm_all)

    titles_all = pd.concat(all_titles, ignore_index=True)
    hold_all = pd.concat(all_hold, ignore_index=True) if all_hold else pd.DataFrame()
    monthly_all = pd.concat(all_monthly, ignore_index=True) if all_monthly else pd.DataFrame()
    rec_all = titles_all.copy()

    OUT.mkdir(parents=True, exist_ok=True)
    titles_all.to_parquet(OUT / "fact_titles_month.parquet", index=False)
    titles_all.to_csv(OUT / "fact_titles_month.csv", index=False)
    hold_all.to_parquet(OUT / "fact_portfolio_holdings.parquet", index=False)
    rec_all.to_parquet(OUT / "fact_recommendations.parquet", index=False)
    monthly_all.to_parquet(OUT / "fact_portfolio_monthly.parquet", index=False)

    summary_path = REPORTS / "bloc_d_summary_14blocks.csv"
    if summary_path.is_file():
        kpi = pd.read_csv(summary_path)
        kpi["strategy_label"] = kpi["model_id"].map(STRATEGY_LABELS)
        kpi.to_csv(OUT / "kpi_c1c5.csv", index=False)

    months_out = sorted(titles_all["month"].unique().tolist())
    nsga_cfg = Stage2Config()
    meta = {
        "strategies": STRATEGY_LABELS,
        "model_ids": list(MODEL_IDS),
        "months": months_out,
        "n_months": len(months_out),
        "window": f"{months_out[0]} → {months_out[-1]}" if months_out else None,
        "source_bloc_d": str(BLOC_D_PARTS),
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
            "pop_size": nsga_cfg.pop_size,
            "n_gen": nsga_cfg.n_gen,
        },
        "read_only": True,
    }
    (OUT / "meta_c1c5.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )

    # dim_ticker partagé
    names.to_parquet(OUT / "dim_ticker.parquet", index=False)

    print(f"Mart C1–C5 écrit dans {OUT}")
    print(f"  titres : {len(titles_all)} lignes | holdings : {len(hold_all)} | mois : {len(months_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
