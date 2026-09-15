"""
Étape 6 — Métriques portefeuille (étage 2) sur la période test.

Sharpe, Sortino, CVaR 95 %, liquidité moyenne pondérée, turnover mensuel moyen.
Portefeuille rapporté : P_equilibre (identique pour les 4 cellules).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bvc_recommender.backtest.metrics import compute_backtest_metrics
from bvc_recommender.benchmarking.config import TRANSACTION_COST
from bvc_recommender.benchmarking.simulation import (
    portfolio_daily_returns,
    turnover_cost,
)
from bvc_recommender.config import DATA_PROCESSED_DIR, FEATURES_DIR
from experiments.factorial_hybrid_adapt.allocation import load_market_panels

logger = logging.getLogger(__name__)

PORTFOLIO_KEY = "P_equilibre"
LIQUIDITY_COL = "indicateur_liquidite"


def _price_panel(cours: pd.DataFrame) -> pd.DataFrame:
    c = cours.copy()
    c["date_cours"] = pd.to_datetime(c["date_cours"], errors="coerce")
    price_col = "prix_cloture" if "prix_cloture" in c.columns else "prix_courant"
    c[price_col] = pd.to_numeric(c[price_col], errors="coerce")
    return (
        c.pivot_table(index="date_cours", columns="ticker", values=price_col, aggfunc="last")
        .sort_index()
    )


def _masi_returns(indices: pd.DataFrame) -> pd.Series:
    df = indices.copy()
    code_col = "code_index" if "code_index" in df.columns else "code"
    date_col = "date_index" if "date_index" in df.columns else "date"
    val_col = "valeur_index" if "valeur_index" in df.columns else "valeur"
    masi = df[df[code_col].astype(str).str.upper() == "MASI"].copy()
    masi[date_col] = pd.to_datetime(masi[date_col], errors="coerce")
    masi[val_col] = pd.to_numeric(masi[val_col], errors="coerce")
    prices = (
        masi.dropna(subset=[date_col, val_col])
        .drop_duplicates(date_col, keep="last")
        .set_index(date_col)[val_col]
        .sort_index()
    )
    return prices.pct_change(fill_method=None).dropna()


def _liq_asof(
    technical: pd.DataFrame,
    tickers: list[str],
    as_of: pd.Timestamp,
    col: str = LIQUIDITY_COL,
) -> pd.Series:
    tech = technical.copy()
    tech["date_cours"] = pd.to_datetime(tech["date_cours"], errors="coerce")
    tech = tech[tech["ticker"].isin(tickers) & (tech["date_cours"] <= as_of)]
    if tech.empty or col not in tech.columns:
        return pd.Series({t: np.nan for t in tickers}, dtype=float)
    snap = (
        tech.sort_values(["ticker", "date_cours"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .set_index("ticker")[col]
    )
    return pd.to_numeric(snap.reindex(tickers), errors="coerce")


def _weights_dict(month_w: pd.DataFrame) -> dict[str, float]:
    return {
        str(r.ticker): float(r.weight)
        for r in month_w.itertuples()
        if pd.notna(r.weight) and float(r.weight) > 0
    }


def simulate_cell_returns(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    technical: pd.DataFrame,
    *,
    cell_id: int,
    portfolio: str = PORTFOLIO_KEY,
    transaction_cost: float = TRANSACTION_COST,
) -> dict[str, Any]:
    """Simule la série journalière + turnover / liquidité mensuels pour une cellule."""
    w = weights[(weights["cell_id"] == cell_id) & (weights["portfolio"] == portfolio)].copy()
    if w.empty:
        raise ValueError(f"Pas de poids pour cell={cell_id} portfolio={portfolio}")

    w["date_rebalance"] = pd.to_datetime(w["date_rebalance"], errors="coerce")
    months = sorted(w["mois"].unique())
    parts: list[pd.Series] = []
    turnovers: list[float] = []
    liqs: list[float] = []
    monthly_rets: list[dict[str, Any]] = []
    prev_w: dict[str, float] = {}

    for i, mois in enumerate(months):
        month_w = w[w["mois"] == mois]
        reb = pd.Timestamp(month_w["date_rebalance"].iloc[0])
        if i + 1 < len(months):
            nxt_mois = months[i + 1]
            nxt = pd.Timestamp(w.loc[w["mois"] == nxt_mois, "date_rebalance"].iloc[0])
        else:
            nxt = prices.index.max()

        weights_dict = _weights_dict(month_w)
        if not weights_dict:
            continue

        # Turnover unilatéral Σ|Δw| (convention backtest projet)
        if prev_w:
            to = float(
                sum(abs(weights_dict.get(t, 0.0) - prev_w.get(t, 0.0)) for t in set(weights_dict) | set(prev_w))
            )
        else:
            to = float(sum(weights_dict.values()))  # entrée initiale
        turnovers.append(to)

        tickers = list(weights_dict.keys())
        liq_vals = _liq_asof(technical, tickers, reb)
        liq_w = 0.0
        w_avail = 0.0
        for t in tickers:
            v = liq_vals.get(t, np.nan)
            if pd.notna(v):
                liq_w += weights_dict[t] * float(v)
                w_avail += weights_dict[t]
        if w_avail > 0:
            # Si quelques titres sans liquidité, renormaliser sur le poids disponible
            liq_w = liq_w / w_avail * sum(weights_dict.values())
        liqs.append(float(liq_w))

        daily = portfolio_daily_returns(weights_dict, prices, reb, nxt)
        cost = turnover_cost(prev_w, weights_dict, transaction_cost)
        if not daily.empty and cost > 0:
            daily = daily.copy()
            daily.iloc[0] -= cost
        if not daily.empty:
            parts.append(daily)
            monthly_rets.append(
                {
                    "cell_id": cell_id,
                    "mois": mois,
                    "date_rebalance": reb,
                    "ret_month": float((1 + daily).prod() - 1),
                    "turnover": to,
                    "liquidite_ponderee": float(liq_w),
                    "n_days": int(len(daily)),
                }
            )
        prev_w = weights_dict

    daily_rets = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    return {
        "daily_returns": daily_rets,
        "monthly": pd.DataFrame(monthly_rets),
        "turnover_mean": float(np.mean(turnovers)) if turnovers else np.nan,
        "liquidite_moy": float(np.mean(liqs)) if liqs else np.nan,
        "n_rebalances": len(turnovers),
    }


def compute_stage2_metrics(
    weights: pd.DataFrame,
    *,
    cell_ids: tuple[int, ...] = (1, 2, 3, 4),
    portfolio: str = PORTFOLIO_KEY,
) -> dict[str, Any]:
    cours, indices, technical = load_market_panels()
    prices = _price_panel(cours)
    masi = _masi_returns(indices)

    cell_rows: list[dict[str, Any]] = []
    monthly_all: list[pd.DataFrame] = []
    daily_map: dict[int, pd.Series] = {}

    for cell_id in cell_ids:
        logger.info("Métriques étage 2 — cellule %s", cell_id)
        sim = simulate_cell_returns(
            weights, prices, technical, cell_id=cell_id, portfolio=portfolio
        )
        daily = sim["daily_returns"]
        daily_map[cell_id] = daily
        monthly_all.append(sim["monthly"])

        bench = masi.reindex(daily.index).fillna(0.0) if not daily.empty else masi
        perf = compute_backtest_metrics(daily, bench, name=f"C{cell_id}")
        cell_rows.append(
            {
                "cellule": cell_id,
                "portfolio": portfolio,
                "sharpe": perf.get("sharpe"),
                "sortino": perf.get("sortino"),
                "CVaR": perf.get("cvar_95"),
                "liquidite_moy": round(sim["liquidite_moy"], 6)
                if pd.notna(sim["liquidite_moy"])
                else None,
                "turnover": round(sim["turnover_mean"], 4)
                if pd.notna(sim["turnover_mean"])
                else None,
                "total_return": perf.get("total_return"),
                "ann_return": perf.get("annualized_return"),
                "max_drawdown": perf.get("max_drawdown"),
                "vol_ann": perf.get("volatility_ann"),
                "n_days": perf.get("days"),
                "n_rebalances": sim["n_rebalances"],
            }
        )
        logger.info(
            "[C%s] Sharpe=%.3f | Sortino=%.3f | CVaR=%.4f | liq=%.4f | TO=%.3f",
            cell_id,
            perf.get("sharpe") or np.nan,
            perf.get("sortino") or np.nan,
            perf.get("cvar_95") or np.nan,
            sim["liquidite_moy"],
            sim["turnover_mean"],
        )

    summary = pd.DataFrame(cell_rows).sort_values("cellule")
    monthly = pd.concat(monthly_all, ignore_index=True) if monthly_all else pd.DataFrame()
    return {
        "summary": summary,
        "monthly": monthly,
        "daily_returns": daily_map,
        "portfolio": portfolio,
        "transaction_cost": TRANSACTION_COST,
        "liquidity_column": LIQUIDITY_COL,
    }


def merge_with_stage1(
    stage2: pd.DataFrame,
    stage1_summary_path: Path,
) -> pd.DataFrame:
    """Joint IC/hit de l'étage 1 pour le tableau récap (colonnes finales partielles)."""
    if not stage1_summary_path.is_file():
        return stage2
    s1 = pd.read_csv(stage1_summary_path)
    # stage1: cellule, modele, regime, IC_val, IC_test, hit_ratio_val, hit_ratio_test
    keep = [
        c
        for c in (
            "cellule",
            "modele",
            "regime",
            "IC_val",
            "IC_test",
            "hit_ratio_test",
            "hit_ratio_val",
        )
        if c in s1.columns
    ]
    merged = s1[keep].merge(stage2, on="cellule", how="right")
    # hit_ratio rapporté = test (période d'évaluation finale)
    if "hit_ratio_test" in merged.columns:
        merged = merged.rename(columns={"hit_ratio_test": "hit_ratio"})
    return merged


def save_stage2_artifacts(
    results: dict[str, Any],
    *,
    reports_dir: Path,
    stage1_summary_path: Path | None = None,
) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "factorial_stage2_metrics.csv"
    monthly_path = reports_dir / "factorial_stage2_monthly.parquet"
    combined_path = reports_dir / "factorial_recap_partial.csv"
    json_path = reports_dir / "factorial_stage2_report.json"
    daily_dir = reports_dir / "daily_returns"
    daily_dir.mkdir(parents=True, exist_ok=True)

    results["summary"].to_csv(summary_path, index=False)
    results["monthly"].to_parquet(monthly_path, index=False)

    for cell_id, series in results["daily_returns"].items():
        s = series.rename("ret").reset_index()
        s.columns = ["date", "ret"]
        s.to_parquet(daily_dir / f"cell{cell_id}_daily.parquet", index=False)

    combined = results["summary"].copy()
    if stage1_summary_path is not None:
        combined = merge_with_stage1(results["summary"], stage1_summary_path)
        # Ordre colonnes proche du format final
        preferred = [
            "cellule",
            "modele",
            "regime",
            "IC_val",
            "IC_test",
            "hit_ratio",
            "sharpe",
            "sortino",
            "CVaR",
            "liquidite_moy",
            "turnover",
        ]
        cols = [c for c in preferred if c in combined.columns] + [
            c for c in combined.columns if c not in preferred
        ]
        combined = combined[cols]
    combined.to_csv(combined_path, index=False)

    payload = {
        "portfolio": results["portfolio"],
        "transaction_cost": results["transaction_cost"],
        "liquidity_column": results["liquidity_column"],
        "summary": results["summary"].to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "summary": summary_path,
        "monthly": monthly_path,
        "combined": combined_path,
        "report": json_path,
        "daily_dir": daily_dir,
    }
