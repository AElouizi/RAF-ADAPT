"""
Étape 8 — Backtest walk-forward (mensuel ou trimestriel).

Compare P_agressif / P_equilibre / P_defensif vs MASI et proxy MASI20.
Coûts de transaction : 0.3 % par trade.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bvc_recommender.backtest.metrics import compute_backtest_metrics
from bvc_recommender.config import RANDOM_STATE, REBALANCE_FREQUENCY
from bvc_recommender.models.liquidity_filter import rank_and_label_universe
from bvc_recommender.models.portfolio_optimizer import optimize_portfolios
from bvc_recommender.rebalance import (
    period_label,
    rebalance_end_dates,
    turnover_annual_factor,
)

logger = logging.getLogger(__name__)

TRANSACTION_COST = 0.003  # 0.3 % par trade (turnover unilatéral × coût)
BACKTEST_START = pd.Timestamp("2023-01-01")
BACKTEST_END = pd.Timestamp("2025-12-31")


def _filter_asof(df: pd.DataFrame, date_col: str, as_of: pd.Timestamp) -> pd.DataFrame:
    """Coupe l'historique à as_of (anti look-ahead optimisation)."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out[out[date_col] <= pd.Timestamp(as_of)]


def _price_panel(cours: pd.DataFrame) -> pd.DataFrame:
    c = cours.copy()
    c["date_cours"] = pd.to_datetime(c["date_cours"], errors="coerce")
    price_col = "prix_cloture" if "prix_cloture" in c.columns else "prix_courant"
    c[price_col] = pd.to_numeric(c[price_col], errors="coerce")
    return c.pivot_table(index="date_cours", columns="ticker", values=price_col, aggfunc="last").sort_index()


def _index_levels(indices: pd.DataFrame, code: str) -> pd.Series:
    df = indices.copy()
    code_col = "code_index" if "code_index" in df.columns else "code"
    date_col = "date_index" if "date_index" in df.columns else "date"
    val_col = "valeur_index" if "valeur_index" in df.columns else "valeur"
    sub = df[df[code_col].astype(str).str.upper() == code.upper()].copy()
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub[val_col] = pd.to_numeric(sub[val_col], errors="coerce")
    return (
        sub.dropna(subset=[date_col, val_col])
        .drop_duplicates(date_col, keep="last")
        .set_index(date_col)[val_col]
        .sort_index()
    )


def _masi20_proxy_weights(cours: pd.DataFrame, as_of: pd.Timestamp, n: int = 20) -> dict[str, float]:
    """Proxy MASI20 : top N capitalisation, pondération égale (indice MASI20 absent en base)."""
    c = cours.copy()
    c["date_cours"] = pd.to_datetime(c["date_cours"], errors="coerce")
    snap = c[c["date_cours"] <= as_of].sort_values("date_cours").groupby("ticker", as_index=False).tail(1)
    if "capitalisation" not in snap.columns:
        tickers = snap["ticker"].dropna().astype(str).head(n).tolist()
    else:
        snap["capitalisation"] = pd.to_numeric(snap["capitalisation"], errors="coerce")
        tickers = snap.nlargest(n, "capitalisation")["ticker"].tolist()
    if not tickers:
        return {}
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


def _quarter_label(ts: pd.Timestamp) -> str:
    return f"{ts.year}-Q{ts.quarter}"


def _mean_quarterly_turnover(period_labels: list[str], turnovers: list[float]) -> float | None:
    """Moyenne du turnover agrégé par trimestre civil."""
    if not turnovers or len(period_labels) != len(turnovers):
        return None
    buckets: dict[str, float] = {}
    for lab, to in zip(period_labels, turnovers):
        try:
            if "-Q" in lab.upper():
                q = lab.upper()
            else:
                ts = pd.Timestamp(f"{lab}-01")
                q = _quarter_label(ts)
        except Exception:
            continue
        buckets[q] = buckets.get(q, 0.0) + float(to)
    if not buckets:
        return None
    return float(np.mean(list(buckets.values())))


def _portfolio_daily_returns(
    weights: dict[str, float],
    prices: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    tickers = [t for t in weights if t in prices.columns]
    if not tickers:
        return pd.Series(dtype=float)
    sub = prices.loc[(prices.index > start) & (prices.index <= end), tickers]
    rets = sub.pct_change(fill_method=None).dropna(how="all")
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()
    port = rets.fillna(0.0).values @ w
    return pd.Series(port, index=rets.index)


def _turnover_cost(old_w: dict[str, float], new_w: dict[str, float]) -> float:
    tickers = set(old_w) | set(new_w)
    turnover = sum(abs(new_w.get(t, 0.0) - old_w.get(t, 0.0)) for t in tickers)
    return turnover * TRANSACTION_COST


def run_walk_forward_backtest(
    scores: pd.DataFrame,
    technical: pd.DataFrame,
    cours: pd.DataFrame,
    indices: pd.DataFrame,
    *,
    start: pd.Timestamp = BACKTEST_START,
    end: pd.Timestamp = BACKTEST_END,
    recommendations_dir: Path,
    score_col: str = "prediction",
    fast_nsga: bool = True,
    frequency: str | None = None,
) -> dict[str, Any]:
    """
    Backtest walk-forward : ranking → optimisation → simulation avec frais.
    """
    freq = frequency or REBALANCE_FREQUENCY
    prices = _price_panel(cours)
    masi_levels = _index_levels(indices, "MASI")
    masi_rets = masi_levels.pct_change(fill_method=None).dropna()

    period_ends = rebalance_end_dates(prices.index, start, end, frequency=freq)
    if len(period_ends) < 2:
        label = "mois" if freq == "monthly" else "trimestres"
        raise ValueError(f"Pas assez de {label} pour le backtest.")

    portfolio_keys = ["P_agressif", "P_equilibre", "P_defensif"]
    paths: dict[str, list[pd.Series]] = {k: [] for k in portfolio_keys}
    paths["MASI"] = []
    paths["MASI20_proxy"] = []

    prev_weights: dict[str, dict[str, float]] = {k: {} for k in portfolio_keys}
    prev_masi20: dict[str, float] = {}
    turnover_log: dict[str, list[float]] = {k: [] for k in portfolio_keys}
    period_labels_log: list[str] = []
    period_records: list[dict[str, Any]] = []
    quarterly_rows: dict[str, list[dict[str, Any]]] = {}

    nsga_pop = 36 if fast_nsga else 60
    nsga_gen = 30 if fast_nsga else 70
    ann_factor = turnover_annual_factor(freq)
    idx_date_col = "date_index" if "date_index" in indices.columns else "date"

    # Réduire le bruit de logs pendant la boucle mensuelle
    liq_logger = logging.getLogger("bvc_recommender.models.liquidity_filter")
    opt_logger = logging.getLogger("bvc_recommender.models.portfolio_optimizer")
    prev_liq_level = liq_logger.level
    prev_opt_level = opt_logger.level
    liq_logger.setLevel(logging.WARNING)
    opt_logger.setLevel(logging.WARNING)

    try:
        for i, rebalance_date in enumerate(period_ends[:-1]):
            next_date = period_ends[i + 1]
            p_label = period_label(rebalance_date, frequency=freq)
            q_label = _quarter_label(rebalance_date)
            period_labels_log.append(p_label)

            ranked = rank_and_label_universe(
                scores, technical, score_col=score_col, as_of_date=rebalance_date
            )
            cours_asof = _filter_asof(cours, "date_cours", rebalance_date)
            indices_asof = _filter_asof(indices, idx_date_col, rebalance_date)
            try:
                opt = optimize_portfolios(
                    ranked,
                    cours_asof,
                    indices_asof,
                    pop_size=nsga_pop,
                    n_gen=nsga_gen,
                )
            except Exception as exc:
                logger.warning("[%s] Optimisation échouée : %s — poids égaux TOP", p_label, exc)
                top = ranked[ranked["label"] == "TOP"]["ticker"].head(10).tolist()
                w = 1.0 / len(top) if top else 0.0
                opt = {
                    k: {
                        "tickers": top,
                        "weights": [w] * len(top),
                        "weights_pct": [w * 100] * len(top),
                    }
                    for k in portfolio_keys
                }

            masi20_w = _masi20_proxy_weights(cours_asof, rebalance_date)

            rec_rows = []
            for key in portfolio_keys:
                p = opt.get(key, {})
                weights = dict(zip(p.get("tickers", []), p.get("weights", [])))
                turnover = sum(
                    abs(weights.get(t, 0.0) - prev_weights[key].get(t, 0.0))
                    for t in set(weights) | set(prev_weights[key])
                )
                turnover_log[key].append(turnover)
                cost = turnover * TRANSACTION_COST
                daily = _portfolio_daily_returns(weights, prices, rebalance_date, next_date)
                if not daily.empty and cost > 0:
                    daily.iloc[0] -= cost
                if not daily.empty:
                    paths[key].append(daily)
                prev_weights[key] = weights

                for t, w in weights.items():
                    row = {
                        "period": p_label,
                        "quarter": q_label,
                        "portfolio": key,
                        "ticker": t,
                        "weight": round(w, 4),
                        "weight_pct": round(w * 100, 2),
                        "rebalance_date": str(rebalance_date.date()),
                    }
                    rec_rows.append(row)

            masi_seg = masi_rets.loc[
                (masi_rets.index > rebalance_date) & (masi_rets.index <= next_date)
            ]
            if not masi_seg.empty:
                paths["MASI"].append(masi_seg)

            m20_daily = _portfolio_daily_returns(masi20_w, prices, rebalance_date, next_date)
            m20_cost = _turnover_cost(prev_masi20, masi20_w)
            if not m20_daily.empty and m20_cost > 0:
                m20_daily.iloc[0] -= m20_cost
            if not m20_daily.empty:
                paths["MASI20_proxy"].append(m20_daily)
            prev_masi20 = masi20_w

            rec_df = pd.DataFrame(rec_rows)
            recommendations_dir.mkdir(parents=True, exist_ok=True)
            # Fichier mensuel (rebalancement mensuel)
            rec_path = recommendations_dir / f"{p_label}.csv"
            rec_df.to_csv(rec_path, index=False)
            # Agrégat trimestriel YYYY-QX.csv (concat des mois du trimestre)
            quarterly_rows.setdefault(q_label, []).extend(rec_rows)
            q_path = recommendations_dir / f"{q_label}.csv"
            pd.DataFrame(quarterly_rows[q_label]).to_csv(q_path, index=False)

            period_records.append(
                {
                    "period": p_label,
                    "quarter": q_label,
                    "rebalance_date": str(rebalance_date.date()),
                    "n_ranked": len(ranked),
                    "n_top": int((ranked["label"] == "TOP").sum()),
                    "recommendations_file": str(rec_path),
                    "recommendations_quarter_file": str(q_path),
                }
            )
            logger.info(
                "[%s] %s positions TOP | %s | %s",
                p_label,
                int((ranked["label"] == "TOP").sum()),
                rec_path.name,
                q_path.name,
            )
    finally:
        liq_logger.setLevel(prev_liq_level)
        opt_logger.setLevel(prev_opt_level)

    combined: dict[str, pd.Series] = {}
    for key, parts in paths.items():
        if parts:
            combined[key] = pd.concat(parts).sort_index()

    masi_aligned = combined.get("MASI", pd.Series(dtype=float))
    metrics: dict[str, Any] = {}
    for key in list(portfolio_keys) + ["MASI20_proxy"]:
        if key in combined and not masi_aligned.empty:
            m = compute_backtest_metrics(combined[key], masi_aligned, name=key)
            if turnover_log.get(key):
                m["turnover_mean_period"] = round(float(np.mean(turnover_log[key])), 4)
                m["turnover_annualized"] = round(
                    float(np.mean(turnover_log[key]) * ann_factor), 4
                )
                tq = _mean_quarterly_turnover(period_labels_log, turnover_log[key])
                if tq is not None:
                    m["turnover_quarterly"] = round(tq, 4)
            metrics[key] = m

    if not masi_aligned.empty:
        metrics["MASI"] = compute_backtest_metrics(masi_aligned, masi_aligned, name="MASI")

    return {
        "random_state": RANDOM_STATE,
        "rebalance_frequency": freq,
        "period": {"start": str(start.date()), "end": str(end.date())},
        "transaction_cost": TRANSACTION_COST,
        "periods": period_records,
        "quarters": period_records,
        "metrics": metrics,
        "cumulative_returns": {
            k: float((1 + s).prod() - 1) for k, s in combined.items() if not s.empty
        },
        "daily_returns": combined,
    }
