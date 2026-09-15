"""
Pipeline A+C+D partagé pour les benchmarks qui ne remplacent que B.

À chaque fin de mois :
  1. rank_and_label_universe (Composante C, scores = alternative à TFT)
  2. optimize_portfolios NSGA-III (Composante D)
  3. simulation P_equilibre avec frais 0.3 %
"""

from __future__ import annotations

import logging

import pandas as pd

from bvc_recommender.benchmarking.config import TEST_END, TEST_START, TRANSACTION_COST
from bvc_recommender.benchmarking.simulation import portfolio_daily_returns, turnover_cost
from bvc_recommender.models.liquidity_filter import rank_and_label_universe
from bvc_recommender.models.portfolio_optimizer import optimize_portfolios
from bvc_recommender.rebalance import rebalance_end_dates

logger = logging.getLogger(__name__)


def run_acd_walk_forward(
    scores: pd.DataFrame,
    technical: pd.DataFrame,
    cours: pd.DataFrame,
    indices: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    portfolio_key: str = "P_equilibre",
    start: str | pd.Timestamp = TEST_START,
    end: str | pd.Timestamp = TEST_END,
    fast_nsga: bool = True,
) -> pd.Series:
    """
    Walk-forward mensuel Composantes C+D à partir d'un panel de scores (B alternatif).
    Retourne les rendements journaliers nets de ``portfolio_key``.
    """
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if not prices.empty:
        end = min(end, prices.index.max())

    ends = rebalance_end_dates(prices.index, start, end, frequency="monthly")
    if len(ends) < 2:
        return pd.Series(dtype=float)

    nsga_pop, nsga_gen = (36, 30) if fast_nsga else (60, 70)
    idx_date_col = "date_index" if "date_index" in indices.columns else "date"

    liq_logger = logging.getLogger("bvc_recommender.models.liquidity_filter")
    opt_logger = logging.getLogger("bvc_recommender.models.portfolio_optimizer")
    prev_liq, prev_opt = liq_logger.level, opt_logger.level
    liq_logger.setLevel(logging.WARNING)
    opt_logger.setLevel(logging.WARNING)

    parts: list[pd.Series] = []
    prev_w: dict[str, float] = {}

    try:
        for i, reb in enumerate(ends[:-1]):
            nxt = ends[i + 1]
            ranked = rank_and_label_universe(scores, technical, as_of_date=reb)
            cours_asof = cours.copy()
            cours_asof["date_cours"] = pd.to_datetime(cours_asof["date_cours"], errors="coerce")
            cours_asof = cours_asof[cours_asof["date_cours"] <= reb]
            indices_asof = indices.copy()
            indices_asof[idx_date_col] = pd.to_datetime(indices_asof[idx_date_col], errors="coerce")
            indices_asof = indices_asof[indices_asof[idx_date_col] <= reb]

            try:
                opt = optimize_portfolios(
                    ranked, cours_asof, indices_asof, pop_size=nsga_pop, n_gen=nsga_gen
                )
            except Exception as exc:
                logger.warning("[%s] NSGA-III échec : %s — equal TOP", reb.date(), exc)
                top = ranked[ranked["label"] == "TOP"]["ticker"].head(10).tolist()
                w = 1.0 / len(top) if top else 0.0
                opt = {
                    portfolio_key: {
                        "tickers": top,
                        "weights": [w] * len(top),
                    }
                }

            p = opt.get(portfolio_key, {})
            weights = dict(zip(p.get("tickers", []), p.get("weights", [])))
            if not weights:
                continue
            daily = portfolio_daily_returns(weights, prices, reb, nxt)
            cost = turnover_cost(prev_w, weights, TRANSACTION_COST)
            if not daily.empty and cost > 0:
                daily = daily.copy()
                daily.iloc[0] -= cost
            if not daily.empty:
                parts.append(daily)
            prev_w = weights
            if (i + 1) % 6 == 0:
                logger.info("ACD progress %s/%s (%s)", i + 1, len(ends) - 1, reb.date())
    finally:
        liq_logger.setLevel(prev_liq)
        opt_logger.setLevel(prev_opt)

    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).sort_index()
