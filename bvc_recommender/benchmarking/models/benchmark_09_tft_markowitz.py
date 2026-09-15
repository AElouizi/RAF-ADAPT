"""
Benchmark 09 — TFT + Markowitz min-variance (Niveau 3 — ablation).

Composantes RAF-ADAPT :
  A (régime)     : intacte (via TFT déjà entraîné avec régime)
  B (TFT)        : intacte (scores tft_predictions existants)
  C (liquidité)  : intacte
  D (NSGA-III)   : REMPLACÉE par Markowitz minimum variance (scipy)
"""

from __future__ import annotations

import logging

import pandas as pd

from bvc_recommender.benchmarking.base import BenchmarkModel, BenchmarkResult
from bvc_recommender.benchmarking.config import TEST_END, TEST_START, TRANSACTION_COST
from bvc_recommender.benchmarking.markowitz import markowitz_min_variance
from bvc_recommender.benchmarking.metrics import compute_metrics
from bvc_recommender.benchmarking.simulation import portfolio_daily_returns, turnover_cost
from bvc_recommender.models.liquidity_filter import rank_and_label_universe
from bvc_recommender.rebalance import rebalance_end_dates

logger = logging.getLogger(__name__)


class TFTMarkowitzBenchmark(BenchmarkModel):
    model_id = "09_tft_markowitz"
    model_name = "TFT + Markowitz"
    level = 3
    replaces = "Composante D (NSGA-III → Markowitz min-variance)"

    def run(
        self,
        *,
        cours: pd.DataFrame,
        indices: pd.DataFrame,
        prices: pd.DataFrame,
        masi_returns: pd.Series,
        scores: pd.DataFrame | None = None,
        technical: pd.DataFrame | None = None,
    ) -> BenchmarkResult:
        if scores is None or scores.empty:
            raise ValueError("Scores TFT requis pour TFT+Markowitz.")
        tech = technical if technical is not None else pd.DataFrame()

        start = pd.Timestamp(TEST_START)
        end = min(pd.Timestamp(TEST_END), prices.index.max())
        ends = rebalance_end_dates(prices.index, start, end, frequency="monthly")

        liq_logger = logging.getLogger("bvc_recommender.models.liquidity_filter")
        prev = liq_logger.level
        liq_logger.setLevel(logging.WARNING)

        parts: list[pd.Series] = []
        prev_w: dict[str, float] = {}
        try:
            for i, reb in enumerate(ends[:-1]):
                nxt = ends[i + 1]
                ranked = rank_and_label_universe(scores, tech, as_of_date=reb)
                weights = markowitz_min_variance(ranked, cours, reb)
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
        finally:
            liq_logger.setLevel(prev)

        daily = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
        metrics = compute_metrics(daily, masi_returns, name=self.model_name)
        return BenchmarkResult(
            model_id=self.model_id,
            model_name=self.model_name,
            level=self.level,
            replaces=self.replaces,
            daily_returns=daily,
            metrics=metrics,
        ).with_monthly()
