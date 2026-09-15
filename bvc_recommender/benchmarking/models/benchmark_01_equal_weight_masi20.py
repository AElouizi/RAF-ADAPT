"""
Benchmark 01 — Equal-Weight MASI20 (Niveau 1 — naïf, pas de ML).

Composantes RAF-ADAPT :
  A (régime)     : désactivée
  B (TFT)        : désactivée
  C (liquidité)  : désactivée
  D (NSGA-III)   : désactivée

Méthode : top 20 capitalisation (proxy MASI20), poids égaux, rebalancement mensuel,
frais 0.3 % par trade.
"""

from __future__ import annotations

import pandas as pd

from bvc_recommender.benchmarking.base import BenchmarkModel, BenchmarkResult
from bvc_recommender.benchmarking.metrics import compute_metrics
from bvc_recommender.benchmarking.simulation import masi20_equal_weights, simulate_monthly_rebalance


class EqualWeightMASI20(BenchmarkModel):
    model_id = "01_equal_weight_masi20"
    model_name = "Equal-Weight MASI20"
    level = 1
    replaces = "Aucune (benchmark naïf hors pipeline)"

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
        def weights_fn(as_of: pd.Timestamp) -> dict[str, float]:
            return masi20_equal_weights(cours, as_of, n=20)

        daily = simulate_monthly_rebalance(prices, weights_fn)
        metrics = compute_metrics(daily, masi_returns, name=self.model_name)
        return BenchmarkResult(
            model_id=self.model_id,
            model_name=self.model_name,
            level=self.level,
            replaces=self.replaces,
            daily_returns=daily,
            metrics=metrics,
        ).with_monthly()
