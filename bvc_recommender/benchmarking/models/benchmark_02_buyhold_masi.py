"""
Benchmark 02 — Buy & Hold MASI (Niveau 1 — naïf, pas de ML).

Composantes RAF-ADAPT :
  A (régime)     : désactivée
  B (TFT)        : désactivée
  C (liquidité)  : désactivée
  D (NSGA-III)   : désactivée

Méthode : rendement brut de l'indice MASI sur la période de test (pas de rebalancement,
pas de frais — détention passive de l'indice).
"""

from __future__ import annotations

import pandas as pd

from bvc_recommender.benchmarking.base import BenchmarkModel, BenchmarkResult
from bvc_recommender.benchmarking.config import TEST_END, TEST_START
from bvc_recommender.benchmarking.metrics import compute_metrics


class BuyHoldMASI(BenchmarkModel):
    model_id = "02_buyhold_masi"
    model_name = "Buy & Hold MASI"
    level = 1
    replaces = "Aucune (benchmark indice)"

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
        start = pd.Timestamp(TEST_START)
        end = pd.Timestamp(TEST_END)
        if not masi_returns.empty:
            end = min(end, masi_returns.index.max())
        daily = masi_returns.loc[(masi_returns.index >= start) & (masi_returns.index <= end)].copy()
        # Alpha vs lui-même = 0 ; utile comme référence de marché
        metrics = compute_metrics(daily, daily, name=self.model_name)
        return BenchmarkResult(
            model_id=self.model_id,
            model_name=self.model_name,
            level=self.level,
            replaces=self.replaces,
            daily_returns=daily,
            metrics=metrics,
        ).with_monthly()
