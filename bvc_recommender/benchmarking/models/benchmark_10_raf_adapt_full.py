"""
Benchmark 10 — RAF-ADAPT complet (Niveau 3 — référence).

Composantes RAF-ADAPT :
  A (régime)     : intacte (HMM discret is_* via features_indices)
  B (TFT)        : intacte (tft_predictions existantes)
  C (liquidité)  : intacte
  D (NSGA-III)   : intacte (P_equilibre)

Ré-exécution du pipeline complet pour la comparaison, sans ré-entraîner le TFT.
"""

from __future__ import annotations

import pandas as pd

from bvc_recommender.benchmarking.base import BenchmarkModel, BenchmarkResult
from bvc_recommender.benchmarking.metrics import compute_metrics
from bvc_recommender.benchmarking.pipeline_acd import run_acd_walk_forward


class RAFAdaptFullBenchmark(BenchmarkModel):
    model_id = "10_raf_adapt_full"
    model_name = "RAF-ADAPT complet"
    level = 3
    replaces = "Aucune (pipeline A+B+C+D de référence)"

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
            raise ValueError("Scores TFT requis pour RAF-ADAPT complet.")
        tech = technical if technical is not None else pd.DataFrame()
        daily = run_acd_walk_forward(scores, tech, cours, indices, prices)
        metrics = compute_metrics(daily, masi_returns, name=self.model_name)
        return BenchmarkResult(
            model_id=self.model_id,
            model_name=self.model_name,
            level=self.level,
            replaces=self.replaces,
            daily_returns=daily,
            metrics=metrics,
        ).with_monthly()
