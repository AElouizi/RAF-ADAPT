"""
Benchmark 07 — TFT sans conditionnement de régime (Niveau 3 — ablation).

Composantes RAF-ADAPT :
  A (régime)     : DÉSACTIVÉE comme input TFT (pas de known-reals régime)
  B (TFT)        : même architecture, sans [is_bull, is_neutral, is_bear]
  C (liquidité)  : intacte
  D (NSGA-III)   : intacte (P_equilibre)
"""

from __future__ import annotations

import logging

import pandas as pd

from bvc_recommender.benchmarking.base import BenchmarkModel, BenchmarkResult
from bvc_recommender.benchmarking.metrics import compute_metrics
from bvc_recommender.benchmarking.pipeline_acd import run_acd_walk_forward
from bvc_recommender.benchmarking.scorers import load_ml_dataset
from bvc_recommender.benchmarking.tft_ablation import predict_tft_ablation, train_tft_ablation

logger = logging.getLogger(__name__)


class TFTNoRegimeBenchmark(BenchmarkModel):
    model_id = "07_tft_no_regime"
    model_name = "TFT sans régime"
    level = 3
    replaces = "Ablation A→B : régime retiré des known-reals TFT"

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
        df = load_ml_dataset()
        logger.info("TFT ablation — entraînement sans régime …")
        bundle = train_tft_ablation(df, regime_mode="none", max_epochs=3)
        pred = predict_tft_ablation(bundle, df, regime_mode="none")
        tech = technical if technical is not None else pd.DataFrame()
        daily = run_acd_walk_forward(pred, tech, cours, indices, prices)
        metrics = compute_metrics(daily, masi_returns, name=self.model_name)
        return BenchmarkResult(
            model_id=self.model_id,
            model_name=self.model_name,
            level=self.level,
            replaces=self.replaces,
            daily_returns=daily,
            metrics=metrics,
        ).with_monthly()
