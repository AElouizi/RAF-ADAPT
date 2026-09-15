"""
Benchmark 08 — TFT + régime flou Mamdani (Niveau 3 — ablation).

Composantes RAF-ADAPT :
  A (régime) : variante scikit-fuzzy / Mamdani (mu_* continu) — ancienne prod
  B (TFT)    : intacte architecturalement, known-reals = mu_bull/sideways/bear
  C (liquidité) : intacte
  D (NSGA-III)  : intacte (P_equilibre)

À comparer au pipeline principal où la Composante A = HMM discret (is_*).
"""

from __future__ import annotations

import logging

import pandas as pd

from bvc_recommender.benchmarking.base import BenchmarkModel, BenchmarkResult
from bvc_recommender.benchmarking.fuzzy_regime import detect_fuzzy_regime_history
from bvc_recommender.benchmarking.metrics import compute_metrics
from bvc_recommender.benchmarking.pipeline_acd import run_acd_walk_forward
from bvc_recommender.benchmarking.scorers import load_ml_dataset
from bvc_recommender.benchmarking.tft_ablation import predict_tft_ablation, train_tft_ablation
from bvc_recommender.config import FEATURES_DIR, REGIME_END_YEAR, REGIME_START_YEAR

logger = logging.getLogger(__name__)


class TFTFuzzyBenchmark(BenchmarkModel):
    model_id = "08_tft_fuzzy"
    model_name = "TFT + régime flou"
    level = 3
    replaces = "Ablation A : HMM discret (prod) → régime flou Mamdani continu"

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
        ctx_path = FEATURES_DIR / "features_indices.parquet"
        market_ctx = pd.read_parquet(ctx_path) if ctx_path.is_file() else pd.DataFrame()
        fuzzy_history = detect_fuzzy_regime_history(
            market_ctx,
            start_year=REGIME_START_YEAR,
            end_year=REGIME_END_YEAR,
        )
        df = load_ml_dataset()
        logger.info("TFT ablation — entraînement avec régime flou Mamdani …")
        bundle = train_tft_ablation(
            df, regime_mode="fuzzy", regime_history=fuzzy_history, max_epochs=3
        )
        pred = predict_tft_ablation(
            bundle, df, regime_mode="fuzzy", regime_history=fuzzy_history
        )
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
