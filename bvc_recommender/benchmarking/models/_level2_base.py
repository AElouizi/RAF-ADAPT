"""Helper commun Niveau 2 : scorer alternatif + pipeline A+C+D."""

from __future__ import annotations

import logging

import pandas as pd

from bvc_recommender.benchmarking.base import BenchmarkModel, BenchmarkResult
from bvc_recommender.benchmarking.metrics import compute_metrics
from bvc_recommender.benchmarking.pipeline_acd import run_acd_walk_forward
from bvc_recommender.benchmarking.scorers import ScorerName, build_scorer_predictions

logger = logging.getLogger(__name__)


class Level2ScorerBenchmark(BenchmarkModel):
    """Remplace B ; conserve A (via features), C et D."""

    scorer_name: ScorerName
    level = 2
    replaces = "Composante B (TFT → scorer alternatif) ; A/C/D intactes"

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
        logger.info("Entraînement / prédiction scorer=%s …", self.scorer_name)
        pred = build_scorer_predictions(self.scorer_name)
        tech = technical if technical is not None else pd.DataFrame()

        daily = run_acd_walk_forward(
            pred,
            tech,
            cours,
            indices,
            prices,
            portfolio_key="P_equilibre",
        )
        metrics = compute_metrics(daily, masi_returns, name=self.model_name)
        return BenchmarkResult(
            model_id=self.model_id,
            model_name=self.model_name,
            level=self.level,
            replaces=self.replaces,
            daily_returns=daily,
            metrics=metrics,
        ).with_monthly()
