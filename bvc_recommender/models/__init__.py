"""Étape 4+ — modèles de régime, scoring et optimisation."""

from bvc_recommender.models.baseline_models import (
    BASELINE_MODEL_NAMES,
    train_and_evaluate_baselines,
)
from bvc_recommender.models.liquidity_filter import rank_and_label_universe
from bvc_recommender.models.portfolio_optimizer import optimize_portfolios
from bvc_recommender.models.stock_scorer import train_and_evaluate_tft
from bvc_recommender.models.metrics import evaluate_scoring_model
from bvc_recommender.models.regime_detector import (
    DOCUMENTED_THRESHOLDS,
    HMMRegimeDetector,
    REBALANCE_FLAG_COLUMN,
    REGIME_COLUMNS,
    RegimeDetector,
    ThresholdRegimeDetector,
    aggregate_monthly_context,
    detect_regime_daily,
    detect_regime_history,
    dominant_regime,
    enrich_market_context_with_regime,
)

__all__ = [
    "REGIME_COLUMNS",
    "REBALANCE_FLAG_COLUMN",
    "DOCUMENTED_THRESHOLDS",
    "ThresholdRegimeDetector",
    "HMMRegimeDetector",
    "RegimeDetector",
    "aggregate_monthly_context",
    "detect_regime_daily",
    "detect_regime_history",
    "dominant_regime",
    "enrich_market_context_with_regime",
    "BASELINE_MODEL_NAMES",
    "train_and_evaluate_baselines",
    "evaluate_scoring_model",
    "train_and_evaluate_tft",
    "rank_and_label_universe",
    "optimize_portfolios",
]
