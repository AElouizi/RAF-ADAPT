"""Étapes 2–3 — features et dataset ML."""

from bvc_recommender.features.dataset_builder import TARGET_COLUMN, build_ml_dataset
from bvc_recommender.features.fundamental import (
    FUNDAMENTAL_FEATURE_COLUMNS,
    build_fundamental_features,
)
from bvc_recommender.features.market_context import (
    MARKET_CONTEXT_COLUMNS,
    build_market_context,
)
from bvc_recommender.features.technical import (
    TECHNICAL_FEATURE_COLUMNS,
    build_technical_features,
)

__all__ = [
    "FUNDAMENTAL_FEATURE_COLUMNS",
    "TECHNICAL_FEATURE_COLUMNS",
    "MARKET_CONTEXT_COLUMNS",
    "TARGET_COLUMN",
    "build_fundamental_features",
    "build_technical_features",
    "build_market_context",
    "build_ml_dataset",
]
