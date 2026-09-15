"""Étapes 2–3 — features et dataset ML (imports lazy pour démarrage Streamlit)."""

from __future__ import annotations

from typing import Any

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


def __getattr__(name: str) -> Any:
    if name in {
        "FUNDAMENTAL_FEATURE_COLUMNS",
        "build_fundamental_features",
    }:
        from bvc_recommender.features.fundamental import (
            FUNDAMENTAL_FEATURE_COLUMNS,
            build_fundamental_features,
        )

        return {
            "FUNDAMENTAL_FEATURE_COLUMNS": FUNDAMENTAL_FEATURE_COLUMNS,
            "build_fundamental_features": build_fundamental_features,
        }[name]
    if name in {"TECHNICAL_FEATURE_COLUMNS", "build_technical_features"}:
        from bvc_recommender.features.technical import (
            TECHNICAL_FEATURE_COLUMNS,
            build_technical_features,
        )

        return {
            "TECHNICAL_FEATURE_COLUMNS": TECHNICAL_FEATURE_COLUMNS,
            "build_technical_features": build_technical_features,
        }[name]
    if name in {"MARKET_CONTEXT_COLUMNS", "build_market_context"}:
        from bvc_recommender.features.market_context import (
            MARKET_CONTEXT_COLUMNS,
            build_market_context,
        )

        return {
            "MARKET_CONTEXT_COLUMNS": MARKET_CONTEXT_COLUMNS,
            "build_market_context": build_market_context,
        }[name]
    if name in {"TARGET_COLUMN", "build_ml_dataset"}:
        from bvc_recommender.features.dataset_builder import TARGET_COLUMN, build_ml_dataset

        return {"TARGET_COLUMN": TARGET_COLUMN, "build_ml_dataset": build_ml_dataset}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
