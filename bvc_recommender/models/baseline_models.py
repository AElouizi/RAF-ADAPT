"""
Étape 5 — Modèles baseline de scoring (Ridge, LightGBM, XGBoost).

Entraînement sur le dataset ML (features z-scorées cross-sectionnellement),
évaluation IC / Hit Ratio / Sharpe sur val et test.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
import pandas as pd

from bvc_recommender.config import RANDOM_STATE
from bvc_recommender.features.dataset_builder import TARGET_COLUMN
from bvc_recommender.models.metrics import evaluate_scoring_model

logger = logging.getLogger(__name__)

BASELINE_MODEL_NAMES = ("ridge", "lightgbm", "xgboost")


class _FittedModel(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def get_z_feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.endswith("_z"))


def _clip_target(y: pd.Series, lower: float, upper: float) -> pd.Series:
    return y.clip(lower=lower, upper=upper)


def prepare_splits(
    df: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    target_col: str = TARGET_COLUMN,
) -> tuple[dict[str, pd.DataFrame], list[str], dict[str, float]]:
    """Prépare train/val/test avec clipping de la cible appris sur train."""
    data = df.copy()
    data["date_cours"] = pd.to_datetime(data["date_cours"], errors="coerce")
    feature_cols = feature_cols or get_z_feature_columns(data)

    splits: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        part = data[data["split"] == name].copy()
        part = part.dropna(subset=[target_col])
        splits[name] = part

    train_y = splits["train"][target_col]
    clip_bounds = {
        "lower": float(train_y.quantile(0.01)),
        "upper": float(train_y.quantile(0.99)),
    }

    for name in splits:
        splits[name][target_col] = _clip_target(splits[name][target_col], **clip_bounds)

    return splits, feature_cols, clip_bounds


def _build_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].fillna(0.0)
    y = df[target_col]
    return X, y


def _fit_ridge(X_train: pd.DataFrame, y_train: pd.Series) -> _FittedModel:
    from sklearn.linear_model import Ridge

    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def _fit_lightgbm(X_train: pd.DataFrame, y_train: pd.Series) -> _FittedModel:
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def _fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> _FittedModel:
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def train_model(name: str, X_train: pd.DataFrame, y_train: pd.Series) -> _FittedModel:
    builders = {
        "ridge": _fit_ridge,
        "lightgbm": _fit_lightgbm,
        "xgboost": _fit_xgboost,
    }
    if name not in builders:
        raise ValueError(f"Modèle inconnu : {name}")
    return builders[name](X_train, y_train)


def predict_model(model: _FittedModel, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)


def train_and_evaluate_baselines(
    df: pd.DataFrame,
    *,
    models: tuple[str, ...] = BASELINE_MODEL_NAMES,
    target_col: str = TARGET_COLUMN,
) -> dict[str, Any]:
    """
    Entraîne les baselines et retourne métriques val/test + prédictions agrégées.
    """
    splits, feature_cols, clip_bounds = prepare_splits(df, target_col=target_col)
    if splits["train"].empty:
        raise ValueError("Split train vide.")

    X_train, y_train = _build_xy(splits["train"], feature_cols, target_col)
    logger.info(
        "Entraînement baselines : train=%s | features=%s | cible clip [%.3f, %.3f]",
        len(X_train),
        len(feature_cols),
        clip_bounds["lower"],
        clip_bounds["upper"],
    )

    results: dict[str, Any] = {
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "clip_bounds": clip_bounds,
        "models": {},
    }

    all_predictions: list[pd.DataFrame] = []

    for model_name in models:
        logger.info("Entraînement %s ...", model_name)
        model = train_model(model_name, X_train, y_train)
        model_result: dict[str, Any] = {"name": model_name, "splits": {}}

        for split_name in ("val", "test"):
            split_df = splits[split_name]
            if split_df.empty:
                continue
            X_split, _ = _build_xy(split_df, feature_cols, target_col)
            preds = predict_model(model, X_split)

            eval_df = split_df[["ticker", "date_cours", target_col]].copy()
            eval_df["prediction"] = preds
            eval_df["model"] = model_name
            eval_df["split"] = split_name
            all_predictions.append(eval_df)

            metrics = evaluate_scoring_model(eval_df, "prediction", target_col)
            model_result["splits"][split_name] = {
                "rows": len(eval_df),
                "metrics": metrics,
            }
            logger.info(
                "[%s/%s] IC=%.4f | Hit=%.1f%% | Sharpe=%.2f",
                model_name,
                split_name,
                metrics.get("ic_mean", np.nan),
                (metrics.get("hit_ratio", np.nan) or 0) * 100,
                metrics.get("sharpe_long_short", np.nan),
            )

        results["models"][model_name] = model_result

    if all_predictions:
        results["predictions"] = pd.concat(all_predictions, ignore_index=True)
    return results
