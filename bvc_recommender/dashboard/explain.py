"""
Explications SHAP — LightGBM baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from bvc_recommender.config import RANDOM_STATE, DATASET_DIR
from bvc_recommender.dashboard.data import load_ml_dataset
from bvc_recommender.features.dataset_builder import TARGET_COLUMN
from bvc_recommender.models.baseline_models import prepare_splits


@st.cache_resource(show_spinner="Chargement du modèle LightGBM…")
def load_lightgbm_explainer():
    df = load_ml_dataset()
    if df.empty:
        path = DATASET_DIR / "ml_dataset.parquet"
        if path.is_file():
            df = pd.read_parquet(path)
    if df.empty:
        return None, [], None

    splits, features, _ = prepare_splits(df)
    train = splits.get("train", pd.DataFrame())
    if train.empty:
        return None, features, None

    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    X_train = train[features].fillna(0.0)
    y_train = train[TARGET_COLUMN]
    model.fit(X_train, y_train)

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        return model, features, explainer
    except Exception:
        return model, features, None


def get_ticker_features(ticker: str, as_of: pd.Timestamp, features: list[str]) -> pd.DataFrame | None:
    df = load_ml_dataset()
    if df.empty:
        return None
    df = df.copy()
    df["date_cours"] = pd.to_datetime(df["date_cours"], errors="coerce")
    sub = df[(df["ticker"] == ticker) & (df["date_cours"] <= as_of)].sort_values("date_cours")
    if sub.empty:
        return None
    row = sub.tail(1)
    return row[features].fillna(0.0)


def compute_shap_values(ticker: str, as_of: pd.Timestamp) -> pd.DataFrame | None:
    model, features, explainer = load_lightgbm_explainer()
    if model is None or not features:
        return None

    X = get_ticker_features(ticker, as_of, features)
    if X is None or X.empty:
        return None

    if explainer is not None:
        import shap

        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[0]
        values = np.asarray(sv).flatten()
    else:
        values = model.feature_importances_ * X.values.flatten()

    out = pd.DataFrame({"feature": features, "shap_value": values})
    out["abs_shap"] = out["shap_value"].abs()
    return out.sort_values("abs_shap", ascending=False).head(15)
