"""
Variantes TFT pour l'étude d'ablation (Niveau 3).

Réutilise la logique de ``stock_scorer`` en ne changeant que les known-reals :
- fuzzy  : mu_* (scikit-fuzzy / Mamdani, benchmarking/fuzzy_regime)
- none   : sans régime
- discrete : is_* (pipeline principal seuils — utile si on force la source)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

from bvc_recommender.benchmarking.fuzzy_regime import FUZZY_REGIME_COLUMNS
from bvc_recommender.config import DATASET_DIR, FEATURES_DIR, RANDOM_STATE
from bvc_recommender.features.dataset_builder import TARGET_COLUMN
from bvc_recommender.models.baseline_models import prepare_splits
from bvc_recommender.models.regime_detector import REGIME_COLUMNS
from bvc_recommender.models.stock_scorer import (
    DEFAULT_ENCODER_LENGTH,
    DEFAULT_MAX_TRAIN_ROWS,
    TFT_QUANTILES,
    _extract_quantile_array,
    _history_frame_for_split,
    _prediction_time_col,
    _prepare_tft_frame,
    attach_regime_features,
)

logger = logging.getLogger(__name__)

RegimeMode = Literal["fuzzy", "none", "discrete"]


def _build_dataset(frame: pd.DataFrame, feature_cols: list[str], known_reals: list[str], max_encoder_length: int):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    return TimeSeriesDataSet(
        frame,
        time_idx="time_idx",
        target=TARGET_COLUMN,
        group_ids=["ticker"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=feature_cols,
        target_normalizer=GroupNormalizer(groups=["ticker"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        randomize_length=None,
    )


def train_tft_ablation(
    df: pd.DataFrame,
    *,
    regime_mode: RegimeMode,
    regime_history: pd.DataFrame | None = None,
    regime_daily: pd.DataFrame | None = None,
    max_epochs: int = 3,
    max_train_rows: int = DEFAULT_MAX_TRAIN_ROWS,
    max_encoder_length: int = DEFAULT_ENCODER_LENGTH,
    batch_size: int = 256,
) -> dict[str, Any]:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss

    pl.seed_everything(RANDOM_STATE, workers=True)

    if regime_mode == "fuzzy":
        if regime_history is None or regime_history.empty:
            raise ValueError("regime_history (mensuel mu_*) requis pour mode fuzzy")
        enriched = attach_regime_features(
            df,
            regime_history,
            columns=FUZZY_REGIME_COLUMNS,
            fill_values={c: 1 / 3 for c in FUZZY_REGIME_COLUMNS},
        )
        known = list(FUZZY_REGIME_COLUMNS)
    elif regime_mode == "discrete":
        source = regime_daily
        if source is None or source.empty:
            path = FEATURES_DIR / "features_indices.parquet"
            if path.is_file():
                source = pd.read_parquet(path)
        if source is None or source.empty:
            raise ValueError("regime_daily / features_indices requis pour mode discrete")
        enriched = attach_regime_features(df, source, columns=REGIME_COLUMNS)
        known = list(REGIME_COLUMNS)
    else:
        enriched = df.copy()
        known = []

    splits, feature_cols, clip_bounds = prepare_splits(enriched, target_col=TARGET_COLUMN)
    train_frame = _prepare_tft_frame(
        splits["train"], feature_cols, known_reals=known, subsample=max_train_rows
    )
    for col in known:
        if col not in train_frame.columns:
            train_frame[col] = 0.0
        train_frame[col] = pd.to_numeric(train_frame[col], errors="coerce").fillna(0.0)

    training = _build_dataset(train_frame, feature_cols, known, max_encoder_length)
    train_tickers = set(train_frame["ticker"].unique())
    val_part = splits.get("val", pd.DataFrame())
    val_part = val_part[val_part["ticker"].isin(train_tickers)]
    val_frame = (
        _prepare_tft_frame(val_part, feature_cols, known_reals=known)
        if not val_part.empty
        else train_frame.iloc[:0]
    )
    for col in known:
        if col not in val_frame.columns:
            val_frame[col] = 0.0

    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_frame if not val_frame.empty else train_frame,
        predict=True,
        stop_randomization=True,
    )

    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(quantiles=TFT_QUANTILES),
        log_interval=-1,
        reduce_on_plateau_patience=2,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="val_loss", patience=2, mode="min")],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(tft, train_loader, val_loader)

    return {
        "model": tft,
        "training_dataset": training,
        "feature_columns": feature_cols,
        "known_reals": known,
        "clip_bounds": clip_bounds,
        "max_encoder_length": max_encoder_length,
        "train_tickers": sorted(train_tickers),
        "regime_mode": regime_mode,
        "enriched_template": enriched,
    }


def predict_tft_ablation(
    bundle: dict[str, Any],
    df: pd.DataFrame,
    *,
    regime_mode: RegimeMode,
    regime_history: pd.DataFrame | None = None,
    regime_daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    from pytorch_forecasting import TimeSeriesDataSet

    if regime_mode == "fuzzy":
        enriched = attach_regime_features(
            df,
            regime_history,
            columns=FUZZY_REGIME_COLUMNS,
            fill_values={c: 1 / 3 for c in FUZZY_REGIME_COLUMNS},
        )
    elif regime_mode == "discrete":
        source = regime_daily
        if source is None or source.empty:
            path = FEATURES_DIR / "features_indices.parquet"
            if path.is_file():
                source = pd.read_parquet(path)
        enriched = attach_regime_features(df, source, columns=REGIME_COLUMNS)
    else:
        enriched = df.copy()

    known = bundle["known_reals"]
    for col in known:
        if col not in enriched.columns:
            enriched[col] = 0.0

    splits, feature_cols, _ = prepare_splits(enriched, target_col=TARGET_COLUMN)
    train_tickers = set(bundle.get("train_tickers", []))
    parts = []
    for split_name in ("val", "test"):
        frame, eval_dates = _history_frame_for_split(
            splits, split_name, feature_cols, train_tickers, known_reals=known
        )
        if frame.empty or not eval_dates:
            continue
        for col in known:
            if col not in frame.columns:
                frame[col] = 0.0
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

        dataset = TimeSeriesDataSet.from_dataset(
            bundle["training_dataset"],
            frame,
            predict=False,
            stop_randomization=True,
        )
        loader = dataset.to_dataloader(train=False, batch_size=256, num_workers=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = bundle["model"].predict(
                loader, mode="quantiles", return_x=False, return_index=True
            )
        if hasattr(prediction, "output") and hasattr(prediction, "index"):
            preds = _extract_quantile_array(prediction.output)
            index_df = prediction.index.reset_index(drop=True)
        else:
            preds = _extract_quantile_array(prediction)
            index_df = dataset.decoded_index.reset_index(drop=True)
        time_col = _prediction_time_col(index_df)
        n = min(len(index_df), len(preds))
        index_df = index_df.iloc[:n].copy()
        preds = preds[:n]
        full_meta = index_df.merge(
            frame[["ticker", "time_idx", "date_cours", TARGET_COLUMN]],
            left_on=["ticker", time_col],
            right_on=["ticker", "time_idx"],
            how="left",
        )
        full_meta["date_cours"] = pd.to_datetime(full_meta["date_cours"], errors="coerce")
        keep = full_meta["date_cours"].isin(eval_dates).to_numpy()
        full_meta = full_meta.loc[keep].reset_index(drop=True)
        preds = preds[keep]
        if full_meta.empty:
            continue
        out = full_meta[["ticker", "date_cours"]].copy()
        out["prediction"] = preds[:, 1] if preds.shape[1] >= 2 else preds[:, 0]
        out["split"] = split_name
        parts.append(out)

    if not parts:
        return pd.DataFrame(columns=["ticker", "date_cours", "prediction", "split"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "date_cours"], keep="last")


def load_regime_history() -> pd.DataFrame:
    path = DATASET_DIR / "regime_history.parquet"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)
