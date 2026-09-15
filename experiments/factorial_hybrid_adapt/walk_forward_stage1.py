"""
Étape 3 WF — réentraînement des 4 cellules sur chaque pli rolling.

Réutilise CELLS / Ridge / RF / hybride de factorial_cells (pas de duplication métier).
Hyperparams RF figés (étape 2 factorielle) — pas de re-tuning par pli.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bvc_recommender.config import RANDOM_STATE
from bvc_recommender.features.dataset_builder import TARGET_COLUMN
from bvc_recommender.models.metrics import evaluate_scoring_model
from experiments.factorial_hybrid_adapt.factorial_cells import (
    CELLS,
    feature_columns_for_cell,
    predict_cell,
    train_base_models,
)
from experiments.factorial_hybrid_adapt.random_forest_scorer import RFHyperParams
from experiments.factorial_hybrid_adapt.walk_forward_folds import Fold

logger = logging.getLogger(__name__)

DEFAULT_RF_PARAMS = RFHyperParams(
    n_estimators=100,
    max_depth=4,
    min_samples_leaf=20,
    random_state=RANDOM_STATE,
)


def _clip_bounds_from_train(y: pd.Series) -> dict[str, float]:
    return {
        "lower": float(y.quantile(0.01)),
        "upper": float(y.quantile(0.99)),
    }


def slice_fold_panels(
    df: pd.DataFrame,
    fold: Fold,
    *,
    target_col: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Découpe train/test calendaires du pli + clip cible appris sur train du pli."""
    data = df.copy()
    data["date_cours"] = pd.to_datetime(data["date_cours"], errors="coerce")
    train = data[
        (data["date_cours"] >= fold.train_start) & (data["date_cours"] <= fold.train_end)
    ].copy()
    test = data[
        (data["date_cours"] >= fold.test_start) & (data["date_cours"] <= fold.test_end)
    ].copy()
    train = train.dropna(subset=[target_col])
    test = test.dropna(subset=[target_col])
    if train.empty or test.empty:
        raise ValueError(
            f"Pli {fold.fold_id}: train={len(train)} test={len(test)} — panel vide"
        )
    bounds = _clip_bounds_from_train(train[target_col])
    train[target_col] = train[target_col].clip(**bounds)
    test[target_col] = test[target_col].clip(**bounds)
    return train, test, bounds


def run_one_fold(
    df: pd.DataFrame,
    fold: Fold,
    *,
    target_col: str = TARGET_COLUMN,
    rf_params: RFHyperParams = DEFAULT_RF_PARAMS,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Entraîne les 4 cellules sur le pli, retourne prédictions test + métriques."""
    train, test, bounds = slice_fold_panels(df, fold, target_col=target_col)

    model_cache: dict[bool, dict[str, Any]] = {}
    pred_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []

    for cell in CELLS:
        if cell.regime not in model_cache:
            feat_cols = feature_columns_for_cell(df, regime=cell.regime)
            # Vérifie présence colonnes sur le train
            missing = [c for c in feat_cols if c not in train.columns]
            if missing:
                raise ValueError(f"Features manquantes pli {fold.fold_id}: {missing}")
            bundle = train_base_models(
                {"train": train},
                feat_cols,
                target_col=target_col,
                rf_params=rf_params,
                need_rf=True,
            )
            model_cache[cell.regime] = bundle

        bundle = model_cache[cell.regime]
        pred = predict_cell(bundle, test, cell, target_col=target_col)
        pred["fold_id"] = fold.fold_id
        pred["split"] = "test"
        pred["train_start"] = str(fold.train_start.date())
        pred["train_end"] = str(fold.train_end.date())
        pred["test_start"] = str(fold.test_start.date())
        pred["test_end"] = str(fold.test_end.date())
        pred["test_start_month"] = fold.test_start_month
        pred["test_end_month"] = fold.test_end_month
        pred_frames.append(pred)

        m = evaluate_scoring_model(pred, "prediction", target_col)
        metrics_rows.append(
            {
                "fold_id": fold.fold_id,
                "cell_id": cell.cell_id,
                "cell_name": cell.name,
                "model": cell.model,
                "regime": cell.regime,
                "train_start_month": fold.train_start_month,
                "train_end_month": fold.train_end_month,
                "test_start_month": fold.test_start_month,
                "test_end_month": fold.test_end_month,
                "n_train": len(train),
                "n_test": len(test),
                "n_features": len(bundle["feature_columns"]),
                "clip_lower": bounds["lower"],
                "clip_upper": bounds["upper"],
                "rank_ic": m.get("rank_ic_mean"),
                "ic_pearson": m.get("ic_mean"),
                "hit_ratio": m.get("hit_ratio"),
                "sharpe_ls": m.get("sharpe_long_short"),
            }
        )

    preds = pd.concat(pred_frames, ignore_index=True)
    return preds, metrics_rows


def run_all_overlapping_folds(
    df: pd.DataFrame,
    folds: list[Fold],
    *,
    parts_dir: Path,
    rf_params: RFHyperParams = DEFAULT_RF_PARAMS,
    resume: bool = True,
) -> dict[str, Any]:
    """
    Boucle sur tous les plis chevauchants.
    Checkpoint par pli sous ``parts_dir/fold_XXXX.parquet`` + metrics CSV partiel.
    """
    parts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = parts_dir / "fold_metrics_partial.csv"

    all_metrics: list[dict[str, Any]] = []
    if resume and metrics_path.is_file():
        prev = pd.read_csv(metrics_path)
        all_metrics = prev.to_dict(orient="records")
        done_folds = set(int(x) for x in prev["fold_id"].unique())
        logger.info("Resume : %s plis déjà faits", len(done_folds))
    else:
        done_folds = set()

    for fold in folds:
        part_pred = parts_dir / f"fold_{fold.fold_id:04d}_preds.parquet"
        part_met = parts_dir / f"fold_{fold.fold_id:04d}_metrics.csv"
        if resume and fold.fold_id in done_folds and part_pred.is_file():
            continue

        logger.info(
            "=== Pli %s/%s | train %s->%s | test %s->%s ===",
            fold.fold_id,
            folds[-1].fold_id,
            fold.train_start_month,
            fold.train_end_month,
            fold.test_start_month,
            fold.test_end_month,
        )
        preds, metrics_rows = run_one_fold(df, fold, rf_params=rf_params)
        preds.to_parquet(part_pred, index=False)
        pd.DataFrame(metrics_rows).to_csv(part_met, index=False)

        # Met à jour metrics agrégées (remplace ce fold_id si relance)
        all_metrics = [m for m in all_metrics if int(m["fold_id"]) != fold.fold_id]
        all_metrics.extend(metrics_rows)
        pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)

        # Log IC C1/C4 du pli
        by_cell = {m["cell_id"]: m for m in metrics_rows}
        logger.info(
            "Pli %s Rank-IC C1=%.4f C2=%.4f C3=%.4f C4=%.4f",
            fold.fold_id,
            by_cell[1]["rank_ic"],
            by_cell[2]["rank_ic"],
            by_cell[3]["rank_ic"],
            by_cell[4]["rank_ic"],
        )

    metrics_df = pd.DataFrame(all_metrics).sort_values(["fold_id", "cell_id"])
    return {
        "metrics": metrics_df,
        "parts_dir": parts_dir,
        "n_folds_expected": len(folds),
        "n_folds_done": int(metrics_df["fold_id"].nunique()) if not metrics_df.empty else 0,
        "rf_params": {
            "n_estimators": rf_params.n_estimators,
            "max_depth": rf_params.max_depth,
            "min_samples_leaf": rf_params.min_samples_leaf,
            "random_state": rf_params.random_state,
        },
    }


def consolidate_predictions(parts_dir: Path, out_path: Path) -> pd.DataFrame:
    files = sorted(parts_dir.glob("fold_*_preds.parquet"))
    if not files:
        raise FileNotFoundError(f"Aucun fold_*_preds.parquet dans {parts_dir}")
    frames = [pd.read_parquet(f) for f in files]
    out = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out
