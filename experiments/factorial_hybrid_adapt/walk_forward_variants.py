"""
Walk-forward variantes A/B/C — retune RF par pli et/ou return_dispersion_z.

A : retune RF seul (dataset original)
B : return_dispersion_z corrigée seule (HP RF fixes)
C : les deux
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
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
from experiments.factorial_hybrid_adapt.random_forest_scorer import (
    RFHyperParams,
    tune_random_forest_on_val,
)
from experiments.factorial_hybrid_adapt.walk_forward_folds import Fold
from experiments.factorial_hybrid_adapt.walk_forward_stage1 import (
    DEFAULT_RF_PARAMS,
    slice_fold_panels,
)

logger = logging.getLogger(__name__)

# Grille allégée (évite depth=8 / n=200 qui bloquaient ~11h/pli)
RETUNE_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [100],
    "max_depth": [3, 4, 6],
    "min_samples_leaf": [10, 20, 30],
}
INNER_VAL_MONTHS = 6


def _inner_train_val(
    train: pd.DataFrame,
    *,
    val_months: int = INNER_VAL_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = train.copy()
    tmp["date_cours"] = pd.to_datetime(tmp["date_cours"], errors="coerce")
    tmp["_mois"] = tmp["date_cours"].dt.to_period("M")
    months = sorted(tmp["_mois"].dropna().unique())
    if len(months) <= val_months + 6:
        # fenetre trop courte : 80/20 temporel
        cutoff = tmp["date_cours"].quantile(0.8)
        fit = tmp[tmp["date_cours"] <= cutoff].drop(columns=["_mois"])
        val = tmp[tmp["date_cours"] > cutoff].drop(columns=["_mois"])
        return fit, val
    val_m = set(months[-val_months:])
    fit = tmp[~tmp["_mois"].isin(val_m)].drop(columns=["_mois"])
    val = tmp[tmp["_mois"].isin(val_m)].drop(columns=["_mois"])
    return fit, val


def tune_rf_for_fold(
    train: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = TARGET_COLUMN,
    param_grid: dict[str, list[Any]] | None = None,
) -> tuple[RFHyperParams, list[dict[str, Any]]]:
    """Recherche grille sur val interne (derniers 6 mois du train du pli)."""
    fit_df, val_df = _inner_train_val(train)
    if fit_df.empty or val_df.empty:
        logger.warning("Inner split vide — fallback DEFAULT_RF_PARAMS")
        return DEFAULT_RF_PARAMS, []

    # Réduit le bruit de logs pendant la grille
    rf_logger = logging.getLogger("experiments.factorial_hybrid_adapt.random_forest_scorer")
    prev = rf_logger.level
    rf_logger.setLevel(logging.WARNING)
    try:
        best, history = tune_random_forest_on_val(
            {"train": fit_df, "val": val_df},
            feature_cols,
            target_col=target_col,
            param_grid=param_grid or RETUNE_PARAM_GRID,
            random_state=RANDOM_STATE,
        )
    finally:
        rf_logger.setLevel(prev)
    return best, history


def run_one_fold_variant(
    df: pd.DataFrame,
    fold: Fold,
    *,
    retune_rf: bool,
    fixed_rf_params: RFHyperParams = DEFAULT_RF_PARAMS,
    target_col: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    train, test, bounds = slice_fold_panels(df, fold, target_col=target_col)

    # Tune une fois sur features sans régime (réutilisé pour regime=True)
    if retune_rf:
        feat_base = feature_columns_for_cell(df, regime=False)
        best_params, hist = tune_rf_for_fold(train, feat_base, target_col=target_col)
        tune_meta = {
            "retuned": True,
            "best_params": asdict(best_params),
            "n_grid_tried": len(hist),
            "best_inner_rank_ic": hist[0]["rank_ic_val"] if hist else None,
        }
        rf_params = best_params
    else:
        rf_params = fixed_rf_params
        tune_meta = {"retuned": False, "best_params": asdict(rf_params)}

    model_cache: dict[bool, dict[str, Any]] = {}
    pred_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []

    for cell in CELLS:
        if cell.regime not in model_cache:
            feat_cols = feature_columns_for_cell(df, regime=cell.regime)
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
                "test_start_month": fold.test_start_month,
                "test_end_month": fold.test_end_month,
                "n_train": len(train),
                "n_test": len(test),
                "rank_ic": m.get("rank_ic_mean"),
                "ic_pearson": m.get("ic_mean"),
                "hit_ratio": m.get("hit_ratio"),
                "rf_n_estimators": rf_params.n_estimators,
                "rf_max_depth": rf_params.max_depth,
                "rf_min_samples_leaf": rf_params.min_samples_leaf,
            }
        )

    preds = pd.concat(pred_frames, ignore_index=True)
    tune_meta["fold_id"] = fold.fold_id
    return preds, metrics_rows, tune_meta


def run_variant(
    df: pd.DataFrame,
    folds: list[Fold],
    *,
    variant_name: str,
    parts_dir: Path,
    retune_rf: bool,
    resume: bool = True,
) -> dict[str, Any]:
    parts_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = parts_dir / "fold_metrics_partial.csv"
    tune_path = parts_dir / "rf_tune_per_fold.csv"

    all_metrics: list[dict[str, Any]] = []
    all_tunes: list[dict[str, Any]] = []
    if resume and metrics_path.is_file():
        prev = pd.read_csv(metrics_path)
        all_metrics = prev.to_dict(orient="records")
        done = set(int(x) for x in prev["fold_id"].unique())
        logger.info("[%s] resume : %s plis faits", variant_name, len(done))
    else:
        done = set()

    if resume and tune_path.is_file():
        all_tunes = pd.read_csv(tune_path).to_dict(orient="records")

    for fold in folds:
        part = parts_dir / f"fold_{fold.fold_id:04d}_preds.parquet"
        if resume and fold.fold_id in done and part.is_file():
            continue

        logger.info(
            "[%s] Pli %s/%s test %s->%s retune=%s",
            variant_name,
            fold.fold_id,
            folds[-1].fold_id,
            fold.test_start_month,
            fold.test_end_month,
            retune_rf,
        )
        preds, metrics_rows, tune_meta = run_one_fold_variant(
            df, fold, retune_rf=retune_rf
        )
        preds.to_parquet(part, index=False)

        all_metrics = [m for m in all_metrics if int(m["fold_id"]) != fold.fold_id]
        all_metrics.extend(metrics_rows)
        pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)

        flat_tune = {
            "fold_id": tune_meta["fold_id"],
            "retuned": tune_meta["retuned"],
            **{f"rf_{k}": v for k, v in tune_meta["best_params"].items()},
            "best_inner_rank_ic": tune_meta.get("best_inner_rank_ic"),
            "n_grid_tried": tune_meta.get("n_grid_tried"),
        }
        all_tunes = [t for t in all_tunes if int(t["fold_id"]) != fold.fold_id]
        all_tunes.append(flat_tune)
        pd.DataFrame(all_tunes).to_csv(tune_path, index=False)

        by_c = {m["cell_id"]: m for m in metrics_rows}
        logger.info(
            "[%s] Pli %s IC C1=%.4f C3=%.4f | RF depth=%s leaf=%s n=%s",
            variant_name,
            fold.fold_id,
            by_c[1]["rank_ic"],
            by_c[3]["rank_ic"],
            rf_params_depth(tune_meta),
            tune_meta["best_params"].get("min_samples_leaf"),
            tune_meta["best_params"].get("n_estimators"),
        )

    metrics_df = pd.DataFrame(all_metrics).sort_values(["fold_id", "cell_id"])
    return {
        "variant": variant_name,
        "retune_rf": retune_rf,
        "metrics": metrics_df,
        "n_folds_done": int(metrics_df["fold_id"].nunique()) if len(metrics_df) else 0,
        "parts_dir": parts_dir,
        "tune_log": pd.DataFrame(all_tunes),
    }


def rf_params_depth(tune_meta: dict[str, Any]) -> Any:
    return tune_meta["best_params"].get("max_depth")
