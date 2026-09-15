"""
Bloc B — boucle walk-forward sur les 5 stratégies C1–C5.

Réutilise le découpage de folds existant (walk_forward_folds / slice_fold_panels)
sans le modifier. Tuning RF = grille interne existante ; LightGBM = Optuna
(config gelée).
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from bvc_recommender.config import RANDOM_STATE
from bvc_recommender.features.dataset_builder import TARGET_COLUMN
from experiments.factorial_hybrid_adapt.bloc_b_metrics import (
    level1_predictive_metrics,
    level2_selection_metrics,
    metrics_to_long,
)
from experiments.factorial_hybrid_adapt.bloc_b_models import (
    MODEL_IDS,
    MODEL_SPECS,
    LGBMHyperParams,
    default_lgbm_params,
    load_lgbm_config,
    predict_model_bundle,
    train_model_bundle,
    tune_lightgbm_optuna,
)
from experiments.factorial_hybrid_adapt.factorial_cells import feature_columns_for_cell
from experiments.factorial_hybrid_adapt.random_forest_scorer import RFHyperParams
from experiments.factorial_hybrid_adapt.walk_forward_folds import Fold
from experiments.factorial_hybrid_adapt.walk_forward_stage1 import (
    DEFAULT_RF_PARAMS,
    slice_fold_panels,
)
from experiments.factorial_hybrid_adapt.walk_forward_variants import (
    INNER_VAL_MONTHS,
    _inner_train_val,
    tune_rf_for_fold,
)

logger = logging.getLogger(__name__)


def _tune_lgbm_for_fold(
    train: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = TARGET_COLUMN,
    n_trials: int | None = None,
) -> tuple[LGBMHyperParams, list[dict[str, Any]]]:
    fit_df, val_df = _inner_train_val(train, val_months=INNER_VAL_MONTHS)
    if fit_df.empty or val_df.empty:
        logger.warning("Inner split LGBM vide — fallback defaults config gelée")
        return default_lgbm_params(), []
    return tune_lightgbm_optuna(
        {"train": fit_df, "val": val_df},
        feature_cols,
        target_col=target_col,
        n_trials=n_trials,
        random_state=RANDOM_STATE,
    )


def run_one_fold_bloc_b(
    df: pd.DataFrame,
    fold: Fold,
    *,
    model_ids: Sequence[str] = MODEL_IDS,
    target_col: str = TARGET_COLUMN,
    retune_rf: bool = True,
    retune_lgbm: bool = True,
    fixed_rf_params: RFHyperParams = DEFAULT_RF_PARAMS,
    fixed_lgbm_params: LGBMHyperParams | None = None,
    lgbm_n_trials: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Entraîne / score les modèles demandés sur un pli (même train/test pour tous).

    Retourne :
      - prédictions test concaténées
      - résultats format long (métriques)
      - méta tuning
    """
    train, test, bounds = slice_fold_panels(df, fold, target_col=target_col)
    fixed_lgbm = fixed_lgbm_params or default_lgbm_params()

    # Tuning une fois sur features sans régime (même logique walk_forward_variants)
    # — hyperparams réutilisés pour les variantes +régime.
    tune_cache: dict[str, Any] = {"fold_id": fold.fold_id}
    rf_params = fixed_rf_params
    lgbm_params = fixed_lgbm

    need_rf = any(MODEL_SPECS[m].needs_rf for m in model_ids if m in MODEL_SPECS)
    need_lgbm = any(MODEL_SPECS[m].needs_lgbm for m in model_ids if m in MODEL_SPECS)
    feat_base = feature_columns_for_cell(df, regime=False)

    if need_rf:
        if retune_rf:
            best_rf, hist_rf = tune_rf_for_fold(
                train, feat_base, target_col=target_col
            )
            rf_params = best_rf
            tune_cache["rf"] = {
                "retuned": True,
                "best_params": asdict(best_rf),
                "n_grid": len(hist_rf),
                "best_inner_rank_ic": hist_rf[0]["rank_ic_val"] if hist_rf else None,
            }
        else:
            tune_cache["rf"] = {
                "retuned": False,
                "best_params": asdict(fixed_rf_params),
            }

    if need_lgbm:
        if retune_lgbm:
            best_l, hist_l = _tune_lgbm_for_fold(
                train,
                feat_base,
                target_col=target_col,
                n_trials=lgbm_n_trials,
            )
            lgbm_params = best_l
            tune_cache["lgbm"] = {
                "retuned": True,
                "best_params": asdict(best_l),
                "n_trials": len(hist_l),
                "best_inner_rank_ic": hist_l[0]["rank_ic_val"] if hist_l else None,
            }
        else:
            tune_cache["lgbm"] = {
                "retuned": False,
                "best_params": asdict(fixed_lgbm),
            }

    pred_frames: list[pd.DataFrame] = []
    long_rows: list[dict[str, Any]] = []
    # Cache Ridge (+ RF / LGBM) par flag régime — un seul fit par composante
    component_cache: dict[bool, dict[str, Any]] = {}

    def _components_for_regime(regime_flag: bool) -> dict[str, Any]:
        if regime_flag in component_cache:
            return component_cache[regime_flag]
        if need_lgbm and need_rf:
            cover = "c5_hybrid_tri_regime" if regime_flag else "c4_hybrid_tri"
        elif need_lgbm:
            cover = "c3_lightgbm"
        elif need_rf:
            cover = "c2_rf"
        else:
            cover = "c1_ridge"
        base = train_model_bundle(
            train,
            cover,
            target_col=target_col,
            rf_params=rf_params,
            lgbm_params=lgbm_params,
            panel_for_features=df,
        )
        if need_rf and "rf" not in base:
            rf_only = train_model_bundle(
                train,
                "c2_rf",
                target_col=target_col,
                rf_params=rf_params,
                lgbm_params=lgbm_params,
                panel_for_features=df,
            )
            base["rf"] = rf_only["rf"]
            base["rf_params"] = rf_only.get("rf_params")
        if need_lgbm and "lgbm" not in base:
            lgb_only = train_model_bundle(
                train,
                "c3_lightgbm",
                target_col=target_col,
                rf_params=rf_params,
                lgbm_params=lgbm_params,
                panel_for_features=df,
            )
            base["lgbm"] = lgb_only["lgbm"]
            base["lgbm_params"] = lgb_only.get("lgbm_params")
        component_cache[regime_flag] = base
        return base

    for model_id in model_ids:
        if model_id not in MODEL_SPECS:
            raise ValueError(f"model_id inconnu : {model_id}")
        spec = MODEL_SPECS[model_id]
        logger.info(
            "[BlocB] fold=%s model=%s groupe=%s regime=%s train=%s->%s test=%s->%s",
            fold.fold_id,
            model_id,
            spec.groupe,
            spec.regime,
            fold.train_start_month,
            fold.train_end_month,
            fold.test_start_month,
            fold.test_end_month,
        )

        shared = _components_for_regime(spec.regime)
        bundle: dict[str, Any] = {
            "model_id": model_id,
            "groupe": spec.groupe,
            "regime": spec.regime,
            "feature_columns": shared["feature_columns"],
            "ridge": shared["ridge"],
        }
        if spec.needs_rf:
            bundle["rf"] = shared["rf"]
            bundle["rf_params"] = shared.get("rf_params")
        if spec.needs_lgbm:
            bundle["lgbm"] = shared["lgbm"]
            bundle["lgbm_params"] = shared.get("lgbm_params")

        pred = predict_model_bundle(bundle, test, target_col=target_col)
        pred["fold_id"] = fold.fold_id
        pred["split"] = "test"
        pred["train_start"] = str(fold.train_start.date())
        pred["train_end"] = str(fold.train_end.date())
        pred["test_start"] = str(fold.test_start.date())
        pred["test_end"] = str(fold.test_end.date())
        pred["test_start_month"] = fold.test_start_month
        pred["test_end_month"] = fold.test_end_month
        pred["clip_lower"] = bounds["lower"]
        pred["clip_upper"] = bounds["upper"]
        pred_frames.append(pred)

        l1 = level1_predictive_metrics(pred, target_col=target_col)
        l2 = level2_selection_metrics(pred, target_col=target_col)
        long_rows.extend(
            metrics_to_long(
                fold_id=fold.fold_id,
                model_id=model_id,
                level1=l1,
                level2=l2,
                groupe=spec.groupe,
            )
        )
        logger.info(
            "[BlocB] fold=%s model=%s | Rank-IC=%.4f IC-IR=%.4f RMSE=%.4f | "
            "spread_BUY_SELL=%.4f turnover=%.3f",
            fold.fold_id,
            model_id,
            l1.get("rank_ic", float("nan")),
            l1.get("ic_ir", float("nan")),
            l1.get("rmse", float("nan")),
            l2.get("spread_BUY_SELL", float("nan")),
            l2.get("turnover", float("nan")),
        )

    preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    results_long = pd.DataFrame(long_rows)
    return preds, results_long, tune_cache


def run_bloc_b_walk_forward(
    df: pd.DataFrame,
    folds: list[Fold],
    *,
    parts_dir: Path,
    model_ids: Sequence[str] = MODEL_IDS,
    retune_rf: bool = True,
    retune_lgbm: bool = True,
    lgbm_n_trials: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """
    Boucle sur les folds fournis (81 overlapping ou 14 blocs — même générateur).
    Résultats long concaténés sous ``parts_dir/results_long.csv``.
    """
    parts_dir.mkdir(parents=True, exist_ok=True)
    long_path = parts_dir / "results_long.csv"
    tune_path = parts_dir / "tune_per_fold.jsonl"

    all_long: list[pd.DataFrame] = []
    done: set[int] = set()
    if resume and long_path.is_file():
        prev = pd.read_csv(long_path)
        all_long.append(prev)
        done = set(int(x) for x in prev["fold_id"].unique())
        logger.info("[BlocB] resume : %s plis déjà faits", len(done))

    cfg_note = load_lgbm_config()
    logger.info(
        "[BlocB] config LGBM gelée v%s @ %s | n_trials=%s",
        cfg_note.get("version"),
        cfg_note.get("frozen_at"),
        lgbm_n_trials if lgbm_n_trials is not None else cfg_note.get("n_trials"),
    )

    for fold in folds:
        part_pred = parts_dir / f"fold_{fold.fold_id:04d}_preds.parquet"
        part_long = parts_dir / f"fold_{fold.fold_id:04d}_long.csv"
        if resume and fold.fold_id in done and part_pred.is_file() and part_long.is_file():
            continue

        preds, results_long, tune_meta = run_one_fold_bloc_b(
            df,
            fold,
            model_ids=model_ids,
            retune_rf=retune_rf,
            retune_lgbm=retune_lgbm,
            lgbm_n_trials=lgbm_n_trials,
        )
        preds.to_parquet(part_pred, index=False)
        results_long.to_csv(part_long, index=False)

        # Remplace ce fold dans l'agrégat
        if all_long:
            concat_prev = pd.concat(all_long, ignore_index=True)
            concat_prev = concat_prev[concat_prev["fold_id"] != fold.fold_id]
            all_long = [concat_prev] if not concat_prev.empty else []
        all_long.append(results_long)
        pd.concat(all_long, ignore_index=True).to_csv(long_path, index=False)

        with tune_path.open("a", encoding="utf-8") as f:
            import json

            f.write(json.dumps(tune_meta, default=str) + "\n")

        done.add(fold.fold_id)

    results = (
        pd.concat(all_long, ignore_index=True)
        if all_long
        else pd.DataFrame(columns=["fold_id", "model_id", "groupe", "metrique", "valeur"])
    )
    return {
        "results_long": results,
        "parts_dir": parts_dir,
        "n_folds_done": int(results["fold_id"].nunique()) if not results.empty else 0,
        "model_ids": list(model_ids),
        "lgbm_config": cfg_note,
    }
