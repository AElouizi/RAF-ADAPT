"""
Étape 2 — Random Forest sur les mêmes features / cible / splits que Ridge.

- Features : colonnes ``*_z`` (via ``prepare_splits``)
- Cible : ``alpha_ajuste_risque``
- Split : train ≤ 2020-12-31 | val 2021–2022 | test ≥ 2023-01-01
- Tuning : grille sur le split **val** uniquement (pas de fuite future)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from bvc_recommender.config import RANDOM_STATE
from bvc_recommender.features.dataset_builder import TARGET_COLUMN
from bvc_recommender.models.baseline_models import (
    get_z_feature_columns,
    prepare_splits,
)
from bvc_recommender.models.metrics import evaluate_scoring_model

logger = logging.getLogger(__name__)

# Grille raisonnable — sélection par Rank-IC (Spearman) moyen sur val
DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 400],
    "max_depth": [4, 8, 12, None],
    "min_samples_leaf": [5, 10, 20],
}


@dataclass(frozen=True)
class RFHyperParams:
    n_estimators: int = 200
    max_depth: int | None = 8
    min_samples_leaf: int = 10
    random_state: int = RANDOM_STATE
    n_jobs: int = -1  # OK avec grille light (depth<=6, n_estimators=100)

    def to_sklearn(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }


def _build_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].fillna(0.0)
    y = df[target_col]
    return X, y


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: RFHyperParams,
) -> RandomForestRegressor:
    model = RandomForestRegressor(**params.to_sklearn())
    model.fit(X_train, y_train)
    return model


def _rank_ic_on_frame(
    df: pd.DataFrame,
    preds: np.ndarray,
    target_col: str,
) -> float:
    eval_df = df[["ticker", "date_cours", target_col]].copy()
    eval_df["prediction"] = preds
    metrics = evaluate_scoring_model(eval_df, "prediction", target_col)
    return float(metrics.get("rank_ic_mean", np.nan))


def tune_random_forest_on_val(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    *,
    target_col: str = TARGET_COLUMN,
    param_grid: dict[str, list[Any]] | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[RFHyperParams, list[dict[str, Any]]]:
    """
    Fit chaque combinaison sur train, score Rank-IC Spearman sur val.
    Retourne les meilleurs hyperparamètres + historique de grille.
    """
    grid = param_grid or DEFAULT_PARAM_GRID
    keys = list(grid.keys())
    combos = list(product(*(grid[k] for k in keys)))

    X_train, y_train = _build_xy(splits["train"], feature_cols, target_col)
    X_val, _ = _build_xy(splits["val"], feature_cols, target_col)

    history: list[dict[str, Any]] = []
    best_score = -np.inf
    best_params = RFHyperParams(random_state=random_state)

    logger.info(
        "RF tuning : %s combinaisons | train=%s | val=%s | features=%s",
        len(combos),
        len(X_train),
        len(X_val),
        len(feature_cols),
    )

    for i, values in enumerate(combos, start=1):
        kwargs = dict(zip(keys, values))
        params = RFHyperParams(random_state=random_state, **kwargs)
        model = fit_random_forest(X_train, y_train, params)
        preds = model.predict(X_val)
        rank_ic = _rank_ic_on_frame(splits["val"], preds, target_col)
        full_metrics = evaluate_scoring_model(
            pd.DataFrame(
                {
                    "ticker": splits["val"]["ticker"].values,
                    "date_cours": splits["val"]["date_cours"].values,
                    target_col: splits["val"][target_col].values,
                    "prediction": preds,
                }
            ),
            "prediction",
            target_col,
        )
        row = {
            **kwargs,
            "rank_ic_val": None if np.isnan(rank_ic) else round(rank_ic, 6),
            "ic_mean_val": full_metrics.get("ic_mean"),
            "hit_ratio_val": full_metrics.get("hit_ratio"),
        }
        history.append(row)
        logger.info(
            "[%s/%s] %s → Rank-IC_val=%.4f",
            i,
            len(combos),
            kwargs,
            rank_ic if not np.isnan(rank_ic) else float("nan"),
        )
        score = rank_ic if not np.isnan(rank_ic) else -np.inf
        if score > best_score:
            best_score = score
            best_params = params

    history.sort(
        key=lambda r: r["rank_ic_val"] if r["rank_ic_val"] is not None else -999,
        reverse=True,
    )
    logger.info("Meilleurs params RF : %s | Rank-IC_val=%.4f", asdict(best_params), best_score)
    return best_params, history


def train_and_evaluate_random_forest(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COLUMN,
    param_grid: dict[str, list[Any]] | None = None,
    include_regime: bool = False,
    regime_cols: tuple[str, ...] = ("is_bull", "is_neutral", "is_bear"),
) -> dict[str, Any]:
    """
    Pipeline complet étape 2 : prepare → tune sur val → refit train → métriques val/test.

    ``include_regime`` est prévu pour l'étape 3 (cellules 2/4) ; défaut False ici.
    """
    feature_cols = get_z_feature_columns(df)
    if include_regime:
        missing = [c for c in regime_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes régime manquantes : {missing}")
        feature_cols = feature_cols + list(regime_cols)

    splits, feature_cols, clip_bounds = prepare_splits(
        df, feature_cols=feature_cols, target_col=target_col
    )
    if splits["train"].empty or splits["val"].empty:
        raise ValueError("Split train ou val vide.")

    best_params, grid_history = tune_random_forest_on_val(
        splits,
        feature_cols,
        target_col=target_col,
        param_grid=param_grid,
    )

    X_train, y_train = _build_xy(splits["train"], feature_cols, target_col)
    model = fit_random_forest(X_train, y_train, best_params)

    results: dict[str, Any] = {
        "model_name": "random_forest",
        "feature_columns": feature_cols,
        "n_features": len(feature_cols),
        "include_regime": include_regime,
        "clip_bounds": clip_bounds,
        "best_params": asdict(best_params),
        "tuning_metric": "rank_ic_mean (Spearman) on val",
        "grid_history": grid_history,
        "splits": {},
    }

    all_predictions: list[pd.DataFrame] = []
    for split_name in ("val", "test"):
        split_df = splits[split_name]
        if split_df.empty:
            continue
        X_split, _ = _build_xy(split_df, feature_cols, target_col)
        preds = model.predict(X_split)
        eval_df = split_df[["ticker", "date_cours", target_col]].copy()
        eval_df["prediction"] = preds
        eval_df["model"] = "random_forest"
        eval_df["split"] = split_name
        all_predictions.append(eval_df)
        metrics = evaluate_scoring_model(eval_df, "prediction", target_col)
        results["splits"][split_name] = {"rows": len(eval_df), "metrics": metrics}
        logger.info(
            "[RF/%s] Rank-IC=%.4f | IC=%.4f | Hit=%.1f%%",
            split_name,
            metrics.get("rank_ic_mean", np.nan),
            metrics.get("ic_mean", np.nan),
            (metrics.get("hit_ratio", np.nan) or 0) * 100,
        )

    results["predictions"] = (
        pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    )
    results["model"] = model
    return results


def save_rf_artifacts(
    results: dict[str, Any],
    *,
    models_dir: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    """Persiste modèle, prédictions et rapport JSON (hors arbre TFT / step5)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "random_forest.joblib"
    preds_path = reports_dir / "rf_predictions.parquet"
    report_path = reports_dir / "rf_etape2_report.json"
    grid_path = reports_dir / "rf_tuning_grid.csv"

    joblib.dump(
        {
            "model": results["model"],
            "feature_columns": results["feature_columns"],
            "best_params": results["best_params"],
            "clip_bounds": results["clip_bounds"],
            "include_regime": results["include_regime"],
        },
        model_path,
    )

    preds = results.get("predictions")
    if isinstance(preds, pd.DataFrame) and not preds.empty:
        preds.to_parquet(preds_path, index=False)

    grid_df = pd.DataFrame(results.get("grid_history") or [])
    if not grid_df.empty:
        grid_df.to_csv(grid_path, index=False)

    serializable = {
        k: v
        for k, v in results.items()
        if k not in ("model", "predictions")
    }
    # Convert numpy types in nested metrics
    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            x = float(obj)
            return None if np.isnan(x) else x
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    report_path.write_text(
        json.dumps(_json_safe(serializable), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "model": model_path,
        "predictions": preds_path,
        "report": report_path,
        "grid": grid_path,
    }
