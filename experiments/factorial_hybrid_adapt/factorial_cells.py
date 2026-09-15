"""
Étapes 3–4 — Plan factoriel 2×2 (Hybridation × Adaptation) + sélection top-25.

Cellules :
  1 Ridge | régime Non
  2 Ridge | régime Oui
  3 Ridge+RF (0.5/0.5, scores z-scorés par date) | régime Non
  4 Ridge+RF | régime Oui
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from bvc_recommender.config import RANDOM_STATE
from bvc_recommender.features.dataset_builder import TARGET_COLUMN
from bvc_recommender.models.baseline_models import get_z_feature_columns, prepare_splits
from bvc_recommender.models.metrics import evaluate_scoring_model
from bvc_recommender.models.regime_detector import REGIME_COLUMNS
from experiments.factorial_hybrid_adapt.random_forest_scorer import (
    RFHyperParams,
    fit_random_forest,
)

logger = logging.getLogger(__name__)

ModelKind = Literal["ridge", "hybrid"]
TOP_N = 25
HYBRID_WEIGHT_RIDGE = 0.5
HYBRID_WEIGHT_RF = 0.5
HYBRID_TRI_WEIGHT_RIDGE = 1.0 / 3.0
HYBRID_TRI_WEIGHT_RF = 1.0 / 3.0
HYBRID_TRI_WEIGHT_LGBM = 1.0 / 3.0


@dataclass(frozen=True)
class CellSpec:
    cell_id: int
    name: str
    model: ModelKind
    regime: bool

    @property
    def label(self) -> str:
        reg = "oui" if self.regime else "non"
        return f"C{self.cell_id}:{self.model}|regime={reg}"


CELLS: tuple[CellSpec, ...] = (
    CellSpec(1, "ridge_static", "ridge", False),
    CellSpec(2, "ridge_adapt", "ridge", True),
    CellSpec(3, "hybrid_static", "hybrid", False),
    CellSpec(4, "hybrid_adapt", "hybrid", True),
)


def feature_columns_for_cell(df: pd.DataFrame, *, regime: bool) -> list[str]:
    cols = get_z_feature_columns(df)
    if regime:
        missing = [c for c in REGIME_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes régime absentes du panel : {missing}")
        cols = cols + list(REGIME_COLUMNS)
    return cols


def _build_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].fillna(0.0)
    y = df[target_col]
    return X, y


def _fit_ridge(X: pd.DataFrame, y: pd.Series) -> Ridge:
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X, y)
    return model


def _cs_zscore(series: pd.Series) -> pd.Series:
    """Z-score cross-sectionnel (par date) — évite qu'une échelle domine l'hybride."""
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma is None or not np.isfinite(sigma) or sigma == 0:
        return series * 0.0
    return (series - mu) / sigma


def combine_hybrid(
    linear_pred: np.ndarray | pd.Series,
    nonlinear_pred: np.ndarray | pd.Series,
    dates: pd.Series | None = None,
    *,
    method: Literal["weighted_cs_zscore"] = "weighted_cs_zscore",
    w_linear: float = HYBRID_WEIGHT_RIDGE,
    w_nonlinear: float = HYBRID_WEIGHT_RF,
) -> np.ndarray:
    """
    Combine prédictions Ridge + composante non-linéaire (RF ou LightGBM).

    ``weighted_cs_zscore`` (méthode historique C3/C4) : z-score cross-sectionnel
    par date puis pondération. Ne pas dupliquer cette logique ailleurs.
    """
    if method != "weighted_cs_zscore":
        raise ValueError(f"method inconnu pour combine_hybrid : {method}")
    if dates is None:
        raise ValueError("dates requis pour method='weighted_cs_zscore'")
    panel = pd.DataFrame(
        {
            "date_cours": pd.to_datetime(dates, errors="coerce").values,
            "linear": np.asarray(linear_pred, dtype=float),
            "nonlinear": np.asarray(nonlinear_pred, dtype=float),
        }
    )
    panel["linear_z"] = panel.groupby("date_cours", sort=False)["linear"].transform(
        _cs_zscore
    )
    panel["nonlinear_z"] = panel.groupby("date_cours", sort=False)["nonlinear"].transform(
        _cs_zscore
    )
    return (w_linear * panel["linear_z"] + w_nonlinear * panel["nonlinear_z"]).to_numpy(
        dtype=float
    )


def combine_hybrid_scores(
    ridge_pred: np.ndarray | pd.Series,
    rf_pred: np.ndarray | pd.Series,
    dates: pd.Series,
    *,
    w_ridge: float = HYBRID_WEIGHT_RIDGE,
    w_rf: float = HYBRID_WEIGHT_RF,
) -> np.ndarray:
    """Alias rétrocompat C3/C4 — délègue à ``combine_hybrid``."""
    return combine_hybrid(
        ridge_pred,
        rf_pred,
        dates,
        method="weighted_cs_zscore",
        w_linear=w_ridge,
        w_nonlinear=w_rf,
    )


def combine_hybrid_tri(
    ridge_pred: np.ndarray | pd.Series,
    rf_pred: np.ndarray | pd.Series,
    lgbm_pred: np.ndarray | pd.Series,
    dates: pd.Series | None = None,
    *,
    w_ridge: float = HYBRID_TRI_WEIGHT_RIDGE,
    w_rf: float = HYBRID_TRI_WEIGHT_RF,
    w_lgbm: float = HYBRID_TRI_WEIGHT_LGBM,
) -> np.ndarray:
    """
  Ridge + RF + LightGBM : z-score cross-sectionnel par date, puis pondération égale.
  Utilisé pour C4 (sans régime) et C5 (avec régime).
    """
    if dates is None:
        raise ValueError("dates requis pour combine_hybrid_tri")
    panel = pd.DataFrame(
        {
            "date_cours": pd.to_datetime(dates, errors="coerce").values,
            "ridge": np.asarray(ridge_pred, dtype=float),
            "rf": np.asarray(rf_pred, dtype=float),
            "lgbm": np.asarray(lgbm_pred, dtype=float),
        }
    )
    panel["ridge_z"] = panel.groupby("date_cours", sort=False)["ridge"].transform(_cs_zscore)
    panel["rf_z"] = panel.groupby("date_cours", sort=False)["rf"].transform(_cs_zscore)
    panel["lgbm_z"] = panel.groupby("date_cours", sort=False)["lgbm"].transform(_cs_zscore)
    return (
        w_ridge * panel["ridge_z"] + w_rf * panel["rf_z"] + w_lgbm * panel["lgbm_z"]
    ).to_numpy(dtype=float)


def train_base_models(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    *,
    target_col: str = TARGET_COLUMN,
    rf_params: RFHyperParams | None = None,
    need_rf: bool = True,
) -> dict[str, Any]:
    X_train, y_train = _build_xy(splits["train"], feature_cols, target_col)
    ridge = _fit_ridge(X_train, y_train)
    out: dict[str, Any] = {"ridge": ridge, "feature_columns": feature_cols}
    if need_rf:
        params = rf_params or RFHyperParams(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        )
        out["rf"] = fit_random_forest(X_train, y_train, params)
        out["rf_params"] = params
    return out


def predict_cell(
    models: dict[str, Any],
    split_df: pd.DataFrame,
    cell: CellSpec,
    *,
    target_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    feature_cols = models["feature_columns"]
    X, _ = _build_xy(split_df, feature_cols, target_col)
    ridge_pred = models["ridge"].predict(X)

    if cell.model == "ridge":
        score = ridge_pred
    else:
        rf_pred = models["rf"].predict(X)
        score = combine_hybrid_scores(ridge_pred, rf_pred, split_df["date_cours"])

    out = split_df[["ticker", "date_cours", target_col]].copy()
    out["prediction"] = score
    out["score_ridge"] = ridge_pred
    if cell.model == "hybrid":
        out["score_rf"] = models["rf"].predict(X)
    out["cell_id"] = cell.cell_id
    out["cell_name"] = cell.name
    out["model"] = cell.model
    out["regime"] = cell.regime
    return out


def month_end_snapshots(df: pd.DataFrame, *, date_col: str = "date_cours") -> pd.DataFrame:
    """Dernier jour de bourse de chaque mois calendaire."""
    tmp = df.copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp["month"] = tmp[date_col].dt.to_period("M")
    last_dates = tmp.groupby("month", sort=True)[date_col].max()
    return tmp[tmp[date_col].isin(last_dates.values)].copy()


def select_top_n_monthly(
    pred_df: pd.DataFrame,
    *,
    n: int = TOP_N,
    score_col: str = "prediction",
    date_col: str = "date_cours",
) -> pd.DataFrame:
    """
    Pour chaque mois : top-n titres au dernier jour de bourse.
    Colonnes : titre | mois | score | rang_percentile (+ métadonnées).
    """
    snaps = month_end_snapshots(pred_df, date_col=date_col)
    rows: list[pd.DataFrame] = []
    for mois, grp in snaps.groupby(snaps[date_col].dt.to_period("M"), sort=True):
        g = (
            grp.sort_values(score_col, ascending=False)
            .drop_duplicates("ticker", keep="first")
            .copy()
        )
        g["rang"] = np.arange(1, len(g) + 1)
        g["rang_percentile"] = 1.0 - (g["rang"] - 1) / max(len(g) - 1, 1)
        top = g.head(n).copy()
        if len(top) != n:
            logger.warning("Mois %s : %s titres (attendu %s)", mois, len(top), n)
        top["mois"] = str(mois)
        top["titre"] = top["ticker"]
        top["score"] = top[score_col]
        rows.append(top)

    if not rows:
        return pd.DataFrame(
            columns=[
                "titre",
                "mois",
                "score",
                "rang_percentile",
                "rang",
                "ticker",
                "date_cours",
                "cell_id",
                "cell_name",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    base_cols = [
        "titre",
        "mois",
        "score",
        "rang_percentile",
        "rang",
        "ticker",
        "date_cours",
        "cell_id",
        "cell_name",
    ]
    extra = [c for c in ("split",) if c in out.columns]
    return out[base_cols + extra]


def run_factorial_stage1(
    df: pd.DataFrame,
    *,
    target_col: str = TARGET_COLUMN,
    top_n: int = TOP_N,
    rf_params: RFHyperParams | None = None,
) -> dict[str, Any]:
    """
    Entraîne / score les 4 cellules, calcule IC/hit val+test, sauve top-n mensuel.
    """
    if rf_params is None:
        rf_params = RFHyperParams(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=20,
            random_state=RANDOM_STATE,
        )

    # Cache modèles par flag régime (évite de réentraîner 2× le même jeu de features)
    model_cache: dict[bool, dict[str, Any]] = {}
    clip_bounds: dict[str, float] | None = None
    cell_results: dict[int, dict[str, Any]] = {}
    all_preds: list[pd.DataFrame] = []
    all_selections: list[pd.DataFrame] = []

    for cell in CELLS:
        logger.info("=== %s ===", cell.label)
        if cell.regime not in model_cache:
            feat_cols = feature_columns_for_cell(df, regime=cell.regime)
            splits, feat_cols, clip_bounds = prepare_splits(
                df, feature_cols=feat_cols, target_col=target_col
            )
            need_rf = any(c.regime == cell.regime and c.model == "hybrid" for c in CELLS)
            # Always train RF for both régime flags (cells 3 and 4)
            model_cache[cell.regime] = {
                **train_base_models(
                    splits,
                    feat_cols,
                    target_col=target_col,
                    rf_params=rf_params,
                    need_rf=True,
                ),
                "splits": splits,
            }
            logger.info(
                "Modèles entraînés (regime=%s) | features=%s | train=%s",
                cell.regime,
                len(feat_cols),
                len(splits["train"]),
            )

        bundle = model_cache[cell.regime]
        splits = bundle["splits"]
        cell_metrics: dict[str, Any] = {
            "cell_id": cell.cell_id,
            "name": cell.name,
            "model": cell.model,
            "regime": cell.regime,
            "n_features": len(bundle["feature_columns"]),
            "splits": {},
        }

        for split_name in ("val", "test"):
            split_df = splits[split_name]
            if split_df.empty:
                continue
            pred = predict_cell(bundle, split_df, cell, target_col=target_col)
            pred["split"] = split_name
            metrics = evaluate_scoring_model(pred, "prediction", target_col)
            cell_metrics["splits"][split_name] = {
                "rows": len(pred),
                "metrics": metrics,
            }
            logger.info(
                "[C%s/%s] Rank-IC=%.4f | Hit=%.1f%%",
                cell.cell_id,
                split_name,
                metrics.get("rank_ic_mean", np.nan),
                (metrics.get("hit_ratio", np.nan) or 0) * 100,
            )
            all_preds.append(pred)

            sel = select_top_n_monthly(pred, n=top_n)
            sel["split"] = split_name
            # Vérif cardinalité
            counts = sel.groupby("mois").size()
            cell_metrics["splits"][split_name]["top_n_months"] = int(counts.shape[0])
            cell_metrics["splits"][split_name]["top_n_ok"] = bool(
                (counts == top_n).all()
            ) if len(counts) else False
            cell_metrics["splits"][split_name]["top_n_min"] = (
                int(counts.min()) if len(counts) else 0
            )
            cell_metrics["splits"][split_name]["top_n_max"] = (
                int(counts.max()) if len(counts) else 0
            )
            all_selections.append(sel)

        cell_results[cell.cell_id] = cell_metrics

    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    selections = (
        pd.concat(all_selections, ignore_index=True) if all_selections else pd.DataFrame()
    )

    return {
        "cells": cell_results,
        "predictions": predictions,
        "selections": selections,
        "clip_bounds": clip_bounds,
        "rf_params": {
            "n_estimators": rf_params.n_estimators,
            "max_depth": rf_params.max_depth,
            "min_samples_leaf": rf_params.min_samples_leaf,
            "random_state": rf_params.random_state,
        },
        "top_n": top_n,
        "model_cache": model_cache,
    }


def save_stage1_artifacts(
    results: dict[str, Any],
    *,
    reports_dir: Path,
    selections_dir: Path,
    models_dir: Path,
) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    selections_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    preds_path = reports_dir / "factorial_predictions.parquet"
    report_path = reports_dir / "factorial_stage1_report.json"
    summary_path = reports_dir / "factorial_stage1_summary.csv"

    results["predictions"].to_parquet(preds_path, index=False)

    # Sélections par cellule
    sel_paths: dict[str, Path] = {}
    selections: pd.DataFrame = results["selections"]
    for cell_id, grp in selections.groupby("cell_id"):
        p = selections_dir / f"cell{cell_id}_top{results['top_n']}.parquet"
        grp.to_parquet(p, index=False)
        # CSV lisible aussi
        csv_p = selections_dir / f"cell{cell_id}_top{results['top_n']}.csv"
        grp.to_csv(csv_p, index=False)
        sel_paths[f"cell{cell_id}"] = p

    # Résumé métriques
    rows = []
    for cell_id, data in results["cells"].items():
        row = {
            "cellule": cell_id,
            "modele": data["model"],
            "regime": data["regime"],
            "name": data["name"],
        }
        for split_name in ("val", "test"):
            m = data["splits"].get(split_name, {}).get("metrics", {})
            row[f"IC_{split_name}"] = m.get("rank_ic_mean")
            row[f"IC_pearson_{split_name}"] = m.get("ic_mean")
            row[f"hit_ratio_{split_name}"] = m.get("hit_ratio")
            row[f"top25_ok_{split_name}"] = data["splits"].get(split_name, {}).get("top_n_ok")
            row[f"n_months_{split_name}"] = data["splits"].get(split_name, {}).get("top_n_months")
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("cellule")
    summary.to_csv(summary_path, index=False)

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

    serializable = {
        "rf_params": results["rf_params"],
        "top_n": results["top_n"],
        "clip_bounds": results["clip_bounds"],
        "cells": results["cells"],
    }
    report_path.write_text(
        json.dumps(_json_safe(serializable), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Persister les 2 bundles (regime True/False)
    for regime_flag, bundle in results["model_cache"].items():
        tag = "with_regime" if regime_flag else "no_regime"
        joblib.dump(
            {
                "ridge": bundle["ridge"],
                "rf": bundle["rf"],
                "feature_columns": bundle["feature_columns"],
                "rf_params": results["rf_params"],
                "regime": regime_flag,
            },
            models_dir / f"factorial_models_{tag}.joblib",
        )

    return {
        "predictions": preds_path,
        "report": report_path,
        "summary": summary_path,
        **sel_paths,
    }
