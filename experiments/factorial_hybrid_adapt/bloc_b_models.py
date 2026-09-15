"""
Bloc B — 5 stratégies de sélection (C1–C5).

C1 : Ridge
C2 : Random Forest
C3 : LightGBM
C4 : Ridge + RF + LightGBM (z-score CS, poids égaux)
C5 : Ridge + RF + LightGBM + régime Bloc A

Tuning : RF grille interne + Optuna LightGBM (config gelée).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from bvc_recommender.config import RANDOM_STATE
from bvc_recommender.features.dataset_builder import TARGET_COLUMN
from bvc_recommender.models.metrics import evaluate_scoring_model
from experiments.factorial_hybrid_adapt.factorial_cells import (
    HYBRID_WEIGHT_RF,
    HYBRID_WEIGHT_RIDGE,
    combine_hybrid,
    combine_hybrid_tri,
    feature_columns_for_cell,
)
from experiments.factorial_hybrid_adapt.random_forest_scorer import (
    RFHyperParams,
    fit_random_forest,
)
from experiments.factorial_hybrid_adapt.walk_forward_stage1 import DEFAULT_RF_PARAMS

logger = logging.getLogger(__name__)

ModelId = Literal[
    "c1_ridge",
    "c2_rf",
    "c3_lightgbm",
    "c4_hybrid_tri",
    "c5_hybrid_tri_regime",
]

# Stratégies actives Bloc B (sélection) et Bloc D (allocation)
MODEL_IDS: tuple[ModelId, ...] = (
    "c1_ridge",
    "c2_rf",
    "c3_lightgbm",
    "c4_hybrid_tri",
    "c5_hybrid_tri_regime",
)

STRATEGY_LABELS: dict[str, str] = {
    "c1_ridge": "C1 — Ridge",
    "c2_rf": "C2 — Random Forest",
    "c3_lightgbm": "C3 — LightGBM",
    "c4_hybrid_tri": "C4 — Ridge + RF + LightGBM",
    "c5_hybrid_tri_regime": "C5 — Ridge + RF + LightGBM + régime",
}

MODEL_GROUP: dict[str, str] = {
    "c1_ridge": "C1_ridge",
    "c2_rf": "C2_rf",
    "c3_lightgbm": "C3_lightgbm",
    "c4_hybrid_tri": "C4_hybrid_tri",
    "c5_hybrid_tri_regime": "C5_hybrid_tri_regime",
}

# Anciens identifiants (artefacts historiques — lecture seule)
LEGACY_MODEL_IDS: tuple[str, ...] = (
    "ridge",
    "rf",
    "lightgbm",
    "hybrid_rf",
    "hybrid_lgbm",
    "hybrid_rf_regime",
    "hybrid_lgbm_regime",
)

LGBM_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "lightgbm_search_space.json"
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: ModelId
    groupe: str
    needs_rf: bool
    needs_lgbm: bool
    regime: bool
    hybrid_tri: bool


MODEL_SPECS: dict[str, ModelSpec] = {
    "c1_ridge": ModelSpec("c1_ridge", "C1_ridge", False, False, False, False),
    "c2_rf": ModelSpec("c2_rf", "C2_rf", True, False, False, False),
    "c3_lightgbm": ModelSpec("c3_lightgbm", "C3_lightgbm", False, True, False, False),
    "c4_hybrid_tri": ModelSpec(
        "c4_hybrid_tri", "C4_hybrid_tri", True, True, False, True
    ),
    "c5_hybrid_tri_regime": ModelSpec(
        "c5_hybrid_tri_regime", "C5_hybrid_tri_regime", True, True, True, True
    ),
}


@dataclass
class LGBMHyperParams:
    n_estimators: int = 200
    max_depth: int = 5
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = RANDOM_STATE
    verbose: int = -1

    def to_lgbm(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }


def load_lgbm_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or LGBM_CONFIG_PATH
    with cfg_path.open(encoding="utf-8") as f:
        return json.load(f)


def default_lgbm_params(cfg: dict[str, Any] | None = None) -> LGBMHyperParams:
    cfg = cfg or load_lgbm_config()
    defaults = {**cfg.get("fixed_params", {}), **cfg.get("defaults", {})}
    return LGBMHyperParams(
        n_estimators=int(defaults.get("n_estimators", 200)),
        max_depth=int(defaults.get("max_depth", 5)),
        learning_rate=float(defaults.get("learning_rate", 0.05)),
        num_leaves=int(defaults.get("num_leaves", 31)),
        min_child_samples=int(defaults.get("min_child_samples", 20)),
        subsample=float(defaults.get("subsample", 0.8)),
        colsample_bytree=float(defaults.get("colsample_bytree", 0.8)),
        random_state=RANDOM_STATE,
        verbose=int(defaults.get("verbose", -1)),
    )


def is_known_model_id(model_id: str) -> bool:
    return model_id in MODEL_SPECS or model_id in LEGACY_MODEL_IDS


def build_model(model_type: str, hyperparams: dict | None = None) -> Any:
    if model_type not in MODEL_SPECS:
        raise ValueError(
            f"model_type inconnu : {model_type}. Attendu l'un de {list(MODEL_IDS)}"
        )
    hp = dict(hyperparams or {})
    spec = MODEL_SPECS[model_type]

    if model_type == "c1_ridge":
        return Ridge(alpha=float(hp.get("alpha", 1.0)), random_state=RANDOM_STATE)

    if model_type == "c2_rf":
        params = RFHyperParams(
            n_estimators=int(hp.get("n_estimators", DEFAULT_RF_PARAMS.n_estimators)),
            max_depth=hp.get("max_depth", DEFAULT_RF_PARAMS.max_depth),
            min_samples_leaf=int(
                hp.get("min_samples_leaf", DEFAULT_RF_PARAMS.min_samples_leaf)
            ),
            random_state=int(hp.get("random_state", RANDOM_STATE)),
            n_jobs=int(hp.get("n_jobs", DEFAULT_RF_PARAMS.n_jobs)),
        )
        return RandomForestRegressor(**params.to_sklearn())

    if model_type == "c3_lightgbm":
        base = asdict(default_lgbm_params())
        kwargs = {**base, **{k: v for k, v in hp.items() if k in base}}
        return _make_lgbm(LGBMHyperParams(**kwargs))

    raise ValueError(f"build_model ne gère pas l'hybride triple : {model_type}")


def _make_lgbm(params: LGBMHyperParams) -> Any:
    import lightgbm as lgb

    return lgb.LGBMRegressor(**params.to_lgbm())


def fit_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: LGBMHyperParams,
) -> Any:
    model = _make_lgbm(params)
    model.fit(X_train, y_train)
    return model


def _build_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].fillna(0.0)
    y = df[target_col]
    return X, y


def _suggest_from_space(trial: Any, space: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, spec in space.items():
        t = spec["type"]
        if t == "categorical":
            out[name] = trial.suggest_categorical(name, spec["choices"])
        elif t == "int":
            out[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        elif t == "float":
            out[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        else:
            raise ValueError(f"Type d'espace Optuna inconnu : {t}")
    return out


def tune_lightgbm_optuna(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    *,
    target_col: str = TARGET_COLUMN,
    config: dict[str, Any] | None = None,
    n_trials: int | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[LGBMHyperParams, list[dict[str, Any]]]:
    import optuna
    from optuna.samplers import TPESampler

    cfg = config or load_lgbm_config()
    n_trials = int(n_trials if n_trials is not None else cfg["n_trials"])
    space = cfg["search_space"]
    fixed = cfg.get("fixed_params", {})

    X_train, y_train = _build_xy(splits["train"], feature_cols, target_col)
    X_val, _ = _build_xy(splits["val"], feature_cols, target_col)
    history: list[dict[str, Any]] = []

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        suggested = _suggest_from_space(trial, space)
        params = LGBMHyperParams(
            random_state=random_state,
            **{**fixed, **suggested},
        )
        model = fit_lightgbm(X_train, y_train, params)
        preds = model.predict(X_val)
        eval_df = splits["val"][["ticker", "date_cours", target_col]].copy()
        eval_df["prediction"] = preds
        metrics = evaluate_scoring_model(eval_df, "prediction", target_col)
        rank_ic = float(metrics.get("rank_ic_mean", np.nan))
        history.append(
            {**suggested, "rank_ic_val": None if np.isnan(rank_ic) else round(rank_ic, 6)}
        )
        if np.isnan(rank_ic):
            return -1.0
        return rank_ic

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = {**fixed, **study.best_params}
    best_params = LGBMHyperParams(random_state=random_state, **best)
    history.sort(
        key=lambda r: r["rank_ic_val"] if r["rank_ic_val"] is not None else -999,
        reverse=True,
    )
    logger.info(
        "Optuna LightGBM : best=%s | Rank-IC_val=%.4f | trials=%s",
        study.best_params,
        study.best_value,
        n_trials,
    )
    return best_params, history


def train_model_bundle(
    train_df: pd.DataFrame,
    model_id: str,
    *,
    target_col: str = TARGET_COLUMN,
    rf_params: RFHyperParams | None = None,
    lgbm_params: LGBMHyperParams | None = None,
    panel_for_features: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if model_id not in MODEL_SPECS:
        raise ValueError(f"model_id inconnu : {model_id}")
    spec = MODEL_SPECS[model_id]
    feat_src = panel_for_features if panel_for_features is not None else train_df
    feature_cols = feature_columns_for_cell(feat_src, regime=spec.regime)
    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Features manquantes ({model_id}) : {missing}")

    X, y = _build_xy(train_df, feature_cols, target_col)
    ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    ridge.fit(X, y)

    bundle: dict[str, Any] = {
        "model_id": model_id,
        "groupe": spec.groupe,
        "regime": spec.regime,
        "feature_columns": feature_cols,
        "ridge": ridge,
    }

    if spec.needs_rf:
        params = rf_params or DEFAULT_RF_PARAMS
        bundle["rf"] = fit_random_forest(X, y, params)
        bundle["rf_params"] = asdict(params)

    if spec.needs_lgbm:
        params_l = lgbm_params or default_lgbm_params()
        bundle["lgbm"] = fit_lightgbm(X, y, params_l)
        bundle["lgbm_params"] = asdict(params_l)

    return bundle


def predict_model_bundle(
    bundle: dict[str, Any],
    split_df: pd.DataFrame,
    *,
    target_col: str = TARGET_COLUMN,
) -> pd.DataFrame:
    model_id = bundle["model_id"]
    spec = MODEL_SPECS[model_id]
    feature_cols = bundle["feature_columns"]
    X, _ = _build_xy(split_df, feature_cols, target_col)
    ridge_pred = bundle["ridge"].predict(X)
    dates = split_df["date_cours"]

    if model_id == "c1_ridge":
        score = ridge_pred
    elif model_id == "c2_rf":
        score = bundle["rf"].predict(X)
    elif model_id == "c3_lightgbm":
        score = bundle["lgbm"].predict(X)
    elif model_id in ("c4_hybrid_tri", "c5_hybrid_tri_regime"):
        rf_pred = bundle["rf"].predict(X)
        lgbm_pred = bundle["lgbm"].predict(X)
        score = combine_hybrid_tri(ridge_pred, rf_pred, lgbm_pred, dates)
    else:
        raise ValueError(f"Prédiction non implémentée : {model_id}")

    out = split_df[["ticker", "date_cours", target_col]].copy()
    out["prediction"] = np.asarray(score, dtype=float)
    out["score_ridge"] = ridge_pred
    if "rf" in bundle:
        out["score_rf"] = bundle["rf"].predict(X)
    if "lgbm" in bundle:
        out["score_lgbm"] = bundle["lgbm"].predict(X)
    out["model_id"] = model_id
    out["groupe"] = spec.groupe
    out["regime"] = spec.regime
    return out


# Rétrocompat imports (combine_hybrid 2 composantes — plan factoriel legacy)
def _legacy_combine_rf(ridge_pred, rf_pred, dates):
    return combine_hybrid(
        ridge_pred, rf_pred, dates, w_linear=HYBRID_WEIGHT_RIDGE, w_nonlinear=HYBRID_WEIGHT_RF
    )
