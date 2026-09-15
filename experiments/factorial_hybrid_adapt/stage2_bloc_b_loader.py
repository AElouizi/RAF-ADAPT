"""
Interface Bloc B → Bloc D : chargement scores par ``model_id`` + fold.

Convertit les prédictions walk-forward Bloc B en inputs NSGA-III
(recommandations BUY/NEUTRAL/SELL + risque + liquidité).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.factorial_hybrid_adapt.bloc_b_models import (
    LEGACY_MODEL_IDS,
    MODEL_GROUP,
    MODEL_IDS,
    MODEL_SPECS,
)
from experiments.factorial_hybrid_adapt.recommendations import (
    RecommendationConfig,
    generate_recommendations,
    month_end_mask,
)
from experiments.factorial_hybrid_adapt.stage2_interface import (
    STAGE2_CORE_COLS,
    build_stage2_inputs,
)
from experiments.factorial_hybrid_adapt.walk_forward_folds import Fold

logger = logging.getLogger(__name__)

# 5 stratégies C1–C5 : même liste pour sélection (Bloc B) et allocation (Bloc D)
BLOC_D_MODEL_IDS: tuple[str, ...] = MODEL_IDS

DEFAULT_PREDS_DIR = (
    Path(__file__).resolve().parent / "outputs" / "bloc_b_wf_overlapping"
)


def load_bloc_b_predictions(
    model_id: str,
    fold_id: int,
    *,
    preds_dir: Path | None = None,
) -> pd.DataFrame:
    """Charge les scores test d'un modèle pour un pli WF depuis ``fold_XXXX_preds.parquet``."""
    if model_id not in MODEL_SPECS and model_id not in BLOC_D_MODEL_IDS and model_id not in LEGACY_MODEL_IDS:
        raise ValueError(f"model_id inconnu : {model_id}")
    root = preds_dir or DEFAULT_PREDS_DIR
    part = root / f"fold_{fold_id:04d}_preds.parquet"
    if not part.is_file():
        raise FileNotFoundError(f"Prédictions introuvables : {part}")
    df = pd.read_parquet(part)
    if "model_id" not in df.columns:
        raise ValueError(f"Colonne model_id absente dans {part}")
    sub = df[df["model_id"] == model_id].copy()
    if sub.empty:
        raise ValueError(f"Aucune ligne model_id={model_id} dans {part}")
    return sub


def load_all_fold_predictions(
    model_id: str,
    fold_ids: list[int] | None = None,
    *,
    preds_dir: Path | None = None,
) -> pd.DataFrame:
    """Concatène les prédictions test d'un modèle sur plusieurs plis (pour tau expanding)."""
    root = preds_dir or DEFAULT_PREDS_DIR
    if fold_ids is None:
        files = sorted(root.glob("fold_*_preds.parquet"))
    else:
        files = [root / f"fold_{fid:04d}_preds.parquet" for fid in fold_ids]
    frames: list[pd.DataFrame] = []
    for f in files:
        if not f.is_file():
            continue
        df = pd.read_parquet(f)
        if "model_id" not in df.columns:
            continue
        m = df[df["model_id"] == model_id].copy()
        if not m.empty:
            frames.append(m)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _tau_reference_preds(
    all_preds: pd.DataFrame,
    model_id: str,
    fold: Fold,
) -> pd.Series:
    """Prédictions mois-fin des plis antérieurs (même modèle) pour calibrer tau."""
    if all_preds.empty:
        return pd.Series(dtype=float)
    tmp = all_preds.copy()
    tmp["date_cours"] = pd.to_datetime(tmp["date_cours"], errors="coerce")
    tmp["test_end"] = pd.to_datetime(tmp.get("test_end", tmp["date_cours"]), errors="coerce")
    t_start = pd.Timestamp(fold.test_start)
    me = tmp.loc[
        month_end_mask(tmp["date_cours"]) & (tmp["test_end"] < t_start),
        "prediction",
    ]
    return pd.to_numeric(me, errors="coerce").dropna()


def predictions_to_recommendations(
    preds: pd.DataFrame,
    *,
    model_id: str,
    fold: Fold | None = None,
    reference_preds: pd.Series | None = None,
    cfg: RecommendationConfig | None = None,
) -> pd.DataFrame:
    """Scores continus → BUY/NEUTRAL/SELL (tau causal, pas de look-ahead)."""
    cfg = cfg or RecommendationConfig(
        benchmark_mode="zero_alpha",
        tau_method="expanding_past_quantile",
        tau_quantile=0.33,
        tau_fixed=0.10,
    )
    df = preds.copy()
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    ref = reference_preds
    if ref is None and fold is not None and cfg.tau_method == "expanding_past_quantile":
        ref = pd.Series(dtype=float)
    recos = generate_recommendations(
        df,
        train_predictions_for_tau=ref,
        cfg=cfg,
        rebalance_only=True,
        pred_col="prediction",
        date_col="date_cours",
        target_col="alpha_ajuste_risque"
        if "alpha_ajuste_risque" in df.columns
        else "prediction",
    )
    recos["model_id"] = model_id
    recos["groupe"] = MODEL_GROUP.get(model_id, "unknown")
    recos["fold_id"] = int(df["fold_id"].iloc[0]) if "fold_id" in df.columns else None
    if fold is not None:
        recos["test_start_month"] = fold.test_start_month
        recos["test_end_month"] = fold.test_end_month
    return recos


def recommendations_to_stage2_inputs(
    recos: pd.DataFrame,
    model_id: str,
) -> pd.DataFrame:
    """Panel enrichi risque/liquidité prêt pour NSGA-III."""
    out = build_stage2_inputs(recos)
    out["model_id"] = model_id
    out["groupe"] = MODEL_GROUP.get(model_id, "unknown")
    if "fold_id" in recos.columns:
        out["fold_id"] = recos["fold_id"].iloc[0]
    # Compat colonnes stage2 (cell laissé NaN — runs modèle utilisent model_id)
    out["cell"] = pd.NA
    for c in STAGE2_CORE_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    return out


def build_stage2_inputs_for_model_fold(
    model_id: str,
    fold: Fold,
    *,
    preds_dir: Path | None = None,
    all_preds_for_tau: pd.DataFrame | None = None,
    cfg: RecommendationConfig | None = None,
) -> pd.DataFrame:
    """
    Pipeline complet : preds Bloc B → recos → stage2 inputs pour un (model, bloc).
    """
    preds = load_bloc_b_predictions(model_id, fold.fold_id, preds_dir=preds_dir)
    if all_preds_for_tau is None:
        all_preds_for_tau = load_all_fold_predictions(model_id, preds_dir=preds_dir)
    ref = _tau_reference_preds(all_preds_for_tau, model_id, fold)
    recos = predictions_to_recommendations(
        preds,
        model_id=model_id,
        fold=fold,
        reference_preds=ref,
        cfg=cfg,
    )
    stage2 = recommendations_to_stage2_inputs(recos, model_id)
    stage2["fold_id"] = fold.fold_id
    stage2["test_start_month"] = fold.test_start_month
    stage2["test_end_month"] = fold.test_end_month
    logger.info(
        "[BlocD loader] model=%s fold=%s | recos=%s | dates=%s",
        model_id,
        fold.fold_id,
        len(recos),
        stage2["date"].nunique(),
    )
    return stage2


def block_fold_ids_from_summary(summary_path: Path | None = None) -> list[int]:
    """14 blocs indépendants (ids plis overlapping)."""
    import json

    path = summary_path or (
        Path(__file__).resolve().parent
        / "outputs"
        / "reports"
        / "bloc_b_summary_all.json"
    )
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        ids = data.get("analysis", {}).get("block_fold_ids")
        if ids:
            return [int(x) for x in ids]
    # fallback : step 6 mois
    return [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78]
