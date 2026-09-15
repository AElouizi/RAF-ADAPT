"""Chargement dataset ML + jointure régime (features_indices), hors TFT."""

from __future__ import annotations

import logging

import pandas as pd

from bvc_recommender.config import DATASET_DIR, FEATURES_DIR
from bvc_recommender.models.regime_detector import REGIME_COLUMNS

logger = logging.getLogger(__name__)


def load_ml_dataset() -> pd.DataFrame:
    path = DATASET_DIR / "ml_dataset.parquet"
    if not path.is_file():
        path = DATASET_DIR / "ml_dataset.csv"
    if not path.is_file():
        raise FileNotFoundError(f"ml_dataset introuvable sous {DATASET_DIR}")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["date_cours"] = pd.to_datetime(df["date_cours"], errors="coerce")
    return df


def load_regime_daily() -> pd.DataFrame:
    """One-hot journalier depuis features_indices (sortie step4, non modifié)."""
    path = FEATURES_DIR / "features_indices.parquet"
    if not path.is_file():
        path = FEATURES_DIR / "features_indices.csv"
    if not path.is_file():
        raise FileNotFoundError(f"features_indices introuvable sous {FEATURES_DIR}")
    cols = ["date_cours", *REGIME_COLUMNS]
    fi = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = [c for c in cols if c not in fi.columns]
    if missing:
        raise ValueError(f"Colonnes régime manquantes dans features_indices : {missing}")
    regime = fi[cols].copy()
    regime["date_cours"] = pd.to_datetime(regime["date_cours"], errors="coerce")
    for c in REGIME_COLUMNS:
        regime[c] = pd.to_numeric(regime[c], errors="coerce").fillna(0).astype(float)
    regime = (
        regime.dropna(subset=["date_cours"])
        .sort_values("date_cours")
        .drop_duplicates("date_cours", keep="last")
    )
    return regime


def attach_regime_to_panel(df: pd.DataFrame, regime: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Joint is_bull / is_neutral / is_bear au panel titres (merge exact sur date_cours).
    Valeurs manquantes → neutre (0, 1, 0).
    """
    regime = load_regime_daily() if regime is None else regime
    out = df.copy()
    out["date_cours"] = pd.to_datetime(out["date_cours"], errors="coerce")
    # Drop éventuelles colonnes déjà présentes pour éviter suffixes
    out = out.drop(columns=[c for c in REGIME_COLUMNS if c in out.columns], errors="ignore")
    out = out.merge(regime, on="date_cours", how="left")
    missing = out[REGIME_COLUMNS[0]].isna().sum()
    out["is_bull"] = out["is_bull"].fillna(0.0)
    out["is_neutral"] = out["is_neutral"].fillna(1.0)
    out["is_bear"] = out["is_bear"].fillna(0.0)
    if missing:
        logger.warning(
            "Régime manquant pour %s lignes — rempli en is_neutral=1",
            int(missing),
        )
    return out
