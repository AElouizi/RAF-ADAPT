"""
Archive / ablation — ancien détecteur GaussianHMM (3 états).

La production utilise désormais la règle à seuils
(``bvc_recommender.models.regime_detector.ThresholdRegimeDetector``).
Ce module conserve l'implémentation HMM pour comparaison éventuelle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from bvc_recommender.config import RANDOM_STATE, REGIME_START_YEAR, SPLIT_TRAIN_END
from bvc_recommender.models.regime_detector import (
    REBALANCE_FLAG_COLUMN,
    enrich_market_context_with_regime,
)

logger = logging.getLogger(__name__)

# Colonnes historiques HMM (is_sideways = neutral)
HMM_COLUMNS = ["is_bull", "is_sideways", "is_bear"]
HMM_INPUTS = ["masi_mom_3m", "return_dispersion", "breadth_ma50"]
REGIME_COLUMNS = HMM_COLUMNS  # alias historique
REGIME_INPUTS = HMM_INPUTS


@dataclass
class GaussianHMMRegimeDetector:
    """GaussianHMM à 3 états — archivé, non utilisé en production."""

    n_components: int = 3
    n_iter: int = 200
    random_state: int = RANDOM_STATE
    train_end: str = SPLIT_TRAIN_END
    train_start_year: int = REGIME_START_YEAR
    min_covar: float = 1e-3
    _model: Any = field(default=None, repr=False, compare=False)
    _scaler: Any = field(default=None, repr=False, compare=False)
    _state_to_label: dict[int, str] = field(default_factory=dict, repr=False, compare=False)

    def fit(self, market_context: pd.DataFrame) -> GaussianHMMRegimeDetector:
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler

        df = self._prepare_inputs(market_context)
        train_end = pd.Timestamp(self.train_end)
        train = df[
            (df["date_cours"].dt.year >= self.train_start_year)
            & (df["date_cours"] <= train_end)
        ]
        if len(train) < 30:
            raise ValueError(f"Trop peu d'obs train HMM (n={len(train)})")

        X_train_raw = train[list(HMM_INPUTS)].to_numpy(dtype=float)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        model = GaussianHMM(
            n_components=self.n_components,
            covariance_type="diag",
            n_iter=self.n_iter,
            random_state=self.random_state,
            min_covar=self.min_covar,
        )
        model.fit(X_train)
        states_train = model.predict(X_train)
        mom = train["masi_mom_3m"].to_numpy(dtype=float)
        state_mom = {
            s: float(mom[states_train == s].mean()) if np.any(states_train == s) else 0.0
            for s in range(self.n_components)
        }
        ordered = sorted(state_mom, key=state_mom.get)
        self._state_to_label = {
            ordered[0]: "bear",
            ordered[1]: "sideways",
            ordered[2]: "bull",
        }
        self._scaler = scaler
        self._model = model
        return self

    def predict_onehot(self, market_context: pd.DataFrame) -> pd.DataFrame:
        if self._model is None or self._scaler is None:
            raise RuntimeError("GaussianHMMRegimeDetector.fit() requis.")
        df = self._prepare_inputs(market_context)
        X = self._scaler.transform(df[list(HMM_INPUTS)].to_numpy(dtype=float))
        states = self._model.predict(X)
        out = pd.DataFrame({"date_cours": df["date_cours"].values})
        for col in HMM_COLUMNS:
            out[col] = 0
        label_to_col = {"bull": "is_bull", "sideways": "is_sideways", "bear": "is_bear"}
        for i, s in enumerate(states):
            out.loc[i, label_to_col[self._state_to_label[int(s)]]] = 1
        return out

    @staticmethod
    def _prepare_inputs(market_context: pd.DataFrame) -> pd.DataFrame:
        df = market_context.copy()
        df["date_cours"] = pd.to_datetime(df["date_cours"], errors="coerce")
        for c in HMM_INPUTS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["date_cours", *HMM_INPUTS]).sort_values("date_cours")
        return df.drop_duplicates("date_cours", keep="last").reset_index(drop=True)


def fit_hmm_regime_panel(market_context: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Panel journalier one-hot HMM (archive). Pour la production, utiliser
    ``enrich_market_context_with_regime`` (seuils).
    """
    from bvc_recommender.config import FEATURES_DIR

    if market_context is None or market_context.empty:
        path = FEATURES_DIR / "features_indices.parquet"
        if not path.is_file():
            raise FileNotFoundError("features_indices.parquet manquant pour HMM")
        market_context = pd.read_parquet(path)

    det = GaussianHMMRegimeDetector()
    det.fit(market_context)
    onehot = det.predict_onehot(market_context)
    return onehot


def fit_production_regime_panel(market_context: pd.DataFrame | None = None) -> pd.DataFrame:
    """Délègue au détecteur de production (seuils)."""
    from bvc_recommender.config import FEATURES_DIR
    from bvc_recommender.models.regime_detector import REGIME_COLUMNS as PROD_COLS

    if market_context is None or market_context.empty:
        path = FEATURES_DIR / "features_indices.parquet"
        if not path.is_file():
            raise FileNotFoundError("features_indices.parquet manquant")
        market_context = pd.read_parquet(path)

    enriched = enrich_market_context_with_regime(market_context)
    cols = ["date_cours", *PROD_COLS]
    if REBALANCE_FLAG_COLUMN in enriched.columns:
        cols.append(REBALANCE_FLAG_COLUMN)
    return enriched[cols].copy()
