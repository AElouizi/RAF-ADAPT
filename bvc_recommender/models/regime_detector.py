"""
Sous-module A — Détection du régime de marché par règle à seuils (3 états).

Remplace le GaussianHMM (bull trop rare ~3–5 % des mois). Règle explicite
calibrée UNIQUEMENT sur le train 2015–2020 (EOM mensuel), appliquée telle
quelle sur validation / test (pas de look-ahead).

Entrées : masi_mom_3m, breadth_ma50.
Sortie journalière one-hot : [is_bull, is_neutral, is_bear]
(+ is_rebalancing_date). Régime calculé à chaque fin de mois, propagé
à tous les jours de bourse du mois.

Seuils documentés (article) — train EOM 2015-2020, quantiles 0.33 / 0.67 :
  p33_momentum = -0.016704
  p67_momentum =  0.042896
  p33_breadth  =  0.392857
  p67_breadth  =  0.596491
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from bvc_recommender.config import REGIME_START_YEAR, SPLIT_TRAIN_END

logger = logging.getLogger(__name__)

REGIME_COLUMNS = ["is_bull", "is_neutral", "is_bear"]
# Alias rétrocompat (anciens rapports / colonnes HMM)
LEGACY_NEUTRAL_COLUMN = "is_sideways"
REGIME_INPUTS = ["masi_mom_3m", "breadth_ma50"]
REBALANCE_FLAG_COLUMN = "is_rebalancing_date"

# Alias pour un éventuel retour à un conditionnement continu (modularité Composante B).
CONTINUOUS_REGIME_COLUMNS = ["mu_bull", "mu_sideways", "mu_bear"]

# Quantiles et seuils retenus — traçabilité méthodologique article
THRESHOLD_Q_LOW = 0.33
THRESHOLD_Q_HIGH = 0.67
DOCUMENTED_THRESHOLDS: dict[str, float] = {
    "p33_momentum": -0.016704,
    "p67_momentum": 0.042896,
    "p33_breadth": 0.392857,
    "p67_breadth": 0.596491,
}


@dataclass
class ThresholdRegimeDetector:
    """
    Règle à seuils both-AND :
      is_bull    = (mom > p67_mom) & (breadth > p67_br)
      is_bear    = (mom < p33_mom) & (breadth < p33_br)
      is_neutral = ~is_bull & ~is_bear
    """

    train_end: str = SPLIT_TRAIN_END
    train_start_year: int = REGIME_START_YEAR
    q_low: float = THRESHOLD_Q_LOW
    q_high: float = THRESHOLD_Q_HIGH
    # True = seuils documentés article (reproductibilité) ; False = recalcul quantile
    use_documented_thresholds: bool = True
    _thresholds: dict[str, float] = field(default_factory=dict, repr=False, compare=False)
    _fitted: bool = field(default=False, repr=False, compare=False)

    def fit(self, market_context: pd.DataFrame) -> ThresholdRegimeDetector:
        """Calibre les seuils sur EOM mensuels train uniquement (ou charge les documentés)."""
        monthly = _monthly_eom(market_context)
        train_end = pd.Timestamp(self.train_end)
        train = monthly[
            (monthly["year"] >= self.train_start_year)
            & (monthly["date_cours"] <= train_end)
        ]
        if len(train) < 12:
            raise ValueError(
                f"Trop peu d'observations train EOM pour seuils (n={len(train)}, "
                f"train_end={self.train_end})."
            )

        computed = {
            "p33_momentum": float(train["masi_mom_3m"].quantile(self.q_low)),
            "p67_momentum": float(train["masi_mom_3m"].quantile(self.q_high)),
            "p33_breadth": float(train["breadth_ma50"].quantile(self.q_low)),
            "p67_breadth": float(train["breadth_ma50"].quantile(self.q_high)),
        }

        if self.use_documented_thresholds:
            self._thresholds = dict(DOCUMENTED_THRESHOLDS)
            logger.info(
                "Seuils DOCUMENTÉS retenus (train EOM n=%s, q=%.2f/%.2f) : %s | "
                "recalculés pour contrôle : %s",
                len(train),
                self.q_low,
                self.q_high,
                {k: round(v, 6) for k, v in self._thresholds.items()},
                {k: round(v, 6) for k, v in computed.items()},
            )
        else:
            self._thresholds = computed
            logger.info(
                "Seuils recalculés sur train EOM n=%s (q=%.2f/%.2f) : %s",
                len(train),
                self.q_low,
                self.q_high,
                {k: round(v, 6) for k, v in self._thresholds.items()},
            )

        self._fitted = True
        return self

    def predict_onehot(self, market_context: pd.DataFrame) -> pd.DataFrame:
        """
        Applique la règle sur les EOM mensuels, puis propage le one-hot
        à tous les jours de bourse de chaque mois.
        """
        if not self._fitted or not self._thresholds:
            raise RuntimeError("ThresholdRegimeDetector.fit() doit être appelé avant predict.")

        daily = self._prepare_daily(market_context)
        monthly = _monthly_eom(daily)
        thr = self._thresholds
        mom = monthly["masi_mom_3m"]
        br = monthly["breadth_ma50"]

        is_bull = (mom > thr["p67_momentum"]) & (br > thr["p67_breadth"])
        is_bear = (mom < thr["p33_momentum"]) & (br < thr["p33_breadth"])
        is_neutral = ~is_bull & ~is_bear

        monthly = monthly.copy()
        monthly["is_bull"] = is_bull.astype(int)
        monthly["is_neutral"] = is_neutral.astype(int)
        monthly["is_bear"] = is_bear.astype(int)
        monthly["_ym"] = monthly["date_cours"].dt.to_period("M")

        daily = daily.copy()
        daily["_ym"] = daily["date_cours"].dt.to_period("M")
        out = daily[["date_cours", "_ym"]].merge(
            monthly[["_ym", *REGIME_COLUMNS]],
            on="_ym",
            how="left",
        )
        out = out.drop(columns=["_ym"])
        for col in REGIME_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
        missing = out[REGIME_COLUMNS].sum(axis=1) <= 0
        out.loc[missing, "is_neutral"] = 1
        out[REBALANCE_FLAG_COLUMN] = _mark_rebalancing_dates(out["date_cours"])
        return out

    @property
    def thresholds(self) -> dict[str, float]:
        return dict(self._thresholds)

    @staticmethod
    def _prepare_daily(market_context: pd.DataFrame) -> pd.DataFrame:
        if market_context is None or market_context.empty:
            raise ValueError("market_context vide pour régime à seuils")
        df = market_context.copy()
        df["date_cours"] = pd.to_datetime(df["date_cours"], errors="coerce")
        for c in REGIME_INPUTS:
            if c not in df.columns:
                raise ValueError(f"Colonne manquante pour régime à seuils : {c}")
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["date_cours"]).sort_values("date_cours")
        df = df.drop_duplicates("date_cours", keep="last").reset_index(drop=True)
        return df


# Alias public / rétrocompat API
RegimeDetector = ThresholdRegimeDetector
# Ancien nom conservé pour imports historiques (délègue désormais aux seuils)
HMMRegimeDetector = ThresholdRegimeDetector


def _monthly_eom(market_context: pd.DataFrame) -> pd.DataFrame:
    """Dernier jour de bourse de chaque mois avec entrées régime non nulles."""
    df = market_context.copy()
    df["date_cours"] = pd.to_datetime(df["date_cours"], errors="coerce")
    for c in REGIME_INPUTS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    needed = ["date_cours", *REGIME_INPUTS]
    df = df.dropna(subset=[c for c in needed if c in df.columns]).sort_values("date_cours")
    df["year"] = df["date_cours"].dt.year
    df["month"] = df["date_cours"].dt.month
    eom = (
        df.groupby(["year", "month"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return eom


def _mark_rebalancing_dates(dates: pd.Series) -> pd.Series:
    """True sur le dernier jour de bourse de chaque mois calendaire."""
    ts = pd.to_datetime(dates, errors="coerce")
    tmp = pd.DataFrame({"date_cours": ts, "_ym": ts.dt.to_period("M")})
    last = tmp.groupby("_ym", sort=False)["date_cours"].transform("max")
    return (tmp["date_cours"] == last).astype(bool)


def enrich_market_context_with_regime(
    market_context: pd.DataFrame,
    *,
    train_end: str | None = None,
    detector: ThresholdRegimeDetector | None = None,
) -> pd.DataFrame:
    """
    Ajoute is_bull / is_neutral / is_bear / is_rebalancing_date au panel
    features_indices (source de vérité quotidienne).
    """
    det = detector or ThresholdRegimeDetector(train_end=train_end or SPLIT_TRAIN_END)
    if not det._fitted:
        det.fit(market_context)
    onehot = det.predict_onehot(market_context)

    out = market_context.copy()
    out["date_cours"] = pd.to_datetime(out["date_cours"], errors="coerce")
    # Retirer ancien encodage (HMM ou seuils) avant merge
    drop_cols = [
        c
        for c in [*REGIME_COLUMNS, LEGACY_NEUTRAL_COLUMN, REBALANCE_FLAG_COLUMN]
        if c in out.columns
    ]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    out = out.merge(onehot, on="date_cours", how="left")

    for col in REGIME_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    missing = out[REGIME_COLUMNS].sum(axis=1) <= 0
    out.loc[missing, "is_neutral"] = 1
    # Miroir rétrocompat colonnes Supabase / dashboards encore sur is_sideways
    out[LEGACY_NEUTRAL_COLUMN] = out["is_neutral"]

    if REBALANCE_FLAG_COLUMN in out.columns:
        flag = out[REBALANCE_FLAG_COLUMN]
        out[REBALANCE_FLAG_COLUMN] = flag.where(flag.notna(), False).astype(bool)
    else:
        out[REBALANCE_FLAG_COLUMN] = _mark_rebalancing_dates(out["date_cours"])

    logger.info(
        "features_indices enrichi (seuils) : %s j | bull=%s neutral=%s bear=%s | rebal=%s | thr=%s",
        len(out),
        int(out["is_bull"].sum()),
        int(out["is_neutral"].sum()),
        int(out["is_bear"].sum()),
        int(out[REBALANCE_FLAG_COLUMN].sum()),
        {k: round(v, 6) for k, v in det.thresholds.items()},
    )
    return out


def detect_regime_daily(
    market_context: pd.DataFrame,
    *,
    start_year: int = 2015,
    end_year: int = 2025,
    train_end: str | None = None,
    detector: ThresholdRegimeDetector | None = None,
) -> pd.DataFrame:
    """Panel journalier filtré [start_year, end_year] avec one-hot à seuils."""
    enriched = enrich_market_context_with_regime(
        market_context, train_end=train_end, detector=detector
    )
    df = enriched.copy()
    df["date_cours"] = pd.to_datetime(df["date_cours"], errors="coerce")
    df = df.dropna(subset=["date_cours"])
    df = df[
        (df["date_cours"].dt.year >= start_year) & (df["date_cours"].dt.year <= end_year)
    ]
    return df.sort_values("date_cours").reset_index(drop=True)


def aggregate_monthly_context(
    market_context: pd.DataFrame,
    *,
    start_year: int = 2015,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Dernier jour de bourse de chaque mois (entrées + one-hot si présents)."""
    if market_context.empty:
        return pd.DataFrame()

    df = market_context.copy()
    df["date_cours"] = pd.to_datetime(df["date_cours"], errors="coerce")
    df = df.dropna(subset=["date_cours"]).sort_values("date_cours")
    # Normaliser is_sideways legacy → is_neutral
    if "is_neutral" not in df.columns and LEGACY_NEUTRAL_COLUMN in df.columns:
        df["is_neutral"] = df[LEGACY_NEUTRAL_COLUMN]
    df["year"] = df["date_cours"].dt.year
    df["month"] = df["date_cours"].dt.month
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    rows: list[dict[str, Any]] = []
    for (year, month), grp in df.groupby(["year", "month"], sort=True):
        last = grp.sort_values("date_cours").iloc[-1]
        row: dict[str, Any] = {
            "year": int(year),
            "month": int(month),
            "month_label": f"{year}-{month:02d}",
            "as_of_date": last["date_cours"],
        }
        for c in REGIME_INPUTS:
            if c in last.index:
                row[c] = last.get(c)
        for c in REGIME_COLUMNS:
            if c in last.index:
                row[c] = int(last.get(c) or 0)
        if REBALANCE_FLAG_COLUMN in last.index:
            row[REBALANCE_FLAG_COLUMN] = bool(last.get(REBALANCE_FLAG_COLUMN))
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)
    logger.info(
        "Contexte mensuel (seuils) : %s mois (%s → %s)",
        len(out),
        out["month_label"].min() if not out.empty else "—",
        out["month_label"].max() if not out.empty else "—",
    )
    return out


def detect_regime_history(
    market_context: pd.DataFrame,
    *,
    start_year: int = 2015,
    end_year: int = 2025,
    train_end: str | None = None,
    detector: ThresholdRegimeDetector | None = None,
) -> pd.DataFrame:
    """
    Historique mensuel (dernier jour de chaque mois) pour rapports / dashboard.
    Le panel quotidien enrichi reste la source de vérité (features_indices).
    """
    daily = detect_regime_daily(
        market_context,
        start_year=start_year,
        end_year=end_year,
        train_end=train_end,
        detector=detector,
    )
    if daily.empty:
        return pd.DataFrame()
    return aggregate_monthly_context(daily, start_year=start_year, end_year=end_year)


def dominant_regime(row: pd.Series) -> str:
    """Libellé économique à partir du one-hot (ou argmax si plusieurs)."""
    cols = {c: float(row.get(c, 0) or 0) for c in REGIME_COLUMNS}
    if sum(cols.values()) <= 0 and LEGACY_NEUTRAL_COLUMN in row.index:
        cols = {
            "is_bull": float(row.get("is_bull", 0) or 0),
            "is_neutral": float(row.get(LEGACY_NEUTRAL_COLUMN, 0) or 0),
            "is_bear": float(row.get("is_bear", 0) or 0),
        }
    if sum(cols.values()) <= 0:
        return "unknown"
    winner = max(cols, key=cols.get)
    return winner.replace("is_", "")
