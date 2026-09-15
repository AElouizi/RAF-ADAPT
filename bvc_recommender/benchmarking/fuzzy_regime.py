"""
Régime flou Mamdani (scikit-fuzzy) — variante d'ablation uniquement.

Ancienne Composante A de production, conservée pour le benchmark
``08_tft_fuzzy`` (TFT + régime flou continu) face au pipeline principal HMM.

Entrées : masi_mom_3m, return_dispersion, breadth_ma50.
Sortie mensuelle : [mu_bull, mu_sideways, mu_bear] normalisé (somme = 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import skfuzzy as fuzz

logger = logging.getLogger(__name__)

FUZZY_REGIME_COLUMNS = ["mu_bull", "mu_sideways", "mu_bear"]
FUZZY_REGIME_INPUTS = ["masi_mom_3m", "return_dispersion", "breadth_ma50"]


@dataclass(frozen=True)
class FuzzyRegimeDetector:
    """Système Mamdani scikit-fuzzy à neuf règles (ablation)."""

    mom_neg: tuple[float, float, float] = (-0.30, -0.30, 0.0)
    mom_neu: tuple[float, float, float] = (-0.04, 0.02, 0.08)
    mom_pos: tuple[float, float, float] = (0.02, 0.30, 0.30)
    br_low: tuple[float, float, float] = (0.0, 0.0, 0.45)
    br_med: tuple[float, float, float] = (0.35, 0.50, 0.65)
    br_high: tuple[float, float, float] = (0.55, 1.0, 1.0)
    disp_low: tuple[float, float, float] = (0.0, 0.0, 0.022)
    disp_med: tuple[float, float, float] = (0.015, 0.025, 0.040)
    disp_high: tuple[float, float, float] = (0.020, 0.06, 0.06)

    @staticmethod
    def _membership(
        value: float,
        universe: np.ndarray,
        triangle: tuple[float, float, float],
    ) -> float:
        curve = fuzz.trimf(universe, triangle)
        clipped = float(np.clip(value, universe[0], universe[-1]))
        return float(fuzz.interp_membership(universe, curve, clipped))

    def fuzzify(self, mom: float, breadth: float, dispersion: float) -> dict[str, float]:
        """Retourne les degrés d'appartenance des trois entrées."""
        mom_u = np.linspace(-0.30, 0.30, 1201)
        breadth_u = np.linspace(0.0, 1.0, 1001)
        dispersion_u = np.linspace(0.0, 0.06, 601)
        return {
            "mom_neg": self._membership(mom, mom_u, self.mom_neg),
            "mom_neu": self._membership(mom, mom_u, self.mom_neu),
            "mom_pos": self._membership(mom, mom_u, self.mom_pos),
            "br_low": self._membership(breadth, breadth_u, self.br_low),
            "br_med": self._membership(breadth, breadth_u, self.br_med),
            "br_high": self._membership(breadth, breadth_u, self.br_high),
            "disp_low": self._membership(dispersion, dispersion_u, self.disp_low),
            "disp_med": self._membership(dispersion, dispersion_u, self.disp_med),
            "disp_high": self._membership(dispersion, dispersion_u, self.disp_high),
        }

    def infer(self, mom: float, breadth: float, dispersion: float) -> dict[str, float]:
        """
        Applique les neuf règles Mamdani (ET = min, agrégation = max).

        R1 négatif + faible  → baissier
        R2 négatif + moyen   → baissier
        R3 négatif + élevé   → latéral
        R4 neutre  + faible  → baissier
        R5 neutre  + moyen   → latéral
        R6 neutre  + élevé   → haussier
        R7 positif + faible  → latéral
        R8 positif + moyen   → haussier
        R9 positif + élevé   → haussier
        """
        f = self.fuzzify(mom, breadth, dispersion)
        rules: list[tuple[float, str]] = [
            (min(f["mom_neg"], f["br_low"]), "bear"),
            (min(f["mom_neg"], f["br_med"]), "bear"),
            (min(f["mom_neg"], f["br_high"]), "sideways"),
            (min(f["mom_neu"], f["br_low"]), "bear"),
            (min(f["mom_neu"], f["br_med"]), "sideways"),
            (min(f["mom_neu"], f["br_high"]), "bull"),
            (min(f["mom_pos"], f["br_low"]), "sideways"),
            (min(f["mom_pos"], f["br_med"]), "bull"),
            (min(f["mom_pos"], f["br_high"]), "bull"),
        ]

        aggregated = {"bull": 0.0, "sideways": 0.0, "bear": 0.0}
        for strength, regime in rules:
            aggregated[regime] = max(aggregated[regime], strength)

        directional_confidence = float(
            np.clip(f["disp_low"] + 0.65 * f["disp_med"] + 0.30 * f["disp_high"], 0, 1)
        )
        aggregated["bull"] *= directional_confidence
        aggregated["bear"] *= directional_confidence
        aggregated["sideways"] = max(
            aggregated["sideways"],
            0.70 * f["disp_high"] + 0.25 * f["disp_med"],
        )

        return self._normalize(
            aggregated["bull"], aggregated["sideways"], aggregated["bear"]
        )

    @staticmethod
    def _normalize(mu_bull: float, mu_side: float, mu_bear: float) -> dict[str, float]:
        vals = np.array([mu_bull, mu_side, mu_bear], dtype=float)
        vals = np.clip(vals, 0.0, None)
        total = vals.sum()
        if total <= 0:
            vals = np.array([1 / 3, 1 / 3, 1 / 3])
        else:
            vals = vals / total
        return {
            "mu_bull": float(vals[0]),
            "mu_sideways": float(vals[1]),
            "mu_bear": float(vals[2]),
        }


def aggregate_monthly_context(
    market_context: pd.DataFrame,
    *,
    start_year: int = 2015,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Prend les entrées au dernier jour de bourse de chaque mois."""
    if market_context.empty:
        return pd.DataFrame()

    df = market_context.copy()
    df["date_cours"] = pd.to_datetime(df["date_cours"], errors="coerce")
    df = df.dropna(subset=["date_cours"]).sort_values("date_cours")

    for col in FUZZY_REGIME_INPUTS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["year"] = df["date_cours"].dt.year
    df["month"] = df["date_cours"].dt.month
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    rows: list[dict[str, Any]] = []
    for (year, month), grp in df.groupby(["year", "month"], sort=True):
        last = grp.sort_values("date_cours").iloc[-1]
        rows.append(
            {
                "year": int(year),
                "month": int(month),
                "month_label": f"{year}-{month:02d}",
                "as_of_date": last["date_cours"],
                **{c: last.get(c) for c in FUZZY_REGIME_INPUTS},
            }
        )

    out = pd.DataFrame(rows).sort_values(["year", "month"]).reset_index(drop=True)
    for col in FUZZY_REGIME_INPUTS:
        if col in out.columns:
            out[col] = out[col].ffill()
    logger.info(
        "Contexte mensuel (fuzzy) : %s mois (%s → %s)",
        len(out),
        out["month_label"].min() if not out.empty else "—",
        out["month_label"].max() if not out.empty else "—",
    )
    return out


def detect_fuzzy_regime_history(
    market_context: pd.DataFrame,
    *,
    start_year: int = 2015,
    end_year: int = 2025,
    detector: FuzzyRegimeDetector | None = None,
) -> pd.DataFrame:
    """Calcule le vecteur de régime flou pour chaque mois (ablation)."""
    monthly = aggregate_monthly_context(
        market_context, start_year=start_year, end_year=end_year
    )
    if monthly.empty:
        return pd.DataFrame()

    det = detector or FuzzyRegimeDetector()
    regimes: list[dict[str, float]] = []
    for row in monthly.itertuples(index=False):
        mom = float(getattr(row, "masi_mom_3m", np.nan))
        disp = float(getattr(row, "return_dispersion", np.nan))
        breadth = float(getattr(row, "breadth_ma50", np.nan))
        if any(np.isnan(v) for v in (mom, disp, breadth)):
            regimes.append({c: np.nan for c in FUZZY_REGIME_COLUMNS})
            continue
        regimes.append(det.infer(mom, breadth, disp))

    regime_df = pd.DataFrame(regimes)
    out = pd.concat([monthly.reset_index(drop=True), regime_df], axis=1)

    valid = out.dropna(subset=FUZZY_REGIME_COLUMNS)
    if not valid.empty:
        sums = valid[FUZZY_REGIME_COLUMNS].sum(axis=1)
        logger.info(
            "Régimes flous (ablation) : %s mois | somme mu [min=%.3f, max=%.3f]",
            len(valid),
            sums.min(),
            sums.max(),
        )
    return out


def dominant_fuzzy_regime(row: pd.Series) -> str:
    cols = {c: row.get(c, np.nan) for c in FUZZY_REGIME_COLUMNS}
    if any(pd.isna(v) for v in cols.values()):
        return "unknown"
    return max(cols, key=cols.get).replace("mu_", "")
