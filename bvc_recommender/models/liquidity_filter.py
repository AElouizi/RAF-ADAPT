"""
Sous-module C — Filtre de liquidité (pénalité sigmoïde continue).

Ne pas exclure brutalement les titres peu liquides.
Appliquer :
    score_final = score_TFT × sigmoid(VMQ_20j, seuil=500K MAD)

Puis classer par score_final décroissant et labelliser par terciles :
TOP / NEUTRE / BOTTOM.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bvc_recommender.config import LIQUIDITY_VMQ_THRESHOLD_MAD

logger = logging.getLogger(__name__)

LABEL_TOP = "TOP"
LABEL_NEUTRE = "NEUTRE"
LABEL_BOTTOM = "BOTTOM"

LIQUIDITY_COLUMNS = ("vmq_20j", "vmq_60j", "indicateur_liquidite")


def sigmoid_liquidity_factor(
    vmq: pd.Series | np.ndarray,
    threshold: float = LIQUIDITY_VMQ_THRESHOLD_MAD,
    steepness: float | None = None,
) -> pd.Series:
    """
    Facteur multiplicatif ∈ (0, 1].

    - VMQ ≫ seuil → facteur ≈ 1 (pénalité ≈ 0, score inchangé)
    - VMQ ≈ seuil → facteur élevé (~0.88) : liquidité acceptable
    - VMQ ≪ seuil → facteur → 0 (pénalité forte, score réduit)

    Le point d'inflexion est placé à 0.5 × seuil pour que le seuil
    réglementaire 500K MAD corresponde déjà à une liquidité « OK »,
    conformément à « très liquide → score inchangé ».
    """
    values = pd.to_numeric(pd.Series(vmq), errors="coerce").fillna(0.0)
    # Pente : ~4 unités de logit entre 0 et le seuil → transition douce
    k = steepness if steepness is not None else (8.0 / max(threshold, 1.0))
    midpoint = 0.5 * threshold
    x = k * (values - midpoint)
    # clip pour stabilité numérique
    x = x.clip(-60, 60)
    factor = 1.0 / (1.0 + np.exp(-x))
    return pd.Series(factor, index=values.index, dtype=float).clip(0.0, 1.0)


def apply_liquidity_penalty(
    scores: pd.DataFrame,
    *,
    score_col: str = "prediction",
    vmq_col: str = "vmq_20j",
    threshold: float = LIQUIDITY_VMQ_THRESHOLD_MAD,
) -> pd.DataFrame:
    """Applique la pénalité sigmoïde et produit score_final."""
    out = scores.copy()
    if vmq_col not in out.columns:
        out[vmq_col] = np.nan
    out["liquidity_factor"] = sigmoid_liquidity_factor(out[vmq_col], threshold=threshold)
    raw = pd.to_numeric(out[score_col], errors="coerce")
    out["score_tft"] = raw
    out["score_final"] = raw * out["liquidity_factor"]
    return out


def assign_tercile_labels(
    df: pd.DataFrame,
    *,
    score_col: str = "score_final",
    label_col: str = "label",
) -> pd.DataFrame:
    """Classe les valeurs en TOP / NEUTRE / BOTTOM par terciles du score_final."""
    out = df.copy()
    scores = pd.to_numeric(out[score_col], errors="coerce")
    valid = scores.notna()
    out[label_col] = pd.NA

    n_valid = int(valid.sum())
    if n_valid == 0:
        return out

    if n_valid < 3:
        out.loc[valid, label_col] = LABEL_NEUTRE
        return out.sort_values(score_col, ascending=False).reset_index(drop=True)

    # qcut sur le rang pour gérer les égalités
    ranked = scores[valid].rank(method="first", ascending=True)
    labels = pd.qcut(
        ranked,
        q=3,
        labels=[LABEL_BOTTOM, LABEL_NEUTRE, LABEL_TOP],
    )
    out.loc[valid, label_col] = labels.astype(str)
    return out.sort_values(score_col, ascending=False).reset_index(drop=True)


def rank_and_label_universe(
    scores: pd.DataFrame,
    technical: pd.DataFrame,
    *,
    score_col: str = "prediction",
    as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Joint VMQ_20j, applique la pénalité sigmoïde, classe par score_final
    décroissant et assigne les labels TOP / NEUTRE / BOTTOM.
    """
    s = scores.copy()
    s["date_cours"] = pd.to_datetime(s["date_cours"], errors="coerce")

    if as_of_date is not None:
        s = s[s["date_cours"] <= pd.Timestamp(as_of_date)]

    s = s.sort_values(["ticker", "date_cours"]).groupby("ticker", as_index=False).tail(1)

    if technical.empty or "ticker" not in technical.columns:
        tech = pd.DataFrame(columns=["ticker", *LIQUIDITY_COLUMNS])
    else:
        tech = technical.copy()
        if "date_cours" in tech.columns:
            tech["date_cours"] = pd.to_datetime(tech["date_cours"], errors="coerce")
            if as_of_date is not None:
                tech = tech[tech["date_cours"] <= pd.Timestamp(as_of_date)]
            tech = (
                tech.sort_values(["ticker", "date_cours"])
                .groupby("ticker", as_index=False)
                .tail(1)
            )
        else:
            tech = tech.drop_duplicates("ticker", keep="last")
        for col in LIQUIDITY_COLUMNS:
            if col not in tech.columns:
                tech[col] = np.nan
        tech = tech[["ticker", *LIQUIDITY_COLUMNS]]

    merged = s.merge(tech, on="ticker", how="left")
    merged = apply_liquidity_penalty(merged, score_col=score_col)
    merged = assign_tercile_labels(merged)
    merged["rank"] = np.arange(1, len(merged) + 1)

    logger.info(
        "Sous-module C — univers classé : %s tickers | TOP=%s NEUTRE=%s BOTTOM=%s | "
        "seuil VMQ=%s MAD",
        len(merged),
        int((merged["label"] == LABEL_TOP).sum()),
        int((merged["label"] == LABEL_NEUTRE).sum()),
        int((merged["label"] == LABEL_BOTTOM).sum()),
        f"{LIQUIDITY_VMQ_THRESHOLD_MAD:,.0f}",
    )
    return merged
