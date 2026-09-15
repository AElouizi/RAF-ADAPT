"""
Nettoyage des données — Étape 1.

Règle anti look-ahead (résultats financiers BVC)
------------------------------------------------
Les états semestriels ne sont pas disponibles à la date de clôture :

- **S1** (clôture fin juin N) : utilisable uniquement à partir de **septembre N**
  (dernier jour du mois).
- **S2** (clôture fin décembre N) : utilisable uniquement à partir d'**avril N+1**
  (dernier jour du mois).

La colonne ``publication_date`` encode cette disponibilité. Toute jointure
point-in-time doit filtrer ``publication_date <= date_observation``.

Valeurs manquantes : **carry-forward (ffill) uniquement**, jamais d'interpolation
vers le futur (pas de bfill, pas de interpolation temporelle).
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from bvc_recommender.config import PUBLICATION_MONTH_S1, PUBLICATION_MONTH_S2

logger = logging.getLogger(__name__)


def publication_date_from_date_fin(date_fin: pd.Timestamp | str) -> pd.Timestamp:
    """
    Calcule la date de première disponibilité d'un résultat semestriel.

    Parameters
    ----------
    date_fin : date de clôture du semestre (typiquement 30/06 ou 31/12).

    Returns
    -------
    Dernier jour du mois de publication (septembre N ou avril N+1).
    """
    dt = pd.to_datetime(date_fin)
    month = int(dt.month)
    year = int(dt.year)

    if month == 6:
        pub = pd.Timestamp(year=year, month=PUBLICATION_MONTH_S1, day=1)
    elif month == 12:
        pub = pd.Timestamp(year=year + 1, month=PUBLICATION_MONTH_S2, day=1)
    else:
        # Périodes atypiques : décalage conservateur de 4 mois (aligné S2)
        logger.debug("date_fin atypique %s — décalage +4 mois appliqué", dt.date())
        pub = dt + pd.DateOffset(months=4)

    return pub + pd.offsets.MonthEnd(0)


def add_publication_dates(df: pd.DataFrame, date_col: str = "date_fin") -> pd.DataFrame:
    """Ajoute ``date_fin`` (si absent) et ``publication_date`` à partir de ``date_col``."""
    out = df.copy()
    if date_col not in out.columns:
        return out
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["publication_date"] = out[date_col].map(publication_date_from_date_fin)
    return out


def apply_point_in_time_filter(
    df: pd.DataFrame,
    as_of: pd.Timestamp | str,
    *,
    publication_col: str = "publication_date",
) -> pd.DataFrame:
    """
    Filtre les lignes dont la publication est connue à la date ``as_of``.

    Empêche l'utilisation de résultats financiers non encore publiés.
    """
    if publication_col not in df.columns or df.empty:
        return df
    cutoff = pd.to_datetime(as_of)
    pub = pd.to_datetime(df[publication_col], errors="coerce")
    return df.loc[pub <= cutoff].copy()


def carry_forward(
    df: pd.DataFrame,
    group_cols: list[str],
    sort_col: str,
    value_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Remplit les valeurs manquantes par carry-forward (ffill) dans chaque groupe.

    Tri ascendant sur ``sort_col`` ; **pas de bfill** (pas d'interpolation future).
    """
    if df.empty:
        return df

    out = df.sort_values(group_cols + [sort_col]).copy()
    cols = list(value_cols) if value_cols is not None else [
        c
        for c in out.select_dtypes(include=[np.number]).columns
        if c not in group_cols
    ]
    if not cols:
        return out

    out[cols] = out.groupby(group_cols, dropna=False)[cols].ffill()
    return out


def _pick_date_column(df: pd.DataFrame, *candidates: str) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def clean_fondamentaux_rs(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie fondamentaux_rs : dates, publication, carry-forward par ticker/item."""
    if df.empty:
        return df

    out = df.copy()
    date_col = _pick_date_column(out, "date", "date_fin") or "date"
    out["date_fin"] = pd.to_datetime(out[date_col], errors="coerce")
    out = add_publication_dates(out, date_col="date_fin")

    if "montant" in out.columns:
        out["montant"] = pd.to_numeric(out["montant"], errors="coerce")

    group = ["ticker"]
    if "item_id" in out.columns:
        group.append("item_id")

    out = carry_forward(
        out.dropna(subset=["date_fin"]),
        group_cols=group,
        sort_col="date_fin",
        value_cols=["montant"] if "montant" in out.columns else None,
    )
    return out


def clean_donnees_financieres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie donnees_financieres (EAV).

    Note : ``publication_date`` nécessite la table ``periodes`` (étape 2).
    Ici on filtre VALIDE et on normalise les valeurs numériques.
    """
    if df.empty:
        return df

    out = df.copy()
    if "statut_validation" in out.columns:
        out = out[out["statut_validation"] == "VALIDE"].copy()

    if "valeur" in out.columns:
        out["valeur"] = pd.to_numeric(out["valeur"], errors="coerce")

    group = [c for c in ("entreprise_id", "indicateur_id") if c in out.columns]
    if group and "periode_id" in out.columns:
        out = out.sort_values(group + ["periode_id"])

    return out


def clean_market_data_cours(df: pd.DataFrame, min_date: str | None = None) -> pd.DataFrame:
    """Nettoie market_data_cours_historique."""
    if df.empty:
        return df

    from bvc_recommender.config import COURS_MIN_DATE

    out = df.copy()
    date_col = _pick_date_column(out, "date_cours", "date") or "date_cours"
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    floor = pd.Timestamp(min_date or COURS_MIN_DATE)
    out = out.loc[out[date_col] >= floor]

    numeric_cols = [
        c
        for c in (
            "prix_ouverture",
            "prix_cloture",
            "prix_courant",
            "prix_haut",
            "prix_bas",
            "volume",
            "titres_echanges",
            "capitalisation",
            "prix_ajuste",
        )
        if c in out.columns
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "ticker" in out.columns:
        out = carry_forward(
            out.dropna(subset=[date_col]),
            group_cols=["ticker"],
            sort_col=date_col,
            value_cols=numeric_cols,
        )
    return out


def clean_market_data_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie market_data_indices_historique."""
    if df.empty:
        return df

    out = df.copy()
    date_col = _pick_date_column(out, "date_index", "date") or "date_index"
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    if "valeur_index" in out.columns:
        out["valeur_index"] = pd.to_numeric(out["valeur_index"], errors="coerce")

    code_col = _pick_date_column(out, "code_index", "code") or "code_index"
    if code_col in out.columns:
        out = carry_forward(
            out.dropna(subset=[date_col]),
            group_cols=[code_col],
            sort_col=date_col,
            value_cols=["valeur_index"] if "valeur_index" in out.columns else None,
        )
    return out


def clean_all_tables(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Applique le nettoyage à chaque table chargée."""
    return {
        "fondamentaux_rs": clean_fondamentaux_rs(raw.get("fondamentaux_rs", pd.DataFrame())),
        "donnees_financieres": clean_donnees_financieres(
            raw.get("donnees_financieres", pd.DataFrame())
        ),
        "market_data_cours_historique": clean_market_data_cours(
            raw.get("market_data_cours_historique", pd.DataFrame())
        ),
        "market_data_indices_historique": clean_market_data_indices(
            raw.get("market_data_indices_historique", pd.DataFrame())
        ),
    }
