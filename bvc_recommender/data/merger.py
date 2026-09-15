"""
Fusion légère des sources — Étape 1.

Assemble les tables nettoyées en jeux prêts pour le rapport qualité et
les étapes suivantes (features). Pas de jointure complexe à ce stade.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from bvc_recommender.data.cleaner import add_publication_dates


def pivot_fondamentaux_wide(fondamentaux: pd.DataFrame) -> pd.DataFrame:
    """Pivot fondamentaux_rs (long ticker × item × date) → format wide."""
    if fondamentaux.empty:
        return fondamentaux

    required = {"ticker", "date_fin"}
    if not required.issubset(fondamentaux.columns):
        return pd.DataFrame()

    df = fondamentaux.copy()
    item_col = "item_id" if "item_id" in df.columns else None
    if item_col is None:
        return df

    df["item_code"] = (
        df[item_col].astype(str).str.strip().str.lower().str.replace(" ", "_")
    )
    value_col = "montant" if "montant" in df.columns else None
    if value_col is None:
        return df

    index_cols = ["ticker", "date_fin"]
    if "publication_date" in df.columns:
        index_cols.append("publication_date")

    wide = (
        df.pivot_table(
            index=index_cols,
            columns="item_code",
            values=value_col,
            aggfunc="last",
        )
        .reset_index()
    )
    wide.columns.name = None
    return wide


def merge_step1_datasets(cleaned: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    Fusion Étape 1 : produit un bundle structuré pour le rapport et l'étape 2.

    Returns
    -------
    dict avec clés ``raw_cleaned``, ``fondamentaux_wide``, ``summary``.
    """
    fond = cleaned.get("fondamentaux_rs", pd.DataFrame())
    cours = cleaned.get("market_data_cours_historique", pd.DataFrame())
    indices = cleaned.get("market_data_indices_historique", pd.DataFrame())
    fin = cleaned.get("donnees_financieres", pd.DataFrame())

    fond_wide = pivot_fondamentaux_wide(fond)

    summary: dict[str, Any] = {}
    if not fond.empty and "ticker" in fond.columns:
        summary["tickers_fondamentaux"] = int(fond["ticker"].nunique())
    if not cours.empty and "ticker" in cours.columns:
        summary["tickers_cours"] = int(cours["ticker"].nunique())
    if not indices.empty and "code_index" in indices.columns:
        summary["indices_codes"] = sorted(indices["code_index"].dropna().unique().tolist())
    if not fin.empty and "entreprise_id" in fin.columns:
        summary["entreprises_financieres"] = int(fin["entreprise_id"].nunique())

    return {
        "raw_cleaned": cleaned,
        "fondamentaux_wide": fond_wide,
        "donnees_financieres": fin,
        "market_data_cours_historique": cours,
        "market_data_indices_historique": indices,
        "summary": summary,
    }


def enrich_with_periodes(
    donnees: pd.DataFrame,
    periodes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Joint donnees_financieres avec periodes pour appliquer la règle look-ahead.

    Utilisable dès que la table ``periodes`` est disponible (référence).
    """
    if donnees.empty or periodes.empty:
        return donnees

    peri = periodes.copy()
    peri["date_fin"] = pd.to_datetime(peri["date_fin"], errors="coerce")
    peri = add_publication_dates(peri, date_col="date_fin")

    id_col = "id" if "id" in peri.columns else "periode_id"
    map_cols = peri[[id_col, "date_fin", "publication_date"]].rename(
        columns={id_col: "periode_id"}
    )

    out = donnees.merge(map_cols, on="periode_id", how="left")
    return out
