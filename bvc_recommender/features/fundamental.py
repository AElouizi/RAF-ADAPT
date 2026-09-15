"""
Étape 2 — Features fondamentales.

Sources :
1. donnees_financieres (EAV, valeurs en milliers MAD) — source principale
2. histo_const_ind — cours et nombre_titre (point-in-time via publication_date)
   Jointure ticker histo_const_ind ↔ ticker entreprises

Règles :
- Comptes CONSOLIDE → RNPG ; SOCIAL → RN ; Assurance → Résultat_Part_groupe
- Croissance YoY : même type de compte sur les deux périodes (conso/conso ou social/social)
- Indicateurs affichés à partir de la 1ère date de cotation dans histo_const_ind (date_fin / date_cours ≥ listing_date)
- Périmètre global : pas d'indicateurs avant INDICATORS_MIN_DATE (3/06/2015 — exclut 2011–2014)
- publication_date pour anti look-ahead (S1/S2)
- PE / PB ≤ 0 → NA (non significatif)
- Assurance PE : bénéfice = RNPG → PER = cours / (RNPG × 1000 / nbt)
- Revenu : CA (industrie) ; PNB (banques) ; Primes émises (assurances)
- Marge nette = resultat_net / revenu × 100 (RNPG, RN ou Résultat_Part_groupe selon secteur/compte)
- Marge EBITDA = (REX + |dot. expl.| + |dot. amort.| + |dot. risque|) / CA × 100
- ROE = RNPG / capitaux_propres × 100 (CP incohérent → report du dernier stock valide ; |ROE| > 100 % → NA)
- EV/EBITDA = capitalisation / EBITDA (EV = cours × nombre_titre ; EBITDA en milliers MAD × 1000) — NA pour banques/assurances
- Piotroski (0-7), score_sante pondéré (1/3 chaque composante) — voir SCORE_SANTE_WEIGHTS
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

from bvc_recommender.config import FINANCIAL_SECTOR_KEYWORDS, INDICATORS_MIN_DATE
from bvc_recommender.data.cleaner import add_publication_dates
from bvc_recommender.data.merger import enrich_with_periodes
from bvc_recommender.features._utils import asof_merge_by_ticker, find_col, get_numeric, safe_div, series_or

logger = logging.getLogger(__name__)

THOUSANDS_MAD = 1_000
MIN_CP_KMAD = 10_000  # plancher crédible pour capitaux_propres (milliers MAD)
MAX_ROE_PCT = 100.0  # |ROE| semestriel au-delà → NA (données incohérentes)

FUNDAMENTAL_FEATURE_COLUMNS = [
    "pe",
    "pb",
    "ev_ebitda",
    "roe",
    "marge_ebitda",
    "marge_nette",
    "ca_growth_yoy",
    "resultat_growth_yoy",
    "ebitda_growth_yoy",
    "gearing",
    "piotroski_score",
    "score_sante",
]

FINANCIER_EXCLUDED = ("ev_ebitda",)

# Mapping code indicateur → champ panel (plusieurs codes peuvent pointer vers le même champ)
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "ca": ("ca",),
    "pnb": ("pnb",),
    "primes_emises": ("primes_émises", "primes_emises"),
    "rnpg": ("rnpg",),
    "rn": ("rn",),
    "resultat_part_groupe": ("résultat_part_groupe", "resultat_part_groupe"),
    "capitaux_propres": ("capitaux_propres", "capitaux_propres_groupe"),
    "rex": ("rex",),
    "dotations_exploitation": ("dotations_exploitation", "dotations_exploitations"),
    "dotations_amortissement": ("dotations_amortissement", "dotations_amortissements"),
    "dotations_risque": ("dotations_risque",),
    "total_actif": ("total_actif",),
    "total_passif": ("total_passif",),
    "actif_circulant": ("actif_circulant",),
    "actif_non_courant": ("actif_non_courant",),
    "treso_actif": ("treso_actif", "tréso_actif"),
    "dettes_financieres_courantes": (
        "dettes_financières_courantes",
        "dettes_financieres_courantes",
        "dettes_financières_ct",
    ),
    "dettes_passif_circulant": ("dettes_passif_circulant",),
    "autres_dettes_financ": ("autres_dettes_financ",),
    "emprunt_obligataire": ("emprunt_obligataire",),
}

# Fallback si indicateurs_financiers inaccessible (RLS)
_FALLBACK_INDICATOR_ID_MAP: dict[str, str] = {
    "3eace438-76e4-4cd4-addc-50bd09b9d6d2": "ca",
    "f0a04583-1206-4d30-b1b3-bc988fbd451c": "rnpg",
    "baf36cef-843f-4f85-8486-56d1893f8284": "rn",
    "b9c2dc9f-e1ba-448b-a84b-aefa42e620ea": "capitaux_propres",
    "9228f72b-3991-4db6-89af-b6e59196c7ab": "capitaux_propres",
    "3e08a94a-a249-4a1f-9328-42cf7bf76a44": "capitaux_propres",
    "64fb591e-a7a8-4f50-864c-b2bd2f17510b": "rex",
    "86a4a181-e1d3-4189-9c79-9d72e3d8eae9": "total_actif",
    "834fd28d-ad8d-4477-b646-674f034d768d": "total_passif",
    "652f6657-acfd-4d10-8885-8d2e2cd89fa5": "total_passif",
    "46f0d784-58eb-46e7-8675-19a0f2d0eb6d": "actif_circulant",
    "17ba7a8f-95bb-4f07-8e5d-100b29d72726": "actif_non_courant",
    "f6153f0d-4906-40d0-9765-75d2eb2f5756": "treso_actif",
    "686ac179-ff50-4219-a67f-aa5ef81529a6": "dettes_financieres_courantes",
    "e774dd13-0aa3-4a21-b929-db2d065607e6": "dettes_passif_circulant",
    "99e53c94-1f58-4536-a2ec-aaa67647d0ae": "autres_dettes_financ",
    "a2c5acb6-12c0-4fa2-a39f-7cc6658e22b6": "emprunt_obligataire",
    "63ad02c7-84a5-4265-b011-9f78b513490f": "pnb",
    "42a0195f-7d5c-49a5-a253-ce4e93ce1308": "primes_emises",
    "c2a13096-015d-47c3-872d-d8a4b54f1a95": "dotations_exploitation",
    "9d13be13-30d1-4c1b-b40e-924ad12221e1": "dotations_exploitation",
    "7cdea8ca-cf9d-4bcb-bc8b-cf91da8b468f": "dotations_amortissement",
    "ef0ca4f3-6b33-4e84-8fe7-01dff019215e": "dotations_amortissement",
    "bfc11ed2-7289-4c0a-a6f3-2357c1518ee1": "dotations_risque",
    "3125b2ee-061f-4410-aec1-4f4bd0ee56f8": "resultat_part_groupe",
}


def _norm_code(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.replace(" ", "_")


def build_indicator_field_map(indicateurs_financiers: pd.DataFrame | None) -> dict[str, str]:
    """indicateur_id → nom de champ panel."""
    alias_to_field: dict[str, str] = {}
    for field, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            alias_to_field[_norm_code(alias)] = field

    if indicateurs_financiers is None or indicateurs_financiers.empty:
        return dict(_FALLBACK_INDICATOR_ID_MAP)

    id_col = find_col(indicateurs_financiers, "id") or "id"
    code_col = find_col(indicateurs_financiers, "code") or "code"
    mapping: dict[str, str] = {}
    for _, row in indicateurs_financiers.iterrows():
        field = alias_to_field.get(_norm_code(row.get(code_col)))
        if field:
            mapping[str(row[id_col])] = field
    if not mapping:
        return dict(_FALLBACK_INDICATOR_ID_MAP)
    return mapping


def assign_sector_type(secteur_nom: str | None) -> str:
    if secteur_nom is None or (isinstance(secteur_nom, float) and np.isnan(secteur_nom)):
        return "non_financier"
    lower = str(secteur_nom).lower()
    if any(kw in lower for kw in FINANCIAL_SECTOR_KEYWORDS):
        return "financier"
    return "non_financier"


def assign_sector_subtype(secteur_nom: str | None) -> str:
    """Sous-type sectoriel pour le choix du dénominateur de revenu."""
    if secteur_nom is None or (isinstance(secteur_nom, float) and np.isnan(secteur_nom)):
        return "non_financier"
    lower = str(secteur_nom).lower()
    if "assurance" in lower:
        return "assurance"
    if "banque" in lower:
        return "banque"
    return "non_financier"


def add_sector_metadata(
    panel: pd.DataFrame,
    entreprises: pd.DataFrame | None,
    secteurs: pd.DataFrame | None,
) -> pd.DataFrame:
    if panel.empty or entreprises is None or entreprises.empty:
        out = panel.copy()
        out["sector_type"] = "non_financier"
        out["sector_subtype"] = "non_financier"
        return out

    ent = entreprises.copy()
    ticker_col = find_col(ent, "ticker") or "ticker"
    id_col = find_col(ent, "id") or "id"
    ent = ent[[id_col, ticker_col, "secteur_id"]].rename(
        columns={id_col: "entreprise_id", ticker_col: "ticker"}
    )

    out = panel.copy()
    if "entreprise_id" not in out.columns and "ticker" in out.columns:
        out = out.merge(ent[["ticker", "secteur_id"]].drop_duplicates(), on="ticker", how="left")
    elif "entreprise_id" in out.columns:
        out = out.merge(ent[["entreprise_id", "secteur_id"]].drop_duplicates(), on="entreprise_id", how="left")

    if secteurs is not None and not secteurs.empty:
        sec = secteurs[["id", "nom"]].rename(columns={"id": "secteur_id", "nom": "secteur_nom"})
        out = out.merge(sec, on="secteur_id", how="left")
    else:
        out["secteur_nom"] = np.nan

    out["sector_type"] = out.get("secteur_nom", pd.Series(dtype=object)).map(assign_sector_type)
    out["sector_subtype"] = out.get("secteur_nom", pd.Series(dtype=object)).map(assign_sector_subtype)
    return out


def apply_account_type_preference(donnees: pd.DataFrame) -> pd.DataFrame:
    """
    CONSOLIDE si disponible sur la période, sinon SOCIAL.
    Décision par (entreprise_id, periode_id).
    """
    if donnees.empty or "type_compte" not in donnees.columns:
        return donnees

    df = donnees.copy()
    keys = ["entreprise_id", "periode_id"]
    has_conso = (
        df.loc[df["type_compte"].astype(str).str.upper() == "CONSOLIDE"]
        .groupby(keys, dropna=False)
        .size()
        .gt(0)
        .rename("_use_conso")
    )
    df = df.merge(has_conso, on=keys, how="left")
    df["_use_conso"] = df["_use_conso"].fillna(False).astype(bool)
    type_up = df["type_compte"].astype(str).str.upper()
    keep = (df["_use_conso"] & (type_up == "CONSOLIDE")) | (~df["_use_conso"] & (type_up == "SOCIAL"))
    return df.loc[keep].drop(columns=["_use_conso"])


def resolve_resultat_net(df: pd.DataFrame) -> pd.DataFrame:
    """
    Résultat net selon type de compte et secteur :
    - CONSOLIDE (hors assurance) → RNPG
    - SOCIAL (hors assurance) → RN
    - Assurance → Résultat_Part_groupe
    """
    out = df.copy()
    rnpg = get_numeric(out, "rnpg", "RNPG")
    rn = get_numeric(out, "rn", "RN")
    part_groupe = get_numeric(out, "resultat_part_groupe", "Résultat_Part_groupe")

    type_compte = out.get("type_compte", pd.Series(dtype=object)).astype(str).str.upper()
    subtype = out.get("sector_subtype", pd.Series("non_financier", index=out.index)).astype(str)
    assurance = subtype.str.contains("assurance", case=False, na=False)

    resultat = pd.Series(np.nan, index=out.index, dtype=float)
    resultat = resultat.mask(assurance, part_groupe)
    resultat = resultat.mask(~assurance & (type_compte == "CONSOLIDE"), rnpg)
    resultat = resultat.mask(~assurance & (type_compte == "SOCIAL"), rn)

    unknown = ~assurance & ~type_compte.isin(["CONSOLIDE", "SOCIAL"])
    resultat = resultat.mask(unknown, rnpg.where(rnpg.notna(), rn))

    out["resultat_net"] = resultat
    return out


def resolve_pe_earnings(df: pd.DataFrame) -> pd.Series:
    """
    Bénéfice pour le PER.
    Assurances : RNPG (revenu sectoriel = primes_émises).
    Autres : resultat_net (RNPG conso / RN social / Résultat_Part_groupe assurance exclu du PER).
    """
    subtype = df.get("sector_subtype", pd.Series("non_financier", index=df.index)).astype(str)
    assurance = subtype.str.contains("assurance", case=False, na=False)
    rnpg = get_numeric(df, "rnpg", "RNPG")
    resultat = series_or(df, "_resultat_net", "resultat_net")
    if resultat.isna().all():
        resultat = get_numeric(df, "resultat_net")
    earnings = resultat.copy()
    return earnings.mask(assurance, rnpg)


def resolve_revenue(df: pd.DataFrame) -> pd.Series:
    """
    Revenu pour les ratios : CA (industrie), PNB (banques), Primes émises (assurances).
    """
    ca = get_numeric(df, "ca", "CA")
    pnb = get_numeric(df, "pnb", "PNB")
    primes = get_numeric(df, "primes_emises", "primes_émises", "Primes_émises")
    subtype = df.get("sector_subtype", pd.Series("non_financier", index=df.index)).astype(str)

    revenue = ca.copy()
    banque = subtype.str.contains("banque", case=False, na=False)
    assurance = subtype.str.contains("assurance", case=False, na=False)
    revenue = revenue.where(~banque, pnb.where(pnb.notna(), ca))
    revenue = revenue.where(~assurance, primes.where(primes.notna(), ca))
    return revenue


def compute_ebitda(df: pd.DataFrame) -> pd.Series:
    """EBITDA = REX + |dot. exploitation| + |dot. amortissement| + |dot. risque|."""
    rex = get_numeric(df, "rex", "REX")
    dot_exp = get_numeric(df, "dotations_exploitation")
    dot_amort = get_numeric(df, "dotations_amortissement")
    dot_risk = get_numeric(df, "dotations_risque")
    return rex + dot_exp.abs() + dot_amort.abs() + dot_risk.abs()


def prepare_histo_market_panel(
    histo_const_ind: pd.DataFrame,
    entreprises: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prépare histo_const_ind pour jointure par ticker.
    La colonne ticker de histo_const_ind est reliée au ticker entreprises.
    """
    if histo_const_ind.empty or entreprises.empty:
        return pd.DataFrame()

    h = histo_const_ind.copy()
    date_col = find_col(h, "date", "date_cours") or "date"
    ticker_col = find_col(h, "ticker") or "ticker"
    if ticker_col not in h.columns:
        logger.warning("histo_const_ind sans colonne ticker — jointure marché impossible")
        return pd.DataFrame()

    h[date_col] = pd.to_datetime(h[date_col], errors="coerce")
    h["ticker"] = h[ticker_col].astype(str).str.strip().str.upper()
    if "indice" in h.columns:
        h = h[h["indice"].astype(str).str.upper() == "MASI"]

    valid_tickers = (
        entreprises[find_col(entreprises, "ticker") or "ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
        .dropna()
        .unique()
    )
    h = h[h["ticker"].isin(valid_tickers)]

    h["cours"] = pd.to_numeric(h.get("cours"), errors="coerce")
    h["nombre_titre"] = pd.to_numeric(h.get("nombre_titre"), errors="coerce")
    h = h.dropna(subset=["ticker", date_col])
    h = h.sort_values(["ticker", date_col])
    h = h.drop_duplicates(["ticker", date_col], keep="last")
    return h.rename(columns={date_col: "date"})


def listing_dates_from_histo(
    histo_const_ind: pd.DataFrame,
    entreprises: pd.DataFrame,
) -> pd.DataFrame:
    """Première date de cotation par ticker (min date dans histo_const_ind / MASI)."""
    if histo_const_ind.empty or entreprises.empty:
        return pd.DataFrame(columns=["ticker", "listing_date"])

    h = prepare_histo_market_panel(histo_const_ind, entreprises)
    if h.empty:
        return pd.DataFrame(columns=["ticker", "listing_date"])

    listing = (
        h.groupby("ticker", as_index=False)["date"]
        .min()
        .rename(columns={"date": "listing_date"})
    )
    return listing


def listing_dates_from_cours(cours: pd.DataFrame) -> pd.DataFrame:
    """Première date de cotation par ticker depuis l'historique des cours."""
    if cours.empty:
        return pd.DataFrame(columns=["ticker", "listing_date"])

    c = cours.copy()
    date_col = find_col(c, "date_cours", "date") or "date_cours"
    c[date_col] = pd.to_datetime(c[date_col], errors="coerce")
    c = c.dropna(subset=["ticker", date_col])
    return (
        c.groupby("ticker", as_index=False)[date_col]
        .min()
        .rename(columns={date_col: "listing_date"})
    )


def merge_listing_dates(*frames: pd.DataFrame) -> pd.DataFrame:
    """Fusionne plusieurs sources de dates de cotation (garde la plus ancienne)."""
    parts = [f for f in frames if f is not None and not f.empty]
    if not parts:
        return pd.DataFrame(columns=["ticker", "listing_date"])
    merged = pd.concat(parts, ignore_index=True)
    merged["listing_date"] = pd.to_datetime(merged["listing_date"], errors="coerce")
    return (
        merged.dropna(subset=["ticker", "listing_date"])
        .sort_values(["ticker", "listing_date"])
        .drop_duplicates("ticker", keep="first")
        .reset_index(drop=True)
    )


def filter_from_min_date(
    df: pd.DataFrame,
    date_col: str,
    *,
    min_date: str | None = None,
    label: str = "lignes",
) -> pd.DataFrame:
    """Exclut les lignes antérieures au périmètre global (défaut : 3/06/2015)."""
    if df.empty or date_col not in df.columns:
        return df

    floor = pd.Timestamp(min_date or INDICATORS_MIN_DATE)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    keep = dates.notna() & (dates >= floor)
    before = len(df)
    out = df.loc[keep].reset_index(drop=True)
    dropped = before - len(out)
    if dropped:
        logger.info(
            "Filtre périmètre global (%s ≥ %s) : %s %s exclues",
            date_col,
            floor.date(),
            dropped,
            label,
        )
    return out


def filter_from_listing_date(
    df: pd.DataFrame,
    listing_dates: pd.DataFrame,
    date_col: str,
    *,
    label: str = "lignes",
) -> pd.DataFrame:
    """Exclut les lignes dont la date est antérieure à la 1ère cotation (histo_const_ind)."""
    if df.empty or listing_dates.empty or date_col not in df.columns:
        return df

    out = df.merge(listing_dates, on="ticker", how="left")
    dates = pd.to_datetime(out[date_col], errors="coerce")
    listing = pd.to_datetime(out["listing_date"], errors="coerce")

    has_listing = listing.notna()
    keep = dates.notna() & (~has_listing | (dates >= listing))
    missing = sorted(out.loc[~has_listing, "ticker"].dropna().unique().tolist())
    if missing:
        logger.warning(
            "Date cotation absente — filtre non appliqué pour : %s",
            ", ".join(missing[:15]) + ("…" if len(missing) > 15 else ""),
        )

    before = len(out)
    out = out.loc[keep].drop(columns=["listing_date"], errors="ignore")
    dropped = before - len(out)
    if dropped:
        logger.info(
            "Filtre date cotation (%s) : %s %s exclues (avant 1ère cote)",
            date_col,
            dropped,
            label,
        )
    return out.reset_index(drop=True)


def filter_panel_from_listing_date(
    panel: pd.DataFrame,
    listing_dates: pd.DataFrame,
) -> pd.DataFrame:
    """Exclut les périodes dont date_fin est antérieure à la 1ère cotation connue."""
    return filter_from_listing_date(panel, listing_dates, "date_fin", label="périodes")


def build_panel_from_donnees_financieres(
    donnees: pd.DataFrame,
    periodes: pd.DataFrame,
    entreprises: pd.DataFrame,
    indicateurs_financiers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """EAV donnees_financieres → panel wide semestriel (valeurs en milliers MAD)."""
    if donnees.empty or periodes.empty or entreprises.empty:
        return pd.DataFrame()

    id_to_field = build_indicator_field_map(indicateurs_financiers)
    df = enrich_with_periodes(donnees, periodes)
    df = apply_account_type_preference(df)
    df["field"] = df["indicateur_id"].astype(str).map(id_to_field)
    df = df.dropna(subset=["field", "date_fin"])
    df["valeur"] = pd.to_numeric(df["valeur"], errors="coerce")

    sort_cols = ["entreprise_id", "periode_id", "field"]
    if "updated_at" in df.columns:
        sort_cols.append("updated_at")
    df = df.sort_values(sort_cols)
    cp_mask = df["field"] == "capitaux_propres"
    if cp_mask.any():
        cp_keys = ["entreprise_id", "periode_id", "field"]
        cp_best = df.loc[cp_mask].groupby(cp_keys, as_index=False)["valeur"].max()
        cp_meta = (
            df.loc[cp_mask]
            .sort_values(sort_cols)
            .drop_duplicates(cp_keys, keep="last")
            .drop(columns=["valeur"])
        )
        cp_merged = cp_best.merge(cp_meta, on=cp_keys, how="left")
        df = pd.concat([df.loc[~cp_mask], cp_merged], ignore_index=True)
    df = df.drop_duplicates(["entreprise_id", "periode_id", "field"], keep="last")

    wide = (
        df.pivot_table(
            index=["entreprise_id", "date_fin", "publication_date"],
            columns="field",
            values="valeur",
            aggfunc="last",
        )
        .reset_index()
    )
    wide.columns.name = None

    meta = (
        df.groupby(["entreprise_id", "date_fin", "publication_date"], as_index=False)["type_compte"]
        .first()
    )
    wide = wide.merge(meta, on=["entreprise_id", "date_fin", "publication_date"], how="left")
    wide["type_compte"] = wide["type_compte"].astype(str).str.upper()

    ent = entreprises.copy()
    id_col = find_col(ent, "id") or "id"
    ticker_col = find_col(ent, "ticker") or "ticker"
    ent = ent[[id_col, ticker_col]].rename(columns={id_col: "entreprise_id", ticker_col: "ticker"})
    wide = wide.merge(ent, on="entreprise_id", how="left")
    wide["date_fin"] = pd.to_datetime(wide["date_fin"], errors="coerce")
    if "publication_date" in wide.columns:
        wide["publication_date"] = pd.to_datetime(wide["publication_date"], errors="coerce")

    return wide.dropna(subset=["ticker", "date_fin"]).sort_values(["ticker", "date_fin"]).reset_index(drop=True)


def attach_market_from_histo(
    panel: pd.DataFrame,
    histo_const_ind: pd.DataFrame,
    entreprises: pd.DataFrame,
) -> pd.DataFrame:
    """Joint cours et nombre_titre depuis histo_const_ind (point-in-time, via ticker)."""
    if panel.empty or histo_const_ind.empty or entreprises.empty:
        return panel

    h = prepare_histo_market_panel(histo_const_ind, entreprises)
    if h.empty:
        return panel

    out = panel.copy()
    merge_on = "publication_date" if "publication_date" in out.columns else "date_fin"
    out["_merge_marche"] = pd.to_datetime(out[merge_on], errors="coerce")

    prices = h[["ticker", "date", "cours", "nombre_titre"]].rename(columns={"date": "_merge_marche"})
    prices["_merge_marche"] = pd.to_datetime(prices["_merge_marche"], errors="coerce")

    merged = asof_merge_by_ticker(out, prices, "_merge_marche", ["cours", "nombre_titre"])
    merged["prix_cloture"] = pd.to_numeric(merged.get("cours"), errors="coerce")
    merged["nombre_titre"] = pd.to_numeric(merged.get("nombre_titre"), errors="coerce")
    if "cours" in merged.columns:
        merged = merged.drop(columns=["cours"])
    return merged.drop(columns=["_merge_marche"], errors="ignore")


def supplement_market_from_cours(panel: pd.DataFrame, cours: pd.DataFrame) -> pd.DataFrame:
    """
    Complète prix_cloture et nombre_titre manquants depuis market_data_cours_historique.
    nombre_titre dérivé de capitalisation / prix quand disponible (ffill par ticker).
    """
    if panel.empty or cours.empty:
        return panel

    out = panel.copy()
    if "prix_cloture" not in out.columns:
        out["prix_cloture"] = np.nan
    if "nombre_titre" not in out.columns:
        out["nombre_titre"] = np.nan

    merge_on = "publication_date" if "publication_date" in out.columns else "date_fin"
    out["_merge_marche"] = pd.to_datetime(out[merge_on], errors="coerce")

    c = cours.copy()
    date_col = find_col(c, "date_cours", "date") or "date_cours"
    c[date_col] = pd.to_datetime(c[date_col], errors="coerce")
    price_col = find_col(c, "prix_cloture", "prix_courant") or "prix_cloture"
    c["_prix"] = pd.to_numeric(c[price_col], errors="coerce")
    c["_cap"] = pd.to_numeric(c.get("capitalisation"), errors="coerce")
    c["_nbt"] = safe_div(c["_cap"], c["_prix"])
    c = c.sort_values(["ticker", date_col])
    c["_nbt"] = c.groupby("ticker", sort=False)["_nbt"].ffill().bfill()

    prices = (
        c[["ticker", date_col, "_prix", "_nbt"]]
        .dropna(subset=["ticker", date_col])
        .rename(columns={date_col: "_merge_marche"})
    )
    prices["_merge_marche"] = pd.to_datetime(prices["_merge_marche"], errors="coerce")

    merged = asof_merge_by_ticker(out, prices, "_merge_marche", ["_prix", "_nbt"])
    merged["prix_cloture"] = merged["prix_cloture"].where(
        merged["prix_cloture"].notna(), pd.to_numeric(merged["_prix"], errors="coerce")
    )
    merged["nombre_titre"] = merged["nombre_titre"].where(
        merged["nombre_titre"].notna(), pd.to_numeric(merged["_nbt"], errors="coerce")
    )
    return merged.drop(columns=["_merge_marche", "_prix", "_nbt"], errors="ignore")


def attach_static_nombre_titre(panel: pd.DataFrame, entreprises: pd.DataFrame) -> pd.DataFrame:
    """Dernier recours : nombre_actions statique depuis entreprises."""
    if panel.empty or entreprises.empty:
        return panel

    ent = entreprises.copy()
    ticker_col = find_col(ent, "ticker") or "ticker"
    nbt_col = find_col(ent, "nombre_actions", "nombre_titre") or "nombre_actions"
    if nbt_col not in ent.columns:
        return panel

    static = (
        ent[[ticker_col, nbt_col]]
        .dropna(subset=[ticker_col])
        .drop_duplicates(ticker_col)
        .rename(columns={ticker_col: "ticker", nbt_col: "_nbt_static"})
    )
    static["_nbt_static"] = pd.to_numeric(static["_nbt_static"], errors="coerce")

    out = panel.merge(static, on="ticker", how="left")
    if "nombre_titre" not in out.columns:
        out["nombre_titre"] = np.nan
    out["nombre_titre"] = out["nombre_titre"].where(
        out["nombre_titre"].notna(), out["_nbt_static"]
    )
    return out.drop(columns=["_nbt_static"], errors="ignore")


def attach_market_data(
    panel: pd.DataFrame,
    histo_const_ind: pd.DataFrame | None,
    cours: pd.DataFrame | None,
    entreprises: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Jointure marché point-in-time : histo_const_ind (source principale).
    Complément cours uniquement si prix ou nombre_titre encore manquants.
    """
    out = panel
    if histo_const_ind is not None and not histo_const_ind.empty and entreprises is not None:
        out = attach_market_from_histo(out, histo_const_ind, entreprises)
    if cours is not None and not cours.empty:
        needs_price = out.get("prix_cloture", pd.Series(dtype=float)).isna()
        needs_nbt = out.get("nombre_titre", pd.Series(dtype=float)).isna()
        if needs_price.any() or needs_nbt.any():
            logger.warning(
                "Complément market_data_cours_historique pour %s périodes (histo_const_ind incomplet)",
                int((needs_price | needs_nbt).sum()),
            )
            out = supplement_market_from_cours(out, cours)
    if entreprises is not None:
        out = attach_static_nombre_titre(out, entreprises)

    if not out.empty:
        missing_price = int(out.get("prix_cloture", pd.Series(dtype=float)).isna().sum())
        missing_nbt = int(out.get("nombre_titre", pd.Series(dtype=float)).isna().sum())
        if missing_price or missing_nbt:
            logger.warning(
                "Marché incomplet après jointure : %s périodes sans prix, %s sans nombre_titre",
                missing_price,
                missing_nbt,
            )
    return out


def attach_market_prices(panel: pd.DataFrame, cours: pd.DataFrame) -> pd.DataFrame:
    """Fallback : prix depuis market_data_cours_historique si histo_const_ind indisponible."""
    if panel.empty or cours.empty:
        return panel

    out = panel.copy()
    merge_on = "publication_date" if "publication_date" in out.columns else "date_fin"
    out["_merge_marche"] = pd.to_datetime(out[merge_on], errors="coerce")

    c = cours.copy()
    date_col = find_col(c, "date_cours", "date") or "date_cours"
    c[date_col] = pd.to_datetime(c[date_col], errors="coerce")
    price_col = find_col(c, "prix_cloture", "prix_courant") or "prix_cloture"
    c["prix_marche"] = pd.to_numeric(c[price_col], errors="coerce")

    prices = c[["ticker", date_col, "prix_marche"]].rename(columns={date_col: "_merge_marche"})
    prices["_merge_marche"] = pd.to_datetime(prices["_merge_marche"], errors="coerce")

    merged = asof_merge_by_ticker(out, prices, "_merge_marche", ["prix_marche"])
    merged["prix_cloture"] = pd.to_numeric(merged.get("prix_marche"), errors="coerce")
    return merged.drop(columns=["_merge_marche", "prix_marche"], errors="ignore")


def _total_debt(df: pd.DataFrame) -> pd.Series:
    parts = [
        get_numeric(df, "dettes_financieres_courantes", "dettes_financières_courantes"),
        get_numeric(df, "dettes_passif_circulant"),
        get_numeric(df, "autres_dettes_financ"),
        get_numeric(df, "emprunt_obligataire"),
    ]
    out = pd.Series(0.0, index=df.index)
    for p in parts:
        out = out.add(p.fillna(0), fill_value=0)
    return out.replace(0, np.nan)


def resolve_capitaux_propres_for_roe(df: pd.DataFrame) -> pd.Series:
    """
    Capitaux propres pour le ROE.
    Si le stock est manifestement incohérent (< MIN_CP_KMAD), reporte le dernier stock valide (ffill).
    """
    cp = get_numeric(df, "capitaux_propres", "Capitaux_propres")
    if cp.isna().all() or "ticker" not in df.columns or "date_fin" not in df.columns:
        return cp

    order = df.sort_values(["ticker", "date_fin"]).index
    cp_sorted = pd.to_numeric(cp.loc[order], errors="coerce")
    tickers = df.loc[order, "ticker"]
    valid = cp_sorted.where(cp_sorted >= MIN_CP_KMAD)
    filled = valid.groupby(tickers, sort=False).ffill()
    resolved = cp_sorted.where(cp_sorted.isna() | (cp_sorted >= MIN_CP_KMAD), filled)
    return resolved.reindex(df.index)


def resolve_total_actif(df: pd.DataFrame) -> pd.Series:
    """
    Total bilan pour ROA et Piotroski.
    Total_Actif est souvent absent (1–2 périodes / valeur) ; fallback :
    1. Total_Actif (actif)
    2. Total_Passif / Total_passif (passif = actif en bilan)
    3. Actif_circulant + Actif_non_courant
    """
    actif = get_numeric(df, "total_actif", "Total_Actif")
    passif = get_numeric(df, "total_passif", "Total_Passif", "Total_passif")
    circ = get_numeric(df, "actif_circulant", "Actif_circulant")
    non_circ = get_numeric(df, "actif_non_courant", "Actif_non_courant")
    components = (circ + non_circ).where(circ.notna() & non_circ.notna())
    return actif.where(actif.notna(), passif.where(passif.notna(), components))


def compute_roe(df: pd.DataFrame) -> pd.Series:
    """ROE = RNPG / capitaux_propres × 100."""
    rnpg = get_numeric(df, "rnpg", "RNPG")
    cp = resolve_capitaux_propres_for_roe(df)
    roe = safe_div(rnpg, cp) * 100
    invalid = rnpg.isna() | cp.isna() | cp.le(0) | roe.abs().gt(MAX_ROE_PCT)
    return roe.where(~invalid)


def compute_profitability(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    revenue = resolve_revenue(out)
    rex = get_numeric(out, "rex", "REX")
    resultat = get_numeric(out, "resultat_net")
    if resultat.isna().all():
        out = resolve_resultat_net(out)
        resultat = get_numeric(out, "resultat_net")
    cp = get_numeric(out, "capitaux_propres", "Capitaux_propres")
    actif = resolve_total_actif(out)
    ebitda = compute_ebitda(out)

    debt = _total_debt(out)
    passif_ct = (
        get_numeric(out, "dettes_passif_circulant").fillna(0)
        + get_numeric(out, "dettes_financieres_courantes", "dettes_financières_courantes").fillna(0)
    ).replace(0, np.nan)

    ca = get_numeric(out, "ca", "CA")
    out["marge_nette"] = safe_div(resultat, revenue) * 100
    out["marge_ebitda"] = safe_div(ebitda, ca) * 100
    out["roe"] = compute_roe(out)
    out["_debt"] = debt
    out["_passif_ct"] = passif_ct
    out["_actif"] = actif
    cp_roe = resolve_capitaux_propres_for_roe(out)
    out["_cp"] = cp_roe.where(cp_roe.notna(), get_numeric(out, "capitaux_propres", "Capitaux_propres"))
    out["_resultat_net"] = resultat
    out["_rex"] = rex
    out["_revenue"] = revenue
    out["_ebitda"] = ebitda
    return out


def compute_solidity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cp = out.get("_cp", get_numeric(out, "capitaux_propres", "Capitaux_propres"))
    debt = out.get("_debt", _total_debt(out))
    out["gearing"] = safe_div(debt, cp) * 100
    return out


def compute_growth_yoy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Croissance YoY vs même semestre N-1 (lag 2), en %.
    Les deux périodes doivent avoir le même type_compte (CONSOLIDE ou SOCIAL).
    """
    out = df.sort_values(["ticker", "date_fin"]).copy()
    lag = 2

    type_compte = out.get("type_compte", pd.Series(dtype=object)).astype(str).str.upper()
    prev_type = type_compte.groupby(out["ticker"]).shift(lag)
    same_type = type_compte.notna() & prev_type.notna() & (type_compte == prev_type)

    revenue = out.get("_revenue", resolve_revenue(out))
    resultat = out.get(
        "_resultat_net",
        get_numeric(out, "resultat_net"),
    )
    ebitda = out.get("_ebitda", compute_ebitda(out))

    out["_revenue_growth"] = revenue
    out["_resultat_growth"] = resultat
    out["_ebitda_growth"] = ebitda

    cur_rev = pd.to_numeric(out["_revenue_growth"], errors="coerce")
    prev_rev = out.groupby("ticker", sort=False)["_revenue_growth"].shift(lag)
    out["ca_growth_yoy"] = (safe_div(cur_rev, prev_rev) - 1) * 100
    out.loc[~same_type, "ca_growth_yoy"] = np.nan

    cur_res = pd.to_numeric(out["_resultat_growth"], errors="coerce")
    prev_res = out.groupby("ticker", sort=False)["_resultat_growth"].shift(lag)
    out["resultat_growth_yoy"] = (safe_div(cur_res, prev_res) - 1) * 100
    out.loc[~same_type, "resultat_growth_yoy"] = np.nan

    cur_ebitda = pd.to_numeric(out["_ebitda_growth"], errors="coerce")
    prev_ebitda = out.groupby("ticker", sort=False)["_ebitda_growth"].shift(lag)
    out["ebitda_growth_yoy"] = (safe_div(cur_ebitda, prev_ebitda) - 1) * 100
    out.loc[~same_type, "ebitda_growth_yoy"] = np.nan

    return out.drop(
        columns=["_revenue_growth", "_resultat_growth", "_ebitda_growth"],
        errors="ignore",
    )


def compute_piotroski(df: pd.DataFrame) -> pd.DataFrame:
    """Score de Piotroski simplifié (0-7, sans critère accruals)."""
    out = df.sort_values(["ticker", "date_fin"]).copy()

    resultat = out.get("_resultat_net", get_numeric(out, "resultat_net"))
    actif = out.get("_actif", resolve_total_actif(out))
    debt = out.get("_debt", _total_debt(out))
    passif_ct = out.get("_passif_ct", get_numeric(out, "dettes_passif_circulant"))
    roa = safe_div(resultat, actif)
    actif_ct = (actif - debt.fillna(0)).clip(lower=0)
    liquidity = safe_div(actif_ct, passif_ct)
    gear = pd.to_numeric(out.get("gearing"), errors="coerce")
    revenue = out.get("_revenue", resolve_revenue(out))
    marge = out.get("marge_nette", safe_div(resultat, revenue) * 100)
    rotation = safe_div(revenue, actif)

    g = out["ticker"]
    scores = pd.Series(0.0, index=out.index)
    scores += (roa > 0).astype(int)
    scores += (resultat > 0).astype(int)
    scores += (roa > roa.groupby(g).shift(1)).fillna(False).astype(int)
    scores += (gear < gear.groupby(g).shift(1)).fillna(False).astype(int)
    scores += (liquidity > liquidity.groupby(g).shift(1)).fillna(False).astype(int)
    scores += (marge > marge.groupby(g).shift(1)).fillna(False).astype(int)
    scores += (rotation > rotation.groupby(g).shift(1)).fillna(False).astype(int)

    out["piotroski_score"] = scores.clip(0, 7)
    return out


def sanitize_valuation_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """PE / PB / EV/EBITDA non significatifs (≤ 0 ou non finis) → NA."""
    out = df.copy()
    for col in ("pe", "pb", "ev_ebitda"):
        if col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        out.loc[~np.isfinite(values) | (values <= 0), col] = np.nan
    return out


def compute_valuation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pe_earnings = resolve_pe_earnings(out)
    cp = series_or(out, "_cp", "capitaux_propres", "Capitaux_propres")
    ebitda = series_or(out, "_ebitda")
    if ebitda.isna().all():
        ebitda = compute_ebitda(out)
    cours = series_or(out, "prix_cloture", "cours")
    nbt = series_or(out, "nombre_titre")

    eps = safe_div(pe_earnings * THOUSANDS_MAD, nbt)
    bps = safe_div(cp * THOUSANDS_MAD, nbt)
    ev = cours * nbt  # capitalisation boursière (histo_const_ind)

    out["pe"] = safe_div(cours, eps)
    pe_invalid = (
        (out["pe"] <= 0)
        | eps.le(0)
        | pe_earnings.le(0)
        | cours.le(0)
        | nbt.le(0)
    )
    out.loc[pe_invalid, "pe"] = np.nan
    out["pb"] = safe_div(cours, bps)
    out.loc[(out["pb"] <= 0) | bps.le(0) | cp.le(0) | cours.le(0) | nbt.le(0), "pb"] = np.nan
    out["ev_ebitda"] = safe_div(ev, ebitda * THOUSANDS_MAD)
    ev_ebitda_invalid = ev.le(0) | ebitda.le(0) | cours.le(0) | nbt.le(0)
    out.loc[ev_ebitda_invalid, "ev_ebitda"] = np.nan
    return sanitize_valuation_ratios(out)


PIOTROSKI_MAX = 7

SCORE_SANTE_WEIGHTS: dict[str, float] = {
    "piotroski": 1 / 3,
    "levier": 1 / 3,
    "rentabilite": 1 / 3,
}


def compute_health_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score de santé [0, 1] — moyenne pondérée (poids égaux, renormalisés si composante manquante).

    - Piotroski (1/3) : piotroski_score / 7
    - Levier (1/3) : 1 / (1 + gearing/100)  — gearing stocké en %
    - Rentabilité (1/3) : (marge_nette bornée −10…30 % + 10) / 40
    """
    out = df.copy()
    components: dict[str, pd.Series] = {}

    if "piotroski_score" in out.columns:
        components["piotroski"] = pd.to_numeric(out["piotroski_score"], errors="coerce") / PIOTROSKI_MAX

    if "gearing" in out.columns:
        g_ratio = pd.to_numeric(out["gearing"], errors="coerce").clip(lower=0) / 100.0
        components["levier"] = 1.0 / (1.0 + g_ratio)

    if "marge_nette" in out.columns:
        mn = pd.to_numeric(out["marge_nette"], errors="coerce")
        mn_clipped = mn.clip(-10, 30)
        components["rentabilite"] = (mn_clipped + 10) / 40.0

    if not components:
        out["score_sante"] = np.nan
        return out

    score = pd.Series(0.0, index=out.index)
    total_weight = pd.Series(0.0, index=out.index)
    for name, values in components.items():
        weight = SCORE_SANTE_WEIGHTS[name]
        valid = values.notna()
        score = score + values.fillna(0) * weight * valid.astype(float)
        total_weight = total_weight + weight * valid.astype(float)

    out["score_sante"] = (score / total_weight.replace(0, np.nan)).round(4)
    return out


def apply_sector_rules(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sector_type" not in out.columns:
        return out
    fin = out["sector_type"] == "financier"
    for col in FINANCIER_EXCLUDED:
        if col in out.columns:
            out.loc[fin, col] = np.nan
    return out


# ---------------------------------------------------------------------------
# Source alternative : table large fondamentaux_rs_bis (ratios déjà calculés)
# ---------------------------------------------------------------------------

# Ratios repris tels quels depuis fondamentaux_rs_bis (pas de recalcul)
BIS_RATIOS_FROM_TABLE = (
    "pe",
    "pb",
    "roe",
    "marge_ebitda",
    "marge_nette",
    "ca_growth_yoy",
    "resultat_growth_yoy",
    "ebitda_growth_yoy",
    "gearing",
)

# Colonnes brutes de fondamentaux_rs_bis (panel_name -> alias source possibles)
BIS_RAW_COLUMNS: dict[str, tuple[str, ...]] = {
    "ca": ("ca",),
    "rnpg": ("rnpg",),
    "rex": ("rex",),
    "capitaux_propres": ("capitaux_propres",),
    "total_passif": ("total_passif",),
    "passif_non_courant": ("passif_non_courant",),
    "dettes_passif_circulant": ("dettes_passif_circulant",),
    "dotations_exploitation": ("dotations_exploitation",),
    "dotations_amortissement": ("dotations_amortissement",),
    "dotations_risque": ("dotations_risque",),
    "dette_nette": ("dette_nette",),
    "treso_actif": ("treso_actif", "tréso_actif"),
    "treso_passif": ("treso_passif", "tréso_passif"),
    "ebitda": ("ebitda", "ebidta"),
    "cours": ("cours",),
    "nombre_titre": ("nombre_titre",),
    "roa": ("roa",),
}


def parse_fr_number(series: pd.Series) -> pd.Series:
    """Convertit les nombres au format FR ('3 186 121', '25,6') en float."""
    s = series.astype(str)
    s = s.str.replace(r"[\s\u00a0\u202f]", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    s = s.replace({"": None, "None": None, "nan": None, "NaN": None, "<NA>": None})
    return pd.to_numeric(s, errors="coerce")


def build_panel_from_fondamentaux_bis(
    fondamentaux_bis: pd.DataFrame,
    *,
    periode: str | None = "SEMESTRIELLE",
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Table large fondamentaux_rs_bis → panel exploitable.

    - Parse les nombres FR et les dates (dd/mm/YYYY).
    - Filtre le type de période (SEMESTRIELLE par défaut).
    - ``type_compte_retenu`` (déjà arbitré) devient ``type_compte``.
    - Ajoute ``publication_date`` (anti look-ahead S1/S2).
    """
    if fondamentaux_bis is None or fondamentaux_bis.empty:
        return pd.DataFrame()

    src = fondamentaux_bis.copy()
    out = pd.DataFrame(index=src.index)

    ticker_col = find_col(src, "ticker") or "ticker"
    out["ticker"] = src[ticker_col].astype(str).str.strip().str.upper()

    date_col = find_col(src, "date_fin", "date") or "date_fin"
    date_fin = pd.to_datetime(src[date_col], format="%d/%m/%Y", errors="coerce")
    unresolved = date_fin.isna()
    if unresolved.any():
        date_fin.loc[unresolved] = pd.to_datetime(
            src.loc[unresolved, date_col], errors="coerce", dayfirst=True
        )
    out["date_fin"] = date_fin

    compte_col = find_col(src, "type_compte_retenu", "type_compte")
    out["type_compte"] = (
        src[compte_col].astype(str).str.upper() if compte_col else "CONSOLIDE"
    )

    periode_col = find_col(src, "type_periode")
    if periode_col is not None:
        out["type_periode"] = src[periode_col].astype(str).str.upper()

    for panel_name, aliases in BIS_RAW_COLUMNS.items():
        col = find_col(src, *aliases)
        out[panel_name] = parse_fr_number(src[col]) if col else np.nan

    for ratio in BIS_RATIOS_FROM_TABLE:
        col = find_col(src, ratio)
        out[ratio] = parse_fr_number(src[col]) if col else np.nan

    if periode and periode_col is not None:
        out = out[out["type_periode"] == periode.upper()].copy()

    if tickers:
        wanted = {t.upper() for t in tickers}
        out = out[out["ticker"].isin(wanted)].copy()

    out = out.dropna(subset=["ticker", "date_fin"])
    out = add_publication_dates(out, date_col="date_fin")
    return out.sort_values(["ticker", "date_fin"]).reset_index(drop=True)


def compute_ev_ebitda_from_bis(df: pd.DataFrame) -> pd.Series:
    """EV/EBITDA = (cours × nombre_titre) / (EBITDA × 1000). EBITDA en milliers MAD."""
    cours = get_numeric(df, "cours", "prix_cloture")
    nbt = get_numeric(df, "nombre_titre")
    ebitda = get_numeric(df, "ebitda")
    ev = cours * nbt
    ratio = safe_div(ev, ebitda * THOUSANDS_MAD)
    invalid = ev.le(0) | ebitda.le(0) | cours.le(0) | nbt.le(0)
    return ratio.where(~invalid)


def compute_piotroski_bis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Piotroski adapté (0-7) aux colonnes disponibles dans fondamentaux_rs_bis.

    Signaux CFO exclus (flux de trésorerie absents de la source). Approximations :
    - dette long terme ≈ passif_non_courant (validé utilisateur)
    - marge brute ≈ marge_ebitda
    - current ratio ≈ tréso_actif / (dettes_passif_circulant + tréso_passif)

    1. ROA > 0
    2. ROA(t) > ROA(t-1)
    3. Levier LT (passif_non_courant / total_passif) en baisse
    4. Liquidité (proxy current ratio) en hausse
    5. Pas de dilution (nombre_titre stable ou en baisse)
    6. Marge EBITDA en hausse
    7. Rotation de l'actif (ca / total_passif) en hausse
    """
    out = df.sort_values(["ticker", "date_fin"]).copy()
    g = out["ticker"]

    roa = get_numeric(out, "roa")
    rnpg = get_numeric(out, "rnpg")
    actif = get_numeric(out, "total_passif")
    roa = roa.where(roa.notna(), safe_div(rnpg, actif) * 100)

    levier_lt = safe_div(get_numeric(out, "passif_non_courant"), actif)
    passif_ct = (
        get_numeric(out, "dettes_passif_circulant").fillna(0)
        + get_numeric(out, "treso_passif").fillna(0)
    ).replace(0, np.nan)
    liquidity = safe_div(get_numeric(out, "treso_actif"), passif_ct)
    nbt = get_numeric(out, "nombre_titre")
    marge = get_numeric(out, "marge_ebitda")
    rotation = safe_div(get_numeric(out, "ca"), actif)

    def _prev(s: pd.Series) -> pd.Series:
        return s.groupby(g).shift(1)

    scores = pd.Series(0.0, index=out.index)
    scores += (roa > 0).astype(int)
    scores += (roa > _prev(roa)).fillna(False).astype(int)
    scores += (levier_lt < _prev(levier_lt)).fillna(False).astype(int)
    scores += (liquidity > _prev(liquidity)).fillna(False).astype(int)
    scores += (nbt <= _prev(nbt)).fillna(False).astype(int)
    scores += (marge > _prev(marge)).fillna(False).astype(int)
    scores += (rotation > _prev(rotation)).fillna(False).astype(int)

    out["piotroski_score"] = scores.clip(0, 7)
    return out


SCORE_SANTE_COMPOSITE_WEIGHTS: dict[str, float] = {
    "piotroski": 0.25,
    "valorisation": 0.25,
    "qualite": 0.25,
    "solidite": 0.25,
}

# Ratios de valorisation : plus bas = mieux (rang percentile inversé par date)
VALUATION_SCORE_COLUMNS = ("pe", "pb", "ev_ebitda")
# Qualité / profitabilité : plus haut = mieux (rang percentile par date)
QUALITY_SCORE_COLUMNS = (
    "roe",
    "marge_ebitda",
    "marge_nette",
    "ca_growth_yoy",
    "resultat_growth_yoy",
    "ebitda_growth_yoy",
)


def _pct_rank_by_date(df: pd.DataFrame, col: str, *, ascending: bool) -> pd.Series:
    """Rang percentile (0-1] de ``col`` au sein de chaque date_fin."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    values = pd.to_numeric(df[col], errors="coerce")
    return values.groupby(df["date_fin"]).rank(pct=True, ascending=ascending)


def _mean_of_available(parts: list[pd.Series], index: pd.Index) -> pd.Series:
    """Moyenne ligne à ligne des sous-scores disponibles (NaN ignorés)."""
    if not parts:
        return pd.Series(np.nan, index=index, dtype=float)
    stacked = pd.concat(parts, axis=1)
    return stacked.mean(axis=1, skipna=True)


def compute_health_score_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score de santé composite [0, 1] — 4 composantes à poids égaux (25 % chacune,
    renormalisés si composante manquante) :

    - Piotroski : piotroski_score / 7
    - Valorisation : rang percentile par date_fin de PE, PB, EV/EBITDA
      (ratio faible = société moins chère = score proche de 1)
    - Qualité / profitabilité : rang percentile par date_fin de ROE,
      marge EBITDA, marge nette et croissances YoY (CA, résultat, EBITDA)
    - Solidité : 1 / (1 + gearing/100) — gearing stocké en %
    """
    out = df.copy()
    components: dict[str, pd.Series] = {}

    if "piotroski_score" in out.columns:
        components["piotroski"] = (
            pd.to_numeric(out["piotroski_score"], errors="coerce") / PIOTROSKI_MAX
        )

    valuation_parts = [
        _pct_rank_by_date(out, col, ascending=False) for col in VALUATION_SCORE_COLUMNS
    ]
    valuation = _mean_of_available(valuation_parts, out.index)
    if valuation.notna().any():
        components["valorisation"] = valuation

    quality_parts = [
        _pct_rank_by_date(out, col, ascending=True) for col in QUALITY_SCORE_COLUMNS
    ]
    quality = _mean_of_available(quality_parts, out.index)
    if quality.notna().any():
        components["qualite"] = quality

    if "gearing" in out.columns:
        g_ratio = pd.to_numeric(out["gearing"], errors="coerce").clip(lower=0) / 100.0
        components["solidite"] = 1.0 / (1.0 + g_ratio)

    if not components:
        out["score_sante"] = np.nan
        return out

    score = pd.Series(0.0, index=out.index)
    total_weight = pd.Series(0.0, index=out.index)
    for name, values in components.items():
        weight = SCORE_SANTE_COMPOSITE_WEIGHTS[name]
        valid = values.notna()
        score = score + values.fillna(0) * weight * valid.astype(float)
        total_weight = total_weight + weight * valid.astype(float)

    out["score_sante"] = (score / total_weight.replace(0, np.nan)).round(4)
    return out


def build_fundamental_features_from_bis(
    fondamentaux_bis: pd.DataFrame,
    *,
    entreprises: pd.DataFrame | None = None,
    secteurs: pd.DataFrame | None = None,
    histo_const_ind: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
    periode: str | None = "SEMESTRIELLE",
) -> pd.DataFrame:
    """
    Features fondamentales à partir de fondamentaux_rs_bis.

    Les ratios pe/pb/roe/marges/gearing/growth sont repris tels quels de la table.
    ev_ebitda, piotroski_score et score_sante (absents) sont recalculés.
    """
    panel = build_panel_from_fondamentaux_bis(
        fondamentaux_bis, periode=periode, tickers=tickers
    )
    if panel.empty:
        logger.warning("Panel fondamental (bis) vide.")
        return panel

    panel = add_sector_metadata(panel, entreprises, secteurs)

    listing_dates = pd.DataFrame()
    if histo_const_ind is not None and not histo_const_ind.empty and entreprises is not None:
        listing_dates = listing_dates_from_histo(histo_const_ind, entreprises)
    if not listing_dates.empty:
        panel = filter_panel_from_listing_date(panel, listing_dates)
    panel = filter_from_min_date(panel, "date_fin", label="périodes")
    if panel.empty:
        return panel

    panel["ev_ebitda"] = compute_ev_ebitda_from_bis(panel)
    panel = compute_piotroski_bis(panel)
    panel = apply_sector_rules(panel)
    # PE/PB/EV-EBITDA ≤ 0 → NA avant le rang percentile (sinon classés "pas chers")
    panel = sanitize_valuation_ratios(panel)
    panel = compute_health_score_composite(panel)

    keep = ["ticker", "date_fin", "publication_date", "sector_type", *FUNDAMENTAL_FEATURE_COLUMNS]
    keep = [c for c in keep if c in panel.columns]
    out = panel[keep].copy()
    out = out.drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore")
    out = sanitize_valuation_ratios(out)

    logger.info(
        "Features fondamentales (bis) : %s lignes, %s tickers | PE=%s%% PB=%s%% ROE=%s%%",
        len(out),
        out["ticker"].nunique() if "ticker" in out.columns else 0,
        round(float(out["pe"].notna().mean() * 100), 1) if "pe" in out.columns else 0,
        round(float(out["pb"].notna().mean() * 100), 1) if "pb" in out.columns else 0,
        round(float(out["roe"].notna().mean() * 100), 1) if "roe" in out.columns else 0,
    )
    return out.sort_values(["ticker", "date_fin"]).reset_index(drop=True)


def build_fundamental_features(
    fondamentaux_rs: pd.DataFrame | None = None,
    cours: pd.DataFrame | None = None,
    donnees_financieres: pd.DataFrame | None = None,
    periodes: pd.DataFrame | None = None,
    entreprises: pd.DataFrame | None = None,
    secteurs: pd.DataFrame | None = None,
    histo_const_ind: pd.DataFrame | None = None,
    indicateurs_financiers: pd.DataFrame | None = None,
    *,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """
    Pipeline complet features fondamentales depuis donnees_financieres + histo_const_ind.

    Returns panel semestriel avec colonnes FUNDAMENTAL_FEATURE_COLUMNS.
    """
    if donnees_financieres is None or periodes is None or entreprises is None:
        logger.warning("donnees_financieres / periodes / entreprises manquants.")
        return pd.DataFrame()

    panel = build_panel_from_donnees_financieres(
        donnees_financieres,
        periodes,
        entreprises,
        indicateurs_financiers,
    )
    if panel.empty:
        logger.warning("Panel fondamental vide.")
        return panel

    if tickers:
        panel = panel[panel["ticker"].isin(tickers)].copy()

    panel = add_sector_metadata(panel, entreprises, secteurs)
    panel = resolve_resultat_net(panel)

    listing_dates = pd.DataFrame()
    if histo_const_ind is not None and not histo_const_ind.empty:
        listing_dates = listing_dates_from_histo(histo_const_ind, entreprises)
    if not listing_dates.empty:
        panel = filter_panel_from_listing_date(panel, listing_dates)
    panel = filter_from_min_date(panel, "date_fin", label="périodes")

    panel = attach_market_data(panel, histo_const_ind, cours, entreprises)

    panel = compute_profitability(panel)
    panel = compute_solidity(panel)
    panel = compute_growth_yoy(panel)
    panel = compute_piotroski(panel)
    panel = compute_valuation(panel)
    panel = apply_sector_rules(panel)
    panel = compute_health_score(panel)

    keep = ["ticker", "date_fin", "publication_date", "sector_type", *FUNDAMENTAL_FEATURE_COLUMNS]
    keep = [c for c in keep if c in panel.columns]
    out = panel[keep].copy()
    out = out.drop(columns=[c for c in out.columns if c.startswith("_")], errors="ignore")
    out = sanitize_valuation_ratios(out)

    logger.info(
        "Features fondamentales : %s lignes, %s tickers | PE=%s%% PB=%s%% marge_nette=%s%%",
        len(out),
        out["ticker"].nunique() if "ticker" in out.columns else 0,
        round(float(out["pe"].notna().mean() * 100), 1) if "pe" in out.columns else 0,
        round(float(out["pb"].notna().mean() * 100), 1) if "pb" in out.columns else 0,
        round(float(out["marge_nette"].notna().mean() * 100), 1) if "marge_nette" in out.columns else 0,
    )
    return out.sort_values(["ticker", "date_fin"]).reset_index(drop=True)
