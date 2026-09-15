"""Prolonge MASI dans market_data_indices_historique via proxy capi. des cours.

Quand Supabase/BVC n'ont plus de clôtures MASI récentes, on enchaîne les
rendements journaliers pondérés par capitalisation (univers cours local).
Corrélation empirique ~0.99 avec le MASI officiel sur 2023–2025.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import DATA_PROCESSED_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INDICES_PATH = DATA_PROCESSED_DIR / "market_data_indices_historique.parquet"
COURS_PATH = DATA_PROCESSED_DIR / "market_data_cours_historique.parquet"
SOURCE_TAG = "proxy_cap_weighted_cours"


def _cap_weighted_returns(cours: pd.DataFrame) -> pd.Series:
    c = cours.dropna(subset=["prix_cloture", "capitalisation"]).copy()
    c = c[c["capitalisation"] > 0]
    c["date_cours"] = pd.to_datetime(c["date_cours"])
    c = c.sort_values(["date_cours", "ticker"]).drop_duplicates(
        ["date_cours", "ticker"], keep="last"
    )
    prices = c.pivot(index="date_cours", columns="ticker", values="prix_cloture").sort_index()
    caps = c.pivot(index="date_cours", columns="ticker", values="capitalisation").sort_index()
    rets = prices.pct_change(fill_method=None)
    weights = caps.shift(1)
    weights = weights.div(weights.sum(axis=1), axis=0)
    proxy = (rets * weights).sum(axis=1, min_count=1)
    return proxy.dropna().sort_index()


def extend_masi(*, until: str | None = None) -> pd.DataFrame:
    if not INDICES_PATH.is_file():
        raise FileNotFoundError(INDICES_PATH)
    if not COURS_PATH.is_file():
        raise FileNotFoundError(COURS_PATH)

    indices = pd.read_parquet(INDICES_PATH)
    cours = pd.read_parquet(COURS_PATH)
    masi = indices[indices["code_index"].astype(str).str.upper() == "MASI"].copy()
    masi["date_index"] = pd.to_datetime(masi["date_index"])
    masi = masi.sort_values("date_index").drop_duplicates("date_index", keep="last")
    if masi.empty:
        raise ValueError("Aucune ligne MASI dans les indices.")

    last_official = pd.Timestamp(masi["date_index"].max())
    last_level = float(masi.loc[masi["date_index"] == last_official, "valeur_index"].iloc[0])
    proxy = _cap_weighted_returns(cours)
    end = pd.Timestamp(until) if until else proxy.index.max()
    ext = proxy.loc[(proxy.index > last_official) & (proxy.index <= end)]
    if ext.empty:
        logger.info(
            "Rien à prolonger (MASI déjà à %s, cours max %s).",
            last_official.date(),
            proxy.index.max().date(),
        )
        return indices

    rows: list[dict] = []
    level = last_level
    prev = last_level
    template = masi.iloc[-1].to_dict()
    for dt, ret in ext.items():
        level = level * (1.0 + float(ret))
        row = {k: template.get(k) for k in indices.columns}
        row["code_index"] = "MASI"
        row["nom_index"] = template.get("nom_index") or "MASI"
        row["date_index"] = pd.Timestamp(dt).normalize()
        row["valeur_index"] = float(level)
        row["valeur_haut"] = max(float(level), float(prev))
        row["valeur_bas"] = min(float(level), float(prev))
        row["variation_veille"] = float(level - prev)
        # garder scraped_at en string ISO (schéma parquet object/string)
        row["scraped_at"] = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        if "id" in row:
            row["id"] = None
        if "drupal_id" in row:
            row["drupal_id"] = None
        rows.append(row)
        prev = level

    added = pd.DataFrame(rows)
    for col in indices.columns:
        if col not in added.columns:
            added[col] = None
    added = added[indices.columns]
    # Retirer d'éventuelles lignes proxy déjà présentes après last_official
    keep = indices[
        ~(
            (indices["code_index"].astype(str).str.upper() == "MASI")
            & (pd.to_datetime(indices["date_index"]) > last_official)
        )
    ].copy()
    out = pd.concat([keep, added], ignore_index=True)
    out["date_index"] = pd.to_datetime(out["date_index"])
    if "scraped_at" in out.columns:
        out["scraped_at"] = out["scraped_at"].astype(str)
    out = out.sort_values(["code_index", "date_index"]).reset_index(drop=True)
    out.to_parquet(INDICES_PATH, index=False)
    logger.info(
        "MASI prolongé (%s) : %s → %s (%s jours, niveau %.2f → %.2f) → %s",
        SOURCE_TAG,
        last_official.date(),
        pd.Timestamp(added["date_index"].max()).date(),
        len(added),
        last_level,
        float(added["valeur_index"].iloc[-1]),
        INDICES_PATH,
    )
    return out


def main() -> int:
    until = None
    if len(sys.argv) > 1:
        until = sys.argv[1]
    extend_masi(until=until)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
