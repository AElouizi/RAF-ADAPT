"""
Interface données pour l'étage 2 (NSGA-III) — pas d'optimisation ici.

Pour chaque titre × date de rebalancement × cellule, expose :
- predicted_return / conviction_score (continus)
- recommendation BUY / NEUTRAL / SELL
- risque (vol_baissiere_20d as-of t)
- liquidité (vmq_20j as-of t)

Les contraintes BUY/NEUTRAL/SELL (privilégier / limiter / exclure) seront
codées plus tard dans l'optimiseur ; ce module ne fait que préparer les inputs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bvc_recommender.config import FEATURES_DIR
from experiments.factorial_hybrid_adapt.data_utils import load_ml_dataset

logger = logging.getLogger(__name__)

STAGE2_CORE_COLS = [
    "date",
    "ticker",
    "cell",
    "model",
    "regime",
    "predicted_return",
    "conviction_score",
    "recommendation",
    "excess_return",
    "benchmark_return",
    "tau",
    "risk_vol_baissiere_20d",
    "liquidity_vmq_20j",
]


def _asof_feature(
    panel: pd.DataFrame,
    tickers: list[str],
    as_of: pd.Timestamp,
    value_col: str,
    date_col: str = "date_cours",
) -> pd.Series:
    """Dernière valeur connue ≤ as_of pour chaque ticker (pas de look-ahead)."""
    sub = panel[
        panel["ticker"].isin(tickers) & (panel[date_col] <= pd.Timestamp(as_of))
    ]
    if sub.empty or value_col not in sub.columns:
        return pd.Series({t: np.nan for t in tickers}, dtype=float)
    snap = (
        sub.sort_values(["ticker", date_col])
        .groupby("ticker", as_index=False)
        .tail(1)
        .set_index("ticker")[value_col]
    )
    return pd.to_numeric(snap.reindex(tickers), errors="coerce")


def load_risk_liquidity_panel() -> pd.DataFrame:
    """
    Panel as-of pour risque / liquidité.
    Préfère ml_dataset ; complète vmq depuis features_techniques si besoin.
    """
    ml = load_ml_dataset()
    cols = ["ticker", "date_cours"]
    for c in ("vol_baissiere_20d", "vmq_20j"):
        if c in ml.columns:
            cols.append(c)
    out = ml[cols].copy()
    out["date_cours"] = pd.to_datetime(out["date_cours"], errors="coerce")

    tech_path = FEATURES_DIR / "features_techniques.parquet"
    if not tech_path.is_file():
        tech_path = FEATURES_DIR / "features_techniques.csv"
    if tech_path.is_file():
        tech = (
            pd.read_parquet(tech_path)
            if tech_path.suffix == ".parquet"
            else pd.read_csv(tech_path)
        )
        tech["date_cours"] = pd.to_datetime(tech["date_cours"], errors="coerce")
        keep = [c for c in ("ticker", "date_cours", "vmq_20j") if c in tech.columns]
        if "vmq_20j" in keep:
            out = out.drop(columns=["vmq_20j"], errors="ignore")
            out = out.merge(tech[keep], on=["ticker", "date_cours"], how="left")
    return out


def build_stage2_inputs(
    recommendations: pd.DataFrame,
    risk_liq: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Enrichit le panel de recommandations avec risque + liquidité as-of date.

    Ne calcule pas de poids NSGA. Ne transforme pas SELL/BUY en contraintes codées.
    """
    if recommendations.empty:
        return pd.DataFrame(columns=STAGE2_CORE_COLS)

    risk_liq = load_risk_liquidity_panel() if risk_liq is None else risk_liq
    risk_liq = risk_liq.copy()
    risk_liq["date_cours"] = pd.to_datetime(risk_liq["date_cours"], errors="coerce")

    rec = recommendations.copy()
    rec["date"] = pd.to_datetime(rec["date"], errors="coerce")
    frames: list[pd.DataFrame] = []

    for as_of, g in rec.groupby("date", sort=True):
        tickers_unique = sorted(g["ticker"].astype(str).unique().tolist())
        vol = (
            _asof_feature(risk_liq, tickers_unique, as_of, "vol_baissiere_20d")
            if "vol_baissiere_20d" in risk_liq.columns
            else pd.Series(np.nan, index=tickers_unique, dtype=float)
        )
        vmq = (
            _asof_feature(risk_liq, tickers_unique, as_of, "vmq_20j")
            if "vmq_20j" in risk_liq.columns
            else pd.Series(np.nan, index=tickers_unique, dtype=float)
        )
        # Index unique obligatoire pour Series.map
        vol = vol[~vol.index.duplicated(keep="last")]
        vmq = vmq[~vmq.index.duplicated(keep="last")]
        gg = g.copy()
        gg["risk_vol_baissiere_20d"] = gg["ticker"].astype(str).map(vol)
        gg["liquidity_vmq_20j"] = gg["ticker"].astype(str).map(vmq)
        frames.append(gg)

    out = pd.concat(frames, ignore_index=True)
    # Colonnes stables pour NSGA
    for c in STAGE2_CORE_COLS:
        if c not in out.columns:
            out[c] = np.nan
    # Champs utiles additionnels
    if "fold_id" not in out.columns:
        out["fold_id"] = np.nan
    logger.info(
        "Stage2 inputs: %s lignes | dates=%s | cells=%s",
        len(out),
        out["date"].nunique(),
        sorted(out["cell"].dropna().unique().tolist()),
    )
    return out


def export_stage2_inputs(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path
