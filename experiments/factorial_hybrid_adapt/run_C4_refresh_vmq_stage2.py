"""Refresh cours + VMQ + Stage2 inputs C4. No model retrain. No NSGA."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bvc_recommender.config import (
    DATA_PROCESSED_DIR,
    FEATURES_DIR,
    LIQUIDITY_VMQ_THRESHOLD_MAD,
    TABLE_COURS_HISTORIQUE,
)
from bvc_recommender.data.cleaner import clean_market_data_cours
from bvc_recommender.data.loader import fetch_table, get_supabase_client
from bvc_recommender.features.technical import build_technical_features
from bvc_recommender.features.writer import save_features_local
from experiments.factorial_hybrid_adapt.paths import REPORTS_DIR
from experiments.factorial_hybrid_adapt.stage2_allocation import EVAL_END_MONTH, EVAL_START_MONTH
from experiments.factorial_hybrid_adapt.stage2_interface import (
    build_stage2_inputs,
    export_stage2_inputs,
)

OUT_CHECKS = REPORTS_DIR / "c4_vmq_refresh_lookahead.json"


def refresh_cours() -> pd.DataFrame:
    client = get_supabase_client()
    raw = fetch_table(client, TABLE_COURS_HISTORIQUE, order_by="id", page_size=1000)
    raw_vol = raw[["ticker", "date_cours", "titres_echanges"]].copy()
    raw_vol["ticker"] = raw_vol["ticker"].astype(str).str.upper().str.strip()
    raw_vol["date_cours"] = pd.to_datetime(raw_vol["date_cours"], errors="coerce").dt.normalize()
    raw_vol["titres_echanges"] = pd.to_numeric(raw_vol["titres_echanges"], errors="coerce")
    cleaned = clean_market_data_cours(raw)
    cleaned["ticker"] = cleaned["ticker"].astype(str).str.upper().str.strip()
    cleaned["date_cours"] = pd.to_datetime(cleaned["date_cours"], errors="coerce").dt.normalize()
    cleaned = cleaned.drop(columns=["titres_echanges"], errors="ignore")
    cleaned = cleaned.merge(raw_vol, on=["ticker", "date_cours"], how="left")
    path = DATA_PROCESSED_DIR / "market_data_cours_historique.parquet"
    cleaned.to_parquet(path, index=False)
    vol = pd.to_numeric(cleaned["titres_echanges"], errors="coerce")
    print(
        f"Cours rafraîchi : {len(cleaned):,} lignes | "
        f"{cleaned['date_cours'].min().date()} -> {cleaned['date_cours'].max().date()} | "
        f"titres_echanges NULL={100*vol.isna().mean():.2f}%"
    )
    return cleaned


def recompute_technical(cours: pd.DataFrame) -> pd.DataFrame:
    indices = pd.read_parquet(DATA_PROCESSED_DIR / "market_data_indices_historique.parquet")
    tech = build_technical_features(cours, indices)
    save_features_local(tech, "features_techniques", FEATURES_DIR)
    print(
        f"Features techniques : {len(tech):,} | "
        f"{tech['date_cours'].min().date()} -> {tech['date_cours'].max().date()}"
    )
    return tech


def lookahead_vmq(tech: pd.DataFrame, cours: pd.DataFrame, n: int = 10, seed: int = 42) -> dict:
    c = cours.copy()
    c["date_cours"] = pd.to_datetime(c["date_cours"], errors="coerce")
    c["ticker"] = c["ticker"].astype(str).str.upper().str.strip()
    c["prix_cloture"] = pd.to_numeric(c["prix_cloture"], errors="coerce")
    vol = pd.to_numeric(c["titres_echanges"], errors="coerce").fillna(0)
    c["_tv"] = c["prix_cloture"] * vol
    c = c.dropna(subset=["date_cours"]).sort_values(["ticker", "date_cours"])
    c["vmq_full"] = c.groupby("ticker", sort=False)["_tv"].transform(
        lambda s: s.rolling(20, min_periods=20).mean()
    )
    rng = np.random.default_rng(seed)
    pool = c[(c["date_cours"] >= "2018-07-01") & (c["date_cours"] <= "2025-06-30") & c["vmq_full"].notna()]
    dates = pool["date_cours"].drop_duplicates().to_numpy()
    pick = sorted(pd.to_datetime(rng.choice(dates, size=min(n, len(dates)), replace=False)))
    rows = []
    ok = True
    for t in pick:
        at = c[c["date_cours"] == t][["ticker", "vmq_full"]]
        past = c[c["date_cours"] <= t].copy()
        past["vmq_cut"] = past.groupby("ticker", sort=False)["_tv"].transform(
            lambda s: s.rolling(20, min_periods=20).mean()
        )
        m = at.merge(past[past["date_cours"] == t][["ticker", "vmq_cut"]], on="ticker")
        mx = float((m["vmq_full"] - m["vmq_cut"]).abs().max()) if len(m) else None
        row_ok = mx is not None and mx < 1e-6
        ok = ok and row_ok
        rows.append({"date": str(pd.Timestamp(t).date()), "n": int(len(m)), "max_abs_diff": mx, "ok": row_ok})
    report = {"formula_no_lookahead": ok, "checks": rows, "n_tech": int(len(tech))}
    OUT_CHECKS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Look-ahead VMQ : {'OK' if ok else 'FAIL'} -> {OUT_CHECKS}")
    return report


def rebuild_stage2() -> pd.DataFrame:
    rec = pd.read_parquet(REPORTS_DIR / "stage1_recommendations_14blocks.parquet")
    s2 = build_stage2_inputs(rec)
    path = REPORTS_DIR / "stage2_inputs_14blocks.parquet"
    export_stage2_inputs(s2, path)
    s2["mois"] = pd.to_datetime(s2["date"]).dt.to_period("M").astype(str)
    c4 = s2[s2["cell"] == 4].copy()
    c4 = c4[(c4["mois"] >= EVAL_START_MONTH) & (c4["mois"] <= EVAL_END_MONTH)]
    vmq = pd.to_numeric(c4["liquidity_vmq_20j"], errors="coerce")
    recs = c4["recommendation"].astype(str).str.upper()
    elig = vmq.notna() & (vmq >= LIQUIDITY_VMQ_THRESHOLD_MAD) & (recs != "SELL")
    print(
        f"Stage2 C4 {EVAL_START_MONTH}->{EVAL_END_MONTH} : "
        f"{c4['mois'].nunique()} mois | %VMQ non-null={100*vmq.notna().mean():.1f}% | "
        f"n_NSGA moyen (SELL exclu + VMQ>=500k)={elig.groupby(c4['mois']).sum().mean():.1f}"
    )
    return s2


def main() -> int:
    print(f"Fenetre figee : {EVAL_START_MONTH} -> {EVAL_END_MONTH}")
    cours = refresh_cours()
    tech = recompute_technical(cours)
    la = lookahead_vmq(tech, cours)
    if not la["formula_no_lookahead"]:
        raise SystemExit("Look-ahead VMQ détecté — arrêt")
    rebuild_stage2()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
