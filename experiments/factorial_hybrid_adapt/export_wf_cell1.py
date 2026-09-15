"""Export Cellule 1 — résultats walk-forward uniquement."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bvc_recommender.features.dataset_builder import TARGET_COLUMN
from experiments.factorial_hybrid_adapt.paths import OUTPUT_DIR, REPORTS_DIR, SELECTIONS_DIR


def top25_from_preds(pred: pd.DataFrame) -> pd.DataFrame:
    """Top 25 au dernier jour de chaque mois de la fenêtre test du pli."""
    p = pred.copy()
    p["date_cours"] = pd.to_datetime(p["date_cours"])
    p["mois"] = p["date_cours"].dt.to_period("M").astype(str)
    rows = []
    for mois, g in p.groupby("mois", sort=True):
        last = g["date_cours"].max()
        snap = g[g["date_cours"] == last].sort_values("prediction", ascending=False)
        snap = snap.drop_duplicates("ticker", keep="first").head(25).copy()
        snap["rang"] = range(1, len(snap) + 1)
        snap["titre"] = snap["ticker"]
        snap["score"] = snap["prediction"]
        snap["mois"] = mois
        rows.append(snap[["mois", "rang", "titre", "score", "date_cours"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> int:
    m = pd.read_csv(REPORTS_DIR / "wf_etape3_fold_metrics.csv")
    c1 = m[m["cell_id"] == 1].sort_values("fold_id")
    blocks = pd.read_csv(REPORTS_DIR / "wf_folds_nonoverlapping.csv")
    keys = set(zip(blocks["test_start_month"], blocks["test_end_month"]))
    c1b = c1[
        c1.apply(lambda r: (r["test_start_month"], r["test_end_month"]) in keys, axis=1)
    ]

    print("=== Cellule 1 Ridge static — WALK-FORWARD (plus de split fixe) ===\n")
    print("--- Descriptif : 81 plis mensuels chevauchants ---")
    print(
        f"IC_mean={c1['rank_ic'].mean():.4f} | IC_median={c1['rank_ic'].median():.4f} | "
        f"IC_std={c1['rank_ic'].std():.4f} | hit_mean={c1['hit_ratio'].mean():.4f} | "
        f"%IC>0={(c1['rank_ic'] > 0).mean() * 100:.1f}%"
    )
    print(
        f"Fenetres test : {c1['test_start_month'].iloc[0]}->{c1['test_end_month'].iloc[0]} "
        f"... {c1['test_start_month'].iloc[-1]}->{c1['test_end_month'].iloc[-1]}"
    )

    print("\n--- 14 blocs non-chevauchants (base DM) ---")
    print(
        f"IC_mean={c1b['rank_ic'].mean():.4f} | IC_median={c1b['rank_ic'].median():.4f} | "
        f"IC_std={c1b['rank_ic'].std():.4f} | hit_mean={c1b['hit_ratio'].mean():.4f}"
    )
    print("\nIC C1 par bloc:")
    print(
        c1b[["test_start_month", "test_end_month", "rank_ic", "hit_ratio"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )

    # Top25 dernier bloc (2025-01 -> 2025-06) depuis predictions WF
    preds = pd.read_parquet(REPORTS_DIR / "wf_stage1_predictions.parquet")
    last = c1b.iloc[-1]
    fold_id = int(last["fold_id"])
    p1 = preds[(preds["fold_id"] == fold_id) & (preds["cell_id"] == 1)].copy()
    tops = top25_from_preds(p1)
    out_csv = SELECTIONS_DIR / "wf_cell1_top25_by_month.csv"
    # Build all months from last few blocks for convenience — full last block months
    tops.to_csv(REPORTS_DIR / "wf_cell1_last_block_top25.csv", index=False)

    # Also export top25 for ALL months appearing in any of the 14 blocks' C1 preds
    all_tops = []
    for _, row in c1b.iterrows():
        fid = int(row["fold_id"])
        sub = preds[(preds["fold_id"] == fid) & (preds["cell_id"] == 1)]
        t = top25_from_preds(sub)
        t["fold_id"] = fid
        t["test_start_month"] = row["test_start_month"]
        t["test_end_month"] = row["test_end_month"]
        all_tops.append(t)
    all_df = pd.concat(all_tops, ignore_index=True)
    # Deduplicate by mois keeping the block that contains that month as end-window? 
    # Better: for each calendar month, use the fold where that month is in test and prefer non-overlap unique months
    # Simplest unique months: take from each block only months in that block's test window once
    unique_months = []
    seen = set()
    for _, row in c1b.iterrows():
        fid = int(row["fold_id"])
        sub = preds[(preds["fold_id"] == fid) & (preds["cell_id"] == 1)]
        t = top25_from_preds(sub)
        for mois, g in t.groupby("mois"):
            if mois in seen:
                continue
            seen.add(mois)
            gg = g.copy()
            gg["fold_id"] = fid
            unique_months.append(gg)
    uniq = pd.concat(unique_months, ignore_index=True)
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    uniq.to_csv(out_csv, index=False)

    last_mois = sorted(tops["mois"].unique())[-1]
    print(f"\n--- Top 25 C1 (WF) — dernier mois du dernier bloc: {last_mois} ---")
    print(
        tops[tops["mois"] == last_mois][["rang", "titre", "score"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )
    print(f"\nFichier titres WF C1 (mois uniques des 14 blocs): {out_csv}")
    print(f"n_mois={uniq['mois'].nunique()} | n_lignes={len(uniq)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
