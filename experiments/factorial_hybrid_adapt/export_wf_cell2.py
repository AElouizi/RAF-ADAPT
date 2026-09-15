"""Export Cellule 2 — walk-forward, 14 blocs non-chevauchants uniquement."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.factorial_hybrid_adapt.paths import REPORTS_DIR, SELECTIONS_DIR


def top25_from_preds(pred: pd.DataFrame) -> pd.DataFrame:
    p = pred.copy()
    p["date_cours"] = pd.to_datetime(p["date_cours"])
    p["mois"] = p["date_cours"].dt.to_period("M").astype(str)
    rows = []
    for mois, g in p.groupby("mois", sort=True):
        last = g["date_cours"].max()
        snap = (
            g[g["date_cours"] == last]
            .sort_values("prediction", ascending=False)
            .drop_duplicates("ticker", keep="first")
            .head(25)
            .copy()
        )
        snap["rang"] = range(1, len(snap) + 1)
        snap["titre"] = snap["ticker"]
        snap["score"] = snap["prediction"]
        snap["mois"] = mois
        rows.append(snap[["mois", "rang", "titre", "score", "date_cours"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def export_cell(cell_id: int, label: str) -> int:
    m = pd.read_csv(REPORTS_DIR / "wf_etape3_fold_metrics.csv")
    blocks = pd.read_csv(REPORTS_DIR / "wf_folds_nonoverlapping.csv")
    keys = set(zip(blocks["test_start_month"], blocks["test_end_month"]))
    cell = m[m["cell_id"] == cell_id].sort_values("fold_id")
    cell_b = cell[
        cell.apply(lambda r: (r["test_start_month"], r["test_end_month"]) in keys, axis=1)
    ].copy()

    print(f"=== Cellule {cell_id} {label} — WF 14 blocs NON-chevauchants ===\n")
    print(
        f"IC_mean={cell_b['rank_ic'].mean():.4f} | IC_median={cell_b['rank_ic'].median():.4f} | "
        f"IC_std={cell_b['rank_ic'].std():.4f} | hit_mean={cell_b['hit_ratio'].mean():.4f} | "
        f"n_blocs={len(cell_b)}"
    )
    print("\nIC par bloc:")
    print(
        cell_b[
            ["test_start_month", "test_end_month", "rank_ic", "hit_ratio"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    preds = pd.read_parquet(REPORTS_DIR / "wf_stage1_predictions.parquet")
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    unique_months = []
    seen: set[str] = set()
    for _, row in cell_b.iterrows():
        fid = int(row["fold_id"])
        sub = preds[(preds["fold_id"] == fid) & (preds["cell_id"] == cell_id)]
        t = top25_from_preds(sub)
        for mois, g in t.groupby("mois"):
            if mois in seen:
                continue
            seen.add(mois)
            gg = g.copy()
            gg["fold_id"] = fid
            gg["cell_id"] = cell_id
            unique_months.append(gg)

    uniq = pd.concat(unique_months, ignore_index=True)
    out_csv = SELECTIONS_DIR / f"wf_cell{cell_id}_top25_by_month_nonoverlap.csv"
    uniq.to_csv(out_csv, index=False)

    # Also save metrics for this cell on 14 blocks
    met_path = REPORTS_DIR / f"wf_cell{cell_id}_metrics_14blocks.csv"
    cell_b.to_csv(met_path, index=False)

    last = cell_b.iloc[-1]
    last_tops = top25_from_preds(
        preds[(preds["fold_id"] == int(last["fold_id"])) & (preds["cell_id"] == cell_id)]
    )
    last_mois = sorted(last_tops["mois"].unique())[-1]
    print(f"\n--- Top 25 — dernier mois du dernier bloc: {last_mois} ---")
    print(
        last_tops[last_tops["mois"] == last_mois][["rang", "titre", "score"]].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )
    print(f"\nFichier titres: {out_csv}")
    print(f"Fichier metrics: {met_path}")
    print(f"n_mois uniques={uniq['mois'].nunique()} | n_lignes={len(uniq)}")
    return 0


def main() -> int:
    return export_cell(2, "Ridge+regime")


if __name__ == "__main__":
    raise SystemExit(main())
