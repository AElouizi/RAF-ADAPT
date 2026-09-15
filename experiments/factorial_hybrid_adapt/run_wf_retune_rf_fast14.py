"""
Correction RF retune — version RAPIDE.

Uniquement les 14 blocs non-chevauchants (échantillon du DM), pas les 81 plis
chevauchants. Grille : 6 combos (depth x leaf), n_estimators=100.
Val interne = 6 derniers mois du train du pli.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import load_env_file  # noqa: E402
from experiments.factorial_hybrid_adapt.data_utils import (  # noqa: E402
    attach_regime_to_panel,
    load_ml_dataset,
)
from experiments.factorial_hybrid_adapt.paths import (  # noqa: E402
    OUTPUT_DIR,
    REPORTS_DIR,
    ensure_output_dirs,
)
from experiments.factorial_hybrid_adapt.run_wf_dm_ic_blocks import (  # noqa: E402
    diebold_mariano,
)
from experiments.factorial_hybrid_adapt.walk_forward_folds import (  # noqa: E402
    Fold,
    available_months_from_dates,
    folds_to_frame,
    generate_nonoverlapping_blocks,
    generate_rolling_folds,
)
from experiments.factorial_hybrid_adapt.walk_forward_variants import (  # noqa: E402
    RETUNE_PARAM_GRID,
    run_one_fold_variant,
)

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Grille ultra-légère : 3 x 2 = 6 combos
FAST_GRID = {
    "n_estimators": [100],
    "max_depth": [3, 4, 6],
    "min_samples_leaf": [10, 30],
}


def _agg(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell_id, g in metrics.groupby("cell_id"):
        rows.append(
            {
                "cellule": int(cell_id),
                "modele": g["model"].iloc[0],
                "regime": bool(g["regime"].iloc[0]),
                "IC_mean": float(g["rank_ic"].mean()),
                "IC_median": float(g["rank_ic"].median()),
                "IC_std": float(g["rank_ic"].std(ddof=1)),
                "hit_mean": float(g["hit_ratio"].mean()),
                "n_folds": int(len(g)),
            }
        )
    return pd.DataFrame(rows).sort_values("cellule")


def _dm(metrics: pd.DataFrame) -> pd.DataFrame:
    wide = metrics.pivot_table(index="fold_id", columns="cell_id", values="rank_ic")
    loss = -wide
    rows = []
    for ref, alt, lab in (
        (1, 2, "adaptation C2vsC1"),
        (1, 3, "hybridation C3vsC1"),
        (1, 4, "combine C4vsC1"),
    ):
        d = diebold_mariano(loss[ref].values, loss[alt].values, h=1)
        rows.append(
            {
                "pair": lab,
                "delta_IC": float(wide[alt].mean() - wide[ref].mean()),
                "dm_stat_hln": d["dm_stat_hln"],
                "pvalue_hln": d["pvalue_hln"],
                "sig_5pct": bool(np.isfinite(d["pvalue_hln"]) and d["pvalue_hln"] < 0.05),
                "n": d["n"],
            }
        )
    return pd.DataFrame(rows)


def baseline_on_same_blocks(block_keys: set) -> pd.DataFrame:
    base = pd.read_csv(REPORTS_DIR / "wf_etape3_fold_metrics.csv")
    return base[
        base.apply(
            lambda r: (r["test_start_month"], r["test_end_month"]) in block_keys,
            axis=1,
        )
    ].copy()


def main() -> int:
    ensure_output_dirs()
    # Override module grid for this run
    import experiments.factorial_hybrid_adapt.walk_forward_variants as wfv

    wfv.RETUNE_PARAM_GRID = FAST_GRID

    df = attach_regime_to_panel(load_ml_dataset())
    months = available_months_from_dates(df["date_cours"])
    blocks = generate_nonoverlapping_blocks(months)
    logger.info(
        "RF retune RAPIDE : %s blocs non-chevauchants | grille=%s",
        len(blocks),
        FAST_GRID,
    )

    parts = OUTPUT_DIR / "wf_variant_A_retune_14blocks"
    parts.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict] = []
    tunes: list[dict] = []
    t0 = time.time()

    for i, fold in enumerate(blocks):
        # Re-number display but keep temporal identity via test months
        logger.info(
            "[%s/%s] test %s->%s",
            i + 1,
            len(blocks),
            fold.test_start_month,
            fold.test_end_month,
        )
        # Use overlapping fold_id matching same test window for comparability
        overlapping = generate_rolling_folds(months)
        match = [
            f
            for f in overlapping
            if f.test_start_month == fold.test_start_month
            and f.test_end_month == fold.test_end_month
        ]
        use_fold = match[0] if match else fold

        preds, metrics_rows, tune_meta = run_one_fold_variant(
            df, use_fold, retune_rf=True
        )
        preds.to_parquet(parts / f"block_{i:02d}_preds.parquet", index=False)
        all_metrics.extend(metrics_rows)
        tunes.append(
            {
                "block_id": i,
                "fold_id": use_fold.fold_id,
                "test_start_month": use_fold.test_start_month,
                "test_end_month": use_fold.test_end_month,
                **{f"rf_{k}": v for k, v in tune_meta["best_params"].items()},
                "best_inner_rank_ic": tune_meta.get("best_inner_rank_ic"),
            }
        )
        by_c = {m["cell_id"]: m for m in metrics_rows}
        logger.info(
            "  IC C1=%.4f C3=%.4f | RF depth=%s leaf=%s | elapsed=%.0fs",
            by_c[1]["rank_ic"],
            by_c[3]["rank_ic"],
            tune_meta["best_params"]["max_depth"],
            tune_meta["best_params"]["min_samples_leaf"],
            time.time() - t0,
        )

    metrics = pd.DataFrame(all_metrics)
    metrics_path = REPORTS_DIR / "wf_variant_A_retune14_fold_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(tunes).to_csv(REPORTS_DIR / "wf_variant_A_retune14_tune_log.csv", index=False)

    keys = {(t["test_start_month"], t["test_end_month"]) for t in tunes}
    base_same = baseline_on_same_blocks(keys)

    print("\n=== Correction RF retune (14 blocs non-chevauchants) ===")
    print(f"Grille: {FAST_GRID}")
    print(f"Duree totale: {(time.time()-t0)/60:.1f} min")
    print("\n--- Baseline (memes 14 blocs, HP fixes) ---")
    print(_agg(base_same).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(_dm(base_same).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n--- Avec retune RF par bloc (val interne au train) ---")
    agg = _agg(metrics)
    dm = _dm(metrics)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(dm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    agg.to_csv(REPORTS_DIR / "wf_variant_A_retune14_ic.csv", index=False)
    dm.to_csv(REPORTS_DIR / "wf_variant_A_retune14_dm.csv", index=False)
    summary = {
        "scope": "14_blocs_non_chevauchants",
        "grid": FAST_GRID,
        "elapsed_min": (time.time() - t0) / 60,
        "note": (
            "Retune a l'interieur du train de chaque bloc (6 derniers mois = val interne). "
            "Pas de val 2021-22 fixe. Pas de fuite du test du bloc. "
            "Echantillon aligne sur le DM (14), pas les 81 plis chevauchants."
        ),
        "aggregation": agg.to_dict(orient="records"),
        "dm": dm.to_dict(orient="records"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (REPORTS_DIR / "wf_variant_A_retune14_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nTermine: {datetime.now(timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
