"""
Smoke test Bloc B — 1 fold × 7 modèles (pas le run OOS complet).

Usage :
    py -m experiments.factorial_hybrid_adapt.run_bloc_b_smoke_test
    py -m experiments.factorial_hybrid_adapt.run_bloc_b_smoke_test --fold-id 0 --no-tune
"""

from __future__ import annotations

import argparse
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
from experiments.factorial_hybrid_adapt.bloc_b_metrics import (  # noqa: E402
    bootstrap_stationary_ci,
    run_omnibus_test,
)
from experiments.factorial_hybrid_adapt.bloc_b_models import (  # noqa: E402
    MODEL_GROUP,
    MODEL_IDS,
    build_model,
    load_lgbm_config,
    train_model_bundle,
)
from experiments.factorial_hybrid_adapt.factorial_cells import combine_hybrid_tri  # noqa: E402
from experiments.factorial_hybrid_adapt.bloc_b_walk_forward import (  # noqa: E402
    run_one_fold_bloc_b,
)
from experiments.factorial_hybrid_adapt.data_utils import (  # noqa: E402
    attach_regime_to_panel,
    load_ml_dataset,
)
from experiments.factorial_hybrid_adapt.paths import (  # noqa: E402
    OUTPUT_DIR,
    REPORTS_DIR,
    ensure_output_dirs,
)
from experiments.factorial_hybrid_adapt.walk_forward_folds import (  # noqa: E402
    available_months_from_dates,
    generate_rolling_folds,
)

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _assert_factory(df: pd.DataFrame) -> None:
    for mid in ("c1_ridge", "c2_rf", "c3_lightgbm"):
        est = build_model(mid, {})
        assert est is not None, mid
        logger.info("build_model(%s) OK → %s", mid, type(est).__name__)

    for mid in ("c4_hybrid_tri", "c5_hybrid_tri_regime"):
        bundle = train_model_bundle(df.head(500), mid, panel_for_features=df)
        assert "ridge" in bundle and "rf" in bundle and "lgbm" in bundle
        logger.info("train_model_bundle(%s) OK", mid)

    dates = pd.Series(pd.to_datetime(["2020-01-31"] * 5 + ["2020-02-28"] * 5))
    a = np.linspace(-1, 1, 10)
    b = np.linspace(1, -1, 10)
    c = np.linspace(0.5, -0.5, 10)
    out = combine_hybrid_tri(a, b, c, dates)
    assert len(out) == 10


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Bloc B (1 fold × 5 stratégies C1–C5)")
    parser.add_argument("--fold-id", type=int, default=0, help="Index du pli overlapping")
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Hyperparams fixes (rapide) — recommandé pour le smoke test",
    )
    parser.add_argument(
        "--lgbm-trials",
        type=int,
        default=None,
        help="Override n_trials Optuna LGBM (sinon config gelée / smoke=2)",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    out_dir = OUTPUT_DIR / "bloc_b_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Smoke factory + combine_hybrid_tri ===")
    df_panel = attach_regime_to_panel(load_ml_dataset())
    _assert_factory(df_panel)
    cfg = load_lgbm_config()
    logger.info(
        "Config LGBM gelée : frozen_at=%s n_trials=%s keys=%s",
        cfg.get("frozen_at"),
        cfg.get("n_trials"),
        list(cfg.get("search_space", {}).keys()),
    )

    logger.info("Chargement panel + régime…")
    df = attach_regime_to_panel(load_ml_dataset())
    months = available_months_from_dates(df["date_cours"])
    folds = generate_rolling_folds(months)
    if not folds:
        raise RuntimeError("Aucun fold généré — dataset trop court ?")
    if args.fold_id < 0 or args.fold_id >= len(folds):
        raise IndexError(f"fold_id={args.fold_id} hors plage [0, {len(folds) - 1}]")
    fold = folds[args.fold_id]
    logger.info(
        "Pli smoke #%s | train %s→%s | test %s→%s | total_folds=%s",
        fold.fold_id,
        fold.train_start_month,
        fold.train_end_month,
        fold.test_start_month,
        fold.test_end_month,
        len(folds),
    )

    retune = not args.no_tune
    lgbm_trials = args.lgbm_trials
    if retune and lgbm_trials is None:
        lgbm_trials = 2
        logger.info("Smoke : Optuna LGBM limité à %s trials (pas le run OOS)", lgbm_trials)

    t0 = time.time()
    preds, results_long, tune_meta = run_one_fold_bloc_b(
        df,
        fold,
        model_ids=MODEL_IDS,
        retune_rf=retune,
        retune_lgbm=retune,
        lgbm_n_trials=lgbm_trials,
    )
    elapsed = time.time() - t0

    models_done = sorted(results_long["model_id"].unique())
    assert set(models_done) == set(MODEL_IDS), (
        f"Modèles manquants : {set(MODEL_IDS) - set(models_done)}"
    )
    assert set(results_long["groupe"].unique()) == {
        "C1_ridge",
        "C2_rf",
        "C3_lightgbm",
        "C4_hybrid_tri",
        "C5_hybrid_tri_regime",
    }

    long_path = out_dir / "results_long_smoke.csv"
    preds_path = out_dir / f"fold_{fold.fold_id:04d}_preds.parquet"
    results_long.to_csv(long_path, index=False)
    preds.to_parquet(preds_path, index=False)

    omnibus = run_omnibus_test(results_long, "rank_ic")
    boot = bootstrap_stationary_ci(
        results_long, "rank_ic", "c1_ridge", n_bootstrap=50, block_size=2.0
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke_1_fold",
        "fold_id": fold.fold_id,
        "train": f"{fold.train_start_month}->{fold.train_end_month}",
        "test": f"{fold.test_start_month}->{fold.test_end_month}",
        "n_models": len(models_done),
        "model_ids": models_done,
        "groupes": sorted(results_long["groupe"].unique().tolist()),
        "n_pred_rows": int(len(preds)),
        "n_metric_rows": int(len(results_long)),
        "elapsed_sec": round(elapsed, 1),
        "retune": retune,
        "lgbm_trials": lgbm_trials,
        "lgbm_config_frozen_at": cfg.get("frozen_at"),
        "omnibus_smoke": omnibus,
        "bootstrap_smoke": boot,
        "tune_meta": tune_meta,
        "note": (
            "Validation pipeline uniquement. "
            "Ne PAS interpréter omnibus/bootstrap sur 1 fold. "
            "Run OOS complet (81 folds / 14 blocs) non lancé."
        ),
    }
    sum_path = out_dir / "smoke_summary.json"
    sum_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    reports_copy = REPORTS_DIR / "bloc_b_results_long_smoke.csv"
    results_long.to_csv(reports_copy, index=False)

    print("\n=== Bloc B smoke test — OK ===")
    print(f"Fold {fold.fold_id} | {len(models_done)} modèles | {elapsed:.1f}s")
    print("\nRank-IC par modèle :")
    for mid in MODEL_IDS:
        row = results_long[
            (results_long["model_id"] == mid) & (results_long["metrique"] == "rank_ic")
        ]
        val = float(row["valeur"].iloc[0]) if len(row) else float("nan")
        print(f"  {mid:22s} [{MODEL_GROUP[mid]:15s}] Rank-IC={val:.4f}")
    print(f"\nArtefacts :\n  - {long_path}\n  - {preds_path}\n  - {sum_path}\n  - {reports_copy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
