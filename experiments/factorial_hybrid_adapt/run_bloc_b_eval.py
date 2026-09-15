"""
CLI Bloc B — évaluation walk-forward 7 modèles (format long).

Usage :
    # Pipeline complet (recommandé) : 81 overlapping + omnibus 14 blocs + bootstrap
    py -m experiments.factorial_hybrid_adapt.run_bloc_b_eval --run --folds all

    # Sous-ensembles
    py -m experiments.factorial_hybrid_adapt.run_bloc_b_eval --run --folds overlapping
    py -m experiments.factorial_hybrid_adapt.run_bloc_b_eval --run --folds blocks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    MODEL_IDS,
    load_lgbm_config,
)
from experiments.factorial_hybrid_adapt.bloc_b_walk_forward import (  # noqa: E402
    run_bloc_b_walk_forward,
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
    Fold,
    available_months_from_dates,
    generate_nonoverlapping_blocks,
    generate_rolling_folds,
)

load_env_file(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OMNIBUS_METRICS = (
    "rank_ic",
    "ic_ir",
    "rmse",
    "spread_BUY_SELL",
    "hit_ratio",
    "turnover",
)


def _block_fold_ids_from_overlapping(
    overlapping: list[Fold],
    blocks: list[Fold],
) -> list[int]:
    """Mappe chaque bloc indépendant vers le fold_id overlapping de même fenêtre test."""
    by_window = {
        (f.test_start_month, f.test_end_month): f.fold_id for f in overlapping
    }
    ids: list[int] = []
    for b in blocks:
        key = (b.test_start_month, b.test_end_month)
        if key not in by_window:
            raise KeyError(f"Bloc test {key} introuvable dans les folds overlapping")
        ids.append(int(by_window[key]))
    return ids


def _postprocess(
    long_df: pd.DataFrame,
    *,
    overlapping: list[Fold],
    blocks: list[Fold],
    metric: str,
) -> dict[str, Any]:
    block_ids = _block_fold_ids_from_overlapping(overlapping, blocks)
    blocks_long = long_df[long_df["fold_id"].isin(block_ids)].copy()

    omnibus = {
        m: run_omnibus_test(blocks_long, m) for m in OMNIBUS_METRICS
    }
    bootstrap = {
        mid: bootstrap_stationary_ci(long_df, metric, mid, n_bootstrap=1000)
        for mid in MODEL_IDS
    }
    # Bootstrap aussi sur spread (métrique économique clé)
    bootstrap_spread = {
        mid: bootstrap_stationary_ci(long_df, "spread_BUY_SELL", mid, n_bootstrap=1000)
        for mid in MODEL_IDS
    }

    means = (
        long_df[long_df["metrique"] == metric]
        .groupby(["model_id", "groupe"], as_index=False)["valeur"]
        .mean()
        .sort_values("valeur", ascending=False)
    )

    return {
        "n_overlapping_folds": int(long_df["fold_id"].nunique()),
        "n_independent_blocks": len(block_ids),
        "block_fold_ids": block_ids,
        "mean_metric_by_model": means.to_dict(orient="records"),
        "omnibus_14blocks": omnibus,
        "bootstrap_81folds": {metric: bootstrap, "spread_BUY_SELL": bootstrap_spread},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bloc B — walk-forward 7 modèles")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Exécute réellement (sinon dry-run)",
    )
    parser.add_argument(
        "--folds",
        choices=("all", "overlapping", "blocks"),
        default="all",
        help="all = 81 overlapping puis omnibus sur 14 blocs (sans double train)",
    )
    parser.add_argument("--no-tune", action="store_true", help="Hyperparams fixes")
    parser.add_argument("--lgbm-trials", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--omnibus-metric", default="rank_ic")
    args = parser.parse_args()
    resume = not args.no_resume

    ensure_output_dirs()
    cfg = load_lgbm_config()
    df = attach_regime_to_panel(load_ml_dataset())
    months = available_months_from_dates(df["date_cours"])
    overlapping = generate_rolling_folds(months)
    blocks = generate_nonoverlapping_blocks(months)

    if args.folds == "blocks":
        folds_to_run = blocks
        parts = OUTPUT_DIR / "bloc_b_wf_blocks"
        tag = "blocks"
    else:
        # overlapping ou all → un seul entraînement sur 81 plis
        folds_to_run = overlapping
        parts = OUTPUT_DIR / "bloc_b_wf_overlapping"
        tag = "overlapping"

    n_trials = args.lgbm_trials if args.lgbm_trials is not None else cfg.get("n_trials")
    logger.info(
        "Plan Bloc B | mode=%s | plis_à_entraîner=%s | blocs_indép=%s | "
        "models=%s | LGBM frozen=%s n_trials=%s | tune=%s | run=%s",
        args.folds,
        len(folds_to_run),
        len(blocks),
        list(MODEL_IDS),
        cfg.get("frozen_at"),
        n_trials,
        not args.no_tune,
        args.run,
    )
    if not args.run:
        print(
            f"Dry-run : {len(folds_to_run)} plis × {len(MODEL_IDS)} modèles. "
            "Relancer avec --run pour exécuter."
        )
        return 0

    results = run_bloc_b_walk_forward(
        df,
        folds_to_run,
        parts_dir=parts,
        model_ids=MODEL_IDS,
        retune_rf=not args.no_tune,
        retune_lgbm=not args.no_tune,
        lgbm_n_trials=args.lgbm_trials,
        resume=resume,
    )
    long_df = results["results_long"]
    out_csv = REPORTS_DIR / f"bloc_b_results_long_{tag}.csv"
    long_df.to_csv(out_csv, index=False)

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.folds,
        "n_folds_done": results["n_folds_done"],
        "model_ids": results["model_ids"],
        "retune": not args.no_tune,
        "lgbm_config": {
            "frozen_at": cfg.get("frozen_at"),
            "n_trials": n_trials,
            "version": cfg.get("version"),
        },
        "results_long": str(out_csv),
    }

    if args.folds in ("all", "overlapping") and not long_df.empty:
        summary["analysis"] = _postprocess(
            long_df,
            overlapping=overlapping,
            blocks=blocks,
            metric=args.omnibus_metric,
        )
        # Export sous-ensemble 14 blocs
        block_ids = summary["analysis"]["block_fold_ids"]
        blocks_csv = REPORTS_DIR / "bloc_b_results_long_blocks_from_overlapping.csv"
        long_df[long_df["fold_id"].isin(block_ids)].to_csv(blocks_csv, index=False)
        summary["results_long_blocks"] = str(blocks_csv)
    elif args.folds == "blocks" and not long_df.empty:
        summary["omnibus_14blocks"] = {
            m: run_omnibus_test(long_df, m) for m in OMNIBUS_METRICS
        }

    sum_path = REPORTS_DIR / f"bloc_b_summary_{tag if args.folds != 'all' else 'all'}.json"
    sum_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # Aperçu console
    if "analysis" in summary:
        om = summary["analysis"]["omnibus_14blocks"].get(args.omnibus_metric, {})
        print("\n=== Omnibus Friedman (14 blocs) ===")
        print(
            f"  metric={args.omnibus_metric} | "
            f"stat={om.get('stat')} | p={om.get('pvalue')} | "
            f"means={om.get('group_means')}"
        )
        print("\n=== Bootstrap IC95 Rank-IC (81 folds) ===")
        for mid, b in summary["analysis"]["bootstrap_81folds"][args.omnibus_metric].items():
            print(
                f"  {mid:22s} mean={b['mean']:.4f} "
                f"[{b['ci_low']:.4f}, {b['ci_high']:.4f}]"
            )

    print(f"\nTerminé : {out_csv} | {sum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
