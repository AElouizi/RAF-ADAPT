"""
Smoke test Bloc D — 1 bloc × 6 modèles (NSGA-III inchangé).

Usage :
    py -m experiments.factorial_hybrid_adapt.run_bloc_d_smoke_test
    py -m experiments.factorial_hybrid_adapt.run_bloc_d_smoke_test --fold-id 0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import LIQUIDITY_VMQ_THRESHOLD_MAD, load_env_file  # noqa: E402
from experiments.factorial_hybrid_adapt.bloc_d_multi_model import (  # noqa: E402
    run_portfolio_optimization_all_models,
)
from experiments.factorial_hybrid_adapt.paths import OUTPUT_DIR, REPORTS_DIR, ensure_output_dirs  # noqa: E402
from experiments.factorial_hybrid_adapt.stage2_allocation import (  # noqa: E402
    DEFAULT_L_MIN,
    Stage2Config,
)
from experiments.factorial_hybrid_adapt.stage2_bloc_b_loader import (  # noqa: E402
    BLOC_D_MODEL_IDS,
    block_fold_ids_from_summary,
)
from experiments.factorial_hybrid_adapt.walk_forward_folds import (  # noqa: E402
    available_months_from_dates,
    generate_nonoverlapping_blocks,
    generate_rolling_folds,
)

load_env_file(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke Bloc D — 1 bloc × 6 modèles")
    parser.add_argument("--fold-id", type=int, default=0, help="Bloc indépendant (fold overlapping id)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()
    block_ids = block_fold_ids_from_summary()
    if args.fold_id not in block_ids:
        logger.warning("fold_id=%s pas dans les 14 blocs standard %s", args.fold_id, block_ids)

    from experiments.factorial_hybrid_adapt.data_utils import load_ml_dataset  # noqa: E402

    months = available_months_from_dates(load_ml_dataset()["date_cours"])
    all_folds = generate_rolling_folds(months)
    blocks_all = generate_nonoverlapping_blocks(months)
    by_id = {f.fold_id: f for f in all_folds}
    fold = by_id.get(args.fold_id)
    if fold is None:
        raise IndexError(f"fold_id={args.fold_id} introuvable")

    cfg = Stage2Config(
        liquidity_mode="pareto",
        apply_vmq_filter=False,
        min_vmq=float(LIQUIDITY_VMQ_THRESHOLD_MAD),
        l_min=DEFAULT_L_MIN,
        sell_mode="exclude",
        selection_rule="knee",
    )

    out_dir = OUTPUT_DIR / "bloc_d_smoke"
    logger.info(
        "Smoke Bloc D | bloc fold=%s test=%s→%s | models=%s",
        fold.fold_id,
        fold.test_start_month,
        fold.test_end_month,
        list(BLOC_D_MODEL_IDS),
    )

    t0 = time.time()
    results = run_portfolio_optimization_all_models(
        BLOC_D_MODEL_IDS,
        [fold],
        cfg=cfg,
        parts_dir=out_dir,
        resume=not args.no_resume,
    )
    elapsed = time.time() - t0

    assert len(results["block_financials"]) == len(BLOC_D_MODEL_IDS), (
        f"Attendu {len(BLOC_D_MODEL_IDS)} runs, got {len(results['block_financials'])}"
    )

    long_path = REPORTS_DIR / "bloc_d_results_long_smoke.csv"
    summary_path = REPORTS_DIR / "bloc_d_summary_smoke.csv"
    json_path = REPORTS_DIR / "bloc_d_smoke_report.json"

    results["results_long"].to_csv(long_path, index=False)
    results["summary"].to_csv(summary_path, index=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke_1_block",
        "fold_id": fold.fold_id,
        "test_window": f"{fold.test_start_month}->{fold.test_end_month}",
        "model_ids": list(BLOC_D_MODEL_IDS),
        "n_models_done": len(results["block_financials"]),
        "elapsed_sec": round(elapsed, 1),
        "nsga_config": cfg.__dict__,
        "benchmark_summary": results.get("benchmark_summary", {}),
        "omnibus_financial": results.get("omnibus_financial", {}),
        "block_financials": results["block_financials"].to_dict(orient="records"),
        "note": "Validation pipeline — pas interprétation. Run 14 blocs : run_bloc_d_eval.py",
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n=== Bloc D smoke — OK ===")
    print(f"Bloc {fold.fold_id} | {len(BLOC_D_MODEL_IDS)} modèles | {elapsed:.0f}s\n")
    print(results["summary"].to_string(index=False))
    print(f"\nBenchmarks : {results.get('benchmark_summary', {})}")
    print(f"\nArtefacts :\n  - {long_path}\n  - {summary_path}\n  - {json_path}\n  - {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
