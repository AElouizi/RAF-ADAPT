"""
CLI Bloc D — NSGA-III × 6 modèles × 14 blocs (dry-run par défaut).

Usage :
    py -m experiments.factorial_hybrid_adapt.run_bloc_d_eval --run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import LIQUIDITY_VMQ_THRESHOLD_MAD, load_env_file  # noqa: E402
from experiments.factorial_hybrid_adapt.bloc_d_multi_model import (  # noqa: E402
    run_portfolio_optimization_all_models,
)
from experiments.factorial_hybrid_adapt.data_utils import load_ml_dataset  # noqa: E402
from experiments.factorial_hybrid_adapt.paths import OUTPUT_DIR, REPORTS_DIR, ensure_output_dirs  # noqa: E402
from experiments.factorial_hybrid_adapt.stage2_allocation import DEFAULT_L_MIN, Stage2Config  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Bloc D multi-modèles")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()
    months = available_months_from_dates(load_ml_dataset()["date_cours"])
    block_ids = block_fold_ids_from_summary()
    all_folds = generate_rolling_folds(months)
    by_id = {f.fold_id: f for f in all_folds}
    blocks = [by_id[fid] for fid in block_ids if fid in by_id]

    cfg = Stage2Config(
        liquidity_mode="pareto",
        apply_vmq_filter=False,
        min_vmq=float(LIQUIDITY_VMQ_THRESHOLD_MAD),
        l_min=DEFAULT_L_MIN,
    )

    logger.info(
        "Plan Bloc D | %s blocs × %s modèles | NSGA pop=%s gen=%s | run=%s",
        len(blocks),
        len(BLOC_D_MODEL_IDS),
        cfg.pop_size,
        cfg.n_gen,
        args.run,
    )
    if not args.run:
        print(f"Dry-run : {len(blocks)} blocs × {len(BLOC_D_MODEL_IDS)} modèles. Relancer avec --run.")
        return 0

    results = run_portfolio_optimization_all_models(
        BLOC_D_MODEL_IDS,
        blocks,
        cfg=cfg,
        parts_dir=OUTPUT_DIR / "bloc_d_wf_14blocks",
        resume=not args.no_resume,
    )

    long_path = REPORTS_DIR / "bloc_d_results_long_14blocks.csv"
    summary_path = REPORTS_DIR / "bloc_d_summary_14blocks.csv"
    json_path = REPORTS_DIR / "bloc_d_report_14blocks.json"

    results["results_long"].to_csv(long_path, index=False)
    results["summary"].to_csv(summary_path, index=False)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_blocks": len(blocks),
        "model_ids": list(BLOC_D_MODEL_IDS),
        "benchmark_summary": results.get("benchmark_summary", {}),
        "omnibus_financial": results.get("omnibus_financial", {}),
        "summary": results["summary"].to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Terminé : {long_path} | {summary_path} | {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
