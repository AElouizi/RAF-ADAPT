"""
CLI Étage 2 — NSGA-III (3 variantes liquidité).

Variantes ``--liquidity-mode`` :
  eligibility  VMQ>=500k filtre + 2 obj (alpha, CVaR)  [protocole principal]
  pareto       pas de filtre + 3 obj (alpha, CVaR, max L)
  none         pas de filtre + 2 obj (alpha, CVaR)

Fenêtre commune : 2018-07 → 2025-06 ; exclusion SELL ; as-of t.

Usage :
  py -3 experiments/factorial_hybrid_adapt/run_stage2_nsga.py --liquidity-mode eligibility
  py -3 experiments/factorial_hybrid_adapt/run_stage2_nsga.py --liquidity-mode pareto
  py -3 experiments/factorial_hybrid_adapt/run_stage2_nsga.py --liquidity-mode none
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bvc_recommender.config import LIQUIDITY_VMQ_THRESHOLD_MAD
from experiments.factorial_hybrid_adapt.paths import OUTPUT_DIR, REPORTS_DIR, ensure_output_dirs
from experiments.factorial_hybrid_adapt.stage2_allocation import (
    DEFAULT_L_MIN,
    EVAL_END_MONTH,
    EVAL_START_MONTH,
    Stage2Config,
    run_stage2_all_cells,
    save_stage2_artifacts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODE_DEFAULT_OUT = {
    "eligibility": "portfolios_stage2_official",
    "pareto": "portfolios_stage2_liq_pareto",
    "none": "portfolios_stage2_liq_none",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Étage 2 NSGA-III — variantes liquidité")
    p.add_argument(
        "--liquidity-mode",
        choices=("eligibility", "pareto", "none"),
        default="pareto",
    )
    p.add_argument("--cells", type=str, default="1,2,3,4", help="ex. 1,2,3,4 ou 3")
    p.add_argument("--sell-mode", choices=("exclude", "penalize"), default="exclude")
    p.add_argument(
        "--selection",
        choices=("knee", "ideal_distance", "max_alpha", "min_cvar", "max_alpha_constrained"),
        default="knee",
    )
    p.add_argument("--pref-buy", type=float, default=1.0)
    p.add_argument("--pref-neutral", type=float, default=0.5)
    p.add_argument("--pref-sell", type=float, default=0.0)
    p.add_argument("--w-max", type=float, default=0.10)
    p.add_argument("--w-min", type=float, default=0.0)
    p.add_argument("--max-turnover", type=float, default=None)
    p.add_argument(
        "--min-vmq",
        type=float,
        default=float(LIQUIDITY_VMQ_THRESHOLD_MAD),
        help="Seuil d'éligibilité VMQ (MAD) si mode=eligibility ou --apply-vmq-filter.",
    )
    p.add_argument(
        "--apply-vmq-filter",
        action="store_true",
        help="Exclure VMQ < min_vmq (et VMQ NaN) même en mode pareto. Pas d'imputation.",
    )
    p.add_argument("--eval-start", type=str, default=EVAL_START_MONTH)
    p.add_argument("--eval-end", type=str, default=EVAL_END_MONTH)
    p.add_argument("--pop-size", type=int, default=36)
    p.add_argument("--n-gen", type=int, default=30)
    p.add_argument(
        "--max-months",
        type=int,
        default=None,
        help="Limite le nombre de mois (smoke test, après filtre de fenêtre)",
    )
    p.add_argument(
        "--inputs",
        type=str,
        default=str(REPORTS_DIR / "stage2_inputs_14blocks.parquet"),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Défaut selon liquidity-mode",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (OUTPUT_DIR / MODE_DEFAULT_OUT[args.liquidity_mode])
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    path = Path(args.inputs)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} manquant — lancer run_stage1_recommendations.py --folds 14"
        )
    stage2 = pd.read_parquet(path)
    stage2["date"] = pd.to_datetime(stage2["date"], errors="coerce")
    stage2["mois"] = stage2["date"].dt.to_period("M").astype(str)
    stage2 = stage2[
        (stage2["mois"] >= args.eval_start) & (stage2["mois"] <= args.eval_end)
    ].copy()
    logger.info(
        "Fenêtre figée %s→%s | mode=%s | n_rows=%s | n_mois=%s",
        args.eval_start,
        args.eval_end,
        args.liquidity_mode,
        len(stage2),
        stage2["mois"].nunique(),
    )

    if args.max_months is not None:
        keep_mois = sorted(stage2["mois"].dropna().unique())[: args.max_months]
        stage2 = stage2[stage2["mois"].isin(keep_mois)].copy()
        logger.info("Smoke : %s mois %s → %s", len(keep_mois), keep_mois[0], keep_mois[-1])

    cells = tuple(int(x) for x in args.cells.split(",") if x.strip())
    cfg = Stage2Config(
        sell_mode=args.sell_mode,
        pref_BUY=args.pref_buy,
        pref_NEUTRAL=args.pref_neutral,
        pref_SELL=args.pref_sell,
        w_min=args.w_min,
        w_max=args.w_max,
        max_turnover=args.max_turnover,
        liquidity_mode=args.liquidity_mode,
        apply_vmq_filter=bool(args.apply_vmq_filter),
        min_vmq=float(args.min_vmq),
        l_min=DEFAULT_L_MIN,
        selection_rule=args.selection,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        eval_start_month=args.eval_start,
        eval_end_month=args.eval_end,
    )

    objs = "alpha, CVaR, max L" if args.liquidity_mode == "pareto" else "alpha, CVaR"
    print(f"\n=== Étage 2 — NSGA-III (liquidity_mode={cfg.liquidity_mode}) ===")
    print(
        f"fenêtre={cfg.eval_start_month}→{cfg.eval_end_month} | "
        f"min_vmq={cfg.min_vmq:.0f} apply_filter={cfg.apply_vmq_filter} "
        f"(eligibility ou --apply-vmq-filter)"
    )
    print(
        f"sell_mode={cfg.sell_mode} | selection={cfg.selection_rule} | "
        f"prefs BUY/NEU/SELL={cfg.pref_BUY}/{cfg.pref_NEUTRAL}/{cfg.pref_SELL}"
    )
    print(
        f"objectifs NSGA: {objs} | "
        f"w_min={cfg.w_min} w_max={cfg.w_max} | pop={cfg.pop_size} gen={cfg.n_gen}"
    )
    print(f"cells={cells} | inputs={path.name} | n_rows={len(stage2):,} | out={out_dir.name}")

    results = run_stage2_all_cells(stage2, cell_ids=cells, cfg=cfg)
    paths = save_stage2_artifacts(results, out_dir)

    meta = results["month_meta"]
    print(f"\n{'cell':>4}  {'mois':>5}  {'ok':>4}  {'fallback':>8}  {'pos_moy':>8}")
    print("-" * 40)
    for cell, grp in meta.groupby("cell"):
        n_ok = int((grp["status"] == "ok").sum())
        n_fb = int((grp["status"] != "ok").sum())
        pos = pd.to_numeric(grp["n_positions_selected"], errors="coerce").mean()
        print(f"{int(cell):>4}  {len(grp):>5}  {n_ok:>4}  {n_fb:>8}  {pos:8.1f}")

    print("\nArtefacts :")
    for k, p in paths.items():
        print(f"  - {k}: {p}")
    print(f"Terminé : {datetime.now(timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
