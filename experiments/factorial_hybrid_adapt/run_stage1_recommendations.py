"""
CLI — Étage 1 : recommandations BUY/NEUTRAL/SELL (univers complet) + métriques.

Règle (voir recommendations.py) :
  excess = predicted_alpha - benchmark  (benchmark=0 par défaut)
  BUY / NEUTRAL / SELL selon ±tau
  tau : expanding_past_quantile (plis antérieurs, anti look-ahead)

Sorties (outputs/reports/) :
  - stage1_recommendations_14blocks.parquet
  - stage1_metrics_by_cell_14blocks.csv
  - stage1_compare_C1C4_14blocks.csv
  - stage1_lookahead_checks.json
  - stage2_inputs_14blocks.parquet  (score + reco + risque + liquidité)
  - (option) stage1_*_81overlapping.*
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.factorial_hybrid_adapt.paths import REPORTS_DIR, ensure_output_dirs
from experiments.factorial_hybrid_adapt.recommendations import (
    RecommendationConfig,
    build_recommendations_walk_forward,
    compare_cells_table,
    look_ahead_checklist,
    merge_fold_signal_metrics,
)
from experiments.factorial_hybrid_adapt.stage2_interface import (
    build_stage2_inputs,
    export_stage2_inputs,
)


def _load_preds() -> pd.DataFrame:
    path = REPORTS_DIR / "wf_stage1_predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Prédictions WF introuvables : {path}")
    return pd.read_parquet(path)


def _fold_ids(kind: str) -> list[int]:
    """
    Retourne les fold_id du parquet de prédictions.

    Important : les 14 blocs non-chevauchants ne sont PAS fold_id 0..13,
    mais les plis WF dont (test_start_month, test_end_month) matchent
    ``wf_folds_nonoverlapping.csv`` (ex. 0, 6, 12, …, 78).
    """
    metrics = pd.read_csv(REPORTS_DIR / "wf_etape3_fold_metrics.csv")
    if kind == "14":
        blocks = pd.read_csv(REPORTS_DIR / "wf_folds_nonoverlapping.csv")
        keys = set(zip(blocks["test_start_month"], blocks["test_end_month"]))
        sub = metrics[
            metrics.apply(
                lambda r: (r["test_start_month"], r["test_end_month"]) in keys,
                axis=1,
            )
        ]
        return sorted(int(x) for x in sub["fold_id"].unique())
    if kind == "81":
        return sorted(int(x) for x in metrics["fold_id"].unique())
    raise ValueError(kind)


def run_panel(
    preds: pd.DataFrame,
    fold_ids: list[int],
    cfg: RecommendationConfig,
    *,
    tag: str,
    build_stage2: bool,
) -> dict:
    ensure_output_dirs()
    print(f"\n=== Panel recommandations ({tag}) | n_folds={len(fold_ids)} ===")
    print(
        f"benchmark={cfg.benchmark_mode} | tau_method={cfg.tau_method} | "
        f"tau_q={cfg.tau_quantile} | tau_fixed={cfg.tau_fixed}"
    )

    recos = build_recommendations_walk_forward(preds, fold_ids=fold_ids, cfg=cfg)
    reco_path = REPORTS_DIR / f"stage1_recommendations_{tag}.parquet"
    recos.to_parquet(reco_path, index=False)
    print(f"Recommandations : {reco_path} ({len(recos):,} lignes)")

    compare = compare_cells_table(recos)
    fold_metrics = pd.read_csv(REPORTS_DIR / "wf_etape3_fold_metrics.csv")
    compare = merge_fold_signal_metrics(compare, fold_metrics, fold_ids)

    metrics_path = REPORTS_DIR / f"stage1_metrics_by_cell_{tag}.csv"
    compare.to_csv(metrics_path, index=False)

    # Table demandée (sous-ensemble de colonnes)
    display_cols = [
        "Configuration",
        "IC",
        "Hit",
        "% BUY",
        "% Neutral",
        "% SELL",
        "Precision BUY",
        "Precision SELL",
    ]
    table = compare[[c for c in display_cols if c in compare.columns]].copy()
    table_path = REPORTS_DIR / f"stage1_compare_C1C4_{tag}.csv"
    table.to_csv(table_path, index=False)
    print("\n--- Comparaison C1–C4 ---")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(table.to_string(index=False))

    checks = look_ahead_checklist(cfg)
    checks_path = REPORTS_DIR / f"stage1_lookahead_checks_{tag}.json"
    checks_path.write_text(
        json.dumps(
            {
                "tag": tag,
                "config": {
                    "benchmark_mode": cfg.benchmark_mode,
                    "tau_method": cfg.tau_method,
                    "tau_quantile": cfg.tau_quantile,
                    "tau_fixed": cfg.tau_fixed,
                },
                "checks": checks,
                "notes": [
                    "predicted_return = alpha_ajuste_risque prédit (déjà vs MASI / vol).",
                    "benchmark_return = 0 sous zero_alpha.",
                    "realized_alpha utilisé uniquement pour Precision / mean alpha.",
                    "IC/Hit de la table = moyenne des plis WF journaliers (fold_metrics).",
                    "% BUY/NEUTRAL/SELL et Precision = panel mois-fin (rebalancement).",
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Look-ahead checks : {checks_path}")

    stage2_path = None
    if build_stage2:
        print("Enrichissement risque/liquidité pour étage 2…")
        s2 = build_stage2_inputs(recos)
        stage2_path = REPORTS_DIR / f"stage2_inputs_{tag}.parquet"
        export_stage2_inputs(s2, stage2_path)
        print(f"Stage2 inputs : {stage2_path} ({len(s2):,} lignes)")

    return {
        "reco_path": str(reco_path),
        "table_path": str(table_path),
        "metrics_path": str(metrics_path),
        "checks_path": str(checks_path),
        "stage2_path": str(stage2_path) if stage2_path else None,
        "n_rows": len(recos),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Étage 1 — recommandations universelles")
    p.add_argument(
        "--folds",
        choices=("14", "81", "both"),
        default="14",
        help="14 blocs non-chevauchants (principal), 81 chevauchants (robustesse), ou both",
    )
    p.add_argument(
        "--benchmark",
        choices=("zero_alpha", "cs_median"),
        default="zero_alpha",
    )
    p.add_argument(
        "--tau-method",
        choices=("expanding_past_quantile", "fixed", "train_abs_quantile"),
        default="expanding_past_quantile",
    )
    p.add_argument("--tau-fixed", type=float, default=0.10)
    p.add_argument("--tau-quantile", type=float, default=0.33)
    p.add_argument("--no-stage2", action="store_true", help="Ne pas construire stage2_inputs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.tau_method == "train_abs_quantile":
        print(
            "ERREUR : train_abs_quantile nécessite les prédictions train par pli "
            "(non stockées dans wf_stage1_predictions). "
            "Utiliser expanding_past_quantile ou fixed."
        )
        return 1

    cfg = RecommendationConfig(
        benchmark_mode=args.benchmark,
        tau_method=args.tau_method,
        tau_fixed=args.tau_fixed,
        tau_quantile=args.tau_quantile,
    )
    preds = _load_preds()
    # Renommer cible pour generate_recommendations
    if "alpha_ajuste_risque" in preds.columns and "realized_alpha" not in preds.columns:
        pass  # TARGET_COLUMN déjà = alpha_ajuste_risque

    tags = []
    if args.folds in ("14", "both"):
        tags.append(("14blocks", _fold_ids("14")))
    if args.folds in ("81", "both"):
        tags.append(("81overlapping", _fold_ids("81")))

    build_s2 = not args.no_stage2
    for tag, fids in tags:
        run_panel(preds, fids, cfg, tag=tag, build_stage2=build_s2)

    print("\nTerminé.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
