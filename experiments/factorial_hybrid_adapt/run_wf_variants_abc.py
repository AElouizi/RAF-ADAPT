"""
Lance les variantes A/B/C puis tableaux IC + DM (14 blocs).

Usage :
    python -m experiments.factorial_hybrid_adapt.run_wf_variants_abc
    python -m experiments.factorial_hybrid_adapt.run_wf_variants_abc --only B
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
from experiments.factorial_hybrid_adapt.fix_return_dispersion_z import (  # noqa: E402
    build_dispersion_fixed_dataset,
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
    available_months_from_dates,
    generate_rolling_folds,
)
from experiments.factorial_hybrid_adapt.walk_forward_variants import (  # noqa: E402
    run_variant,
)

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _dm_table(metrics: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    keys = set(zip(blocks["test_start_month"], blocks["test_end_month"]))
    sub = metrics[
        metrics.apply(
            lambda r: (r["test_start_month"], r["test_end_month"]) in keys, axis=1
        )
    ].copy()
    wide = sub.pivot_table(index="fold_id", columns="cell_id", values="rank_ic")
    loss = -wide
    rows = []
    for ref, alt, label in (
        (1, 2, "adaptation C2vsC1"),
        (1, 3, "hybridation C3vsC1"),
        (1, 4, "combine C4vsC1"),
    ):
        dm = diebold_mariano(loss[ref].values, loss[alt].values, h=1)
        rows.append(
            {
                "pair": label,
                "delta_IC": float(wide[alt].mean() - wide[ref].mean()),
                "dm_stat_hln": dm["dm_stat_hln"],
                "pvalue_hln": dm["pvalue_hln"],
                "sig_5pct": bool(
                    np.isfinite(dm["pvalue_hln"]) and dm["pvalue_hln"] < 0.05
                ),
                "n": dm["n"],
            }
        )
    return pd.DataFrame(rows)


def _agg_ic(metrics: pd.DataFrame) -> pd.DataFrame:
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


def evaluate_and_save(variant: str, metrics: pd.DataFrame) -> dict:
    blocks = pd.read_csv(REPORTS_DIR / "wf_folds_nonoverlapping.csv")
    agg = _agg_ic(metrics)
    dm = _dm_table(metrics, blocks)
    out = {
        "variant": variant,
        "aggregation": agg.to_dict(orient="records"),
        "dm_14_blocks": dm.to_dict(orient="records"),
    }
    agg.to_csv(REPORTS_DIR / f"wf_variant_{variant}_ic.csv", index=False)
    dm.to_csv(REPORTS_DIR / f"wf_variant_{variant}_dm.csv", index=False)
    (REPORTS_DIR / f"wf_variant_{variant}_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n===== VARIANTE {variant} — IC (81 plis) =====")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n===== VARIANTE {variant} — DM (14 blocs) =====")
    print(dm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["A", "B", "C", "ALL"],
        default="ALL",
        help="Lancer une variante ou toutes",
    )
    args = parser.parse_args()
    ensure_output_dirs()

    logger.info("Chargement panel…")
    base = attach_regime_to_panel(load_ml_dataset())
    months = available_months_from_dates(base["date_cours"])
    folds = generate_rolling_folds(months)

    # Dataset B/C : dispersion corrigée
    fixed_path = OUTPUT_DIR / "ml_dataset_return_dispersion_z_fixed.parquet"
    if not fixed_path.is_file():
        logger.info("Construction dataset return_dispersion_z corrigé…")
        fixed = build_dispersion_fixed_dataset(load_ml_dataset())
        fixed = attach_regime_to_panel(fixed)
        fixed.to_parquet(fixed_path, index=False)
        logger.info(
            "Saved %s | return_dispersion_z non-null=%.1f%%",
            fixed_path,
            100 * fixed["return_dispersion_z"].notna().mean(),
        )
    else:
        fixed = pd.read_parquet(fixed_path)
        logger.info("Dataset fixé rechargé : %s", fixed_path)

    variants = []
    if args.only in ("A", "ALL"):
        variants.append(("A", base, True, "retune_rf_light"))
    if args.only in ("B", "ALL"):
        variants.append(("B", fixed, False, "dispersion_z_fixed_only"))
    if args.only in ("C", "ALL"):
        variants.append(("C", fixed, True, "retune_rf_and_dispersion_z"))

    # Baseline déjà calculée
    if args.only == "ALL":
        base_metrics = pd.read_csv(REPORTS_DIR / "wf_etape3_fold_metrics.csv")
        print("\n===== BASELINE (HP fixes, disp_z NaN) — rappel =====")
        evaluate_and_save("baseline", base_metrics)

    results = {}
    for code, df, retune, slug in variants:
        parts = OUTPUT_DIR / f"wf_variant_{code}_{slug}"
        logger.info("=== START variante %s retune=%s ===", code, retune)
        out = run_variant(
            df, folds, variant_name=code, parts_dir=parts, retune_rf=retune, resume=True
        )
        metrics_path = REPORTS_DIR / f"wf_variant_{code}_fold_metrics.csv"
        out["metrics"].to_csv(metrics_path, index=False)
        if not out["tune_log"].empty:
            out["tune_log"].to_csv(
                REPORTS_DIR / f"wf_variant_{code}_rf_tune_log.csv", index=False
            )
        results[code] = evaluate_and_save(code, out["metrics"])

    # Tableau comparatif final
    if len(results) >= 1:
        comp_rows = []
        sources = {}
        if (REPORTS_DIR / "wf_variant_baseline_ic.csv").is_file():
            sources["baseline"] = pd.read_csv(REPORTS_DIR / "wf_variant_baseline_ic.csv")
        for code in results:
            sources[code] = pd.read_csv(REPORTS_DIR / f"wf_variant_{code}_ic.csv")
        for name, agg in sources.items():
            for _, r in agg.iterrows():
                comp_rows.append(
                    {
                        "variant": name,
                        "cellule": int(r["cellule"]),
                        "IC_mean": r["IC_mean"],
                        "IC_median": r["IC_median"],
                    }
                )
        comp = pd.DataFrame(comp_rows)
        comp.to_csv(REPORTS_DIR / "wf_variants_ABC_comparison_ic.csv", index=False)

        dm_rows = []
        for name in sources:
            dm_path = REPORTS_DIR / f"wf_variant_{name}_dm.csv"
            if dm_path.is_file():
                d = pd.read_csv(dm_path)
                d.insert(0, "variant", name)
                dm_rows.append(d)
        if dm_rows:
            pd.concat(dm_rows, ignore_index=True).to_csv(
                REPORTS_DIR / "wf_variants_ABC_comparison_dm.csv", index=False
            )

    print(f"\nTermine : {datetime.now(timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
