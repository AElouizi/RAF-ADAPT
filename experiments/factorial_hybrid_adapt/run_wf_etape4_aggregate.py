"""
CLI Étape 4 WF — agrégation descriptive sur plis chevauchants.

Usage :
    python -m experiments.factorial_hybrid_adapt.run_wf_etape4_aggregate
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import load_env_file  # noqa: E402
from experiments.factorial_hybrid_adapt.paths import REPORTS_DIR, ensure_output_dirs  # noqa: E402
from experiments.factorial_hybrid_adapt.walk_forward_aggregate import (  # noqa: E402
    run_etape4_aggregation,
)

load_env_file(ROOT / ".env")


def main() -> int:
    ensure_output_dirs()
    metrics_path = REPORTS_DIR / "wf_etape3_fold_metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Manquant : {metrics_path} — lancer etape 3.")

    results = run_etape4_aggregation(metrics_path, REPORTS_DIR)
    agg = results["aggregation"]
    effects = results["effects"]

    print("=== Walk-forward Etape 4 — Aggregation (plis chevauchants) ===")
    print("Perimetre : metriques DESCRIPTIVES sur 81 plis mensuels chevauchants.")
    print("DM / Sharpe portefeuille = blocs non-chevauchants (etapes 5-6).\n")

    cols = [
        "cellule",
        "modele",
        "regime",
        "IC_mean",
        "IC_median",
        "IC_std",
        "pct_folds_IC_positive",
        "hit_mean",
        "hit_std",
    ]
    print(agg[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nEffets factoriels sur IC_mean (descriptif) :")
    print(f"  adaptation   (C2-C1) = {effects['effet_adaptation']:+.5f}")
    print(f"  hybridation  (C3-C1) = {effects['effet_hybridation']:+.5f}")
    print(f"  interaction          = {effects['effet_interaction']:+.5f}")

    # Rang des cellules
    ranking = agg.sort_values("IC_mean", ascending=False)
    print("\nClassement par IC_mean :")
    for i, r in enumerate(ranking.itertuples(), start=1):
        print(f"  {i}. C{int(r.cellule)} ({r.modele}, regime={r.regime}) IC={r.IC_mean:.4f}")

    print("\nArtefacts :")
    for k, p in results["paths"].items():
        print(f"  - {k}: {p}")
    print(f"Termine : {datetime.now(timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
