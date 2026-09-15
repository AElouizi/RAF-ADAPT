"""
CLI Étape 6 — métriques portefeuille (Sharpe, Sortino, CVaR, liquidité, turnover).

Usage :
    python -m experiments.factorial_hybrid_adapt.run_etape6_metrics
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import load_env_file  # noqa: E402
from experiments.factorial_hybrid_adapt.paths import (  # noqa: E402
    OUTPUT_DIR,
    REPORTS_DIR,
    ensure_output_dirs,
)
from experiments.factorial_hybrid_adapt.stage2_metrics import (  # noqa: E402
    compute_stage2_metrics,
    save_stage2_artifacts,
)

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    ensure_output_dirs()
    weights_path = OUTPUT_DIR / "portfolios" / "portfolios_weights.parquet"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Poids manquants : {weights_path} — lancer étape 5.")

    weights = pd.read_parquet(weights_path)
    logger.info("Poids chargés : %s lignes", len(weights))

    results = compute_stage2_metrics(weights)
    paths = save_stage2_artifacts(
        results,
        reports_dir=REPORTS_DIR,
        stage1_summary_path=REPORTS_DIR / "factorial_stage1_summary.csv",
    )

    combined = pd.read_csv(paths["combined"])
    print("\n=== Étape 6 — Métriques étage 2 (P_equilibre, test, frais 0.3%) ===")
    cols = [
        c
        for c in (
            "cellule",
            "modele",
            "regime",
            "IC_val",
            "IC_test",
            "hit_ratio",
            "sharpe",
            "sortino",
            "CVaR",
            "liquidite_moy",
            "turnover",
        )
        if c in combined.columns
    ]
    print(combined[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Alertes résultats aberrants
    for _, row in combined.iterrows():
        ic = row.get("IC_test")
        sh = row.get("sharpe")
        if pd.notna(ic) and abs(float(ic)) > 0.5:
            print(f"⚠ C{int(row['cellule'])}: IC_test={ic:.3f} > 0.5 en absolu — à vérifier")
        if pd.notna(sh) and abs(float(sh)) > 5:
            print(f"⚠ C{int(row['cellule'])}: Sharpe={sh:.2f} > 5 en absolu — à vérifier")

    print("\nArtefacts :")
    for k, p in paths.items():
        print(f"  - {k}: {p}")
    print(f"Terminé : {datetime.now(timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
