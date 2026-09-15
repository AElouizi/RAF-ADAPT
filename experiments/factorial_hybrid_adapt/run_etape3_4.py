"""
CLI Étapes 3–4 — plan factoriel + sélection top-25 mensuelle.

Usage :
    python -m experiments.factorial_hybrid_adapt.run_etape3_4
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import load_env_file  # noqa: E402
from experiments.factorial_hybrid_adapt.data_utils import (  # noqa: E402
    attach_regime_to_panel,
    load_ml_dataset,
)
from experiments.factorial_hybrid_adapt.factorial_cells import (  # noqa: E402
    CELLS,
    TOP_N,
    run_factorial_stage1,
    save_stage1_artifacts,
)
from experiments.factorial_hybrid_adapt.paths import (  # noqa: E402
    MODELS_DIR,
    REPORTS_DIR,
    SELECTIONS_DIR,
    ensure_output_dirs,
)
from experiments.factorial_hybrid_adapt.random_forest_scorer import (  # noqa: E402
    RFHyperParams,
)

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    ensure_output_dirs()
    logger.info("Chargement ml_dataset + régime…")
    df = load_ml_dataset()
    df = attach_regime_to_panel(df)
    logger.info(
        "Panel : %s lignes | régime bull=%s neutral=%s bear=%s",
        len(df),
        int(df["is_bull"].sum()),
        int(df["is_neutral"].sum()),
        int(df["is_bear"].sum()),
    )

    # Hyperparams RF = meilleurs de l'étape 2 (pas de re-tuning pour éviter fuite / coût)
    rf_params = RFHyperParams(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=20,
    )
    results = run_factorial_stage1(df, top_n=TOP_N, rf_params=rf_params)
    paths = save_stage1_artifacts(
        results,
        reports_dir=REPORTS_DIR,
        selections_dir=SELECTIONS_DIR,
        models_dir=MODELS_DIR,
    )

    print("\n=== Étapes 3–4 — Plan factoriel (étage 1) ===")
    print(f"Top-N fixe : {TOP_N}")
    print(f"RF params  : {results['rf_params']}")
    print()
    hdr = (
        f"{'cellule':>7}  {'modele':<8}  {'regime':>6}  "
        f"{'IC_val':>8}  {'IC_test':>8}  {'hit_val':>8}  {'hit_test':>8}  "
        f"{'top25_val':>9}  {'top25_test':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for cell in CELLS:
        data = results["cells"][cell.cell_id]
        mv = data["splits"]["val"]["metrics"]
        mt = data["splits"]["test"]["metrics"]
        ok_v = data["splits"]["val"]["top_n_ok"]
        ok_t = data["splits"]["test"]["top_n_ok"]
        n_v = data["splits"]["val"]["top_n_months"]
        n_t = data["splits"]["test"]["top_n_months"]
        print(
            f"{cell.cell_id:>7}  {cell.model:<8}  {str(cell.regime):>6}  "
            f"{mv.get('rank_ic_mean', float('nan')):8.4f}  "
            f"{mt.get('rank_ic_mean', float('nan')):8.4f}  "
            f"{100 * (mv.get('hit_ratio') or 0):7.1f}%  "
            f"{100 * (mt.get('hit_ratio') or 0):7.1f}%  "
            f"{'OK' if ok_v else 'KO'}({n_v}m)  "
            f"{'OK' if ok_t else 'KO'}({n_t}m)"
        )

    # Contrôle cardinalité détaillé
    print("\nContrôle top-25 / mois :")
    sel = results["selections"]
    for cell_id in sorted(sel["cell_id"].unique()):
        for split_name in ("val", "test"):
            sub = sel[(sel["cell_id"] == cell_id) & (sel["split"] == split_name)]
            counts = sub.groupby("mois").size()
            print(
                f"  C{cell_id}/{split_name}: mois={len(counts)} | "
                f"min={counts.min() if len(counts) else 0} | "
                f"max={counts.max() if len(counts) else 0} | "
                f"tous=25? {(counts == TOP_N).all() if len(counts) else False}"
            )

    print("\nArtefacts :")
    for k, p in paths.items():
        print(f"  - {k}: {p}")
    print(f"Terminé : {datetime.now(timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
