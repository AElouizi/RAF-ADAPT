"""
CLI Étape 5 — allocation NSGA-III mensuelle pour les 4 cellules.

Usage :
    python -m experiments.factorial_hybrid_adapt.run_etape5_allocation
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
from experiments.factorial_hybrid_adapt.allocation import (  # noqa: E402
    NSGA_GEN_WALKFORWARD,
    NSGA_POP_WALKFORWARD,
    run_allocation_all_cells,
    save_allocation_artifacts,
)
from experiments.factorial_hybrid_adapt.paths import (  # noqa: E402
    OUTPUT_DIR,
    SELECTIONS_DIR,
    ensure_output_dirs,
)

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_all_selections() -> pd.DataFrame:
    frames = []
    for cell_id in (1, 2, 3, 4):
        path = SELECTIONS_DIR / f"cell{cell_id}_top25.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Sélection manquante : {path}")
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ensure_output_dirs()
    portfolios_dir = OUTPUT_DIR / "portfolios"
    portfolios_dir.mkdir(parents=True, exist_ok=True)

    selections = load_all_selections()
    logger.info(
        "Sélections chargées : %s lignes | cells=%s",
        len(selections),
        sorted(selections["cell_id"].unique().tolist()),
    )

    results = run_allocation_all_cells(
        selections,
        split="test",
        pop_size=NSGA_POP_WALKFORWARD,
        n_gen=NSGA_GEN_WALKFORWARD,
    )
    paths = save_allocation_artifacts(results, output_dir=portfolios_dir)

    meta = results["month_meta"]
    print("\n=== Étape 5 — Allocation NSGA-III (test) ===")
    print(
        f"Params (identiques 4 cellules) : pop={NSGA_POP_WALKFORWARD}, "
        f"gen={NSGA_GEN_WALKFORWARD} (mode walk-forward projet)"
    )
    print(f"{'cellule':>7}  {'mois':>5}  {'ok':>4}  {'fallback':>8}  "
          f"{'pos_moy':>8}  {'sum_w_moy':>9}")
    print("-" * 55)
    for cell_id, grp in meta.groupby("cell_id"):
        n_ok = int((grp["status"] == "ok").sum())
        n_fb = int((grp["status"] != "ok").sum())
        pos = pd.to_numeric(grp["n_positions_equilibre"], errors="coerce").mean()
        sw = pd.to_numeric(grp["sum_weights_equilibre"], errors="coerce").mean()
        print(
            f"{cell_id:>7}  {len(grp):>5}  {n_ok:>4}  {n_fb:>8}  "
            f"{pos:8.1f}  {sw:9.4f}"
        )

    # Contrôle : chaque mois a bien des poids P_equilibre
    w = results["weights"]
    eq = w[w["portfolio"] == "P_equilibre"]
    print("\nContrôle P_equilibre (titres pondérés / mois) :")
    for cell_id, grp in eq.groupby("cell_id"):
        counts = grp.groupby("mois")["ticker"].nunique()
        print(
            f"  C{cell_id}: mois={len(counts)} | "
            f"pos min={counts.min()} max={counts.max()} mean={counts.mean():.1f}"
        )

    print("\nArtefacts :")
    for k, p in paths.items():
        print(f"  - {k}: {p}")
    print(f"Terminé : {datetime.now(timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
