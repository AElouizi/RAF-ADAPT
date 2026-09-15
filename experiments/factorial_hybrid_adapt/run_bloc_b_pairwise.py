"""
Contrasts pairwise Bloc B — 14 blocs indépendants.

Usage :
    py -m experiments.factorial_hybrid_adapt.run_bloc_b_pairwise
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.factorial_hybrid_adapt.bloc_b_metrics import run_pairwise_battery  # noqa: E402
from experiments.factorial_hybrid_adapt.paths import REPORTS_DIR, ensure_output_dirs  # noqa: E402

# Paires demandées + compléments utiles
DEFAULT_PAIRS: list[tuple[str, str, str]] = [
    ("c3_lightgbm", "c1_ridge", "C3 LightGBM vs C1 Ridge"),
    ("c4_hybrid_tri", "c3_lightgbm", "C4 Hybride triple vs C3 LightGBM"),
    ("c5_hybrid_tri_regime", "c4_hybrid_tri", "C5 Hybride+régime vs C4 Hybride"),
    ("c2_rf", "c1_ridge", "C2 RF vs C1 Ridge"),
    ("c3_lightgbm", "c2_rf", "C3 LightGBM vs C2 RF"),
]


def main() -> int:
    ensure_output_dirs()
    blocks_path = REPORTS_DIR / "bloc_b_results_long_blocks_from_overlapping.csv"
    if not blocks_path.is_file():
        blocks_path = REPORTS_DIR / "bloc_b_results_long_overlapping.csv"
    df = pd.read_csv(blocks_path)
    n_blocks = int(df["fold_id"].nunique())
    print(f"=== Pairwise Bloc B — {n_blocks} blocs indépendants ===\n")

    table = run_pairwise_battery(df, DEFAULT_PAIRS)
    out_csv = REPORTS_DIR / "bloc_b_pairwise_14blocks.csv"
    out_json = REPORTS_DIR / "bloc_b_pairwise_14blocks.json"
    table.to_csv(out_csv, index=False)

    # Console : Rank-IC uniquement
    ic = table[table["metric"] == "rank_ic"].copy()
    print("Rank-IC (Wilcoxon + DM-HLN, n=14 blocs) :\n")
    for _, r in ic.iterrows():
        sig_w = "OUI" if r["sig_5pct_wilcoxon"] else "non"
        sig_d = "OUI" if r["sig_5pct_dm_hln"] else "non"
        print(
            f"  {r['label']}\n"
            f"    delta_IC={r['delta_mean']:+.4f} (med={r['delta_median']:+.4f}) | "
            f"Wilcoxon p={r['pvalue_wilcoxon']:.4f} ({sig_w}) | "
            f"DM-HLN p={r['pvalue_dm_hln']:.4f} ({sig_d})\n"
        )

    spread = table[table["metric"] == "spread_BUY_SELL"].copy()
    print("Spread BUY-SELL :\n")
    for _, r in spread.iterrows():
        if r["pair"] in {
            "c3_lightgbm_vs_c1_ridge",
            "c4_hybrid_tri_vs_c3_lightgbm",
            "c5_hybrid_tri_regime_vs_c4_hybrid_tri",
        }:
            print(
                f"  {r['label']}: delta={r['delta_mean']:+.4f} | "
                f"Wilcoxon p={r['pvalue_wilcoxon']:.4f}"
            )

    payload = {
        "scope": f"{n_blocks}_blocs_independants",
        "source": str(blocks_path),
        "pairs": [{"ref": a, "alt": b, "label": lab} for a, b, lab in DEFAULT_PAIRS],
        "results": table.to_dict(orient="records"),
        "recommendation_note": (
            "Décision Stage 2 : privilégier le modèle avec delta Rank-IC positif "
            "ET significativité (Wilcoxon ou DM-HLN @5%) sur 14 blocs, "
            "confirmé par bootstrap IC95 sur 81 folds."
        ),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nArtefacts :\n  - {out_csv}\n  - {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
