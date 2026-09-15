"""
Comparaisons appariées 2x2 (Hybridation x Adaptation) — 14 blocs non-chevauchants.

Contrastes :
  (1) regime | Ridge   : C2 - C1
  (2) regime | Hybrid  : C4 - C3
  (3) Hybrid vs Ridge  : C3 - C1   (sans regime)
  (4) interaction      : C4 - C3 - (C2 - C1) = C4 - C2 - C3 + C1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.factorial_hybrid_adapt.paths import REPORTS_DIR, ensure_output_dirs


CONTRASTS = [
    ("regime_on_ridge", "C2-C1", "Effet regime | Ridge", (2, 1)),
    ("regime_on_hybrid", "C4-C3", "Effet regime | Hybrid", (4, 3)),
    ("hybrid_vs_ridge", "C3-C1", "Hybrid vs Ridge (sans regime)", (3, 1)),
    ("interaction", "C4-C3-(C2-C1)", "Interaction modele x regime", None),
]


def _paired_series(wide: pd.DataFrame, key: str) -> pd.Series:
    if key == "regime_on_ridge":
        return wide[2] - wide[1]
    if key == "regime_on_hybrid":
        return wide[4] - wide[3]
    if key == "hybrid_vs_ridge":
        return wide[3] - wide[1]
    if key == "interaction":
        return wide[4] - wide[3] - (wide[2] - wide[1])
    raise KeyError(key)


def summarize_contrast(d: pd.Series, name: str, formula: str, label: str) -> dict:
    x = d.dropna().astype(float).values
    n = len(x)
    mean = float(np.mean(x))
    median = float(np.median(x))
    std = float(np.std(x, ddof=1)) if n > 1 else float("nan")
    se = std / np.sqrt(n) if n > 1 else float("nan")
    # IC 95% Student
    if n > 1:
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        ci_low = mean - tcrit * se
        ci_high = mean + tcrit * se
    else:
        ci_low = ci_high = float("nan")

    # Wilcoxon signed-rank (bilateral) — zero_method wilcox ; si tous ~0, p=1
    if n >= 1 and np.any(np.abs(x) > 1e-15):
        try:
            w_stat, p_w = stats.wilcoxon(x, alternative="two-sided", zero_method="wilcox")
            w_stat = float(w_stat)
            p_w = float(p_w)
        except ValueError:
            w_stat, p_w = float("nan"), float("nan")
    else:
        w_stat, p_w = float("nan"), 1.0

    # Conclusion
    if np.isfinite(p_w) and p_w < 0.05:
        direction = "positif" if mean > 0 else "negatif"
        conclusion = f"Effet {direction} significatif (p<0.05)"
    elif np.isfinite(p_w) and p_w < 0.10:
        conclusion = "Tendance faible (p<0.10), non significatif a 5%"
    else:
        conclusion = "Pas d'effet significatif"

    return {
        "contrast": name,
        "formula": formula,
        "label": label,
        "n": n,
        "mean": mean,
        "median": median,
        "std": std,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "wilcoxon_stat": w_stat,
        "pvalue_wilcoxon": p_w,
        "significant_5pct": bool(np.isfinite(p_w) and p_w < 0.05),
        "conclusion": conclusion,
    }


def load_wide(scope: str) -> pd.DataFrame:
    m = pd.read_csv(REPORTS_DIR / "wf_etape3_fold_metrics.csv")
    if scope == "nonoverlap":
        blocks = pd.read_csv(REPORTS_DIR / "wf_folds_nonoverlapping.csv")
        keys = set(zip(blocks["test_start_month"], blocks["test_end_month"]))
        m = m[
            m.apply(
                lambda r: (r["test_start_month"], r["test_end_month"]) in keys,
                axis=1,
            )
        ].copy()
        # index by block order
        order = {
            (r.test_start_month, r.test_end_month): int(r.fold_id)
            for r in blocks.itertuples()
        }
        m["block_id"] = m.apply(
            lambda r: order[(r["test_start_month"], r["test_end_month"])], axis=1
        )
        wide = m.pivot_table(index="block_id", columns="cell_id", values="rank_ic")
    else:
        wide = m.pivot_table(index="fold_id", columns="cell_id", values="rank_ic")
    return wide.sort_index()


def run_scope(scope: str) -> pd.DataFrame:
    wide = load_wide(scope)
    rows = []
    for key, formula, label, _ in CONTRASTS:
        d = _paired_series(wide, key)
        rows.append(summarize_contrast(d, key, formula, label))
    return pd.DataFrame(rows), wide


def main() -> int:
    ensure_output_dirs()

    primary, wide14 = run_scope("nonoverlap")
    secondary, wide81 = run_scope("overlap")

    # Descriptive IC by cell on 14
    desc14 = pd.DataFrame(
        {
            "cellule": [1, 2, 3, 4],
            "config": [
                "Ridge sans regime",
                "Ridge + regime",
                "Hybrid sans regime",
                "Hybrid + regime",
            ],
            "IC_mean": [wide14[c].mean() for c in (1, 2, 3, 4)],
            "IC_median": [wide14[c].median() for c in (1, 2, 3, 4)],
            "IC_std": [wide14[c].std(ddof=1) for c in (1, 2, 3, 4)],
        }
    )

    primary.to_csv(REPORTS_DIR / "wf_paired_contrasts_14blocks.csv", index=False)
    secondary.to_csv(REPORTS_DIR / "wf_paired_contrasts_81overlapping.csv", index=False)
    desc14.to_csv(REPORTS_DIR / "wf_cell_ic_14blocks_descriptive.csv", index=False)

    # Per-block deltas for transparency
    deltas = pd.DataFrame(
        {
            "block_id": wide14.index,
            "d_regime_ridge": (wide14[2] - wide14[1]).values,
            "d_regime_hybrid": (wide14[4] - wide14[3]).values,
            "d_hybrid_vs_ridge": (wide14[3] - wide14[1]).values,
            "d_interaction": (
                wide14[4] - wide14[3] - (wide14[2] - wide14[1])
            ).values,
        }
    )
    deltas.to_csv(REPORTS_DIR / "wf_paired_deltas_per_block.csv", index=False)

    print("=== Analyse PRINCIPALE — 14 blocs non-chevauchants ===\n")
    print("IC descriptif par configuration:")
    print(desc14.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nContrastes apparies (Wilcoxon signed-rank):")
    show = primary[
        [
            "label",
            "formula",
            "mean",
            "median",
            "std",
            "ci95_low",
            "ci95_high",
            "pvalue_wilcoxon",
            "significant_5pct",
            "conclusion",
        ]
    ]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== Coherence — 81 plis chevauchants (secondaire) ===\n")
    show81 = secondary[
        [
            "label",
            "formula",
            "mean",
            "median",
            "std",
            "ci95_low",
            "ci95_high",
            "pvalue_wilcoxon",
            "significant_5pct",
            "conclusion",
        ]
    ]
    print(show81.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Consistency note
    print("\n=== Coherence 14 vs 81 (signe de la moyenne) ===")
    for _, r14 in primary.iterrows():
        r81 = secondary[secondary["contrast"] == r14["contrast"]].iloc[0]
        same_sign = np.sign(r14["mean"]) == np.sign(r81["mean"]) or (
            abs(r14["mean"]) < 1e-6 and abs(r81["mean"]) < 1e-6
        )
        print(
            f"  {r14['label']}: mean14={r14['mean']:+.5f} (p={r14['pvalue_wilcoxon']:.3f}) | "
            f"mean81={r81['mean']:+.5f} (p={r81['pvalue_wilcoxon']:.3f}) | "
            f"meme_signe={same_sign}"
        )

    payload = {
        "primary_scope": "14_blocs_non_chevauchants",
        "secondary_scope": "81_plis_chevauchants",
        "metric": "Rank-IC (Spearman) par fenetre test",
        "primary": primary.to_dict(orient="records"),
        "secondary": secondary.to_dict(orient="records"),
        "descriptive_14": desc14.to_dict(orient="records"),
    }
    (REPORTS_DIR / "wf_paired_contrasts_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
