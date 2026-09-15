"""
Diebold-Mariano sur blocs non-chevauchants (IC étage 1).

Perte = -Rank-IC (plus haut IC = meilleure perte plus basse).
Paires : 1vs2 (adaptation), 1vs3 (hybridation), 1vs4 (combiné).
Échantillon = 14 blocs de test 6 mois sans chevauchement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.factorial_hybrid_adapt.paths import REPORTS_DIR, ensure_output_dirs  # noqa: E402


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    h: int = 1,
) -> dict[str, float]:
    """
    DM test H0: E[loss_a - loss_b] = 0.

    Statistique classique + correction HLN (Harvey et al.) pour petit n.
    Pour blocs non-chevauchants, h=1 (pas d'autocorr de chevauchement de fenêtres).
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    d = a - b
    n = len(d)
    if n < 3:
        return {
            "n": n,
            "mean_d": float("nan"),
            "dm_stat": float("nan"),
            "dm_stat_hln": float("nan"),
            "pvalue": float("nan"),
            "pvalue_hln": float("nan"),
        }

    mean_d = float(np.mean(d))
    # Variance Newey-West avec lag h-1 ; h=1 => var empirique
    gamma0 = float(np.var(d, ddof=1))
    var_d = gamma0
    for lag in range(1, h):
        w = 1.0 - lag / h
        cov = float(np.cov(d[lag:], d[:-lag], ddof=1)[0, 1])
        var_d += 2.0 * w * cov
    se = np.sqrt(var_d / n) if var_d > 0 else np.nan
    dm = mean_d / se if se and se > 0 else np.nan

    # Correction Harvey-Leybourne-Newbold
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln_factor if np.isfinite(dm) else np.nan

    # p-values bilatérales (Student df=n-1)
    p = float(2 * stats.t.sf(abs(dm), df=n - 1)) if np.isfinite(dm) else np.nan
    p_hln = (
        float(2 * stats.t.sf(abs(dm_hln), df=n - 1)) if np.isfinite(dm_hln) else np.nan
    )
    return {
        "n": int(n),
        "mean_d": mean_d,
        "dm_stat": float(dm) if np.isfinite(dm) else np.nan,
        "dm_stat_hln": float(dm_hln) if np.isfinite(dm_hln) else np.nan,
        "pvalue": p,
        "pvalue_hln": p_hln,
    }


def load_nonoverlapping_ic() -> pd.DataFrame:
    metrics = pd.read_csv(REPORTS_DIR / "wf_etape3_fold_metrics.csv")
    blocks = pd.read_csv(REPORTS_DIR / "wf_folds_nonoverlapping.csv")
    keys = set(zip(blocks["test_start_month"], blocks["test_end_month"]))
    sub = metrics[
        metrics.apply(
            lambda r: (r["test_start_month"], r["test_end_month"]) in keys, axis=1
        )
    ].copy()
    # ordre temporel des blocs
    order = {
        (r.test_start_month, r.test_end_month): int(r.fold_id)
        for r in blocks.itertuples()
    }
    sub["block_id"] = sub.apply(
        lambda r: order[(r["test_start_month"], r["test_end_month"])], axis=1
    )
    return sub.sort_values(["block_id", "cell_id"])


def main() -> int:
    ensure_output_dirs()
    sub = load_nonoverlapping_ic()
    wide = sub.pivot(index="block_id", columns="cell_id", values="rank_ic").sort_index()
    # loss = -IC
    loss = -wide

    pairs = [
        (1, 2, "adaptation (C2 vs C1)"),
        (1, 3, "hybridation (C3 vs C1)"),
        (1, 4, "combine (C4 vs C1)"),
        (2, 4, "hybridation | regime (C4 vs C2)"),
        (3, 4, "adaptation | hybrid (C4 vs C3)"),
    ]

    rows = []
    print("=== DM sur Rank-IC — 14 blocs NON-chevauchants ===")
    print("Perte = -Rank-IC | H0: E[loss_ref - loss_alt] = 0")
    print("(mean_d > 0 => alt meilleure que ref en moyenne)\n")

    desc = wide.agg(["mean", "median", "std"]).T
    desc.columns = ["IC_mean", "IC_median", "IC_std"]
    print("Descriptif IC sur les 14 blocs :")
    print(desc.to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    for ref, alt, label in pairs:
        # d = loss_ref - loss_alt ; >0 si alt meilleure
        dm = diebold_mariano(loss[ref].values, loss[alt].values, h=1)
        delta_ic = float(wide[alt].mean() - wide[ref].mean())
        row = {
            "pair": f"C{ref}_vs_C{alt}",
            "label": label,
            "delta_IC_mean": delta_ic,
            "mean_loss_diff": dm["mean_d"],
            "dm_stat": dm["dm_stat"],
            "dm_stat_hln": dm["dm_stat_hln"],
            "pvalue": dm["pvalue"],
            "pvalue_hln": dm["pvalue_hln"],
            "n_blocks": dm["n"],
            "significant_5pct_hln": bool(
                np.isfinite(dm["pvalue_hln"]) and dm["pvalue_hln"] < 0.05
            ),
        }
        rows.append(row)
        print(
            f"{label}: delta_IC={delta_ic:+.4f} | "
            f"DM_HLN={dm['dm_stat_hln']:.3f} | "
            f"p_HLN={dm['pvalue_hln']:.3f} | "
            f"sig@5%={'OUI' if row['significant_5pct_hln'] else 'non'}"
        )

    # Effets factoriels sur les 14 blocs
    ic = {c: float(wide[c].mean()) for c in (1, 2, 3, 4)}
    effects = {
        "adaptation": ic[2] - ic[1],
        "hybridation": ic[3] - ic[1],
        "interaction": ic[4] - (ic[2] + ic[3] - ic[1]),
    }
    print("\nEffets IC_mean (14 blocs) :")
    for k, v in effects.items():
        print(f"  {k}: {v:+.5f}")

    out_csv = REPORTS_DIR / "wf_dm_ic_nonoverlapping.csv"
    out_json = REPORTS_DIR / "wf_dm_ic_nonoverlapping.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    payload: dict[str, Any] = {
        "scope": "14_blocs_non_chevauchants",
        "metric": "Rank-IC (Spearman) per 6m test block",
        "loss": "-Rank-IC",
        "n_blocks": int(len(wide)),
        "ic_by_cell": {
            str(c): {
                "mean": float(wide[c].mean()),
                "median": float(wide[c].median()),
                "std": float(wide[c].std(ddof=1)),
            }
            for c in (1, 2, 3, 4)
        },
        "effects": effects,
        "dm_tests": rows,
        "interpretation_note": (
            "Echantillon petit (n=14). IC_std ~0.11-0.12 domine les ecarts entre cellules ; "
            "une moyenne/mediane plus haute n'implique pas significativite DM."
        ),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nArtefacts :\n  - {out_csv}\n  - {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
