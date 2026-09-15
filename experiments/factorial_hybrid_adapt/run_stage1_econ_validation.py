"""
Validation économique des recommandations BUY / NEUTRAL / SELL (étage 1).

Métrique principale d'évaluation (futur, horizon modèle) :
  - ``future_alpha`` = alpha_ajuste_risque réalisé (déjà dans le panel)
  - ``future_excess`` = future_alpha × vol_baissiere_20d(t)
    (= excess_vs_masi reconstruit ; vol connue à t, alpha réalisé futur)

Les classes BUY/NEUTRAL/SELL ont été formées SANS ces grandeurs futures
(voir contrôles look-ahead). Ici elles servent uniquement à mesurer la
séparation économique ex-post.

Ne lance PAS NSGA-III.
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
sys.path.insert(0, str(ROOT))

from experiments.factorial_hybrid_adapt.data_utils import load_ml_dataset
from experiments.factorial_hybrid_adapt.paths import REPORTS_DIR, ensure_output_dirs
from experiments.factorial_hybrid_adapt.recommendations import CELL_META, look_ahead_checklist, RecommendationConfig

CLASS_ORDER = ("BUY", "NEUTRAL", "SELL")


def _ci95(x: pd.Series) -> tuple[float, float, float, float, float, int]:
    """mean, median, std, ci_low, ci_high, n."""
    s = pd.to_numeric(x, errors="coerce").dropna()
    n = int(len(s))
    if n == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    mean = float(s.mean())
    median = float(s.median())
    std = float(s.std(ddof=1)) if n > 1 else 0.0
    if n < 2:
        return (mean, median, std, np.nan, np.nan, n)
    se = std / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return (mean, median, std, mean - tcrit * se, mean + tcrit * se, n)


def attach_future_excess(recos: pd.DataFrame) -> pd.DataFrame:
    """
    Attache les outcomes futurs depuis ml_dataset (non clipés) + reconstruit excess.

    - ``future_alpha`` = alpha_ajuste_risque (ml_dataset, non clipé)
    - ``future_excess`` = future_alpha × vol_baissiere_20d(t)

    Note : ``realized_alpha`` du panel WF peut différer (clip 1–99 % appris sur le
    train du pli). Pour la validation économique on privilégie la cible brute.
    """
    out = recos.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    ml = load_ml_dataset()
    need = ["ticker", "date_cours", "vol_baissiere_20d", "alpha_ajuste_risque"]
    missing = [c for c in need if c not in ml.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes ml_dataset : {missing}")
    vol = ml[need].copy()
    vol["date_cours"] = pd.to_datetime(vol["date_cours"], errors="coerce")
    vol["ticker"] = vol["ticker"].astype(str)
    out["ticker"] = out["ticker"].astype(str)

    merged = out.merge(
        vol.rename(
            columns={
                "date_cours": "date",
                "alpha_ajuste_risque": "future_alpha",
                "vol_baissiere_20d": "vol_t",
            }
        ),
        on=["ticker", "date"],
        how="left",
    )
    merged["future_alpha"] = pd.to_numeric(merged["future_alpha"], errors="coerce")
    merged["vol_t"] = pd.to_numeric(merged["vol_t"], errors="coerce")
    merged["future_excess"] = merged["future_alpha"] * merged["vol_t"]

    # Diagnostic clip WF vs brut (info, pas d'échec)
    if "realized_alpha" in merged.columns:
        gap = (
            pd.to_numeric(merged["realized_alpha"], errors="coerce")
            - merged["future_alpha"]
        ).abs()
        p99 = float(gap.quantile(0.99)) if gap.notna().any() else np.nan
        print(
            f"  Note : |realized_alpha_WF − alpha_brut| p99={p99:.4g} "
            "(écart attendu si clip train du pli)."
        )
    return merged


def class_stats(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    rows = []
    for cell in sorted(df["cell"].dropna().unique()):
        for rec in CLASS_ORDER:
            sub = df[(df["cell"] == cell) & (df["recommendation"] == rec)][value_col]
            mean, median, std, lo, hi, n = _ci95(sub)
            meta = CELL_META.get(int(cell), {})
            rows.append(
                {
                    "Configuration": meta.get("label", f"C{int(cell)}"),
                    "cell": int(cell),
                    "recommendation": rec,
                    "n": n,
                    "mean": mean,
                    "median": median,
                    "std": std,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "metric": value_col,
                }
            )
    return pd.DataFrame(rows)


def block_spreads(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Spreads moyens par (cell, fold_id / bloc)."""
    rows = []
    for (cell, fold_id), g in df.groupby(["cell", "fold_id"], sort=True):
        means = g.groupby("recommendation")[value_col].mean()
        buy = float(means.get("BUY", np.nan))
        neu = float(means.get("NEUTRAL", np.nan))
        sell = float(means.get("SELL", np.nan))
        meta = CELL_META.get(int(cell), {})
        rows.append(
            {
                "Configuration": meta.get("label", f"C{int(cell)}"),
                "cell": int(cell),
                "fold_id": int(fold_id),
                "test_start_month": g["test_start_month"].iloc[0],
                "test_end_month": g["test_end_month"].iloc[0],
                "mean_BUY": buy,
                "mean_NEUTRAL": neu,
                "mean_SELL": sell,
                "spread_BUY_NEUTRAL": buy - neu,
                "spread_NEUTRAL_SELL": neu - sell,
                "spread_BUY_SELL": buy - sell,
                "order_ok": bool(buy > neu > sell) if np.isfinite([buy, neu, sell]).all() else False,
                "metric": value_col,
            }
        )
    return pd.DataFrame(rows)


def summarize_spreads(spreads: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell, g in spreads.groupby("cell"):
        meta = CELL_META.get(int(cell), {})
        bs = g["spread_BUY_SELL"]
        rows.append(
            {
                "Configuration": meta.get("label", f"C{int(cell)}"),
                "cell": int(cell),
                "n_blocks": int(len(g)),
                "mean_BUY": float(g["mean_BUY"].mean()),
                "mean_NEUTRAL": float(g["mean_NEUTRAL"].mean()),
                "mean_SELL": float(g["mean_SELL"].mean()),
                "spread_BUY_NEUTRAL_mean": float(g["spread_BUY_NEUTRAL"].mean()),
                "spread_NEUTRAL_SELL_mean": float(g["spread_NEUTRAL_SELL"].mean()),
                "spread_BUY_SELL_mean": float(bs.mean()),
                "spread_BUY_SELL_median": float(bs.median()),
                "spread_BUY_SELL_std": float(bs.std(ddof=1)) if len(bs) > 1 else np.nan,
                "pct_blocks_BUY_gt_SELL": float((bs > 0).mean()),
                "pct_blocks_full_order": float(g["order_ok"].mean()),
                "metric": g["metric"].iloc[0],
            }
        )
    out = pd.DataFrame(rows).sort_values("cell")
    # Rang séparation économique = plus grand spread BUY−SELL moyen
    out["rank_BUY_SELL"] = out["spread_BUY_SELL_mean"].rank(ascending=False, method="min").astype(int)
    return out


def verify_no_lookahead(recos: pd.DataFrame) -> list[dict[str, str]]:
    """Contrôles stricts : la classe ne dépend pas du futur."""
    cfg = RecommendationConfig(
        benchmark_mode=str(recos["benchmark_mode"].iloc[0])
        if "benchmark_mode" in recos.columns
        else "zero_alpha",
        tau_method=str(recos["tau_method"].iloc[0])
        if "tau_method" in recos.columns
        else "expanding_past_quantile",
    )
    checks = look_ahead_checklist(cfg)

    # Reconstruction locale de la règle
    excess = pd.to_numeric(recos["excess_return"], errors="coerce")
    tau = pd.to_numeric(recos["tau"], errors="coerce")
    rebuilt = np.where(excess > tau, "BUY", np.where(excess < -tau, "SELL", "NEUTRAL"))
    match = float((rebuilt == recos["recommendation"].to_numpy()).mean())
    checks.append(
        {
            "check": "recommendation_reproducible_from_excess_and_tau",
            "status": "OK" if match > 0.999 else "FAIL",
            "detail": f"Taux de reproduction règle excess±tau = {match:.6f}",
        }
    )

    # Corrélation artificielle ? La reco ne doit pas utiliser realized_alpha
    # (test : predire la classe depuis excess seul doit coller ; realized ne doit
    #  pas être dans la formule — déjà OK si reproduction exacte)
    if "realized_alpha" in recos.columns and "predicted_return" in recos.columns:
        # Sur un sous-échantillon, classer avec realized_alpha±tau ≠ recommendation
        # (si égalité systématique → suspicion de fuite)
        ra = pd.to_numeric(recos["realized_alpha"], errors="coerce")
        fake = np.where(ra > tau, "BUY", np.where(ra < -tau, "SELL", "NEUTRAL"))
        leak_rate = float((fake == recos["recommendation"].to_numpy()).mean())
        checks.append(
            {
                "check": "recommendation_not_equal_to_realized_thresholding",
                "status": "OK" if leak_rate < 0.85 else "WARN",
                "detail": (
                    f"Accord reco vs seuil(realized_alpha,tau)={leak_rate:.3f} "
                    "(élevé ⇒ suspicion ; attendu nettement < 1 car pred ≠ realized)."
                ),
            }
        )

    checks.append(
        {
            "check": "future_outcomes_eval_only",
            "status": "OK",
            "detail": (
                "future_alpha / future_excess utilisés uniquement pour stats "
                "ex-post ; non injectés dans recommendation."
            ),
        }
    )
    return checks


def _fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{100 * x:.2f}%"


def _fmt(x: float, nd: int = 4) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x:.{nd}f}"


def print_cell_report(
    cell: int,
    stats_alpha: pd.DataFrame,
    stats_ex: pd.DataFrame,
    spreads_ex: pd.DataFrame,
    summary_ex: pd.DataFrame,
) -> None:
    meta = CELL_META[cell]
    print(f"\n{'=' * 72}")
    print(f"{meta['label']}  (cell={cell}, model={meta['model']}, regime={meta['regime']})")
    print(f"{'=' * 72}")

    print("\n— Rendement futur (excess vs MASI, horizon modèle) par classe —")
    sub = stats_ex[stats_ex["cell"] == cell]
    print(
        sub[
            ["recommendation", "n", "mean", "median", "std", "ci95_low", "ci95_high"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n— Même découpage sur alpha_ajuste_risque réalisé —")
    sub_a = stats_alpha[stats_alpha["cell"] == cell]
    print(
        sub_a[
            ["recommendation", "n", "mean", "median", "std", "ci95_low", "ci95_high"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    sp = spreads_ex[spreads_ex["cell"] == cell].sort_values("fold_id")
    print("\n— Spreads excess par bloc (BUY−NEUTRAL, NEUTRAL−SELL, BUY−SELL) —")
    show = sp[
        [
            "test_start_month",
            "test_end_month",
            "mean_BUY",
            "mean_NEUTRAL",
            "mean_SELL",
            "spread_BUY_NEUTRAL",
            "spread_NEUTRAL_SELL",
            "spread_BUY_SELL",
            "order_ok",
        ]
    ]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    sm = summary_ex[summary_ex["cell"] == cell].iloc[0]
    order = (
        sm["mean_BUY"] > sm["mean_NEUTRAL"] > sm["mean_SELL"]
    )
    print("\n— Synthèse —")
    print(
        f"  Moyennes (avg des 14 blocs) : "
        f"BUY={sm['mean_BUY']:.4f} | NEUTRAL={sm['mean_NEUTRAL']:.4f} | "
        f"SELL={sm['mean_SELL']:.4f}"
    )
    print(
        f"  Spreads moyens : BUY−NEU={sm['spread_BUY_NEUTRAL_mean']:.4f} | "
        f"NEU−SELL={sm['spread_NEUTRAL_SELL_mean']:.4f} | "
        f"BUY−SELL={sm['spread_BUY_SELL_mean']:.4f}"
    )
    print(
        f"  Ordre BUY>NEU>SELL (moyenne blocs) : {'OUI' if order else 'NON'} | "
        f"% blocs ordre strict : {100 * sm['pct_blocks_full_order']:.1f}% | "
        f"% blocs BUY>SELL : {100 * sm['pct_blocks_BUY_gt_SELL']:.1f}%"
    )


def main() -> int:
    ensure_output_dirs()
    path = REPORTS_DIR / "stage1_recommendations_14blocks.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} introuvable — lancer d'abord run_stage1_recommendations.py --folds 14"
        )

    recos = pd.read_parquet(path)
    print(f"Panel : {path.name} | n={len(recos):,} | cells={sorted(recos['cell'].unique())}")

    checks = verify_no_lookahead(recos)
    print("\n=== Contrôles look-ahead ===")
    for c in checks:
        print(f"  [{c['status']}] {c['check']}: {c['detail']}")

    panel = attach_future_excess(recos)
    n_ex = int(panel["future_excess"].notna().sum())
    print(f"\nfuture_excess non-null : {n_ex:,}/{len(panel):,}")

    stats_alpha = class_stats(panel, "future_alpha")
    stats_ex = class_stats(panel, "future_excess")
    spreads_alpha = block_spreads(panel, "future_alpha")
    spreads_ex = block_spreads(panel, "future_excess")
    summary_alpha = summarize_spreads(spreads_alpha)
    summary_ex = summarize_spreads(spreads_ex)

    for cell in (1, 2, 3, 4):
        print_cell_report(cell, stats_alpha, stats_ex, spreads_ex, summary_ex)

    print(f"\n{'=' * 72}")
    print("COMPARAISON C1–C4 — séparation économique BUY vs SELL (excess)")
    print(f"{'=' * 72}")
    cmp_cols = [
        "Configuration",
        "mean_BUY",
        "mean_NEUTRAL",
        "mean_SELL",
        "spread_BUY_SELL_mean",
        "spread_BUY_SELL_median",
        "pct_blocks_BUY_gt_SELL",
        "pct_blocks_full_order",
        "rank_BUY_SELL",
    ]
    print(summary_ex[cmp_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best = summary_ex.sort_values("spread_BUY_SELL_mean", ascending=False).iloc[0]
    print(
        f"\n>>> Meilleure séparation BUY−SELL : {best['Configuration']} "
        f"(spread moyen excess = {best['spread_BUY_SELL_mean']:.4f}, "
        f"{100 * best['pct_blocks_BUY_gt_SELL']:.1f}% des blocs avec BUY>SELL)"
    )

    # Exports
    stats_alpha.to_csv(REPORTS_DIR / "stage1_econ_class_stats_alpha_14blocks.csv", index=False)
    stats_ex.to_csv(REPORTS_DIR / "stage1_econ_class_stats_excess_14blocks.csv", index=False)
    spreads_ex.to_csv(REPORTS_DIR / "stage1_econ_spreads_by_block_excess_14blocks.csv", index=False)
    spreads_alpha.to_csv(REPORTS_DIR / "stage1_econ_spreads_by_block_alpha_14blocks.csv", index=False)
    summary_ex.to_csv(REPORTS_DIR / "stage1_econ_summary_excess_14blocks.csv", index=False)
    summary_alpha.to_csv(REPORTS_DIR / "stage1_econ_summary_alpha_14blocks.csv", index=False)

    payload: dict[str, Any] = {
        "horizon": "FORWARD_HORIZON_DAYS du dataset (alpha / excess vs MASI)",
        "primary_metric": "future_excess = realized_alpha * vol_baissiere_20d(t)",
        "secondary_metric": "future_alpha = alpha_ajuste_risque réalisé",
        "best_BUY_SELL_separation": {
            "configuration": best["Configuration"],
            "cell": int(best["cell"]),
            "spread_BUY_SELL_mean_excess": float(best["spread_BUY_SELL_mean"]),
            "pct_blocks_BUY_gt_SELL": float(best["pct_blocks_BUY_gt_SELL"]),
        },
        "lookahead_checks": checks,
        "nsga_launched": False,
    }
    (REPORTS_DIR / "stage1_econ_validation_14blocks.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nExports écrits sous {REPORTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
