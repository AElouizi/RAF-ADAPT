"""
Étape 4 WF — agrégation descriptive sur plis mensuels chevauchants.

IC moyen / médiane / σ inter-plis, hit ratio, courbe temporelle Rank-IC.
(Les tests DM / allocation = blocs non-chevauchants, étapes 5–6.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CELL_LABELS = {
    1: "C1 Ridge static",
    2: "C2 Ridge+regime",
    3: "C3 Hybrid static",
    4: "C4 Hybrid+regime",
}


def aggregate_fold_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell_id, g in metrics.groupby("cell_id"):
        rows.append(
            {
                "cellule": int(cell_id),
                "modele": g["model"].iloc[0],
                "regime": bool(g["regime"].iloc[0]),
                "n_folds": int(len(g)),
                "IC_mean": float(g["rank_ic"].mean()),
                "IC_median": float(g["rank_ic"].median()),
                "IC_std": float(g["rank_ic"].std(ddof=1)),
                "IC_q25": float(g["rank_ic"].quantile(0.25)),
                "IC_q75": float(g["rank_ic"].quantile(0.75)),
                "pct_folds_IC_positive": float((g["rank_ic"] > 0).mean()),
                "hit_mean": float(g["hit_ratio"].mean()),
                "hit_median": float(g["hit_ratio"].median()),
                "hit_std": float(g["hit_ratio"].std(ddof=1)),
                "sharpe_ls_mean": float(g["sharpe_ls"].mean()),
                "sharpe_ls_std": float(g["sharpe_ls"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows).sort_values("cellule")


def effect_decomposition_ic(agg: pd.DataFrame) -> dict[str, float]:
    """Effets factoriels sur IC_mean (descriptif, plis chevauchants)."""
    ic = {int(r.cellule): float(r.IC_mean) for r in agg.itertuples()}
    return {
        "effet_adaptation": ic[2] - ic[1],
        "effet_hybridation": ic[3] - ic[1],
        "effet_interaction": ic[4] - (ic[2] + ic[3] - ic[1]),
        "note": "Descriptif sur plis chevauchants — significativite = etape 6 (blocs).",
    }


def plot_ic_timeseries(metrics: pd.DataFrame, out_path: Path) -> Path:
    """Courbe Rank-IC par pli (axe = test_start_month)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for cell_id in sorted(metrics["cell_id"].unique()):
        g = metrics[metrics["cell_id"] == cell_id].sort_values("fold_id")
        x = pd.to_datetime(g["test_start_month"].astype(str))
        ax.plot(
            x,
            g["rank_ic"].values,
            label=CELL_LABELS.get(int(cell_id), f"C{cell_id}"),
            linewidth=1.2,
            alpha=0.9,
        )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Walk-forward — Rank-IC (Spearman) par pli (fenetre test 6m)")
    ax.set_xlabel("Debut fenetre test")
    ax.set_ylabel("Rank-IC")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_ic_boxplot(metrics: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [
        metrics.loc[metrics["cell_id"] == c, "rank_ic"].dropna().values
        for c in (1, 2, 3, 4)
    ]
    ax.boxplot(data, tick_labels=[CELL_LABELS[c] for c in (1, 2, 3, 4)], showmeans=True)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Distribution inter-plis du Rank-IC (81 plis chevauchants)")
    ax.set_ylabel("Rank-IC")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def run_etape4_aggregation(
    metrics_path: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    metrics = pd.read_csv(metrics_path)
    agg = aggregate_fold_metrics(metrics)
    effects = effect_decomposition_ic(agg)

    curve_path = reports_dir / "wf_etape4_ic_timeseries.png"
    box_path = reports_dir / "wf_etape4_ic_boxplot.png"
    plot_ic_timeseries(metrics, curve_path)
    plot_ic_boxplot(metrics, box_path)

    # Série longue pour réutilisation
    ts = metrics[
        [
            "fold_id",
            "cell_id",
            "test_start_month",
            "test_end_month",
            "rank_ic",
            "hit_ratio",
            "sharpe_ls",
        ]
    ].sort_values(["cell_id", "fold_id"])
    ts_path = reports_dir / "wf_etape4_ic_by_fold.csv"
    ts.to_csv(ts_path, index=False)

    agg_path = reports_dir / "wf_etape4_aggregation.csv"
    agg.to_csv(agg_path, index=False)

    summary = {
        "scope": "plis_mensuels_chevauchants",
        "n_folds": int(metrics["fold_id"].nunique()),
        "aggregation": agg.to_dict(orient="records"),
        "effects_IC_descriptive": {
            k: (round(v, 6) if isinstance(v, float) else v) for k, v in effects.items()
        },
        "distinction": (
            "Metriques descriptives = plis chevauchants (n=81). "
            "Diebold-Mariano + allocation portefeuille = blocs non-chevauchants (etapes 5-6)."
        ),
        "artifacts": {
            "aggregation": str(agg_path),
            "timeseries_csv": str(ts_path),
            "timeseries_png": str(curve_path),
            "boxplot_png": str(box_path),
        },
    }
    sum_path = reports_dir / "wf_etape4_summary.json"
    # sanitize numpy
    def _safe(o: Any) -> Any:
        if isinstance(o, dict):
            return {str(k): _safe(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_safe(v) for v in o]
        if isinstance(o, (np.floating, float)):
            x = float(o)
            return None if np.isnan(x) else x
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        return o

    sum_path.write_text(
        json.dumps(_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "aggregation": agg,
        "effects": effects,
        "metrics": metrics,
        "paths": {
            "aggregation": agg_path,
            "timeseries_csv": ts_path,
            "timeseries_png": curve_path,
            "boxplot_png": box_path,
            "summary": sum_path,
        },
    }
