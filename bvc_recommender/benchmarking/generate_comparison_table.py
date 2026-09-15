"""
Génère le tableau comparatif des benchmarks (CSV + figure).

Usage :
    py -m bvc_recommender.benchmarking.generate_comparison_table
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import DATASET_DIR, REPORTS_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

METRICS_COLS = [
    "model_name",
    "level",
    "alpha_annualized",
    "sharpe",
    "sortino",
    "cvar_95",
    "max_drawdown",
    "hit_ratio",
    "calmar",
    "omega",
]


def main() -> int:
    metrics_path = DATASET_DIR / "benchmarking" / "benchmark_metrics.csv"
    if not metrics_path.is_file():
        logger.error("Fichier manquant : %s — lancer run_all_benchmarks d'abord.", metrics_path)
        return 2

    df = pd.read_csv(metrics_path)
    if "error" in df.columns:
        df = df[df["error"].isna()] if df["error"].notna().any() else df

    out_dir = REPORTS_DIR / "benchmarking"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = df[[c for c in METRICS_COLS if c in df.columns]].copy()
    csv_out = out_dir / "comparison_table.csv"
    table.to_csv(csv_out, index=False)

    # Figure alpha + Sharpe
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    names = table["model_name"].tolist()
    alphas = (table["alpha_annualized"].fillna(0) * 100).tolist()
    sharpes = table["sharpe"].fillna(0).tolist()

    colors = ["#0e7490", "#0c2340", "#15803d", "#b45309", "#b91c1c"]
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]

    axes[0].barh(names, alphas, color=bar_colors)
    axes[0].axvline(0, color="#94a3b8", lw=1)
    axes[0].set_xlabel("Alpha annualisé (%)")
    axes[0].set_title("Alpha vs MASI")

    axes[1].barh(names, sharpes, color=bar_colors)
    axes[1].set_xlabel("Sharpe")
    axes[1].set_title("Ratio de Sharpe")

    fig.suptitle("RAF-ADAPT — Benchmarking (comparaison multi-modèles)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig_path = out_dir / "comparison_alpha_sharpe.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Tableau : %s", csv_out)
    logger.info("Figure  : %s", fig_path)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
