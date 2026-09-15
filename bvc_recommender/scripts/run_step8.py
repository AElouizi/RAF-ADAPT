"""
Pipeline Étape 8 — backtest walk-forward vs MASI (rebalancement mensuel par défaut).

Usage (depuis la racine du projet) :
    py -m bvc_recommender.scripts.run_step8
    py -m bvc_recommender.scripts.run_step8 --frequency quarterly
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.backtest.engine import BACKTEST_END, BACKTEST_START, run_walk_forward_backtest  # noqa: E402
from bvc_recommender.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    DATASET_DIR,
    FEATURES_DIR,
    RANDOM_STATE,
    REBALANCE_FREQUENCY,
    REPORTS_DIR,
    load_env_file,
)

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RECOMMENDATIONS_DIR = REPORTS_DIR / "recommendations"


def _load_parquet(name: str, directory: Path) -> pd.DataFrame:
    for ext in (".parquet", ".csv"):
        path = directory / f"{name}{ext}"
        if path.is_file():
            return pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
    return pd.DataFrame()


def load_scores() -> tuple[pd.DataFrame, str]:
    """Scores point-in-time : TFT (Q50) prioritaire, puis baselines."""
    for name in ("tft_predictions", "baseline_predictions"):
        path = DATASET_DIR / f"{name}.parquet"
        if not path.is_file():
            continue
        df = pd.read_parquet(path)
        # Conserver val+test pour le walk-forward (pas de train in-sample).
        if "split" in df.columns:
            holdout = df[df["split"].isin(["val", "test"])]
            if not holdout.empty:
                df = holdout
        if "q50" in df.columns:
            df["prediction"] = pd.to_numeric(df["q50"], errors="coerce")
            source = "tft_q50"
        elif "prediction" in df.columns:
            df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
            source = name
        else:
            continue
        logger.info("Scores backtest : %s (%s lignes)", source, len(df))
        return df, source
    raise FileNotFoundError("Prédictions manquantes — lancer run_step5 ou run_step6.")


def write_markdown_report(report: dict, path: Path) -> None:
    freq = report.get("rebalance_frequency", "monthly")
    freq_label = "Mensuel" if freq == "monthly" else "Trimestriel"
    n_periods = len(report.get("periods", report.get("quarters", [])))
    score_src = report.get("score_source", "—")

    lines = [
        "# Backtest walk-forward — Étape 8",
        "",
        f"- **Période** : {report['period']['start']} → {report['period']['end']}",
        f"- **Rebalancement** : {freq_label}",
        f"- **Scores** : {score_src}",
        f"- **Coût transaction** : {report['transaction_cost']:.1%}",
        f"- **Périodes** : {n_periods}",
        f"- **Recommandations** : `reports/recommendations/YYYY-MM.csv` (+ agrégats `YYYY-QX.csv`)",
        "",
        "## Métriques vs MASI",
        "",
        "| Stratégie | Rend. ann. | Alpha | Sharpe | Sortino | Max DD | Calmar | CVaR 95% | Hit Ratio | Turnover trim. |",
        "|-----------|------------|-------|--------|---------|--------|--------|----------|-----------|----------------|",
    ]
    for name, m in report.get("metrics", {}).items():
        if "error" in m:
            continue
        tq = m.get("turnover_quarterly")
        tq_str = f"{tq:.0%}" if tq is not None else "—"
        lines.append(
            f"| {name} | {m.get('annualized_return', 0):.1%} | "
            f"{m.get('alpha_annualized', 0):.1%} | {m.get('sharpe') or '—'} | "
            f"{m.get('sortino') or '—'} | {m.get('max_drawdown', 0):.1%} | "
            f"{m.get('calmar') or '—'} | {m.get('cvar_95', 0):.2%} | "
            f"{m.get('hit_ratio', 0):.1%} | {tq_str} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def main() -> int:
    parser = argparse.ArgumentParser(description="BVC Recommender — Étape 8")
    parser.add_argument("--start", default=str(BACKTEST_START.date()))
    parser.add_argument("--end", default=str(BACKTEST_END.date()))
    parser.add_argument(
        "--frequency",
        choices=("monthly", "quarterly"),
        default=REBALANCE_FREQUENCY,
        help="Fréquence de rebalancement (défaut: config REBALANCE_FREQUENCY)",
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    scores, score_source = load_scores()
    technical = _load_parquet("features_techniques", FEATURES_DIR)
    cours = _load_parquet("market_data_cours_historique", DATA_PROCESSED_DIR)
    indices = _load_parquet("market_data_indices_historique", DATA_PROCESSED_DIR)

    # Borner la fin aux données de marché / scores disponibles
    if not cours.empty and "date_cours" in cours.columns:
        max_cours = pd.to_datetime(cours["date_cours"], errors="coerce").max()
        if pd.notna(max_cours) and max_cours < end:
            end = max_cours
    if "date_cours" in scores.columns:
        max_score = pd.to_datetime(scores["date_cours"], errors="coerce").max()
        if pd.notna(max_score) and max_score < end:
            end = max_score

    logger.info(
        "Backtest %s → %s | rebalancement %s | scores=%s",
        start.date(),
        end.date(),
        args.frequency,
        score_source,
    )
    report = run_walk_forward_backtest(
        scores,
        technical,
        cours,
        indices,
        start=start,
        end=end,
        recommendations_dir=RECOMMENDATIONS_DIR,
        frequency=args.frequency,
    )
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["project"] = "BVC Recommender — Étape 8"
    report["score_source"] = score_source

    daily = report.pop("daily_returns", {})
    curves_path = DATASET_DIR / "backtest_equity_curves.parquet"
    if daily:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        cum = pd.DataFrame({k: (1 + s).cumprod() for k, s in daily.items() if len(s)}).sort_index()
        cum.to_parquet(curves_path)
        report["equity_curves_path"] = str(curves_path)

    json_path = REPORTS_DIR / "step8_validation_report.json"
    md_path = REPORTS_DIR / "backtest_report.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_markdown_report(report, md_path)

    logger.info("Rapport étape 8 : %s", json_path)
    for name, m in report.get("metrics", {}).items():
        if "annualized_return" in m:
            logger.info(
                "[%s] rend. ann.=%.1f%% | alpha=%.1f%% | Sharpe=%s",
                name,
                m["annualized_return"] * 100,
                m.get("alpha_annualized", 0) * 100,
                m.get("sharpe"),
            )

    logger.info("Étape 8 terminée (%s périodes).", len(report.get("periods", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
