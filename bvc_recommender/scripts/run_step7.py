"""
Sous-module C — Filtre de liquidité (pénalité sigmoïde continue).

Optionnellement enchaîne l'optimisation NSGA-III (portefeuilles) si --with-optimize.

Usage (depuis la racine du projet) :
    python -m bvc_recommender.scripts.run_step7
    python -m bvc_recommender.scripts.run_step7 --as-of 2025-08-13
    python -m bvc_recommender.scripts.run_step7 --with-optimize
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import (  # noqa: E402
    DATA_PROCESSED_DIR,
    DATASET_DIR,
    FEATURES_DIR,
    LIQUIDITY_VMQ_THRESHOLD_MAD,
    RANDOM_STATE,
    REPORTS_DIR,
    load_env_file,
)
from bvc_recommender.models.liquidity_filter import (  # noqa: E402
    LABEL_BOTTOM,
    LABEL_NEUTRE,
    LABEL_TOP,
    rank_and_label_universe,
    sigmoid_liquidity_factor,
)
from bvc_recommender.models.portfolio_optimizer import optimize_portfolios  # noqa: E402

load_env_file(ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _load_parquet(name: str, directory: Path) -> pd.DataFrame:
    for ext in (".parquet", ".csv"):
        path = directory / f"{name}{ext}"
        if path.is_file():
            return pd.read_parquet(path) if ext == ".parquet" else pd.read_csv(path)
    return pd.DataFrame()


def load_tft_scores() -> pd.DataFrame:
    """Charge le score TFT (Q50) — prioritaire ; repli sur baselines."""
    for name in ("tft_predictions", "baseline_predictions"):
        path = DATASET_DIR / f"{name}.parquet"
        if not path.is_file():
            continue
        df = pd.read_parquet(path)
        if "split" in df.columns:
            test = df[df["split"] == "test"]
            if not test.empty:
                df = test
        if "q50" in df.columns:
            df["prediction"] = pd.to_numeric(df["q50"], errors="coerce")
            score_source = "tft_q50"
        elif "prediction" in df.columns:
            df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
            score_source = name
        else:
            raise ValueError(f"Colonne score absente dans {name}")
        logger.info(
            "Scores chargés depuis %s (%s lignes) | source=%s",
            name,
            len(df),
            score_source,
        )
        df.attrs["score_source"] = score_source
        return df
    raise FileNotFoundError(
        "Aucune prédiction TFT/baseline trouvée. Lancer sous-module B (run_step6) ou run_step5."
    )


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Sous-module C — Filtre de liquidité",
        "",
        f"- **Généré** : {report['generated_at']}",
        f"- **Formule** : `score_final = score_TFT × sigmoid(VMQ_20j, seuil={LIQUIDITY_VMQ_THRESHOLD_MAD:,.0f} MAD)`",
        f"- **Date as-of** : {report.get('as_of_date')}",
        f"- **Univers** : {report.get('universe_size')} tickers",
        f"- **Source score** : {report.get('score_source')}",
        "",
        "## Labels (terciles)",
        "",
    ]
    for lab in (LABEL_TOP, LABEL_NEUTRE, LABEL_BOTTOM):
        lines.append(f"- **{lab}** : {report.get('label_counts', {}).get(lab, 0)}")

    lines.extend(
        [
            "",
            "## Top 15 (score_final décroissant)",
            "",
            "| Rang | Ticker | Score TFT | VMQ 20j | Facteur liq. | Score final | Label |",
            "|------|--------|-----------|---------|--------------|-------------|-------|",
        ]
    )
    for row in report.get("top_15", []):
        vmq = row.get("vmq_20j")
        vmq_txt = f"{vmq:,.0f}" if vmq is not None and pd.notna(vmq) else "—"
        lines.append(
            f"| {row.get('rank', '')} | {row.get('ticker')} | "
            f"{row.get('score_tft', float('nan')):.4f} | {vmq_txt} | "
            f"{row.get('liquidity_factor', float('nan')):.3f} | "
            f"{row.get('score_final', float('nan')):.4f} | {row.get('label')} |"
        )

    portfolios = report.get("portfolios")
    if portfolios:
        relaxed = (portfolios.get("constraints") or {}).get("cvar_constraint_relaxed")
        lines.extend(
            [
                "",
                "# Sous-module D — Optimisation NSGA-III",
                "",
                f"- **Candidats TOP** : {portfolios.get('candidates')}",
                f"- **Solutions Pareto** : {portfolios.get('n_pareto')}",
                f"- **CVaR MASI** : {portfolios.get('cvar_masi')}",
                f"- **Limite CVaR cible (×0.9)** : {portfolios.get('cvar_limit_target')}",
                f"- **Limite CVaR effective** : {portfolios.get('cvar_limit')}"
                + (" *(assouplie — cible inatteignable)*" if relaxed else ""),
                f"- **CVaR min réalisable** : {portfolios.get('min_cvar_achievable')}",
                "",
            ]
        )
        for key, title in (
            ("P_agressif", "P_agressif — max alpha"),
            ("P_equilibre", "P_equilibre — point genou"),
            ("P_defensif", "P_defensif — min CVaR"),
        ):
            p = portfolios.get(key, {})
            lines.extend(
                [
                    f"## {title}",
                    "",
                    f"- Positions : {p.get('n_positions')}",
                    f"- Alpha attendu : {p.get('expected_alpha')}",
                    f"- CVaR 95 % : {p.get('cvar_95')}",
                    f"- Score liquidité : {p.get('liquidity_score')}",
                    f"- CVaR / limite : {p.get('cvar_vs_limit')}",
                    f"- Contrainte CVaR respectée : {p.get('cvar_ok')}",
                    "",
                    "| Ticker | Poids % |",
                    "|--------|---------|",
                ]
            )
            for t, w in zip(p.get("tickers", []), p.get("weights_pct", [])):
                lines.append(f"| {t} | {w:.2f} |")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="BVC Recommender — Sous-modules C (liquidité) et D (NSGA-III)"
    )
    parser.add_argument("--as-of", default=None, help="Date de rebalancement (YYYY-MM-DD)")
    parser.add_argument(
        "--with-optimize",
        action="store_true",
        default=True,
        help="Enchaîner l'optimisation NSGA-III (défaut : activé)",
    )
    parser.add_argument(
        "--liquidity-only",
        action="store_true",
        help="Sous-module C uniquement (pas d'optimisation)",
    )
    args = parser.parse_args()
    run_optimize = args.with_optimize and not args.liquidity_only

    np.random.seed(RANDOM_STATE)
    as_of = pd.Timestamp(args.as_of) if args.as_of else None

    # Illustration du facteur sigmoïde aux points-clés
    demo_vmq = pd.Series([50_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000])
    demo_f = sigmoid_liquidity_factor(demo_vmq)
    logger.info(
        "Courbe sigmoïde (seuil=%s MAD) : %s",
        f"{LIQUIDITY_VMQ_THRESHOLD_MAD:,.0f}",
        ", ".join(f"{v:,.0f}→{f:.2f}" for v, f in zip(demo_vmq, demo_f)),
    )

    scores = load_tft_scores()
    technical = _load_parquet("features_techniques", FEATURES_DIR)

    ranked = rank_and_label_universe(scores, technical, as_of_date=as_of)
    if ranked.empty:
        logger.error("Univers classé vide.")
        return 2

    ranked_path = DATASET_DIR / "ranked_universe.parquet"
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    ranked.to_parquet(ranked_path, index=False)

    top_cols = [
        c
        for c in (
            "rank",
            "ticker",
            "score_tft",
            "prediction",
            "vmq_20j",
            "liquidity_factor",
            "score_final",
            "label",
            "date_cours",
        )
        if c in ranked.columns
    ]
    top_15 = ranked.head(15)[top_cols].copy()
    if "score_tft" not in top_15.columns and "prediction" in top_15.columns:
        top_15["score_tft"] = top_15["prediction"]

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "BVC Recommender — Sous-modules C/D",
        "random_state": RANDOM_STATE,
        "method": {
            "formula": "score_final = score_TFT × sigmoid(VMQ_20j, seuil=500000 MAD)",
            "threshold_mad": LIQUIDITY_VMQ_THRESHOLD_MAD,
            "labels": [LABEL_TOP, LABEL_NEUTRE, LABEL_BOTTOM],
            "exclusion": "aucune (pénalité continue)",
        },
        "score_source": scores.attrs.get("score_source", "tft_q50"),
        "as_of_date": str(pd.Timestamp(ranked["date_cours"].max()).date()),
        "universe_size": len(ranked),
        "label_counts": ranked["label"].value_counts().to_dict(),
        "sigmoid_curve_demo": {
            str(int(v)): round(float(f), 4) for v, f in zip(demo_vmq, demo_f)
        },
        "ranked_path": str(ranked_path),
        "top_15": top_15.to_dict(orient="records"),
    }

    if run_optimize:
        logger.info("Sous-module D — optimisation NSGA-III sur labels TOP ...")
        cours = _load_parquet("market_data_cours_historique", DATA_PROCESSED_DIR)
        indices = _load_parquet("market_data_indices_historique", DATA_PROCESSED_DIR)
        portfolios = optimize_portfolios(ranked, cours, indices)
        report["portfolios"] = portfolios

        port_path = DATASET_DIR / "portfolios_nsga3.json"
        port_path.write_text(
            json.dumps(portfolios, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        report["portfolios_path"] = str(port_path)

        for key in ("P_agressif", "P_equilibre", "P_defensif"):
            p = portfolios[key]
            logger.info(
                "%s : %s pos | alpha=%.4f | CVaR=%.4f | liq=%.3f | top=%s",
                key,
                p["n_positions"],
                p.get("expected_alpha", float("nan")),
                p.get("cvar_95", float("nan")),
                p.get("liquidity_score", float("nan")),
                list(zip(p["tickers"][:3], p["weights_pct"][:3])),
            )

    json_path = REPORTS_DIR / "step7_validation_report.json"
    md_path = REPORTS_DIR / "liquidity_filter_report.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_markdown_report(report, md_path)

    logger.info("Rapport : %s", json_path)
    logger.info("Markdown : %s", md_path)
    logger.info("Sous-modules C/D terminés.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
