"""
Pipeline Étape 9 — dashboard Streamlit.

Usage (depuis la racine du projet) :
    py -m bvc_recommender.scripts.run_step9
    py -m bvc_recommender.scripts.run_step9 --report-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.config import REPORTS_DIR  # noqa: E402
from bvc_recommender.dashboard.data import list_periods, list_quarters  # noqa: E402

APP_PATH = ROOT / "bvc_recommender" / "app.py"


def write_validation_report() -> Path:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": "BVC Recommender — Étape 9",
        "app_path": str(APP_PATH),
        "launch_command": "py -m streamlit run bvc_recommender/app.py",
        "periods_available": list_periods(),
        "quarters_available": list_quarters(),
        "tabs": [
            {
                "id": 1,
                "name": "Recommandations",
                "features": [
                    "Top 20 classées (score TFT × liquidité)",
                    "Indicateurs clés (P/E, P/B, ROE, RSI, Mom 3M, VMQ)",
                    "Badges NOUVEAU / SORTI vs trimestre précédent",
                ],
            },
            {
                "id": 2,
                "name": "Portefeuilles",
                "features": [
                    "3 cartes Agressif / Équilibré / Défensif",
                    "Métriques backtest + répartition des poids",
                ],
            },
            {
                "id": 3,
                "name": "Performance",
                "features": [
                    "Courbes cumulées vs MASI / MASI20",
                    "Tableau métriques (Sharpe, Sortino, Calmar, CVaR, Hit, TO trim.)",
                ],
            },
            {
                "id": 4,
                "name": "Historique",
                "features": [
                    "Table des trimestres cliquable",
                    "Graphique de rotation trimestrielle",
                ],
            },
            {
                "id": 5,
                "name": "Explication IA",
                "features": [
                    "Valeurs SHAP LightGBM",
                    "Note de régime de marché (HMM discret)",
                ],
            },
        ],
        "constraints": [
            "no look-ahead",
            "carry-forward only",
            "z-score cross-sectional",
            "sigmoid liquidity",
            "random_state=42",
        ],
        "dependencies": ["streamlit", "plotly", "shap", "lightgbm"],
    }
    path = REPORTS_DIR / "step9_validation_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="BVC Recommender — Étape 9")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Générer le rapport de validation sans lancer Streamlit",
    )
    args = parser.parse_args()

    if not APP_PATH.is_file():
        print(f"App introuvable : {APP_PATH}")
        return 1

    report_path = write_validation_report()
    print(f"Rapport étape 9 : {report_path}")

    if args.report_only:
        print("Mode --report-only : Streamlit non lancé.")
        return 0

    print("Lancement du dashboard Streamlit…")
    print("URL locale : http://localhost:8501")

    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(APP_PATH), "--server.headless", "true"],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    raise SystemExit(main())
