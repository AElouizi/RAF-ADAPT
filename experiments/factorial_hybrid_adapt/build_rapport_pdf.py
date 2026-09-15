"""
Génère le rapport PDF RAF-ADAPT (étapes, résultats, architecture web).

Usage :
    py experiments/factorial_hybrid_adapt/build_rapport_pdf.py

Sortie :
    outputs/reports/rapport_projet_RAF-ADAPT.pdf
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from fpdf import FPDF

from experiments.factorial_hybrid_adapt.bloc_b_models import MODEL_IDS, STRATEGY_LABELS

BASE = Path(__file__).resolve().parent
REPORTS = BASE / "outputs" / "reports"
MART = BASE / "outputs" / "platform_mart"
MART_C1C5 = MART / "c1c5"

FONT_DIR = Path("C:/Windows/Fonts")
ARIAL = FONT_DIR / "arial.ttf"
ARIAL_BOLD = FONT_DIR / "arialbd.ttf"


class RapportPDF(FPDF):
    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        if ARIAL.is_file():
            self.add_font("Arial", "", str(ARIAL))
            self.add_font("Arial", "B", str(ARIAL_BOLD))
            self._font = "Arial"
        else:
            self._font = "Helvetica"

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_x(10)
        self.set_font(self._font, "B", 9)
        self.set_text_color(100, 116, 139)
        self.cell(95, 8, "RAF-ADAPT - Rapport projet", align="L")
        self.cell(95, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(14, 116, 144)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font(self._font, "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, "Document généré automatiquement — relecture méthodologique recommandée", align="C")

    def cover(self, title: str, subtitle: str) -> None:
        self.add_page()
        self.set_fill_color(12, 35, 64)
        self.rect(0, 0, 210, 90, style="F")
        self.set_xy(10, 28)
        self.set_font(self._font, "B", 26)
        self.set_text_color(255, 255, 255)
        self.multi_cell(190, 12, title, align="C")
        self.set_x(10)
        self.set_font(self._font, "", 13)
        self.set_text_color(200, 230, 240)
        self.multi_cell(190, 8, subtitle, align="C")
        self.set_y(100)
        self.set_text_color(30, 30, 30)
        self.set_font(self._font, "", 11)
        self.set_x(10)
        self.multi_cell(
            190,
            6,
            f"Date : {date.today().isoformat()}\n"
            "Periode d'evaluation : juillet 2018 - juin 2025 (84 mois)\n"
            "Donnees cours MASI : janvier 2010 - juin 2025\n"
            "Application web : http://localhost:8501",
        )

    def h1(self, text: str) -> None:
        self.ln(4)
        self.set_x(10)
        self.set_font(self._font, "B", 16)
        self.set_text_color(12, 35, 64)
        self.multi_cell(190, 8, text)
        self.set_draw_color(14, 116, 144)
        self.line(10, self.get_y() + 1, 200, self.get_y() + 1)
        self.ln(6)

    def h2(self, text: str) -> None:
        self.ln(2)
        self.set_x(10)
        self.set_font(self._font, "B", 13)
        self.set_text_color(26, 74, 92)
        self.multi_cell(190, 7, text)
        self.ln(2)

    def h3(self, text: str) -> None:
        self.set_x(10)
        self.set_font(self._font, "B", 11)
        self.set_text_color(40, 40, 40)
        self.multi_cell(190, 6, text)
        self.ln(1)

    def body(self, text: str) -> None:
        self.set_x(10)
        self.set_font(self._font, "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(190, 5, text)
        self.ln(2)

    def bullet(self, text: str) -> None:
        self.set_x(10)
        self.set_font(self._font, "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(190, 5, f"- {text}")

    def table(self, headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None) -> None:
        if not rows:
            return
        self.set_x(10)
        n = len(headers)
        widths = col_widths or [190 / n] * n
        self.set_font(self._font, "B", 9)
        self.set_fill_color(232, 238, 244)
        for i, h in enumerate(headers):
            self.cell(widths[i], 7, h, border=1, fill=True)
        self.ln()
        self.set_x(10)
        self.set_font(self._font, "", 9)
        for row in rows:
            for i, val in enumerate(row):
                self.cell(widths[i], 6, str(val)[:80], border=1)
            self.ln()
            self.set_x(10)
        self.ln(3)

    def code_block(self, text: str) -> None:
        self.set_x(10)
        self.set_fill_color(244, 246, 248)
        self.set_font(self._font, "", 8)
        self.multi_cell(190, 4, text, fill=True)
        self.ln(2)


def _load_bloc_d_summary() -> pd.DataFrame:
    p = REPORTS / "bloc_d_summary_14blocks.csv"
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    order = list(MODEL_IDS)
    df["_o"] = df["model_id"].map({m: i for i, m in enumerate(order)})
    return df.sort_values("_o").drop(columns=["_o"])


def _load_meta() -> dict:
    p = MART_C1C5 / "meta_c1c5.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    p2 = MART / "meta_platform.json"
    if p2.is_file():
        return json.loads(p2.read_text(encoding="utf-8"))
    return {}


def build_pdf(out_path: Path) -> Path:
    pdf = RapportPDF()
    pdf.cover(
        "RAF-ADAPT",
        "Recommandation d'actions BVC - Blocs A-D - NSGA-III - Plateforme web",
    )

    pdf.add_page()
    pdf.h1("Sommaire")
    toc = [
        "1. Vue d'ensemble et architecture globale",
        "2. Étape 0 — Données et features",
        "3. Étape 1 — Bloc A (régime de marché)",
        "4. Étape 2 — Walk-forward et plis",
        "5. Stratégies C1–C5 (sélection Bloc B)",
        "6. Étape 5 — Recommandations BUY / NEUTRAL / SELL",
        "7. Étape 6 — Bloc C (liquidité VMQ)",
        "8. Bloc D — NSGA-III × C1–C5 × 14 blocs",
        "9. Mart plateforme C1–C5",
        "10. Application web (sélecteur C1–C5)",
        "11. Synthèse décisionnelle et livrables",
    ]
    for line in toc:
        pdf.bullet(line)

    # --- Vue d'ensemble ---
    pdf.add_page()
    pdf.h1("1. Vue d'ensemble et architecture globale")
    pdf.body(
        "RAF-ADAPT est un système de recommandation d'actions pour la Bourse de Casablanca (BVC). "
        "Le pipeline est structuré en blocs fonctionnels avec protocole walk-forward strict "
        "(train 36 mois, test 6 mois, anti look-ahead)."
    )
    pdf.table(
        ["Bloc", "Rôle", "Statut"],
        [
            ["A", "Régime de marché (haussier / neutre / baissier)", "Validé"],
            ["B", "Scoring C1–C5 (sélection)", "5 stratégies actives"],
            ["C", "Liquidité (filtre VMQ + sigmoïde L)", "Intégré NSGA"],
            ["D", "Allocation NSGA-III (α ↑, CVaR ↓)", "5 × 14 blocs"],
        ],
        [25, 110, 35],
    )
    pdf.h2("Chaîne de traitement")
    pdf.code_block(
        "Données BVC (Supabase)\n"
        "    → Features techniques / fondamentales (z-score cross-sectionnel)\n"
        "    → Bloc A : régime de marché\n"
        "    → Bloc B : scores ML (walk-forward 81 plis + 14 blocs)\n"
        "    → Étage 1 : recommandations BUY / NEUTRAL / SELL (τ causal)\n"
        "    → Bloc C : filtre liquidité VMQ ≥ 500 k MAD\n"
        "    → Bloc D : NSGA-III (knee, wᵢ ≤ 10 %)\n"
        "    → Backtest + mart plateforme + dashboard Streamlit"
    )

    # --- Étapes condensées ---
    pdf.add_page()
    pdf.h1("2. Étape 0 — Données et features")
    pdf.h3("Description")
    pdf.body(
        "Chargement Supabase, nettoyage des splits, construction des features techniques et "
        "fondamentales z-scorées cross-sectionnellement → ml_dataset.parquet."
    )
    pdf.h3("Résultats")
    pdf.bullet("Panel ML depuis juin 2015 (features z-scorées)")
    pdf.bullet("Cible : alpha_ajuste_risque (excès vs MASI / vol baissière 20j)")
    pdf.bullet("Cours et MASI disponibles depuis janvier 2010")

    pdf.h1("3. Étape 1 — Bloc A : Régime de marché")
    pdf.body(
        "Détection mensuelle via masi_mom_3m et breadth_ma50 (terciles calibrés train 2015–2020). "
        "Variables : is_bull, is_neutral, is_bear. Jointure au panel pour C2/C4 et dashboard."
    )

    pdf.h1("4. Étape 2 — Walk-forward")
    pdf.body(
        "81 plis chevauchants (pas 1 mois) pour métriques descriptives. "
        "14 blocs indépendants (pas 6 mois, fold_ids 0, 6, 12, …, 78) pour tests statistiques. "
        "Premier mois test OOS : juillet 2018."
    )

    pdf.add_page()
    pdf.h1("5. Stratégies C1–C5 (sélection Bloc B)")
    pdf.table(
        ["Strat.", "model_id", "Algorithme", "Régime"],
        [
            ["C1", "c1_ridge", "Ridge", "Non"],
            ["C2", "c2_rf", "Random Forest", "Non"],
            ["C3", "c3_lightgbm", "LightGBM (Optuna)", "Non"],
            ["C4", "c4_hybrid_tri", "Ridge+RF+LGBM (z-score CS)", "Non"],
            ["C5", "c5_hybrid_tri_regime", "Ridge+RF+LGBM+régime", "Oui"],
        ],
        [22, 42, 80, 25],
    )
    pdf.body(
        "Chaque stratégie produit des scores puis BUY/NEUTRAL/SELL (τ expanding causal). "
        "Le même identifiant alimente NSGA-III en Bloc D (un portefeuille par stratégie)."
    )

    pdf.h1("6. Recommandations BUY / NEUTRAL / SELL")
    pdf.body(
        "Transformation scores → recommandations via seuil τ (quantile expanding causal). "
        "Titres SELL exclus avant NSGA. Artefact : stage2_inputs_14blocks.parquet."
    )

    pdf.h1("7. Bloc C : Liquidité")
    pdf.body(
        "Filtre VMQ optionnel ; facteur sigmoïde L(VMQ). Mode protocole : pareto "
        "(NSGA 3 objectifs : α, CVaR, liquidité L(VMQ) — pas de filtre VMQ ≥ 500 k)."
    )

    pdf.add_page()
    pdf.h1("8. Bloc D — NSGA-III × C1–C5 × 14 blocs")
    pdf.body(
        "NSGA-III en boucle indépendante pour chaque modèle Bloc B × chaque bloc de test. "
        "Scores isolés par bloc (pas de fuite OOS). Benchmarks : MASI Sharpe 0,67 ; EW MASI20 0,99."
    )
    bloc_d = _load_bloc_d_summary()
    if not bloc_d.empty:
        rows = []
        labels = STRATEGY_LABELS
        for _, r in bloc_d.iterrows():
            mid = str(r["model_id"])
            rows.append(
                [
                    labels.get(mid, mid),
                    str(r["groupe"]),
                    f"{float(r['sharpe']):.3f}",
                    f"{100 * float(r['ann_return']):.1f} %",
                    f"{100 * float(r['max_drawdown']):.1f} %",
                ]
            )
        pdf.table(
            ["Modèle", "Groupe", "Sharpe moy.", "Rend. ann.", "Max DD"],
            rows,
            [45, 35, 25, 25, 25],
        )
    pdf.body(
        "Friedman omnibus (financier) : Sharpe/rendement p ≈ 0,42 ; CVaR p = 0,026 ; turnover p = 0,002."
    )

    pdf.h1("9. Mart plateforme C1–C5")
    pdf.body(
        "outputs/platform_mart/c1c5/ : recommandations, titres, holdings et historique "
        "mensuel pour chaque model_id (colonne model_id dans les parquets)."
    )

    pdf.add_page()
    pdf.h1("10. Application web")
    pdf.h2("Stack technique")
    pdf.table(
        ["Composant", "Technologie"],
        [
            ["UI", "Streamlit (layout wide, sidebar calendrier)"],
            ["Graphiques", "Plotly (courbes richesse, drawdown, histogrammes)"],
            ["Données reco / portefeuille", "Parquet mart (platform_mart/, lecture seule)"],
            ["Performance 5 stratégies", "bloc_d_wf_14blocks/ + bloc_d_summary_14blocks.csv"],
            ["Simulation buy-and-hold", "simulation_engine.py (prix journaliers)"],
            ["Agents (optionnel)", "bvc_recommender/agents/ + admin_agents.py"],
        ],
        [55, 135],
    )

    pdf.h2("Fichiers principaux")
    pdf.code_block(
        "experiments/factorial_hybrid_adapt/web/\n"
        "  app_c4.py           — Dashboard principal (8 onglets)\n"
        "  data_c4.py          — Chargeurs mart + Bloc D (cache Streamlit)\n"
        "  simulation_ui.py    — Onglet simulation investisseur\n"
        "  simulation_engine.py — Moteur backtest utilisateur\n"
        "  admin_agents.py     — Orchestration multi-agents\n"
        "Lancer_RAF_ADAPT.bat  — Double-clic → port 8501"
    )

    pdf.h2("Lancement")
    pdf.code_block(
        "cd Projet_Cursor_RS_final\n"
        "py -m streamlit run experiments/factorial_hybrid_adapt/web/app_c4.py\n"
        "URL : http://localhost:8501"
    )
    pdf.body("Ne pas utiliser bvc_recommender/app.py (ancienne version TFT + 3 portefeuilles).")

    pdf.h2("Onglets du dashboard")
    pdf.table(
        ["#", "Onglet", "Contenu"],
        [
            ["1", "Recommandations", "Titres stratégie sélectionnée, BUY/NEUTRAL/SELL, alpha, régime, liquidité"],
            ["2", "Portefeuille", "Poids NSGA knee, contributions, camembert top positions"],
            ["3", "Simulation", "Buy-and-hold utilisateur vs IA vs MASI"],
            ["4", "Historique 2018–2025", "Richesse mensuelle par stratégie C1–C5"],
            ["5", "Performance", "5 stratégies Bloc D + MASI : KPI, courbes, drawdowns"],
            ["6", "C1–C5 & tests", "Tableau Bloc D par model_id ; tests statistiques"],
            ["7", "Marché depuis 2010", "MASI long terme, bande verte période OOS"],
            ["8", "Administration", "Workflow agents, logs, statuts jobs"],
        ],
        [12, 45, 133],
    )

    pdf.h2("Flux de données web")
    pdf.code_block(
        "platform_mart/c1c5/ (sélecteur sidebar)\n"
        "  fact_recommendations_c1c5.parquet  → Recommandations\n"
        "  fact_portfolio_holdings_c1c5.parquet → Portefeuille\n"
        "  fact_portfolio_monthly_c1c5.parquet  → Historique / KPI mois\n"
        "  fact_masi_since_2010.csv             → Marché long terme\n"
        "  meta_c1c5.json                       → calendrier + métadonnées\n\n"
        "bloc_d_wf_14blocks/ (5 stratégies C1–C5)\n"
        "  fold_XXXX_<model_id>_weights.parquet → simulation bloc\n"
        "  → load_bloc_d_daily_wealth()         → courbes Performance\n\n"
        "reports/bloc_d_summary_14blocks.csv → KPI Sharpe moyen 14 blocs"
    )

    pdf.h2("Modèles affichés")
    pdf.body(
        "Sidebar : sélecteur C1–C5 (model_id). Recommandations, portefeuille et historique "
        "suivent la stratégie active. Onglet Performance : courbes NSGA des 5 stratégies + MASI."
    )

    pdf.add_page()
    pdf.h1("11. Synthèse décisionnelle")
    pdf.table(
        ["Question", "Réponse"],
        [
            ["Stratégies actives", "C1 Ridge … C5 Hybrid triple + régime"],
            ["Sélection vs allocation", "Bloc B (scores/reco) puis Bloc D (NSGA)"],
            ["Mart dashboard", "platform_mart/c1c5/ (une ligne par model_id)"],
            ["Recalcul complet", "run_c1c5_pipeline.py"],
        ],
        [70, 120],
    )

    pdf.h2("Arborescence des livrables")
    pdf.code_block(
        "experiments/factorial_hybrid_adapt/outputs/\n"
        "  reports/\n"
        "    rapport_projet_RAF-ADAPT.pdf    ← ce document\n"
        "    rapport_projet_RAF-ADAPT.md\n"
        "    bloc_d_summary_14blocks.csv\n"
        "    bloc_b_summary_all.json\n"
        "  platform_mart/c1c5/               ← dashboard web (C1–C5)\n"
        "  final_experiment_2010_2025/       ← livraison historique\n"
        "  bloc_b_wf_overlapping/            ← prédictions 81 folds\n"
        "  bloc_d_wf_14blocks/               ← NSGA 5 stratégies"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    return out_path


def main() -> int:
    out = REPORTS / "rapport_projet_RAF-ADAPT.pdf"
    path = build_pdf(out)
    print(f"PDF écrit : {path}")
    print(f"Ouvrir : {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
