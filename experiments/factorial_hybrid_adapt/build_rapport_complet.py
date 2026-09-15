"""
Rapport complet RAF-ADAPT : étapes, équations, tableaux et graphiques.

Usage :
    py experiments/factorial_hybrid_adapt/build_rapport_complet.py

Sorties :
    outputs/reports/rapport_complet_RAF-ADAPT.pdf
    outputs/reports/rapport_complet_RAF-ADAPT.md
    outputs/reports/rapport_complet_RAF-ADAPT.html
    outputs/reports/figures/*.png
"""

from __future__ import annotations

import json
import sys
import textwrap
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.factorial_hybrid_adapt.bloc_b_models import MODEL_IDS, STRATEGY_LABELS
from experiments.factorial_hybrid_adapt.build_rapport_pdf import RapportPDF

BASE = Path(__file__).resolve().parent
REPORTS = BASE / "outputs" / "reports"
FIGURES = REPORTS / "figures"
MART = BASE / "outputs" / "platform_mart"
MART_BEST = MART / "best"
MART_C1C5 = MART / "c1c5"


class RapportCompletPDF(RapportPDF):
    def equation(self, text: str) -> None:
        self.set_x(12)
        self.set_fill_color(248, 250, 252)
        self.set_font(self._font, "", 9)
        self.set_text_color(20, 20, 20)
        self.multi_cell(186, 5, text, fill=True)
        self.ln(3)

    def figure(self, path: Path, width: float = 175, title: str | None = None) -> None:
        if not path.is_file():
            self.body(f"[Figure manquante : {path.name}]")
            return
        if title:
            self.h3(title)
        if self.get_y() > 200:
            self.add_page()
        self.set_x(10)
        self.image(str(path), w=width)
        self.ln(6)


def _bloc_d() -> pd.DataFrame:
    p = REPORTS / "bloc_d_summary_14blocks.csv"
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    order = {m: i for i, m in enumerate(MODEL_IDS)}
    df["_o"] = df["model_id"].map(order)
    return df.sort_values("_o").drop(columns=["_o"])


def _bloc_b_rank_ic() -> pd.Series:
    p = REPORTS / "bloc_b_results_long_overlapping.csv"
    if not p.is_file():
        return pd.Series(dtype=float)
    df = pd.read_csv(p)
    sub = df[df["metrique"] == "rank_ic"]
    return sub.groupby("model_id")["valeur"].mean()


def _bootstrap_rank_ic() -> dict[str, dict]:
    p = REPORTS / "bloc_b_summary_all.json"
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    boot = data.get("analysis", {}).get("bootstrap_81folds", {}).get("rank_ic", {})
    return boot if isinstance(boot, dict) else {}


def _gen_figures() -> dict[str, Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    labels = [STRATEGY_LABELS[m] for m in MODEL_IDS]

    # Sharpe Bloc D
    bd = _bloc_d()
    if not bd.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#ff7f0e", "#15803d", "#7c3aed", "#1f77b4", "#b45309"]
        ax.barh(
            [STRATEGY_LABELS[str(r["model_id"])] for _, r in bd.iterrows()],
            bd["sharpe"],
            color=colors[: len(bd)],
        )
        ax.axvline(0.67, color="#1d4ed8", linestyle="--", label="MASI ~0,67")
        ax.set_xlabel("Sharpe moyen (14 blocs post-NSGA)")
        ax.set_title("Bloc D — performance financière C1–C5")
        ax.legend()
        fig.tight_layout()
        p = FIGURES / "sharpe_bloc_d.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out["sharpe"] = p

    # Rank-IC Bloc B
    ric = _bloc_b_rank_ic()
    if not ric.empty:
        ric = ric.reindex(MODEL_IDS).dropna()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(range(len(ric)), ric.values, color="#0e7490")
        ax.set_xticks(range(len(ric)))
        ax.set_xticklabels([STRATEGY_LABELS[m] for m in ric.index], rotation=20, ha="right")
        ax.set_ylabel("Rank-IC moyen (81 folds)")
        ax.set_title("Bloc B — qualité prédictive")
        fig.tight_layout()
        p = FIGURES / "rank_ic_bloc_b.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        out["rank_ic"] = p

    # Richesse champion
    monthly_p = MART_BEST / "fact_portfolio_monthly.parquet"
    if monthly_p.is_file():
        mm = pd.read_parquet(monthly_p)
        if "wealth100_net" in mm.columns:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(mm["month"], mm["wealth100_net"], color="#0e7490", lw=2)
            ax.set_title("Portefeuille champion (C3 LightGBM) — richesse base 100")
            ax.set_ylabel("Richesse")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            p = FIGURES / "wealth_champion.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            out["wealth"] = p

    # Part des algos sélectionnés (meta)
    titles_p = MART_BEST / "fact_titles_month.parquet"
    if titles_p.is_file():
        t = pd.read_parquet(titles_p)
        if "selected_model_id" in t.columns:
            counts = t["selected_model_id"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(
                counts.values,
                labels=[STRATEGY_LABELS.get(k, k) for k in counts.index],
                autopct="%1.1f%%",
                startangle=140,
            )
            ax.set_title("Meta-sélection : part des algos gagnants (tous mois × titres)")
            fig.tight_layout()
            p = FIGURES / "algo_selection_share.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            out["algo_share"] = p

    return out


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_markdown(figs: dict[str, Path]) -> Path:
    bd = _bloc_d()
    ric = _bloc_b_rank_ic()
    boot = _bootstrap_rank_ic()

    lines = [
        "# Rapport complet — RAF-ADAPT",
        f"**Date** : {date.today().isoformat()}",
        "**Période OOS** : juillet 2018 → juin 2025 (84 mois)",
        "",
        "## Sommaire",
        "1. Architecture et protocole",
        "2. Données et features",
        "3. Bloc A — Régime",
        "4. Walk-forward",
        "5. Bloc B — C1–C5 et équations ML",
        "6. Recommandations BUY / NEUTRAL / SELL",
        "7. Bloc C — Liquidité",
        "8. Bloc D — NSGA-III",
        "9. Meta-sélection et mart plateforme",
        "10. Synthèse des résultats",
        "",
        "---",
        "",
        "## 1. Architecture",
        "",
        "Pipeline : Données BVC → features z-scorées → Bloc A → Bloc B (5 algos) → "
        "reco τ causal → Bloc C → Bloc D NSGA → mart `best/` → dashboard Streamlit.",
        "",
        "## 5. Bloc B — Équations",
        "",
        "**Cible (alpha ajusté risque)** :",
        "```",
        "α_i,t = (r_i,t - r_MASI,t) / σ_down_i,t",
        "```",
        "",
        "**Ridge (C1)** : `min_w ||y - Xw||² + λ||w||²`",
        "",
        "**Hybride triple (C4/C5)** : z-score cross-sectionnel puis",
        "```",
        "score = (z_Ridge + z_RF + z_LGBM) / 3",
        "```",
        "",
        "**Rank-IC moyen (81 folds)** :",
    ]

    if not ric.empty:
        rows = []
        for mid in MODEL_IDS:
            if mid not in ric.index:
                continue
            b = boot.get(mid, {})
            lo = b.get("ci_low")
            hi = b.get("ci_high")
            ic95 = f"[{lo:.3f}, {hi:.3f}]" if lo is not None and hi is not None else "—"
            rows.append([STRATEGY_LABELS[mid], f"{ric[mid]:.4f}", ic95])
        lines.append(_md_table(["Stratégie", "Rank-IC", "IC95 bootstrap"], rows))

    if "rank_ic" in figs:
        lines += ["", f"![Rank-IC](figures/{figs['rank_ic'].name})", ""]

    lines += [
        "## 8. Bloc D — NSGA-III",
        "",
        "**Objectifs** (pareto — 3 objectifs NSGA) :",
        "```",
        "max  α_port = Σ w_i · α_i",
        "min  CVaR_95(w)",
        "max  L_port = Σ w_i · L(VMQ_i)   avec L = sigmoid(VMQ)",
        "s.c. Σ w_i = 1,  0 ≤ w_i ≤ 0,10,  CVaR(w) ≤ 0,9 × CVaR(MASI)",
        "```",
        "",
        "Pas de filtre VMQ ≥ 500 k : la liquidité est arbitrée sur le front de Pareto.",
        "",
        "**Sélection portefeuille** : point knee sur le front de Pareto (distance utopie).",
        "",
    ]

    if not bd.empty:
        rows = []
        for _, r in bd.iterrows():
            mid = str(r["model_id"])
            rows.append(
                [
                    STRATEGY_LABELS[mid],
                    f"{float(r['sharpe']):.3f}",
                    f"{100 * float(r['ann_return']):.1f} %",
                    f"{100 * float(r['max_drawdown']):.1f} %",
                    f"{float(r['turnover_mean']):.3f}",
                ]
            )
        lines.append(
            _md_table(["Stratégie", "Sharpe", "Rend. ann.", "Max DD", "Turnover"], rows)
        )

    if "sharpe" in figs:
        lines += ["", f"![Sharpe Bloc D](figures/{figs['sharpe'].name})", ""]
    if "wealth" in figs:
        lines += ["", f"![Richesse champion](figures/{figs['wealth'].name})", ""]
    if "algo_share" in figs:
        lines += ["", f"![Part algos](figures/{figs['algo_share'].name})", ""]

    lines += [
        "",
        "## 9. Livrable dashboard",
        "",
        "- **Sélection** : meilleur alpha parmi C1–C5 par titre et par mois.",
        "- **Portefeuille** : NSGA du champion Sharpe (C3 LightGBM).",
        "- Mart : `outputs/platform_mart/best/`",
        "",
        "## 10. Synthèse",
        "",
        "| Question | Réponse |",
        "|----------|---------|",
        "| Meilleur prédictif (Rank-IC) | C3 LightGBM |",
        "| Meilleur financier post-NSGA | C3 LightGBM (Sharpe 1,31) |",
        "| Dashboard | Meta-sélection + portefeuille champion |",
        "",
    ]

    out = REPORTS / "rapport_complet_RAF-ADAPT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def build_html(md_path: Path, figs: dict[str, Path]) -> Path:
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>RAF-ADAPT — Rapport complet</title>",
        "<style>body{font-family:Segoe UI,Arial;margin:2rem;max-width:900px;line-height:1.5}",
        "h1{color:#0c2340}h2{color:#1a4a5c;margin-top:2rem}img{max-width:100%;border:1px solid #ddd}",
        "pre{background:#f4f6f8;padding:1rem}table{border-collapse:collapse;width:100%}",
        "th,td{border:1px solid #ccc;padding:6px}</style></head><body>",
        "<h1>Rapport complet RAF-ADAPT</h1>",
        f"<p>Généré le {date.today().isoformat()}</p>",
    ]
    for key, title in [
        ("sharpe", "Sharpe Bloc D"),
        ("rank_ic", "Rank-IC Bloc B"),
        ("wealth", "Richesse portefeuille champion"),
        ("algo_share", "Part des algorithmes en meta-sélection"),
    ]:
        if key in figs:
            html.append(f"<h2>{title}</h2><img src='figures/{figs[key].name}' alt='{title}'/>")

    html.append(f"<h2>Document Markdown</h2><pre>{md_path.read_text(encoding='utf-8')}</pre>")
    html.append("</body></html>")
    out = REPORTS / "rapport_complet_RAF-ADAPT.html"
    out.write_text("\n".join(html), encoding="utf-8")
    return out


def build_pdf(figs: dict[str, Path], out_path: Path) -> Path:
    pdf = RapportCompletPDF()
    pdf.cover(
        "RAF-ADAPT — Rapport complet",
        "Étapes · Équations · Tableaux · Graphiques — C1–C5 · NSGA-III",
    )

    pdf.add_page()
    pdf.h1("Sommaire")
    for i, t in enumerate(
        [
            "Architecture globale",
            "Étape 0 — Données & features",
            "Étape 1 — Bloc A (régime)",
            "Étape 2 — Walk-forward",
            "Étape 3 — Bloc B C1–C5 (équations ML)",
            "Étape 4 — Recommandations BUY/NEUTRAL/SELL",
            "Étape 5 — Bloc C liquidité",
            "Étape 6 — Bloc D NSGA-III",
            "Étape 7 — Meta-sélection & mart",
            "Résultats — graphiques",
            "Synthèse décisionnelle",
        ],
        start=1,
    ):
        pdf.bullet(f"{i}. {t}")

    pdf.add_page()
    pdf.h1("1. Architecture globale")
    pdf.body(
        "RAF-ADAPT recommande des actions BVC via un pipeline walk-forward strict "
        "(train 36 mois, test 6 mois, pas 1 mois). Cinq algorithmes de scoring (C1–C5) "
        "alimentent une meta-sélection par titre ; le portefeuille NSGA utilise le champion financier."
    )
    pdf.table(
        ["Bloc", "Rôle"],
        [
            ["A", "Régime haussier / neutre / baissier"],
            ["B", "Scoring ML C1–C5"],
            ["C", "Filtre liquidité VMQ"],
            ["D", "Allocation NSGA-III (α ↑, CVaR ↓)"],
        ],
        [30, 160],
    )

    pdf.h1("2. Étape 0 — Données & features")
    pdf.equation("α_i,t = (r_i,t - r_MASI,t) / σ_down_i,t   (cible alpha_ajuste_risque)")
    pdf.equation("Features : z-score cross-sectionnel par date sur indicateurs techniques et fondamentaux")
    pdf.bullet("Panel ML depuis juin 2015 · MASI depuis janvier 2010")

    pdf.h1("3. Bloc A — Régime")
    pdf.body("Terciles sur masi_mom_3m et breadth_ma50 (calibrage 2015–2020). Variables is_bull, is_neutral, is_bear.")

    pdf.h1("4. Walk-forward")
    pdf.bullet("81 plis overlapping (métriques descriptives)")
    pdf.bullet("14 blocs indépendants (tests statistiques, fold_id 0,6,…,78)")
    pdf.bullet("Premier OOS : juillet 2018")

    pdf.add_page()
    pdf.h1("5. Bloc B — Stratégies C1–C5")
    pdf.table(
        ["ID", "Algorithme"],
        [(m, STRATEGY_LABELS[m]) for m in MODEL_IDS],
        [50, 140],
    )
    pdf.h2("Équations")
    pdf.equation("C1 Ridge : min_w ||y - Xw||² + λ||w||²")
    pdf.equation("C2 RF : agrégation de arbres (grille hyperparamètres)")
    pdf.equation("C3 LGBM : gradient boosting (Optuna 9 trials, config gelée)")
    pdf.equation("C4/C5 : score = moyenne des z-scores CS de Ridge, RF et LGBM (C5 + features régime)")

    ric = _bloc_b_rank_ic()
    boot = _bootstrap_rank_ic()
    if not ric.empty:
        rows = []
        for mid in MODEL_IDS:
            if mid not in ric.index:
                continue
            b = boot.get(mid, {})
            lo, hi = b.get("ci_low"), b.get("ci_high")
            ic95 = f"{lo:.3f} – {hi:.3f}" if lo is not None and hi is not None else "—"
            rows.append([STRATEGY_LABELS[mid], f"{ric[mid]:.4f}", ic95])
        pdf.table(["Stratégie", "Rank-IC moy.", "IC95 bootstrap"], rows, [70, 35, 35])

    pdf.h1("6. Recommandations")
    pdf.equation("τ_t = quantile_expanding_past(scores, q)   (causal, pas de look-ahead)")
    pdf.equation("BUY si score > τ_t ; NEUTRAL si |score| faible ; SELL si score < -τ_t")
    pdf.body("Titres SELL exclus avant NSGA (sell_mode = exclude).")

    pdf.h1("7. Bloc C — Liquidité")
    pdf.equation("L(VMQ) = 1 / (1 + exp(-k·(VMQ - VMQ_ref)))")
    pdf.body(
        "Mode protocole plateforme : pareto — liquidité = 3ᵉ objectif NSGA (pas de filtre VMQ ≥ 500 k)."
    )

    pdf.add_page()
    pdf.h1("8. Bloc D — NSGA-III")
    pdf.equation("max α_port(w) = Σ_i w_i · α_i")
    pdf.equation("min CVaR_95(w) = moyenne des pertes au 5e percentile le plus défavorable")
    pdf.equation("max L_port(w) = Σ_i w_i · L(VMQ_i)")
    pdf.equation(
        "Σ w_i = 1 ; 0 ≤ w_i ≤ 0,10 ; CVaR(w) ≤ 0,9×CVaR(MASI) ; sélection knee sur front Pareto 3D"
    )
    pdf.body("Paramètres figés : pop=36, gen=30, 14 blocs × 5 stratégies.")

    bd = _bloc_d()
    if not bd.empty:
        rows = []
        for _, r in bd.iterrows():
            mid = str(r["model_id"])
            rows.append(
                [
                    STRATEGY_LABELS[mid],
                    f"{float(r['sharpe']):.3f}",
                    f"{100 * float(r['ann_return']):.1f}%",
                    f"{100 * float(r['max_drawdown']):.1f}%",
                    f"{float(r['sortino']):.2f}",
                    f"{float(r['turnover_mean']):.3f}",
                ]
            )
        pdf.table(
            ["Stratégie", "Sharpe", "Rend.ann.", "MaxDD", "Sortino", "Turnover"],
            rows,
            [55, 22, 25, 22, 22, 22],
        )
    pdf.body("Benchmarks : MASI Sharpe ~0,67 · EW MASI20 ~0,99")

    pdf.h1("9. Meta-sélection & mart")
    pdf.body(
        "Pour chaque (mois, ticker) : algo retenu = argmax alpha parmi C1–C5. "
        "Portefeuille dashboard = NSGA du champion Sharpe (C3 LightGBM). "
        "Mart : platform_mart/best/"
    )

    pdf.add_page()
    pdf.h1("10. Graphiques — résultats")
    if "rank_ic" in figs:
        pdf.figure(figs["rank_ic"], title="Rank-IC moyen (81 folds)")
    if "sharpe" in figs:
        pdf.figure(figs["sharpe"], title="Sharpe moyen post-NSGA (14 blocs)")
    if "wealth" in figs:
        pdf.figure(figs["wealth"], title="Richesse portefeuille champion")
    if "algo_share" in figs:
        pdf.figure(figs["algo_share"], title="Part des algos en meta-sélection")

    pdf.add_page()
    pdf.h1("11. Synthèse")
    pdf.table(
        ["Question", "Réponse"],
        [
            ["Meilleur prédictif (Rank-IC)", "C3 LightGBM"],
            ["Meilleur financier (Sharpe NSGA)", "C3 LightGBM (1,31)"],
            ["Sélection dashboard", "Max alpha par titre (C1–C5)"],
            ["Portefeuille dashboard", "NSGA C3 LightGBM"],
            ["URL application", "http://localhost:8501"],
        ],
        [65, 125],
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    return out_path


def main() -> int:
    print("Génération des figures…")
    figs = _gen_figures()
    md = build_markdown(figs)
    html = build_html(md, figs)
    pdf_path = REPORTS / "rapport_complet_RAF-ADAPT.pdf"
    print("Génération du PDF…")
    build_pdf(figs, pdf_path)
    # Mise à jour du rapport standard aussi
    from experiments.factorial_hybrid_adapt.build_rapport_pdf import build_pdf as build_std

    build_std(REPORTS / "rapport_projet_RAF-ADAPT.pdf")

    print(f"PDF complet : {pdf_path.resolve()}")
    print(f"HTML        : {html.resolve()}")
    print(f"Markdown    : {md.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
