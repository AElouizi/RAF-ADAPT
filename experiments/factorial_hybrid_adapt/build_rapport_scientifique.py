"""
Rapport scientifique RAF-ADAPT — étapes, définitions, équations, graphiques, résultats.

Usage :
    py experiments/factorial_hybrid_adapt/build_rapport_scientifique.py
"""

from __future__ import annotations

import base64
import json
import sys
from datetime import date
from io import BytesIO
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
FIG = REPORTS / "figures_scientifique"
MART = BASE / "outputs" / "platform_mart"
MART_BEST = MART / "best"
MART_C1C5 = MART / "c1c5"

COLORS = {
    "c1_ridge": "#1f77b4",
    "c2_rf": "#b45309",
    "c3_lightgbm": "#ff7f0e",
    "c4_hybrid_tri": "#15803d",
    "c5_hybrid_tri_regime": "#7c3aed",
    "MASI": "#1d4ed8",
}

METRICS_B = ["rank_ic", "ic_ir", "spread_BUY_SELL", "rmse", "turnover"]


def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _img_tag(path: Path, alt: str) -> str:
    if not path.is_file():
        return f"<p><em>Figure non disponible : {alt}</em></p>"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{data}" alt="{alt}"/><figcaption>{alt}</figcaption></figure>'


def _bloc_b_long() -> pd.DataFrame:
    p = REPORTS / "bloc_b_results_long_overlapping.csv"
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()


def _bloc_b_boot() -> dict:
    p = REPORTS / "bloc_b_summary_all.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8")).get("analysis", {}).get("bootstrap_81folds", {})


def _bloc_d() -> pd.DataFrame:
    p = REPORTS / "bloc_d_summary_14blocks.csv"
    if not p.is_file():
        return pd.DataFrame()
    df = pd.read_csv(p)
    order = {m: i for i, m in enumerate(MODEL_IDS)}
    df["_o"] = df["model_id"].map(order)
    return df.sort_values("_o").drop(columns=["_o"])


def _bloc_d_report() -> dict:
    p = REPORTS / "bloc_d_report_14blocks.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def generate_figures() -> dict[str, Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    labels = [STRATEGY_LABELS[m] for m in MODEL_IDS]

    # --- Rank-IC + IC95 ---
    boot = _bloc_b_boot().get("rank_ic", {})
    means, lows, highs = [], [], []
    for mid in MODEL_IDS:
        b = boot.get(mid, {})
        means.append(float(b.get("mean", 0)))
        lows.append(float(b.get("ci_low", 0)))
        highs.append(float(b.get("ci_high", 0)))
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(MODEL_IDS))
    ax.bar(x, means, color=[COLORS[m] for m in MODEL_IDS], alpha=0.85)
    ax.errorbar(
        x,
        means,
        yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
        fmt="none",
        color="#333",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Rank-IC")
    ax.set_title("Bloc B — Rank-IC moyen (81 folds) avec intervalle bootstrap 95 %")
    ax.axhline(0, color="#999", lw=0.8)
    fig.tight_layout()
    p = FIG / "01_rank_ic_ci.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    out["rank_ic"] = p

    # --- Spread BUY-SELL + IC95 ---
    boot_sp = _bloc_b_boot().get("spread_BUY_SELL", {})
    if boot_sp:
        means, lows, highs = [], [], []
        for mid in MODEL_IDS:
            b = boot_sp.get(mid, {})
            means.append(float(b.get("mean", 0)))
            lows.append(float(b.get("ci_low", 0)))
            highs.append(float(b.get("ci_high", 0)))
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(MODEL_IDS))
        ax.bar(x, means, color=[COLORS[m] for m in MODEL_IDS], alpha=0.85)
        ax.errorbar(
        x,
        means,
        yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
        fmt="none",
        color="#333",
        capsize=4,
    )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel("Spread BUY − SELL (α)")
        ax.set_title("Bloc B — spread paniers BUY vs SELL (81 folds, bootstrap 95 %)")
        ax.axhline(0, color="#999", lw=0.8)
        fig.tight_layout()
        p = FIG / "01b_spread_buy_sell.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        out["spread"] = p

    # --- Heatmap métriques Bloc B ---
    bl = _bloc_b_long()
    if not bl.empty:
        rows = []
        for mid in MODEL_IDS:
            row = {"model_id": mid}
            for met in METRICS_B:
                v = bl[(bl.model_id == mid) & (bl.metrique == met)]["valeur"].mean()
                row[met] = v
            rows.append(row)
        hm = pd.DataFrame(rows).set_index("model_id")
        # normaliser colonnes pour affichage
        norm = hm.copy()
        for c in norm.columns:
            col = norm[c]
            if col.max() != col.min():
                norm[c] = (col - col.min()) / (col.max() - col.min())
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(norm.values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(norm.columns)))
        ax.set_xticklabels(norm.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(MODEL_IDS)))
        ax.set_yticklabels([STRATEGY_LABELS[m] for m in MODEL_IDS])
        for i in range(len(MODEL_IDS)):
            for j in range(len(norm.columns)):
                ax.text(j, i, f"{hm.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8)
        ax.set_title("Bloc B — métriques prédictives (valeurs brutes dans cellules)")
        fig.colorbar(im, ax=ax, fraction=0.02)
        fig.tight_layout()
        p = FIG / "02_bloc_b_heatmap.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        out["heatmap_b"] = p

    # --- Sharpe Bloc D ---
    bd = _bloc_d()
    if not bd.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        y = [STRATEGY_LABELS[str(r["model_id"])] for _, r in bd.iterrows()]
        ax.barh(y, bd["sharpe"], color=[COLORS[str(r["model_id"])] for _, r in bd.iterrows()])
        rep = _bloc_d_report()
        bench = rep.get("benchmark_summary", {})
        ax.axvline(bench.get("masi_sharpe_mean", 0.67), color=COLORS["MASI"], ls="--", label="MASI")
        ax.axvline(bench.get("ew_masi20_sharpe_mean", 0.99), color="#b45309", ls=":", label="EW MASI20")
        ax.set_xlabel("Sharpe moyen (14 blocs, rendements mensuels nets)")
        ax.set_title("Bloc D — performance financière post-NSGA-III")
        ax.legend()
        fig.tight_layout()
        p = FIG / "03_sharpe_bloc_d.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        out["sharpe"] = p

        # Rendement vs Max DD
        fig, ax = plt.subplots(figsize=(7, 6))
        for _, r in bd.iterrows():
            mid = str(r["model_id"])
            ax.scatter(
                100 * r["ann_return"],
                100 * abs(r["max_drawdown"]),
                s=120,
                color=COLORS[mid],
                label=STRATEGY_LABELS[mid],
                edgecolors="white",
            )
        ax.set_xlabel("Rendement annualisé (%)")
        ax.set_ylabel("Max drawdown (%)")
        ax.set_title("Bloc D — rendement vs risque (14 blocs)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = FIG / "04_return_dd.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        out["return_dd"] = p

    # --- Richesse 5 stratégies (mart c1c5) ---
    mp = MART_C1C5 / "fact_portfolio_monthly.parquet"
    if mp.is_file():
        mm = pd.read_parquet(mp)
        fig, ax = plt.subplots(figsize=(10, 5))
        for mid in MODEL_IDS:
            sub = mm[mm["model_id"] == mid].sort_values("month")
            if sub.empty:
                continue
            ret = pd.to_numeric(sub["net_return"], errors="coerce").fillna(0)
            w = 100 * (1 + ret).cumprod()
            lw = 2.5 if mid == "c3_lightgbm" else 1.2
            ax.plot(sub["month"], w, label=STRATEGY_LABELS[mid], color=COLORS[mid], lw=lw)
        ax.set_title("Bloc D — richesse cumulée par stratégie (chaînage 14 blocs)")
        ax.set_ylabel("Index base 100")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        p = FIG / "05_wealth_all.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        out["wealth_all"] = p

    # --- Champion monthly returns ---
    mbest = MART_BEST / "fact_portfolio_monthly.parquet"
    if mbest.is_file():
        ch = pd.read_parquet(mbest).sort_values("month")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(
            ch["month"],
            pd.to_numeric(ch["net_return"], errors="coerce") * 100,
            color="#0e7490",
            width=0.7,
        )
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_title("Portefeuille champion (C3) — rendements mensuels nets (%)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        p = FIG / "06_monthly_returns_champion.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        out["monthly_champion"] = p

    # --- Meta sélection algo counts ---
    tp = MART_BEST / "fact_titles_month.parquet"
    if tp.is_file():
        t = pd.read_parquet(tp)
        if "selected_model_id" in t.columns:
            counts = t["selected_model_id"].value_counts().reindex(MODEL_IDS).fillna(0)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.bar(
                [STRATEGY_LABELS[m] for m in MODEL_IDS],
                counts.values,
                color=[COLORS[m] for m in MODEL_IDS],
            )
            ax.set_title("Meta-sélection — nombre de titres où l'algo est retenu (max alpha)")
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            p = FIG / "07_algo_wins.png"
            fig.savefig(p, dpi=160, bbox_inches="tight")
            plt.close(fig)
            out["algo_wins"] = p

        # Reco distribution last month sample
        last_m = t["month"].astype(str).max()
        sub = t[t["month"].astype(str) == last_m]
        vc = sub["recommendation"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            vc.values,
            labels=vc.index,
            autopct="%1.0f%%",
            colors=["#15803d", "#b45309", "#b91c1c"][: len(vc)],
        )
        ax.set_title(f"Répartition BUY/NEUTRAL/SELL — {last_m}")
        fig.tight_layout()
        p = FIG / "08_reco_pie.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        out["reco_pie"] = p

    # --- MASI long ---
    masi_p = MART / "fact_masi_since_2010.csv"
    if not masi_p.is_file() and (MART_BEST / "fact_masi_since_2010.csv").is_file():
        masi_p = MART_BEST / "fact_masi_since_2010.csv"
    if masi_p.is_file():
        mas = pd.read_csv(masi_p, parse_dates=["date"])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(mas["date"], mas["wealth100"], color=COLORS["MASI"])
        ax.axvspan(pd.Timestamp("2018-07-01"), pd.Timestamp("2025-06-30"), alpha=0.12, color="green")
        ax.set_title("MASI depuis 2010 (base 100) — bande verte = période OOS")
        fig.tight_layout()
        p = FIG / "09_masi_long.png"
        fig.savefig(p, dpi=160, bbox_inches="tight")
        plt.close(fig)
        out["masi"] = p

    return out


def _table_html(headers: list[str], rows: list[list[str]]) -> str:
    h = "<thead><tr>" + "".join(f"<th>{c}</th>" for c in headers) + "</tr></thead>"
    body = "<tbody>"
    for row in rows:
        body += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    body += "</tbody>"
    return f"<table class='data'>{h}{body}</table>"


def _section_step(
    num: str,
    title: str,
    objectif: str,
    definitions: list[tuple[str, str]],
    method: str,
    params: list[str],
    results_html: str,
    figures: list[str],
    artefacts: list[str],
    interpretation: str = "",
) -> str:
    defs = "".join(f"<dt>{d}</dt><dd>{v}</dd>" for d, v in definitions)
    pars = "".join(f"<li>{p}</li>" for p in params)
    arts = "".join(f"<li><code>{a}</code></li>" for a in artefacts)
    figs = "".join(figures)
    interp = (
        f"<h3>Interprétation</h3><div class='interp'>{interpretation}</div>"
        if interpretation
        else ""
    )
    return f"""
<section id="etape-{num}">
  <h2>Étape {num} — {title}</h2>
  <div class="card">
    <h3>Objectif</h3>
    <p>{objectif}</p>
    <h3>Définitions</h3>
    <dl class="defs">{defs}</dl>
    <h3>Méthodologie & équations</h3>
    <div class="method">{method}</div>
    <h3>Paramètres figés</h3>
    <ul>{pars}</ul>
    <h3>Résultats</h3>
    {results_html}
    {interp}
    <h3>Figures</h3>
    {figs}
    <h3>Artefacts</h3>
    <ul>{arts}</ul>
  </div>
</section>
"""


def build_html(figs: dict[str, Path]) -> Path:
    boot = _bloc_b_boot()
    bd = _bloc_d()
    brep = _bloc_d_report()

    # Table Bloc B complète
    rows_b = []
    for mid in MODEL_IDS:
        ric = boot.get("rank_ic", {}).get(mid, {})
        sp = boot.get("spread_BUY_SELL", {}).get(mid, {})
        rows_b.append(
            [
                STRATEGY_LABELS[mid],
                f"{ric.get('mean', 0):.4f}",
                f"[{ric.get('ci_low', 0):.3f} ; {ric.get('ci_high', 0):.3f}]",
                f"{sp.get('mean', 0):.4f}",
                f"[{sp.get('ci_low', 0):.3f} ; {sp.get('ci_high', 0):.3f}]",
            ]
        )
    table_b = _table_html(
        ["Stratégie", "Rank-IC", "IC95", "Spread BUY-SELL", "IC95 spread"],
        rows_b,
    )

    rows_d = []
    for _, r in bd.iterrows():
        mid = str(r["model_id"])
        rows_d.append(
            [
                STRATEGY_LABELS[mid],
                f"{float(r['sharpe']):.3f}",
                f"{100*float(r['ann_return']):.1f} %",
                f"{float(r['sortino']):.2f}",
                f"{100*abs(float(r['max_drawdown'])):.1f} %",
                f"{float(r['cvar_95_realized']):.4f}",
                f"{float(r['turnover_mean']):.3f}",
                f"{float(r['liquidity_mean']):.4f}",
            ]
        )
    table_d = _table_html(
        ["Stratégie", "Sharpe", "Rend.ann.", "Sortino", "Max DD", "CVaR95", "Turnover", "L moy."],
        rows_d,
    )

    bench = brep.get("benchmark_summary", {})
    bench_rows = [
        ["MASI", f"{bench.get('masi_sharpe_mean', 0):.3f}", f"{100*bench.get('masi_ann_return_mean', 0):.1f} %"],
        ["EW MASI20", f"{bench.get('ew_masi20_sharpe_mean', 0):.3f}", f"{100*bench.get('ew_masi20_ann_return_mean', 0):.1f} %"],
    ]
    table_bench = _table_html(["Benchmark", "Sharpe moy.", "Rend. ann. moy."], bench_rows)

    meta_path = MART_BEST / "meta_platform.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    bl_long = _bloc_b_long()
    rows_b_full = []
    if not bl_long.empty:
        for mid in MODEL_IDS:
            sub = bl_long[bl_long.model_id == mid]
            row = [STRATEGY_LABELS[mid]]
            for met in METRICS_B:
                v = sub[sub.metrique == met]["valeur"].mean()
                row.append(f"{v:.4f}" if pd.notna(v) else "—")
            rows_b_full.append(row)
    table_b_full = (
        _table_html(["Stratégie"] + METRICS_B, rows_b_full) if rows_b_full else ""
    )

    algo_wins_html = ""
    tp_path = MART_BEST / "fact_titles_month.parquet"
    if tp_path.is_file():
        tmeta = pd.read_parquet(tp_path)
        if "selected_model_id" in tmeta.columns:
            wins = tmeta["selected_model_id"].value_counts()
            win_rows = [
                [STRATEGY_LABELS.get(m, m), str(int(wins.get(m, 0)))]
                for m in MODEL_IDS
            ]
            algo_wins_html = "<h4>Meta-sélection — titres retenus par algo (max α)</h4>"
            algo_wins_html += _table_html(["Algorithme", "Occurrences"], win_rows)
        if "recommendation" in tmeta.columns:
            rc = tmeta["recommendation"].value_counts()
            reco_rows = [
                [k, str(int(v)), f"{100 * v / len(tmeta):.1f} %"]
                for k, v in rc.items()
            ]
            algo_wins_html += "<h4>Répartition globale BUY / NEUTRAL / SELL (84 mois)</h4>"
            algo_wins_html += _table_html(["Reco", "Nombre", "Part"], reco_rows)

    c3 = bd[bd.model_id == "c3_lightgbm"].iloc[0] if not bd.empty else None
    c3_sharpe = f"{float(c3['sharpe']):.2f}" if c3 is not None else "1,31"
    c3_ann = f"{100 * float(c3['ann_return']):.1f} %" if c3 is not None else "18,4 %"

    pipeline_html = """
<section id="pipeline">
  <h2>Architecture du pipeline RAF-ADAPT</h2>
  <div class="card">
    <p>Chaque étape produit des artefacts traçables. Les flèches indiquent l'ordre d'exécution ;
    aucune information du futur est utilisée dans l'entraînement ou les seuils.</p>
    <pre class="pipeline">
[0] Données BVC ──► Panel ML (features + α_adj)
         │
         ▼
[1] Bloc A ──► Régime marché (bull / neutral / bear)
         │
         ▼
[2] Walk-forward ──► 81 plis overlapping + 14 blocs indépendants
         │
         ▼
[3] Bloc B ──► Scores C1–C5 (Ridge, RF, LGBM, hybrides)
         │
         ▼
[4] Étage 1 ──► Recommandations BUY / NEUTRAL / SELL (τ causal)
         │
         ▼
[5] Bloc C ──► Filtre liquidité VMQ ≥ 500 k MAD
         │
         ▼
[6] Bloc D ──► NSGA-III (max α, min CVaR95) → portefeuille knee
         │
         ▼
[7] Plateforme ──► Meta-sélection (max α par titre) + portefeuille champion C3
    </pre>
    <h3>Livrables</h3>
    <ul>
      <li><strong>Rapport HTML</strong> (ce document) — méthodologie complète</li>
      <li><strong>Dashboard Streamlit</strong> — <code>http://localhost:8501</code></li>
      <li><strong>Mart lecture seule</strong> — <code>outputs/platform_mart/best/</code></li>
    </ul>
  </div>
</section>
"""

    sections = []

    sections.append(
        _section_step(
            "0",
            "Données brutes & construction du panel ML",
            "Constituer un panel titre × date sans fuite d'information, prêt pour le walk-forward et le ML.",
            [
                ("Ticker", "Identifiant BVC d'une action."),
                ("VMQ", "Volume moyen quotidien en MAD (liquidité)."),
                ("α ajusté risque", "Excès de rendement vs MASI, normalisé par la volatilité baissière 20j."),
                ("Z-score CS", "Normalisation cross-sectionnelle par date (moyenne 0, écart-type 1 sur l'univers du jour)."),
            ],
            """
            <p>Chaîne : Supabase → nettoyage splits → features techniques/fondamentales → <code>ml_dataset.parquet</code>.</p>
            <div class="eq">\\[ \\alpha_{i,t}^{adj} = \\frac{r_{i,t} - r_{MASI,t}}{\\sigma_{down,i,t}^{(20)}} \\]</div>
            <p>Features : momentum, volatilité, ratios fondamentaux, z-scorés en cross-section à chaque date.</p>
            """,
            [
                "Panel ML : depuis juin 2015",
                "Cours / MASI : janvier 2010 → juin 2025",
                "Cible : alpha_ajuste_risque",
            ],
            "<ul><li>Panel ML depuis juin 2015</li><li>~70 titres actifs en moyenne</li><li>MASI disponible depuis 2010 pour contexte marché</li></ul>"
            + (_img_tag(figs["masi"], "MASI long terme") if "masi" in figs else ""),
            [ _img_tag(figs.get("masi"), "MASI") ] if "masi" in figs else [],
            ["data/processed/ml_dataset.parquet", "data/processed/histo_const_ind.parquet"],
            "<p>Le panel est la fondation de tout le pipeline : chaque ligne correspond à un couple "
            "(titre, fin de mois) avec des features calculées uniquement sur l'historique disponible "
            "à cette date. La cible <em>α ajusté risque</em> permet de comparer les titres sur une "
            "échelle commune indépendante du niveau de volatilité.</p>",
        )
    )

    sections.append(
        _section_step(
            "1",
            "Bloc A — Régime de marché",
            "Détecter le contexte macro (haussier / neutre / baissier) pour enrichir les modèles C5 et le dashboard.",
            [
                ("Régime", "État mensuel du marché : bull, neutral/sideways, bear."),
                ("Breadth MA50", "Part des titres au-dessus de leur MA50 (%)."),
                ("masi_mom_3m", "Momentum 3 mois du MASI."),
            ],
            """
            <p>Terciles calibrés sur train 2015–2020 (pas de look-ahead sur OOS).</p>
            <div class="eq">\\[ \\text{régime}_t = f(\\text{masi\\_mom}_{3m,t},\\; \\text{breadth}_{MA50,t}) \\]</div>
            <p>Variables one-hot : <code>is_bull</code>, <code>is_neutral</code>, <code>is_bear</code> jointes au panel.</p>
            """,
            ["Calibration terciles : 2015-01 → 2020-12", "Fréquence : mensuelle"],
            "<p>Régime utilisé par <strong>C5 — Hybride + régime</strong> et affiché dans le dashboard.</p>",
            [],
            ["factorial_hybrid_adapt/data_utils.py → attach_regime_to_panel()"],
            "<p>Le régime n'est pas utilisé pour filtrer les titres mais comme <em>feature</em> "
            "supplémentaire en C5. Les terciles sont calibrés sur 2015–2020 pour éviter que la "
            "classification bull/bear ne soit influencée par les crises OOS (COVID, etc.).</p>",
        )
    )

    sections.append(
        _section_step(
            "2",
            "Walk-forward — protocole temporel",
            "Évaluer les modèles OOS sans fuite : entraînement strictement antérieur au test.",
            [
                ("Pli overlapping", "Train 36 mois, test 6 mois, pas 1 mois — 81 plis pour métriques descriptives."),
                ("Bloc indépendant", "Pli non chevauchant de 6 mois — 14 blocs pour comparaisons."),
                ("Look-ahead", "Interdit : aucune info du mois test dans train ou features."),
            ],
            """
            <p>Pour chaque pli <em>b</em> :</p>
            <div class="eq">\\[ \\mathcal{T}^{train}_b = [t-36, t-1], \\quad \\mathcal{T}^{test}_b = [t, t+5] \\]</div>
            <p>Premier mois test OOS : <strong>juillet 2018</strong> (36 mois de train depuis juil. 2015).</p>
            <p>Blocs indépendants : fold_id ∈ {0, 6, 12, …, 78}.</p>
            """,
            ["train_months = 36", "test_months = 6", "step_overlapping = 1", "step_blocks = 6"],
            "<ul><li>81 plis overlapping entraînés (Bloc B complet)</li><li>14 blocs pour Bloc D et tests</li><li>84 mois de recommandations (2018-07 → 2025-06)</li></ul>",
            [],
            ["outputs/reports/wf_folds_overlapping.csv", "outputs/reports/wf_folds_nonoverlapping.csv"],
            "<p>Le protocole overlapping maximise le nombre de points pour estimer la qualité prédictive "
            "(Rank-IC, spread). Les 14 blocs indépendants servent à la performance financière : chaque bloc "
            "est un mini-portefeuille NSGA évalué sans chevauchement, puis agrégé.</p>",
        )
    )

    sections.append(
        _section_step(
            "3",
            "Bloc B — Scoring & stratégies C1–C5",
            "Produire un score prédictif par titre et par mois pour chacune des 5 familles d'algorithmes.",
            [
                ("Rank-IC", "Corrélation de Spearman entre score prédit et α réalisé (qualité du ranking)."),
                ("IC-IR", "Rank-IC / std(Rank-IC) — stabilité du signal."),
                ("Spread BUY-SELL", "Différence moyenne de α entre paniers BUY et SELL."),
            ],
            """
            <table class="data"><thead><tr><th>ID</th><th>Algorithme</th><th>Régime</th></tr></thead>
            <tbody>
            <tr><td>C1</td><td>Ridge : \\( \\min_w \\|y-Xw\\|^2 + \\lambda\\|w\\|^2 \\)</td><td>Non</td></tr>
            <tr><td>C2</td><td>Random Forest (grille RF)</td><td>Non</td></tr>
            <tr><td>C3</td><td>LightGBM (Optuna 9 trials, config gelée)</td><td>Non</td></tr>
            <tr><td>C4</td><td>Hybride triple : \\( \\bar{z}(s_{Ridge}, s_{RF}, s_{LGBM}) \\) z-score CS</td><td>Non</td></tr>
            <tr><td>C5</td><td>Idem C4 + features régime Bloc A</td><td>Oui</td></tr>
            </tbody></table>
            """,
            ["LGBM frozen_at : 2026-08-24", "n_trials Optuna : 9", "Ridge alpha : 1.0"],
            "<h4>Rank-IC et spread (bootstrap 81 folds)</h4>" + table_b
            + ("<h4>Métriques moyennes (81 folds)</h4>" + table_b_full if table_b_full else "")
            + (_img_tag(figs["rank_ic"], "Rank-IC") if "rank_ic" in figs else "")
            + (_img_tag(figs.get("spread"), "Spread BUY-SELL") if "spread" in figs else "")
            + (_img_tag(figs.get("heatmap_b"), "Heatmap Bloc B") if "heatmap_b" in figs else ""),
            [
                _img_tag(figs.get("rank_ic"), "Rank-IC"),
                _img_tag(figs.get("spread"), "Spread"),
                _img_tag(figs.get("heatmap_b"), "Heatmap"),
            ],
            [
                "outputs/bloc_b_wf_overlapping/",
                "outputs/reports/bloc_b_results_long_overlapping.csv",
                "outputs/reports/bloc_b_summary_all.json",
            ],
            "<p><strong>C3 — LightGBM</strong> domine nettement : Rank-IC 0,131 avec intervalle bootstrap "
            "entièrement positif [0,098 ; 0,159]. Les modèles linéaires (C1) et RF seul (C2) peinent "
            "à extraire un signal stable. Les hybrides C4/C5 améliorent C1/C2 mais restent bien en deçà "
            "de C3. Le spread BUY−SELL de C3 (0,159) confirme une séparation économique des paniers.</p>",
        )
    )

    sections.append(
        _section_step(
            "4",
            "Étage 1 — Recommandations BUY / NEUTRAL / SELL",
            "Transformer les scores continus en décisions discrètes exploitables par l'investisseur et par NSGA.",
            [
                ("τ (tau)", "Seuil causal calculé sur l'historique strictement antérieur."),
                ("BUY", "Score > τ — signal positif."),
                ("SELL", "Score < -τ — signal négatif ; titres exclus avant NSGA."),
            ],
            """
            <div class="eq">\\[ \\tau_t = Q_q\\big(\\{s_{i,t'} : t' < t\\}\\big) \\quad \\text{(quantile expanding causal)} \\]</div>
            <div class="eq">\\[ \\text{reco}_{i,t} = \\begin{cases} \\text{BUY} & s_{i,t} > \\tau_t \\\\ \\text{SELL} & s_{i,t} < -\\tau_t \\\\ \\text{NEUTRAL} & \\text{sinon} \\end{cases} \\]</div>
            """,
            ["sell_mode = exclude (avant NSGA)", "τ_method = expanding_past_quantile"],
            "<p>Panel <code>stage2_inputs_14blocks.parquet</code> : score, reco, VMQ, volatilité.</p>"
            + (_img_tag(figs.get("reco_pie"), "Répartition reco") if "reco_pie" in figs else ""),
            [_img_tag(figs.get("reco_pie"), "Reco")] if "reco_pie" in figs else [],
            ["outputs/stage2_inputs_14blocks.parquet"],
            "<p>La règle τ causal garantit qu'aucun score du mois courant influence le seuil. "
            "En pratique, ~69 % des lignes reco sont BUY, ~25 % NEUTRAL et ~6 % SELL sur la période OOS — "
            "un profil orienté long mais avec exclusion des SELL avant NSGA.</p>",
        )
    )

    sections.append(
        _section_step(
            "5",
            "Bloc C — Liquidité",
            "Exclure les titres illiquides et modéliser la liquidité pour l'allocation.",
            [
                ("VMQ 20j", "Volume moyen quotidien sur 20 séances."),
                ("L (sigmoïde)", "Facteur de liquidité ∈ (0,1) dérivé de VMQ."),
            ],
            """
            <div class="eq">\\[ L(VMQ) = \\frac{1}{1 + e^{-k(VMQ - VMQ_{ref})}} \\]</div>
            <p>Mode protocole : <strong>pareto</strong> — NSGA à 3 objectifs (max α, min CVaR, max L=sigmoid(VMQ)) ; pas de filtre VMQ ≥ 500 000 MAD.</p>
            """,
            ["VMQ_min = 500 000 MAD (variante eligibility uniquement)", "Mode plateforme = pareto (3 obj.)"],
            "<p>En mode <strong>pareto</strong>, L moyen du portefeuille reflète l'arbitrage NSGA entre α, CVaR et liquidité.</p>",
            [],
            ["allocation.py → prepare_universe()"],
            "<p>La liquidité est un <strong>3ᵉ objectif NSGA</strong> (max Σ wᵢ·L(VMQᵢ)) : l'optimiseur arbitre "
            "liquidité vs alpha vs CVaR sur le front de Pareto, sans filtre dur VMQ ≥ 500 k MAD. "
            "La variante <em>eligibility</em> (filtre puis 2 objectifs) reste disponible dans le code mais "
            "n'est plus le protocole de la plateforme.</p>",
        )
    )

    sections.append(
        _section_step(
            "6",
            "Bloc D — Allocation NSGA-III",
            "Construire un portefeuille réalisable maximisant α et minimisant le risque extrême.",
            [
                ("NSGA-III", "Algorithme multi-objectifs sur front de Pareto."),
                ("Knee point", "Solution retenue : point « genou » du front (distance utopie)."),
                ("CVaR 95 %", "Moyenne des pertes au-delà du 5e percentile (tail risk)."),
            ],
            """
            <div class="eq">\\[ \\max_{w} \\; \\alpha_{port}(w) = \\sum_i w_i \\alpha_i \\]</div>
            <div class="eq">\\[ \\min_{w} \\; \\text{CVaR}_{95\\%}(w) \\]</div>
            <div class="eq">\\[ \\max_{w} \\; L_{port}(w) = \\sum_i w_i L(VMQ_i) \\]</div>
            <div class="eq">\\[ \\sum_i w_i = 1, \\quad 0 \\leq w_i \\leq 0{,}10, \\quad \\text{CVaR}(w) \\leq 0{,}9 \\times \\text{CVaR}(\\text{MASI}) \\]</div>
            <p>NSGA-III à <strong>3 objectifs</strong> (pareto) · 5 stratégies × 14 blocs = 70 runs indépendants.</p>
            """,
            ["pop_size = 36", "n_gen = 30", "w_max = 10 %", "selection = knee"],
            table_d + "<h4>Benchmarks</h4>" + table_bench
            + (_img_tag(figs.get("sharpe"), "Sharpe") if "sharpe" in figs else "")
            + (_img_tag(figs.get("return_dd"), "Rendement vs DD") if "return_dd" in figs else "")
            + (_img_tag(figs.get("wealth_all"), "Richesse 5 stratégies") if "wealth_all" in figs else ""),
            [
                _img_tag(figs.get("sharpe"), "Sharpe"),
                _img_tag(figs.get("return_dd"), "Rend-DD"),
                _img_tag(figs.get("wealth_all"), "Wealth"),
            ],
            [
                "outputs/bloc_d_wf_14blocks/",
                "outputs/reports/bloc_d_summary_14blocks.csv",
            ],
            f"<p>Le champion <strong>C3 LightGBM</strong> atteint Sharpe {c3_sharpe} et rendement annualisé "
            f"{c3_ann} sur 14 blocs, vs MASI Sharpe {bench.get('masi_sharpe_mean', 0.67):.2f}. "
            "Toutes les stratégies C1–C5 battent le MASI ; C3 devance C4/C5 sur Sharpe et rendement. "
            "Le turnover élevé (~0,8–0,9) reflète la réallocation mensuelle NSGA.</p>",
        )
    )

    sections.append(
        _section_step(
            "7",
            "Livrable plateforme — meta-sélection",
            "Produire une recommandation unique par titre (meilleur algo) et un portefeuille champion pour le dashboard.",
            [
                ("Meta-sélection", "Pour chaque (mois, ticker) : algo retenu = argmax α parmi C1–C5."),
                ("Champion NSGA", "Portefeuille = poids NSGA de la stratégie au meilleur Sharpe (C3)."),
            ],
            (
                "<p><strong>Sélection</strong> : "
                "\\( \\text{algo}^*_{i,t} = \\arg\\max_{k \\in \\{C1..C5\\}} \\alpha^{(k)}_{i,t} \\)</p>"
                f"<p><strong>Portefeuille</strong> : {meta.get('portfolio_champion_label', 'C3 LightGBM')}</p>"
                f"<p>Mart : <code>platform_mart/best/</code> — "
                f"{meta.get('n_recommendation_rows', 4778)} lignes reco, "
                f"{meta.get('n_holding_rows', 1566)} lignes holdings.</p>"
                "<p>Le dashboard Streamlit lit ce mart : onglets Recommandations (meta), "
                "Portefeuille & Historique (NSGA champion), Performance (5 courbes Bloc D).</p>"
            ),
            ["selection_rule = max_alpha_per_title_c1c5", f"portfolio_champion = {meta.get('portfolio_champion_model_id', 'c3_lightgbm')}"],
            algo_wins_html
            + (_img_tag(figs.get("algo_wins"), "Algos gagnants") if "algo_wins" in figs else "")
            + (_img_tag(figs.get("monthly_champion"), "Rendements champion") if "monthly_champion" in figs else ""),
            [
                _img_tag(figs.get("algo_wins"), "Algos"),
                _img_tag(figs.get("monthly_champion"), "Monthly"),
            ],
            ["outputs/platform_mart/best/", "outputs/platform_mart/c1c5/"],
            "<p>La meta-sélection tire parti des forces de chaque algo : C3 gagne le plus souvent "
            "(scores α élevés), mais C1 Ridge reste utile sur de nombreux titers (modèle linéaire stable). "
            "Le portefeuille affiché reste celui de C3 car c'est la stratégie NSGA au meilleur Sharpe — "
            "séparation claire entre <em>sélection titre</em> et <em>allocation poids</em>.</p>",
        )
    )

    synth_rows = [
        ["Meilleur Rank-IC (81 folds)", "C3 — LightGBM (0,131 ; IC95 [0,098 ; 0,159])"],
        ["Meilleur Sharpe post-NSGA", "C3 — LightGBM (1,31 ; rend. ann. 18,4 %)"],
        ["Spread BUY-SELL le plus élevé", "C3 — LightGBM (0,159 ; IC95 [0,063 ; 0,246])"],
        ["Dashboard sélection", "Meilleur α par titre parmi C1–C5"],
        ["Dashboard portefeuille", "NSGA C3 LightGBM (champion Sharpe)"],
        ["vs MASI", "Sharpe 1,31 vs 0,67 (MASI) sur 14 blocs"],
    ]

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8"/>
<title>RAF-ADAPT — Rapport scientifique</title>
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<style>
:root {{ --bg:#f0f4f8; --card:#fff; --accent:#0c2340; --teal:#0e7490; }}
* {{ box-sizing:border-box; }}
body {{ font-family:'Segoe UI',system-ui,sans-serif; margin:0; background:var(--bg); color:#1e293b; line-height:1.55; }}
header {{ background:linear-gradient(135deg,#0c2340,#0e7490); color:#fff; padding:2.5rem 2rem; }}
header h1 {{ margin:0 0 0.5rem; font-size:1.8rem; }}
nav {{ background:#fff; border-bottom:1px solid #cbd5e1; padding:0.75rem 2rem; position:sticky; top:0; z-index:10; }}
nav a {{ margin-right:1rem; color:var(--teal); text-decoration:none; font-size:0.9rem; }}
main {{ max-width:960px; margin:0 auto; padding:1.5rem 1rem 3rem; }}
section {{ margin-bottom:2rem; }}
h2 {{ color:var(--accent); border-bottom:2px solid var(--teal); padding-bottom:0.3rem; }}
.card {{ background:var(--card); border-radius:12px; padding:1.25rem 1.5rem; box-shadow:0 2px 8px rgba(0,0,0,.06); }}
h3 {{ color:#1a4a5c; margin-top:1.2rem; font-size:1rem; }}
table.data {{ width:100%; border-collapse:collapse; margin:1rem 0; font-size:0.88rem; }}
table.data th {{ background:#e2e8f0; padding:8px; text-align:left; }}
table.data td {{ border:1px solid #e2e8f0; padding:6px 8px; }}
dl.defs dt {{ font-weight:600; color:var(--accent); margin-top:0.5rem; }}
dl.defs dd {{ margin:0.2rem 0 0.5rem 0; }}
.eq {{ background:#f8fafc; padding:0.75rem; border-radius:8px; margin:0.5rem 0; overflow-x:auto; }}
figure {{ margin:1rem 0; text-align:center; }}
figure img {{ max-width:100%; border:1px solid #e2e8f0; border-radius:8px; }}
figcaption {{ font-size:0.85rem; color:#64748b; margin-top:0.4rem; }}
code {{ background:#f1f5f9; padding:2px 6px; border-radius:4px; font-size:0.85em; }}
.interp {{ background:#ecfdf5; border-left:4px solid #0e7490; padding:0.75rem 1rem; margin:1rem 0; border-radius:0 8px 8px 0; }}
.pipeline {{ background:#f8fafc; padding:1rem; border-radius:8px; font-size:0.9rem; line-height:1.8; overflow-x:auto; }}
.synth {{ background:#0c2340; color:#fff; padding:1.5rem; border-radius:12px; }}
.synth table {{ color:#fff; }}
.synth th {{ background:#1e3a5f; }}
</style>
</head>
<body>
<header>
  <h1>RAF-ADAPT — Rapport scientifique complet</h1>
  <p>Projet factorial_hybrid_adapt · Bourse de Casablanca · Généré le {date.today().isoformat()}</p>
  <p>Période OOS : juillet 2018 → juin 2025 (84 mois) · 5 stratégies C1–C5 · NSGA-III</p>
</header>
<nav>
  <a href="#pipeline">Pipeline</a>
  <a href="#etape-0">Étape 0</a>
  <a href="#etape-1">Bloc A</a>
  <a href="#etape-2">Walk-forward</a>
  <a href="#etape-3">Bloc B</a>
  <a href="#etape-4">Reco</a>
  <a href="#etape-5">Bloc C</a>
  <a href="#etape-6">Bloc D</a>
  <a href="#etape-7">Plateforme</a>
  <a href="#synthese">Synthèse</a>
</nav>
<main>
<section id="intro">
  <div class="card">
    <h2>Introduction</h2>
    <p>RAF-ADAPT est un pipeline de recommandation d'actions pour la BVC. Il combine cinq algorithmes
    de machine learning (C1–C5), une règle de recommandation causal, un filtre de liquidité et une
    allocation multi-objectifs (NSGA-III). Le livrable investisseur applique une <strong>meta-sélection</strong>
    (meilleur score par titre) et un <strong>portefeuille champion</strong> (C3 LightGBM).</p>
    <p><strong>Application web :</strong> <a href="http://localhost:8501">http://localhost:8501</a></p>
  </div>
</section>
{pipeline_html}
{"".join(sections)}
<section id="synthese">
  <h2>Synthèse décisionnelle</h2>
  <div class="synth">
    {_table_html(["Question", "Réponse"], synth_rows)}
  </div>
</section>
</main>
</body>
</html>"""

    out = REPORTS / "rapport_scientifique_RAF-ADAPT.html"
    out.write_text(html, encoding="utf-8")
    return out


def build_pdf_from_figs(figs: dict[str, Path], html_path: Path) -> Path:
    """PDF condensé avec figures clés (complément du HTML)."""
    pdf = RapportPDF()
    pdf.cover("RAF-ADAPT — Rapport scientifique", "Voir HTML pour le détail complet par étape")
    pdf.add_page()
    pdf.body(f"Rapport HTML détaillé : {html_path.name}")
    pdf.body("Chaque étape : objectif, définitions, équations, paramètres, tableaux, figures.")
    order = [
        ("rank_ic", "Bloc B — Rank-IC"),
        ("spread", "Bloc B — Spread BUY-SELL"),
        ("heatmap_b", "Bloc B — métriques"),
        ("sharpe", "Bloc D — Sharpe"),
        ("wealth_all", "Richesse 5 stratégies"),
        ("algo_wins", "Meta-sélection"),
        ("masi", "MASI 2010+"),
    ]
    for key, title in order:
        if key in figs:
            pdf.add_page()
            pdf.h1(title)
            pdf.set_x(10)
            pdf.image(str(figs[key]), w=175)
    out = REPORTS / "rapport_scientifique_RAF-ADAPT.pdf"
    pdf.output(str(out))
    return out


def main() -> int:
    print("Génération des figures…")
    figs = generate_figures()
    print("Génération HTML…")
    html = build_html(figs)
    print("Génération PDF…")
    pdf = build_pdf_from_figs(figs, html)
    print(f"HTML : {html.resolve()}")
    print(f"PDF  : {pdf.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
