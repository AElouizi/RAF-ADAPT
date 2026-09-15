"""
Plateforme web — modèle principal C2 Ridge + régime (NSGA-III knee, w_i ≤ 10 %).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.factorial_hybrid_adapt.web import data_c4 as data  # noqa: E402
import experiments.factorial_hybrid_adapt.bloc_b_models as bloc_b_models  # noqa: E402
importlib.reload(bloc_b_models)  # avant data_c4 (STRATEGY_LABELS C1–C5)
importlib.reload(data)
from experiments.factorial_hybrid_adapt.web import simulation_ui  # noqa: E402
importlib.reload(simulation_ui)
from experiments.factorial_hybrid_adapt.web.admin_agents import render_admin  # noqa: E402
from experiments.factorial_hybrid_adapt.web.simulation_ui import render_simulation  # noqa: E402

st.set_page_config(
    page_title="RAF-ADAPT — Meta sélection · NSGA-III",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

REC_COLORS = {"BUY": "#15803d", "NEUTRAL": "#b45309", "SELL": "#b91c1c"}


def _pick_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Sélectionne les colonnes présentes (mart C1–C5 peut varier selon la version)."""
    return df[[c for c in columns if c in df.columns]].copy()


def _ensure_columns(
    df: pd.DataFrame, columns: list[str], default: float | str | None = None
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = default
    return out

def _wealth_series_config() -> tuple[tuple[str, str, float], ...]:
    """5 stratégies C1–C5 + MASI (trait plus épais sur C3 LightGBM)."""
    out: list[tuple[str, str, float]] = []
    for mid in data.BLOC_D_MODEL_IDS:
        label = data.BLOC_D_LABELS[mid]
        width = 2.8 if mid == data.PRINCIPAL_BLOC_D_MODEL else 1.7
        out.append((label, data.BLOC_D_COLORS[mid], width))
    out.append(("MASI", data.BLOC_D_COLORS["MASI"], 2.2))
    return tuple(out)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@500;700&family=Source+Serif+4:wght@600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
        .stApp {
            background: linear-gradient(165deg, #eef3f8 0%, #f7f5f0 45%, #f4f7fb 100%);
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .hero {
            background: linear-gradient(125deg, #0c2340 0%, #1a4a5c 48%, #0e7490 100%);
            color: #f8fafc;
            border-radius: 16px;
            padding: 1.15rem 1.4rem;
            margin-bottom: 1rem;
        }
        .hero h1 {
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 1.55rem;
            font-weight: 600;
            margin: 0 0 0.25rem 0;
            letter-spacing: -0.01em;
        }
        .hero p { margin: 0; opacity: 0.88; font-size: 0.92rem; }
        .badge {
            display: inline-block;
            padding: 0.15rem 0.55rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }
        .badge-buy { background: #dcfce7; color: #166534; }
        .badge-neutral { background: #ffedd5; color: #9a3412; }
        .badge-sell { background: #fee2e2; color: #991b1b; }
        .section-title {
            font-family: 'DM Sans', sans-serif;
            font-weight: 700;
            color: #0c2340;
            font-size: 1.05rem;
            margin: 0.4rem 0 0.7rem 0;
        }
        div[data-testid="stMetric"] {
            background: #fff;
            border: 1px solid #d8e0ea;
            border-radius: 12px;
            padding: 0.55rem 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fmt_pct(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{100 * float(x):.{digits}f}%"


def fmt_num(x, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{float(x):.{digits}f}"


def reco_badge(rec: str) -> str:
    r = str(rec).upper()
    cls = {"BUY": "badge-buy", "NEUTRAL": "badge-neutral", "SELL": "badge-sell"}.get(
        r, "badge-neutral"
    )
    return f'<span class="badge {cls}">{r}</span>'


def month_label(m: str) -> str:
    try:
        ts = pd.Timestamp(f"{m}-01")
        return ts.strftime("%B %Y").capitalize()
    except Exception:
        return m


inject_css()

meta = data.load_meta()
months = data.list_months()
if not months:
    st.error(
        "Mart C1–C5 introuvable. Exécuter :\n"
        "`py experiments/factorial_hybrid_adapt/seed_c1c5_interim_mart.py` (provisoire)\n"
        "ou `py experiments/factorial_hybrid_adapt/run_c1c5_pipeline.py` (complet)"
    )
    st.stop()

if not months:
    st.error(
        "Mart introuvable. Exécuter :\n"
        "`py experiments/factorial_hybrid_adapt/build_best_algo_platform_mart.py`"
    )
    st.stop()

portfolio_label = meta.get("portfolio_champion_label", data.BLOC_D_LABELS[data.PRINCIPAL_BLOC_D_MODEL])
selection_mode = meta.get("selection_rule", "max_alpha_per_title_c1c5")

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### Livrable investisseur")
    st.info(
        "**Sélection** : meilleur score (alpha) parmi C1–C5, **par titre et par mois**.\n\n"
        f"**Portefeuille** : NSGA-III 3 objectifs (α / CVaR / L) du champion financier ({portfolio_label})."
    )
    st.divider()
    st.markdown("### Calendrier")
    month = st.selectbox(
        "Mois de décision",
        options=months,
        index=0,
        format_func=month_label,
        help="Premier mois = juillet 2018 (début des recommandations walk-forward). 84 mois jusqu'à juin 2025.",
    )
    st.markdown(f"**{len(months)} mois** · {months[0]} → {months[-1]}")
    st.caption("Cours / MASI disponibles dès janvier 2010.")
    st.caption("Juillet 2018 → juin 2025 · 5 algorithmes en comparaison (onglet Performance).")
    st.divider()
    st.markdown("### Filtres titres")
    reco_filter = st.multiselect(
        "Recommandation",
        options=["BUY", "NEUTRAL", "SELL"],
        default=["BUY", "NEUTRAL", "SELL"],
    )
    only_portfolio = st.toggle("Uniquement titres du portefeuille", value=False)
    ticker_q = st.text_input("Rechercher un ticker / nom", value="").strip()
    st.divider()
    st.caption(f"Portefeuille NSGA : {portfolio_label}")
    st.caption("NSGA-III · α / CVaR · knee · w_i ≤ 10 %")
    st.caption("Onglet Administration / Agents : lancer le workflow, logs, statuts")

# ---------- Data slices ----------
titles = data.load_titles()
holdings = data.load_holdings()
monthly = data.load_monthly()
wealth_series = _wealth_series_config()
champion_id = meta.get("portfolio_champion_model_id", data.PRINCIPAL_BLOC_D_MODEL)

t_month = titles[titles["month"] == month].copy()
h_month = holdings[holdings["month"] == month].copy()
m_row = monthly[monthly["month"] == month]
if not m_row.empty:
    m = m_row.iloc[0]
else:
    m = None

if reco_filter:
    t_month = t_month[t_month["recommendation"].isin(reco_filter)]
if only_portfolio:
    t_month = t_month[t_month["in_portfolio"] == 1]
if ticker_q:
    q = ticker_q.lower()
    t_month = t_month[
        t_month["ticker"].str.lower().str.contains(q, na=False)
        | t_month["name"].astype(str).str.lower().str.contains(q, na=False)
    ]

# ---------- Hero ----------
st.markdown(
    f"""
    <div class="hero">
      <h1>RAF-ADAPT — Recommandation BVC</h1>
      <p><b>Version actuelle</b> (pas TFT, pas 3 portefeuilles Agressif/Équilibré/Défensif).</p>
      <p><b>Sélection</b> : meilleur algorithme C1–C5 par titre (score alpha maximal chaque mois).</p>
      <p><b>Portefeuille</b> : <b>{portfolio_label}</b> (champion Sharpe post-NSGA) · comparaison 5 algos en onglet Performance</p>
      <p>Allocation : <b>NSGA-III</b> (α / CVaR / L · knee · wᵢ ≤ 10 %)</p>
      <p>Historique OOS : <b>juillet 2018 → juin 2025 ({meta.get('n_months', 84)} mois)</b> · Mois : <b>{month_label(month)}</b></p>
      <p>Onglets : Recommandations · Portefeuille · Simulation · Historique · Performance C1–C5 · Comparaisons · Marché · Administration</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- KPIs mois ----------
k1, k2, k3, k4, k5, k6 = st.columns(6)
if m is not None:
    k1.metric("Rendement net réalisé", fmt_pct(m.get("net_return")))
    k2.metric("Turnover", fmt_num(m.get("turnover"), 3))
    k3.metric("Liquidité L", fmt_num(m.get("liquidite_ponderee"), 3))
    k4.metric("Richesse (base 100)", fmt_num(m.get("wealth100_net"), 1))
    k5.metric("Positions", int(len(h_month)))
    k6.metric("Portefeuille", portfolio_label.split("—")[0].strip())
else:
    st.warning("Pas de synthèse portefeuille pour ce mois.")

_tab_names = [
    "Recommandations",
    "Portefeuille",
    "Simulation",
    "Historique 2018–2025",
    "Performance",
    "C1–C5 & tests",
    "Marché depuis 2010",
    "Administration / Agents",
]
_tabs = st.tabs(_tab_names)
tab_reco, tab_pf, tab_sim, tab_hist, tab_perf, tab_cmp, tab_mkt, tab_admin = _tabs

# ========== TAB 1 : Recommandations ==========
with tab_reco:
    st.markdown(
        f'<div class="section-title">Recommandations — meilleur algo par titre · {month_label(month)}</div>',
        unsafe_allow_html=True,
    )
    c_a, c_b, c_c, c_d = st.columns(4)
    vc = t_month["recommendation"].value_counts()
    c_a.metric("Titres affichés", len(t_month))
    c_b.metric("BUY", int(vc.get("BUY", 0)))
    c_c.metric("NEUTRAL", int(vc.get("NEUTRAL", 0)))
    c_d.metric("SELL", int(vc.get("SELL", 0)))

    show = _pick_columns(
        t_month,
        [
            "ticker",
            "name",
            "recommendation",
            "alpha",
            "selected_strategy",
            "regime_market",
            "liquidity_L",
            "vmq",
            "in_portfolio",
            "weight_pct",
        ],
    )
    show = show.sort_values(
        ["in_portfolio", "recommendation", "alpha"],
        ascending=[False, True, False],
    )
    show["in_portfolio"] = show["in_portfolio"].map({1: "Oui", 0: "Non"})
    show["weight_pct"] = show["weight_pct"].round(2)
    show["alpha"] = show["alpha"].round(4)
    show["liquidity_L"] = show["liquidity_L"].round(4)
    show["vmq"] = show["vmq"].round(0)
    show = show.rename(
        columns={
            "ticker": "Ticker",
            "name": "Nom",
            "recommendation": "Recommandation",
            "alpha": "Alpha / score",
            "selected_strategy": "Algo sélectionné",
            "regime_market": "Régime",
            "liquidity_L": "Liquidité L",
            "vmq": "VMQ 20j",
            "in_portfolio": "Dans portefeuille",
            "weight_pct": "Poids %",
        }
    )
    st.dataframe(show, use_container_width=True, hide_index=True, height=480)

    # mini chart distribution
    if not t_month.empty:
        fig = px.histogram(
            t_month,
            x="recommendation",
            color="recommendation",
            color_discrete_map=REC_COLORS,
            title=f"Répartition des recommandations — {month_label(month)}",
        )
        fig.update_layout(showlegend=False, height=280, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ========== TAB : Simulation ==========
with tab_sim:
    render_simulation()

# ========== TAB : Portefeuille ==========
with tab_pf:
    st.markdown(
        f'<div class="section-title">Portefeuille NSGA — {portfolio_label} · {month_label(month)}</div>',
        unsafe_allow_html=True,
    )
    if h_month.empty:
        st.info("Aucun portefeuille pour ce mois.")
    else:
        left, right = st.columns([1.35, 1])
        with left:
            pf = _ensure_columns(
                h_month,
                [
                    "ticker",
                    "name",
                    "recommendation",
                    "weight_pct",
                    "alpha",
                    "liquidity_L",
                    "vmq",
                    "regime_market",
                    "realized_return_holding",
                    "contribution_gross",
                ],
            )
            if pf["regime_market"].isna().all() and "regime_market" in t_month.columns:
                reg = t_month[["ticker", "regime_market"]].drop_duplicates("ticker")
                pf = pf.drop(columns=["regime_market"]).merge(reg, on="ticker", how="left")
            pf = pf[
                [
                    "ticker",
                    "name",
                    "recommendation",
                    "weight_pct",
                    "alpha",
                    "liquidity_L",
                    "vmq",
                    "regime_market",
                    "realized_return_holding",
                    "contribution_gross",
                ]
            ].copy()
            pf["weight_pct"] = pf["weight_pct"].round(2)
            pf["alpha"] = pf["alpha"].round(4)
            pf["liquidity_L"] = pf["liquidity_L"].round(4)
            if pf["realized_return_holding"].notna().any():
                pf["realized_return_holding"] = (
                    pd.to_numeric(pf["realized_return_holding"], errors="coerce") * 100
                ).round(2)
            if pf["contribution_gross"].notna().any():
                pf["contribution_gross"] = (
                    pd.to_numeric(pf["contribution_gross"], errors="coerce") * 100
                ).round(3)
            rename_map = {
                "ticker": "Ticker",
                "name": "Nom",
                "recommendation": "Reco",
                "weight_pct": "Poids %",
                "alpha": "Alpha",
                "liquidity_L": "L",
                "vmq": "VMQ",
                "regime_market": "Régime",
                "realized_return_holding": "Rend. réalisé %",
                "contribution_gross": "Contribution pp",
            }
            pf = pf.rename(columns={k: v for k, v in rename_map.items() if k in pf.columns})
            st.dataframe(pf, use_container_width=True, hide_index=True, height=520)
        with right:
            top = h_month.nlargest(12, "weight").copy()
            fig = px.bar(
                top,
                x="weight_pct",
                y="ticker",
                orientation="h",
                color="recommendation",
                color_discrete_map=REC_COLORS,
                title="Top pondérations",
                labels={"weight_pct": "Poids %", "ticker": ""},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
            st.plotly_chart(fig, use_container_width=True)

            if m is not None:
                st.markdown("**Indicateurs du mois**")
                st.write(
                    {
                        "Alpha attendu": fmt_num(m.get("portfolio_alpha")),
                        "CVaR": fmt_num(m.get("portfolio_cvar"), 4),
                        "Liquidité moyenne": fmt_num(m.get("portfolio_liquidity")),
                        "Nb positions": int(m.get("n_positions") or 0),
                        "Turnover": fmt_num(m.get("turnover")),
                        "Rendement brut": fmt_pct(m.get("gross_return")),
                        "Coût transaction": fmt_pct(m.get("transaction_cost"), 3),
                        "Rendement net": fmt_pct(m.get("net_return")),
                    }
                )

# ========== TAB 3 : Historique ==========
with tab_hist:
    st.markdown(
        f'<div class="section-title">Historique mensuel — {portfolio_label}</div>',
        unsafe_allow_html=True,
    )
    hist = monthly.copy()
    if hist.empty:
        st.info("Historique mensuel non disponible pour cette stratégie.")
    else:
        show_cols = [c for c in (
            "month", "turnover", "liquidite_ponderee", "net_return", "wealth100_net"
        ) if c in hist.columns]
        hist_show = hist[show_cols].copy()
        if "net_return" in hist_show.columns:
            hist_show["net_return"] = (
                pd.to_numeric(hist_show["net_return"], errors="coerce") * 100
            ).round(2)
        if "wealth100_net" in hist_show.columns:
            hist_show["wealth100_net"] = pd.to_numeric(
                hist_show["wealth100_net"], errors="coerce"
            ).round(2)
        rename = {
            "month": "Mois",
            "turnover": "Turnover",
            "liquidite_ponderee": "Liquidité L",
            "net_return": "Net %",
            "wealth100_net": "Richesse 100",
        }
        hist_show = hist_show.rename(columns={k: v for k, v in rename.items() if k in hist_show.columns})
        st.dataframe(hist_show, use_container_width=True, hide_index=True, height=420)

        if "net_return" in hist.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=hist["month"],
                    y=pd.to_numeric(hist["net_return"], errors="coerce") * 100,
                    name="Rendement net %",
                    marker_color=data.BLOC_D_COLORS.get(champion_id, "#0e7490"),
                )
            )
            fig.update_layout(
                title=f"Rendements mensuels nets — {portfolio_label}",
                yaxis_title="%",
                height=320,
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Détail du mois sélectionné (composition)**")
    st.caption(f"{month_label(month)} — {len(h_month)} positions")
    if not h_month.empty:
        st.dataframe(
            h_month[["ticker", "name", "recommendation", "weight_pct", "alpha"]]
            .assign(weight_pct=lambda d: d["weight_pct"].round(2), alpha=lambda d: d["alpha"].round(4))
            .rename(
                columns={
                    "ticker": "Ticker",
                    "name": "Nom",
                    "recommendation": "Reco",
                    "weight_pct": "Poids %",
                    "alpha": "Alpha",
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=300,
        )

# ========== TAB 4 : Performance (C1–C5) ==========
with tab_perf:
    st.markdown(
        '<div class="section-title">Performance — stratégies C1–C5 et MASI</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "C1 Ridge · C2 RF · C3 LightGBM · C4 Ridge+RF+LGBM · C5 Ridge+RF+LGBM+régime. "
        "Sharpe / rendement : moyenne 14 blocs. Richesse : courbes chaînées post-NSGA."
    )
    kpi6 = data.load_bloc_d_kpi_table()
    if not kpi6.empty:
        showc = kpi6.copy()
        pct_cols = ["Rendement", "Max_DD", "CVaR", "Turnover", "Liquidite"]
        for c in pct_cols:
            if c in showc.columns:
                showc[c] = (pd.to_numeric(showc[c], errors="coerce") * 100).round(2)
        st.dataframe(
            showc.drop(columns=["model_id"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )
        n_metrics = min(6, len(kpi6))
        cols = st.columns(n_metrics)
        for i, (_, row) in enumerate(kpi6.head(n_metrics).iterrows()):
            with cols[i]:
                st.metric(str(row["Strategie"]), fmt_pct(row["Rendement"]))
                st.caption(
                    f"Sharpe {fmt_num(row['Sharpe'], 3)} · richesse {fmt_num(row['wealth_final_100'], 1)}"
                )
    else:
        st.warning(
            "Résultats Bloc D introuvables. Exécuter "
            "`py -m experiments.factorial_hybrid_adapt.run_bloc_d_eval --run`."
        )

    sharpe_chart = data.load_bloc_d_sharpe_chart()
    if not sharpe_chart.empty:
        fig_sh = go.Figure(
            data=[
                go.Bar(
                    x=sharpe_chart["Strategie"],
                    y=sharpe_chart["Sharpe"],
                    marker_color=sharpe_chart["color"],
                    text=[f"{v:.3f}" for v in sharpe_chart["Sharpe"]],
                    textposition="outside",
                    hovertemplate="%{x}<br>Sharpe %{y:.3f}<extra></extra>",
                )
            ]
        )
        fig_sh.update_layout(
            title="Sharpe moyen (14 blocs) — C1–C5 vs MASI",
            yaxis_title="Sharpe",
            height=380,
            margin=dict(t=60, b=40),
            yaxis=dict(range=[0, float(sharpe_chart["Sharpe"].max()) * 1.18]),
            showlegend=False,
        )
        st.plotly_chart(fig_sh, use_container_width=True)
        st.caption(
            "Critère de sélection Bloc D : Sharpe moyen sur les 14 blocs walk-forward OOS. "
            "MASI ≈ 0,67 — toutes les stratégies C1–C5 le dépassent."
        )

    wealth_bloc_d = data.load_bloc_d_daily_wealth()
    if wealth_bloc_d.empty:
        st.info("Courbes de richesse Bloc D non disponibles (poids NSGA manquants).")
    else:
        fig = go.Figure()
        for col, color, width in wealth_series:
            if col in wealth_bloc_d.columns:
                dash = "dash" if col == "MASI" else "solid"
                fig.add_trace(
                    go.Scatter(
                        x=wealth_bloc_d["date"],
                        y=wealth_bloc_d[col],
                        name=col,
                        line=dict(color=color, width=width, dash=dash),
                    )
                )
        fig.update_layout(
            title="Richesse cumulée — C1–C5 vs MASI (C3 = trait plus épais)",
            yaxis_title="Index (base 100)",
            height=420,
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        for col, color, width in wealth_series:
            dcol = f"dd_{col}"
            if dcol in wealth_bloc_d.columns:
                dash = "dash" if col == "MASI" else "solid"
                fig2.add_trace(
                    go.Scatter(
                        x=wealth_bloc_d["date"],
                        y=wealth_bloc_d[dcol] * 100,
                        name=col,
                        line=dict(color=color, width=width, dash=dash),
                    )
                )
        fig2.update_layout(
            title="Drawdowns — C1–C5 vs MASI (%)",
            yaxis_title="%",
            height=320,
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig2, use_container_width=True)

# ========== TAB 5 : C1–C5 & tests (+ legacy) ==========
with tab_cmp:
    st.markdown(
        '<div class="section-title">Comparaison — stratégies C1–C5 et legacy factoriel</div>',
        unsafe_allow_html=True,
    )
    summary6 = data.load_bloc_d_summary()
    if not summary6.empty:
        st.markdown("**Bloc D — Sharpe moyen (14 blocs) — C1–C5**")
        show_s = summary6.copy()
        show_s["model_label"] = show_s["model_id"].map(data.BLOC_D_LABELS)
        show_s["ann_return"] = (show_s["ann_return"] * 100).round(2)
        show_s["max_drawdown"] = (show_s["max_drawdown"] * 100).round(2)
        st.dataframe(
            show_s[
                [
                    "model_label",
                    "groupe",
                    "sharpe",
                    "ann_return",
                    "sortino",
                    "max_drawdown",
                    "turnover_mean",
                    "liquidity_mean",
                    "n_blocks",
                ]
            ].rename(
                columns={
                    "model_label": "Modèle",
                    "groupe": "Groupe",
                    "sharpe": "Sharpe",
                    "ann_return": "Rend. ann. %",
                    "sortino": "Sortino",
                    "max_drawdown": "Max DD %",
                    "turnover_mean": "Turnover",
                    "liquidity_mean": "Liquidité L",
                    "n_blocks": "Blocs",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Benchmarks Bloc D (moyenne 14 blocs) : MASI Sharpe 0,67 · EW MASI20 Sharpe 0,99. "
            "Tous les modèles NSGA les battent."
        )

    with st.expander("Legacy — plan factoriel C1–C4 (livraison initiale)", expanded=False):
        st.caption(
            "Pas de filtre VMQ ≥ 500k. Liquidité = objectif Pareto uniquement. "
            "Significativité legacy : écarts non significatifs à 5 % vs MASI."
        )
        cells = data.load_kpi_cells()
        if not cells.empty:
            showc = cells.copy()
            pct_cols = ["Rendement_total", "Rendement", "Volatilite", "Max_DD", "pct_mois_positifs", "ret_mensuel_moyen"]
            for c in pct_cols:
                if c in showc.columns:
                    showc[c] = (pd.to_numeric(showc[c], errors="coerce") * 100).round(2)
            st.dataframe(showc, use_container_width=True, hide_index=True)
        tests = data.load_tests()
        if not tests.empty:
            st.markdown("**Tests statistiques (C1–C4)**")
            st.dataframe(tests, use_container_width=True, hide_index=True)
        subp = data.load_subperiods()
        if not subp.empty:
            st.markdown("**Sous-périodes C1–C4**")
            st.dataframe(subp, use_container_width=True, hide_index=True)
        liq = data.load_liquidity_robustness()
        if not liq.empty:
            st.markdown("**Robustesse liquidité — Version A vs Version B**")
            st.dataframe(liq, use_container_width=True, hide_index=True)

with tab_mkt:
    st.markdown(
        '<div class="section-title">Marché BVC depuis 2010 — données historiques brutes</div>',
        unsafe_allow_html=True,
    )
    st.info(
        "Les cours et le MASI existent dès janvier 2010. "
        "Les recommandations BUY/NEUTRAL/SELL et les portefeuilles C1–C4 commencent en juillet 2018 : "
        "le dataset d'apprentissage ne démarre qu'en juin 2015 (indicateurs fondamentaux), "
        "puis 36 mois de train walk-forward. "
        "On n'invente pas de scores C1–C4 pour 2010–2015."
    )
    masi_long = data.load_masi_long()
    if masi_long.empty:
        st.warning("Série MASI 2010 manquante dans le mart.")
    else:
        figm = go.Figure()
        figm.add_trace(
            go.Scatter(
                x=masi_long["date"],
                y=masi_long["wealth100"],
                name="MASI (base 100 au 2010-01)",
                line=dict(color="#1d4ed8", width=2),
            )
        )
        figm.add_vrect(
            x0="2018-07-01",
            x1="2025-06-30",
            fillcolor="green",
            opacity=0.08,
            line_width=0,
            annotation_text="Fenêtre OOS C1–C5",
        )
        figm.update_layout(
            title="MASI depuis 2010 (base 100) — bande verte = période OOS Bloc D / recommandations",
            yaxis_title="Index",
            height=380,
        )
        st.plotly_chart(figm, use_container_width=True)
        st.caption(
            f"Premier point : {masi_long['date'].min().date()} · "
            f"Dernier point : {masi_long['date'].max().date()} · "
            f"{len(masi_long)} séances"
        )

with tab_admin:
    render_admin()

st.caption(
    "Plateforme RAF-ADAPT · meta sélection C1–C5 + NSGA champion · juil. 2018 → juin 2025 · lecture seule."
)
