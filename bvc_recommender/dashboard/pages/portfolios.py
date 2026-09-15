"""Onglet 2 — Portefeuilles (3 cartes Agressif / Équilibré / Défensif)."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from bvc_recommender.dashboard.data import PORTFOLIO_LABELS, get_period_metrics, load_recommendations
from bvc_recommender.rebalance import period_display_label

CARD_CLASS = {
    "P_agressif": "agressif",
    "P_equilibre": "equilibre",
    "P_defensif": "defensif",
}

COLORS = {
    "P_agressif": px.colors.sequential.Reds_r,
    "P_equilibre": px.colors.sequential.Teal,
    "P_defensif": px.colors.sequential.Greens_r,
}


def render(period: str) -> None:
    rec = load_recommendations(period)
    if rec.empty:
        st.warning(f"Aucune recommandation pour {period_display_label(period)}.")
        return

    st.markdown(
        f'<div class="section-title">Allocation NSGA-III — {period_display_label(period)}</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for i, (key, label) in enumerate(PORTFOLIO_LABELS.items()):
        sub = rec[rec["portfolio"] == key].sort_values("weight", ascending=False)
        metrics = get_period_metrics(key)

        with cols[i]:
            st.markdown(
                f'<div class="portfolio-card {CARD_CLASS[key]}"><h3>{label}</h3></div>',
                unsafe_allow_html=True,
            )

            m1, m2 = st.columns(2)
            m1.metric("Positions", len(sub) if not sub.empty else 0)
            if not sub.empty:
                m2.metric("Poids max", f"{sub['weight_pct'].max():.1f}%")
            else:
                m2.metric("Poids max", "—")

            ann = metrics.get("annualized_return")
            sharpe = metrics.get("sharpe")
            cvar = metrics.get("cvar_95")
            alpha = metrics.get("alpha_annualized")
            st.caption(
                f"Rend. ann. {ann:.1%} · Alpha {alpha:.1%} · "
                f"Sharpe {sharpe if sharpe is not None else '—'} · "
                f"CVaR {cvar:.2%}"
                if ann is not None and cvar is not None
                else "Métriques backtest indisponibles"
            )

            if sub.empty:
                st.info("Pas de positions.")
                continue

            fig = px.pie(
                sub.head(12),
                values="weight_pct",
                names="ticker",
                hole=0.42,
                color_discrete_sequence=COLORS[key],
            )
            fig.update_traces(textposition="inside", textinfo="percent+label", insidetextorientation="radial")
            fig.update_layout(
                margin=dict(t=10, b=0, l=0, r=0),
                height=300,
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True, key=f"pie_{key}_{period}")

            with st.expander(f"Détail des poids — {label}"):
                st.dataframe(
                    sub[["ticker", "weight_pct"]].rename(
                        columns={"ticker": "Ticker", "weight_pct": "Poids %"}
                    ),
                    hide_index=True,
                    use_container_width=True,
                    height=280,
                )
