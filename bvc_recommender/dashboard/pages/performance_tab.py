"""Onglet 3 — Performance (courbes cumulées + métriques)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from bvc_recommender.dashboard.data import load_step8_report
from bvc_recommender.dashboard.performance import compute_equity_curves


def render() -> None:
    st.markdown(
        '<div class="section-title">Performance cumulée — Portefeuilles vs MASI / MASI20</div>',
        unsafe_allow_html=True,
    )

    curves = compute_equity_curves()
    if curves.empty:
        st.warning("Courbes de performance indisponibles — lancer `run_step8`.")
        return

    plot_df = curves.reset_index().melt(id_vars="date", var_name="Stratégie", value_name="Valeur")
    color_map = {
        "agressif": "#b91c1c",
        "equilibre": "#0e7490",
        "defensif": "#15803d",
        "MASI": "#0c2340",
        "MASI20": "#b45309",
    }
    fig = px.line(
        plot_df,
        x="date",
        y="Valeur",
        color="Stratégie",
        color_discrete_map=color_map,
        labels={"date": "Date", "Valeur": "Valeur cumulée (base 1)"},
    )
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, title=""),
        height=460,
        margin=dict(t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.6)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0")
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)

    report = load_step8_report()
    metrics = report.get("metrics", {})
    if not metrics:
        return

    rows = []
    for name, m in metrics.items():
        if "error" in m:
            continue
        rows.append(
            {
                "Stratégie": name,
                "Rend. ann.": m.get("annualized_return"),
                "Alpha": m.get("alpha_annualized"),
                "Sharpe": m.get("sharpe"),
                "Sortino": m.get("sortino"),
                "Max DD": m.get("max_drawdown"),
                "Calmar": m.get("calmar"),
                "CVaR 95 %": m.get("cvar_95"),
                "Hit Ratio": m.get("hit_ratio"),
                "Turnover mens.": m.get("turnover_mean_period"),
            }
        )

    df = pd.DataFrame(rows)
    pct_cols = [
        "Rend. ann.",
        "Alpha",
        "Max DD",
        "CVaR 95 %",
        "Hit Ratio",
        "Turnover mens.",
    ]
    display = df.copy()
    for col in pct_cols:
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    for col in ("Sharpe", "Sortino", "Calmar"):
        if col in display.columns:
            display[col] = display[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")

    st.markdown('<div class="section-title">Métriques backtest (2023 → 2025)</div>', unsafe_allow_html=True)
    st.dataframe(display, use_container_width=True, hide_index=True)

    cost = report.get("transaction_cost")
    n_per = len(report.get("periods", []))
    src = report.get("score_source", "—")
    st.caption(
        f"Coût de transaction {cost:.1%} · {n_per} périodes · scores {src} · "
        f"fin {report.get('period', {}).get('end', '—')}"
    )
