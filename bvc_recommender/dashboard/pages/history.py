"""Onglet 4 — Historique (mois cliquables + rotation mensuelle)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from bvc_recommender.dashboard.data import (
    get_ranked_universe,
    get_regime_for_period,
    list_periods,
    load_recommendations,
    load_step8_report,
)
from bvc_recommender.rebalance import period_display_label


def _turnover_between(p1: str, p2: str, portfolio: str = "P_equilibre") -> float:
    """Turnover unilatéral Σ|Δw| / 2 entre deux mois."""
    r1 = load_recommendations(p1)
    r2 = load_recommendations(p2)
    if r1.empty or r2.empty:
        return 0.0
    w1 = dict(
        zip(
            r1.loc[r1["portfolio"] == portfolio, "ticker"],
            r1.loc[r1["portfolio"] == portfolio, "weight"],
        )
    )
    w2 = dict(
        zip(
            r2.loc[r2["portfolio"] == portfolio, "ticker"],
            r2.loc[r2["portfolio"] == portfolio, "weight"],
        )
    )
    tickers = set(w1) | set(w2)
    if not tickers:
        return 0.0
    return 0.5 * sum(abs(w2.get(t, 0.0) - w1.get(t, 0.0)) for t in tickers)


def render(period: str) -> None:
    report = load_step8_report()
    periods_meta = {
        row.get("period", row.get("quarter")): row
        for row in report.get("periods", report.get("quarters", []))
        if row.get("period") or row.get("quarter")
    }
    periods = list_periods()
    metrics = report.get("metrics", {})
    m_eq = metrics.get("P_equilibre", {})

    st.markdown(
        '<div class="section-title">Historique mensuel des recommandations</div>',
        unsafe_allow_html=True,
    )
    st.caption("Cliquez une ligne pour afficher ce mois dans toute l’application.")

    rows = []
    for p in periods:
        meta = periods_meta.get(p, {})
        regime = get_regime_for_period(p)
        ranked = get_ranked_universe(p)
        n_top = int((ranked["label"] == "TOP").sum()) if not ranked.empty else meta.get("n_top")
        n_univ = len(ranked) if not ranked.empty else meta.get("n_ranked")
        rec = load_recommendations(p)
        n_pos = (
            int(rec.loc[rec["portfolio"] == "P_equilibre", "ticker"].nunique())
            if not rec.empty
            else 0
        )
        rows.append(
            {
                "Mois": period_display_label(p),
                "ID": p,
                "Date rebalance": meta.get("rebalance_date", "—"),
                "Univers": n_univ,
                "TOP": n_top,
                "Positions (Équilibré)": n_pos,
                "Régime": {
                    "bull": "Haussier",
                    "bear": "Baissier",
                    "sideways": "Latéral",
                }.get(regime.get("dominant"), regime.get("dominant", "—")),
                "Sélectionné": p == period,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("Aucun mois disponible — lancer `run_step8`.")
        return

    display = df.copy()

    try:
        event = st.dataframe(
            display.drop(columns=["ID"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="history_months_table",
        )
        selected_rows: list[int] = []
        selection = getattr(event, "selection", None)
        if selection is not None:
            selected_rows = list(getattr(selection, "rows", []) or [])
        if selected_rows:
            row_idx = selected_rows[0]
            if 0 <= row_idx < len(df):
                target = df.iloc[row_idx]["ID"]
                if target in periods:
                    st.session_state.period_idx = periods.index(target)
                    st.rerun()
    except TypeError:
        st.dataframe(
            display.drop(columns=["ID"], errors="ignore"),
            hide_index=True,
            use_container_width=True,
        )
        p_sel = st.selectbox(
            "Aller au mois",
            periods,
            index=periods.index(period) if period in periods else len(periods) - 1,
            format_func=period_display_label,
        )
        if st.button("Ouvrir ce mois", key="goto_month"):
            st.session_state.period_idx = periods.index(p_sel)
            st.rerun()

    # Rotation mensuelle
    if len(periods) >= 2:
        rotations = []
        for i in range(1, len(periods)):
            rotations.append(
                {
                    "mois": period_display_label(periods[i]),
                    "turnover_pct": _turnover_between(periods[i - 1], periods[i]) * 100,
                }
            )
        rot_df = pd.DataFrame(rotations)
        fig = px.bar(
            rot_df,
            x="mois",
            y="turnover_pct",
            title="Rotation mensuelle — Portefeuille équilibré (turnover Σ|Δw|/2)",
            labels={"turnover_pct": "Turnover %", "mois": "Mois"},
            color="turnover_pct",
            color_continuous_scale=["#a5f3fc", "#0e7490", "#0c2340"],
        )
        fig.update_layout(
            height=420,
            coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.6)",
            xaxis_tickangle=-45,
        )
        st.plotly_chart(fig, use_container_width=True)

    alpha = m_eq.get("alpha_annualized")
    st.caption(
        f"{len(periods)} mois · Alpha équilibré (backtest global) : "
        f"{alpha:.1%}" if alpha is not None else f"{len(periods)} mois"
    )
