"""Onglet 5 — Explication IA (SHAP + régime)."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from bvc_recommender.dashboard.data import get_ranked_universe, get_rebalance_date, get_regime_for_period
from bvc_recommender.dashboard.explain import compute_shap_values
from bvc_recommender.rebalance import period_display_label


def _regime_note(regime: dict) -> str:
    dominant = regime.get("dominant", "sideways")
    labels = {"bull": "Haussier", "bear": "Baissier", "sideways": "Latéral"}
    dom = labels.get(dominant, str(dominant))

    if dominant == "bull":
        advice = (
            "Le modèle favorise les valeurs à fort momentum et beta élevé. "
            "La pénalité de liquidité (sigmoïde) reste continue — aucune coupure binaire."
        )
    elif dominant == "bear":
        advice = (
            "En régime baissier, le filtre de liquidité et le CVaR pèsent davantage. "
            "Les allocations défensives NSGA-III sont privilégiées."
        )
    else:
        advice = (
            "En régime latéral, la diversification et la sélection alpha cross-sectionnelle "
            "dominent la prise de risque directionnelle."
        )

    return f"**Régime : {dom}** (détecteur HMM discret).\n\n{advice}"


def render(period: str) -> None:
    regime = get_regime_for_period(period)
    st.markdown(
        f'<div class="section-title">Explication IA — {period_display_label(period)}</div>',
        unsafe_allow_html=True,
    )
    st.info(_regime_note(regime))

    ranked = get_ranked_universe(period)
    if ranked.empty:
        st.warning("Univers non disponible pour les explications SHAP.")
        return

    tickers = ranked["ticker"].head(20).tolist()
    c1, c2 = st.columns([2, 1])
    with c1:
        ticker = st.selectbox("Valeur à expliquer", tickers, key="shap_ticker")
    with c2:
        row = ranked[ranked["ticker"] == ticker]
        if not row.empty:
            st.metric("Score final", f"{row['score_final'].iloc[0]:.4f}")
            st.caption(f"Label : {row['label'].iloc[0]}")

    as_of = get_rebalance_date(period)
    if as_of is None:
        st.warning("Date de rebalancement introuvable.")
        return

    with st.spinner(f"Calcul SHAP pour {ticker}…"):
        shap_df = compute_shap_values(ticker, as_of)

    if shap_df is None or shap_df.empty:
        st.warning(
            "SHAP indisponible — vérifier `ml_dataset.parquet` et le package `shap`."
        )
        return

    st.markdown(f"#### Facteurs SHAP — **{ticker}**")
    plot_df = shap_df.sort_values("shap_value").copy()
    plot_df["sens"] = plot_df["shap_value"].map(lambda v: "Positif" if v >= 0 else "Négatif")
    fig = px.bar(
        plot_df,
        x="shap_value",
        y="feature",
        orientation="h",
        color="sens",
        color_discrete_map={"Positif": "#15803d", "Négatif": "#b91c1c"},
        labels={"shap_value": "Contribution SHAP", "feature": "Feature"},
        title="Contribution des features à la prédiction d'alpha (LightGBM)",
    )
    fig.update_layout(
        height=500,
        yaxis=dict(categoryorder="total ascending"),
        legend_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,252,0.6)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Valeurs SHAP positives → poussent la prédiction d'alpha à la hausse. "
        "Features z-scorées cross-sectionnellement par date · random_state=42 · "
        "pas de look-ahead (as-of ≤ date de rebalance)."
    )
