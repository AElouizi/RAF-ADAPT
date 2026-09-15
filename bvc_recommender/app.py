"""
BVC Recommender — Dashboard Streamlit (Étape 9).

Lancement :
    streamlit run bvc_recommender/app.py
    py -m bvc_recommender.scripts.run_step9
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bvc_recommender.dashboard.components import (  # noqa: E402
    inject_styles,
    render_hero,
    render_kpis,
    render_period_nav,
    render_regime_gauges,
)
from bvc_recommender.dashboard.data import (  # noqa: E402
    get_period_metrics,
    get_regime_for_period,
    list_periods,
    rebalance_frequency,
)
from bvc_recommender.dashboard.pages import (  # noqa: E402
    benchmarking_tab,
    explanations,
    history,
    performance_tab,
    portfolios,
    recommendations,
)

PORTFOLIO_NAMES = {
    "P_agressif": "Agressif",
    "P_equilibre": "Équilibré",
    "P_defensif": "Défensif",
}

st.set_page_config(
    page_title="BVC Recommender",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.error(
    "**Ancienne version (TFT + 3 portefeuilles Agressif / Équilibré / Défensif).** "
    "Pour le dashboard officiel RAF-ADAPT (C2 Ridge + régime · NSGA-III), lancez "
    "`Lancer_RAF_ADAPT.bat` ou : "
    "`py -m streamlit run experiments/factorial_hybrid_adapt/web/app_c4.py`"
)

inject_styles()

periods = list_periods()
freq = rebalance_frequency()
freq_label = "mensuelle" if freq == "monthly" else "trimestrielle"

with st.sidebar:
    st.markdown("### Paramètres")
    portfolio_kpi = st.selectbox(
        "Portefeuille (KPIs)",
        list(PORTFOLIO_NAMES.keys()),
        format_func=lambda x: PORTFOLIO_NAMES[x],
    )
    st.divider()
    st.markdown("**Contraintes pipeline**")
    st.caption(
        "No look-ahead · carry-forward · z-score cross-sectionnel · "
        "liquidité sigmoïde · random_state=42"
    )
    st.divider()
    st.markdown("**Pipeline**")
    st.caption(f"Rebalancement {freq_label} · Étapes 1–9")

period = render_period_nav(periods, key_prefix="main")
if not period:
    st.stop()

render_hero(period, freq_label)

regime = get_regime_for_period(period)
metrics = get_period_metrics(portfolio_kpi)

st.markdown('<div class="section-title">Régime de marché (discret)</div>', unsafe_allow_html=True)
render_regime_gauges(regime)

st.markdown(
    f'<div class="section-title">Métriques backtest — {PORTFOLIO_NAMES[portfolio_kpi]}</div>',
    unsafe_allow_html=True,
)
render_kpis(metrics)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "① Recommandations",
        "② Portefeuilles",
        "③ Performance",
        "④ Historique",
        "⑤ Explication IA",
        "⑥ Benchmarking",
    ]
)

with tab1:
    recommendations.render(period)

with tab2:
    portfolios.render(period)

with tab3:
    performance_tab.render()

with tab4:
    history.render(period)

with tab5:
    explanations.render(period)

with tab6:
    benchmarking_tab.render()
