"""
Composants UI réutilisables — dashboard Streamlit (design finance BVC).
"""

from __future__ import annotations

import streamlit as st

from bvc_recommender.rebalance import period_display_label


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

        :root {
            --bvc-navy: #0c2340;
            --bvc-teal: #0e7490;
            --bvc-gold: #b45309;
            --bvc-bg: #f4f7fb;
            --bvc-card: #ffffff;
            --bvc-border: #d8e0ea;
            --bvc-muted: #64748b;
            --bvc-green: #15803d;
            --bvc-red: #b91c1c;
        }

        .stApp {
            background: linear-gradient(180deg, #e8eef5 0%, #f4f7fb 28%, #f8fafc 100%);
            font-family: 'IBM Plex Sans', 'DM Sans', sans-serif;
        }

        .main-header {
            font-family: 'DM Sans', sans-serif;
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--bvc-navy);
            letter-spacing: -0.02em;
            margin-bottom: 0.15rem;
        }
        .sub-header {
            color: var(--bvc-muted);
            font-size: 0.95rem;
            margin-bottom: 1.1rem;
        }

        .hero-band {
            background: linear-gradient(120deg, #0c2340 0%, #164e63 55%, #0e7490 100%);
            color: #f8fafc;
            border-radius: 14px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 28px rgba(12, 35, 64, 0.18);
        }
        .hero-band .title { font-size: 1.15rem; font-weight: 700; }
        .hero-band .meta { opacity: 0.85; font-size: 0.88rem; margin-top: 0.2rem; }

        .regime-card {
            background: var(--bvc-card);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            border: 1px solid var(--bvc-border);
            text-align: center;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .regime-title { font-size: 0.78rem; color: var(--bvc-muted); margin-bottom: 0.25rem; text-transform: uppercase; letter-spacing: 0.04em; }
        .regime-value { font-size: 1.45rem; font-weight: 700; color: var(--bvc-navy); }
        .bull { color: var(--bvc-green) !important; }
        .side { color: var(--bvc-gold) !important; }
        .bear { color: var(--bvc-red) !important; }

        .portfolio-card {
            background: var(--bvc-card);
            border: 1px solid var(--bvc-border);
            border-radius: 14px;
            padding: 1rem 1.1rem 0.6rem;
            height: 100%;
            box-shadow: 0 4px 14px rgba(12, 35, 64, 0.06);
        }
        .portfolio-card h3 {
            margin: 0 0 0.55rem 0;
            color: var(--bvc-navy);
            font-size: 1.15rem;
        }
        .portfolio-card.agressif { border-top: 4px solid #b91c1c; }
        .portfolio-card.equilibre { border-top: 4px solid #0e7490; }
        .portfolio-card.defensif { border-top: 4px solid #15803d; }

        .badge-new {
            background: #dcfce7; color: #166534;
            padding: 3px 10px; border-radius: 999px; font-size: 0.72rem;
            font-weight: 600; display: inline-block;
        }
        .badge-out {
            background: #fee2e2; color: #991b1b;
            padding: 3px 10px; border-radius: 999px; font-size: 0.72rem;
            font-weight: 600; display: inline-block;
        }
        .badge-top {
            background: #e0f2fe; color: #075985;
            padding: 3px 10px; border-radius: 999px; font-size: 0.72rem;
            font-weight: 600; display: inline-block;
        }
        .badge-neutre {
            background: #f1f5f9; color: #475569;
            padding: 3px 10px; border-radius: 999px; font-size: 0.72rem;
            font-weight: 600; display: inline-block;
        }
        .badge-bottom {
            background: #ffedd5; color: #9a3412;
            padding: 3px 10px; border-radius: 999px; font-size: 0.72rem;
            font-weight: 600; display: inline-block;
        }

        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--bvc-navy);
            margin: 0.6rem 0 0.75rem;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid var(--bvc-border);
            border-radius: 12px;
            padding: 0.55rem 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(period: str, freq_label: str) -> None:
    st.markdown(
        f"""
        <div class="hero-band">
            <div class="title">BVC Recommender — Casablanca Stock Exchange</div>
            <div class="meta">Période affichée : {period_display_label(period)}
            · Rebalancement {freq_label} · Pipeline Étapes 1–9</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_period_nav(periods: list[str], key_prefix: str = "nav") -> str:
    if not periods:
        st.warning("Aucune période disponible — lancer `run_step8`.")
        return ""

    if "period_idx" not in st.session_state:
        st.session_state.period_idx = len(periods) - 1

    idx = max(0, min(st.session_state.period_idx, len(periods) - 1))

    col_prev, col_label, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("←", key=f"{key_prefix}_prev", use_container_width=True):
            st.session_state.period_idx = max(0, idx - 1)
            st.rerun()
    with col_label:
        st.markdown(
            f"<div style='text-align:center;font-size:1.25rem;font-weight:700;"
            f"padding-top:0.35rem;color:#0c2340'>"
            f"{period_display_label(periods[idx])}</div>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("→", key=f"{key_prefix}_next", use_container_width=True):
            st.session_state.period_idx = min(len(periods) - 1, idx + 1)
            st.rerun()

    st.session_state.period_idx = idx
    return periods[idx]


def render_quarter_nav(quarters: list[str], key_prefix: str = "nav") -> str:
    return render_period_nav(quarters, key_prefix=key_prefix)


def render_regime_gauges(regime: dict) -> None:
    """Affiche l'état de régime discret unique (règle à seuils)."""
    dominant = str(regime.get("dominant") or "")
    if not dominant or dominant == "—":
        # Fallback si timeline one-hot sans champ dominant
        if int(regime.get("is_bull") or 0) == 1:
            dominant = "bull"
        elif int(regime.get("is_bear") or 0) == 1:
            dominant = "bear"
        elif int(regime.get("is_neutral") or regime.get("is_sideways") or 0) == 1:
            dominant = "neutral"
        else:
            dominant = "neutral"

    if dominant == "sideways":
        dominant = "neutral"

    label = {"bull": "Haussier", "bear": "Baissier", "neutral": "Neutre"}.get(
        dominant, dominant
    )
    color = {"bull": "bull", "bear": "bear", "neutral": "side"}.get(dominant, "")

    st.markdown(
        f'<div class="regime-card" style="max-width:320px;padding:1rem 1.25rem">'
        f'<div class="regime-title">État de régime (seuils)</div>'
        f'<div class="regime-value {color}" style="font-size:1.75rem">Régime : {label}</div>'
        f'<div style="margin-top:0.35rem;font-size:0.85rem;opacity:0.75">'
        f'Un seul état actif par mois — règle momentum ET breadth</div></div>',
        unsafe_allow_html=True,
    )


def render_kpis(metrics: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    alpha = metrics.get("alpha_annualized")
    sharpe = metrics.get("sharpe")
    cvar = metrics.get("cvar_95")
    hit = metrics.get("hit_ratio")

    with c1:
        st.metric("Alpha annuel", f"{alpha:.1%}" if alpha is not None else "—")
    with c2:
        st.metric("Sharpe", f"{sharpe:.2f}" if sharpe is not None else "—")
    with c3:
        st.metric("CVaR 95 %", f"{cvar:.2%}" if cvar is not None else "—")
    with c4:
        st.metric("Hit Ratio", f"{hit:.1%}" if hit is not None else "—")


def html_label_badge(label: str) -> str:
    cls = {
        "TOP": "badge-top",
        "NEUTRE": "badge-neutre",
        "BOTTOM": "badge-bottom",
    }.get(str(label).upper(), "badge-neutre")
    return f'<span class="{cls}">{label}</span>'


def html_move_badge(badge: str) -> str:
    if badge == "NOUVEAU":
        return '<span class="badge-new">NOUVEAU</span>'
    if badge == "SORTI":
        return '<span class="badge-out">SORTI</span>'
    return "—"
